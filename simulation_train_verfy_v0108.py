import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import time

# ==============================================================================
# 0. 全局配置与工具函数
# ==============================================================================
# 锁定随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
# 自动选择计算设备 (GPU优先)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def numpy_to_torch(arr, device):
    """辅助函数：将numpy数组转为PyTorch张量，并调整为 [Batch, Channel, Time] 格式"""
    return torch.FloatTensor(arr).unsqueeze(1).to(device)

# ==============================================================================
# 1. 物理仿真引擎 (TIADC Simulator)
# ==============================================================================
class TIADCSimulator:
    """
    模拟真实的 TIADC 物理通道特性。
    包含：模拟带宽限制、非线性失真、增益误差、采样时间偏斜。
    """
    def __init__(self, fs=20e9):
        self.fs = fs

    def fractional_delay(self, sig, delay):
        """
        基于 FFT 的精确分数延时 (Fractional Delay)。
        原理：时域延时 <-> 频域线性相移
        """
        N = len(sig)
        X = np.fft.rfft(sig)
        k = np.arange(len(X))
        # 相位旋转因子: e^(-j * 2pi * k * delay / N)
        phase_shift = np.exp(-1j * 2 * np.pi * k * delay / N)
        return np.fft.irfft(X * phase_shift, n=N)

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0, hd3=0.0):
        """
        核心仿真函数：模拟单通道的物理退化过程。
        
        参数:
            sig: 输入信号
            cutoff_freq: 模拟带宽 (Hz)
            delay_samples: 采样延时 (Samples)
            gain: 增益
            hd3: 三次谐波失真系数 (模拟非线性)
        
        顺序很重要（模拟物理信号流）：
        Input -> Nonlinearity(Amp) -> Bandwidth(Filter) -> Gain -> Timing Skew -> Noise
        注意：带宽限制位于非线性之后，这会导致“记忆效应”，单纯的多项式无法校正。
        """
        nyquist = self.fs / 2
        
        # 1. 非线性失真 (Input Buffer Stage)
        # y = x + k * x^3
        sig_nonlinear = sig + hd3 * (sig**3)
        
        # 2. 模拟带宽限制 (Analog Bandwidth)
        # 使用 5 阶巴特沃斯低通滤波器模拟滚降
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig_nonlinear)
        
        # 3. 增益误差 (Gain Error)
        sig_gain = sig_bw * gain
        
        # 4. 时间偏斜 (Timing Skew)
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 5. 热噪声 (Thermal Noise)
        noise = np.random.normal(0, 1e-5, len(sig_delayed))
        
        return sig_delayed + noise

    def prepare_training_batch(self, N=32768):
        """
        生成包含丰富频率成分的训练数据。
        只有见过各种频率，模型才能学会全频带校正。
        """
        t = np.arange(N) / self.fs
        
        # A. 线性 Chirp (DC -> Nyquist)
        chirp = signal.chirp(t, f0=1e6, t1=t[-1], f1=self.fs*0.48, method='linear')
        
        # B. 阶跃信号 (Step) - 用于训练瞬态响应
        step = np.zeros(N)
        step[:N//2] = 0.5
        step[N//2:] = -0.5
        
        # C. 宽带粉红噪声 (Pink Noise) - 填补频谱缝隙
        white = np.random.randn(N)
        b, a = signal.butter(1, 0.05, btype='low') # 简单的有色噪声
        pink = signal.lfilter(b, a, white)
        pink = pink / np.std(pink) * 0.5
        
        # 拼接成一个 Batch
        return np.stack([chirp, step, pink], axis=0)

    def generate_tone(self, freq, N=8192):
        """生成测试单音信号"""
        t = np.arange(N) / self.fs
        return np.sin(2 * np.pi * freq * t) * 0.9

# ==============================================================================
# 2. 神经网络模型组件 (Differentiable DSP Modules)
# ==============================================================================

class DifferentiableDelay(nn.Module):
    """
    可微分延时层。
    通过学习参数 'delay'，在频域对信号进行精确的时移校正。
    """
    def __init__(self, init_delay=0.0):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor([init_delay]))

    def forward(self, x):
        # x shape: [Batch, 1, Time]
        N = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        k = torch.arange(X.shape[-1], device=x.device)
        # 相位因子: exp(-j * 2pi * k * d / N)
        phase = torch.exp(-1j * 2 * np.pi * k * self.delay / N)
        return torch.fft.irfft(X * phase, n=N, dim=-1)

class VolterraCorrection(nn.Module):
    """
    【升级版】Volterra 非线性校正层。
    相比简单的多项式 (PolyCorrection)，它增加了卷积核 (FIR)。
    原理：y = x - FIR(x^3)
    优势：可以处理带记忆的非线性 (Memory Effect)，即校正不同频率的不同延时失真。
    """
    def __init__(self, taps=31):
        super().__init__()
        # 专门用于处理 x^3 项的滤波器
        # taps 不需要很大，通常记忆效应只持续几十个采样点
        self.conv3 = nn.Conv1d(1, 1, kernel_size=taps, padding='same', bias=False)
        
        # 初始化为0，让模型从“无校正”状态开始学习
        with torch.no_grad():
            self.conv3.weight.zero_()

    def forward(self, x):
        # 1. 构建非线性基函数 (Basis Function)
        x_cubed = x ** 3
        
        # 2. 通过 FIR 调整非线性项的相位和幅度 (Memory Compensation)
        correction = self.conv3(x_cubed)
        
        # 3. 消除失真
        return x - correction

class HybridCalibrationModel(nn.Module):
    """
    最终的混合校正模型架构 (V5)。
    链路顺序：线性逆滤波(BW) -> 延时(Delay) -> 增益(Gain) -> 非线性(Volterra)
    """
    def __init__(self, lin_taps=511, nonlin_taps=31):
        super().__init__()
        
        # 1. 线性 FIR: 负责频响均衡 (Equalization) 和去嵌入
        # 它是解决“中间凹陷”的主力
        self.linear_fir = nn.Conv1d(1, 1, kernel_size=lin_taps, padding='same', bias=False)
        
        # 2. 全局延时: 负责对齐 Timing Skew
        self.global_delay = DifferentiableDelay(init_delay=0.0)
        
        # 3. 增益: 负责幅度归一化
        self.gain = nn.Parameter(torch.tensor([1.0]))
        
        # 4. Volterra: 负责消除谐波 (HD3)
        self.volterra = VolterraCorrection(taps=nonlin_taps)
        
        # 初始化线性 FIR 为直通 (Identity)，稍后会被物理初始化覆盖
        with torch.no_grad():
            self.linear_fir.weight.zero_()
            self.linear_fir.weight[0, 0, lin_taps // 2] = 1.0

    def forward(self, x):
        # Step 1: 恢复带宽限制 (Inverse Filtering)
        x = self.linear_fir(x)
        
        # Step 2: 对齐时间 (Timing Alignment)
        x = self.global_delay(x)
        
        # Step 3: 对齐幅度 (Gain Alignment)
        x = x * self.gain
        
        # Step 4: 消除非线性 (Nonlinear Cancellation)
        # 注意：要在波形被线性层修复得差不多之后再做非线性消除，效果最好
        x = self.volterra(x)
        
        return x

# ==============================================================================
# 3. 物理感知初始化 (Physics-Aware Initialization) - 核心技术
# ==============================================================================
def physics_aware_initialization(model, fs, fc_ref, fc_bad, taps):
    """
    计算理想逆滤波器系数，并赋值给模型的 linear_fir 层。
    这是解决“频响凹陷”和“训练不收敛”的杀手锏。
    
    原理: H_ideal(f) = H_ref(f) / H_bad(f)
    """
    print(f"[{time.strftime('%H:%M:%S')}] 正在计算物理逆滤波器 (Ref={fc_ref/1e9}G, Bad={fc_bad/1e9}G)...")
    
    nyquist = fs / 2
    # 1. 获取物理通道的滤波器系数 (巴特沃斯)
    b_ref, a_ref = signal.butter(5, fc_ref / nyquist, btype='low')
    b_bad, a_bad = signal.butter(5, fc_bad / nyquist, btype='low')
    
    # 2. 计算复数频响
    w, h_ref = signal.freqz(b_ref, a_ref, worN=4096)
    _, h_bad = signal.freqz(b_bad, a_bad, worN=4096)
    
    # 3. 计算补偿频响 (除法)
    # 添加 epsilon 防止除以零
    h_comp = h_ref / (h_bad + 1e-12)
    
    # 4. 限制增益 (防止在高频阻带过度放大噪声)
    MAX_BOOST_DB = 20
    max_gain = 10**(MAX_BOOST_DB/20)
    h_abs = np.abs(h_comp)
    h_phase = np.angle(h_comp)
    h_abs = np.clip(h_abs, 0, max_gain) # 限幅
    h_comp_clipped = h_abs * np.exp(1j * h_phase)
    
    # 5. IFFT 转回时域得到 FIR 系数
    # 构造双边谱以获得实数脉冲响应
    h_full = np.concatenate([h_comp_clipped, np.conj(h_comp_clipped[-2:0:-1])])
    impulse = np.real(np.fft.ifft(h_full))
    
    # 6. 截取中间部分并加窗
    # 将脉冲中心移到 buffer 中心
    impulse = np.roll(impulse, len(impulse)//2)
    start = len(impulse)//2 - taps//2
    end = start + taps
    fir_coeffs = impulse[start:end]
    
    # 加 Blackman 窗平滑截断效应
    fir_coeffs = fir_coeffs * np.blackman(taps)
    
    # 7. 赋值给 PyTorch 模型
    with torch.no_grad():
        weight_tensor = torch.from_numpy(fir_coeffs).float().view(1, 1, -1).to(device)
        model.linear_fir.weight.data.copy_(weight_tensor)
        print(f"[{time.strftime('%H:%M:%S')}] 物理初始化完成！模型已具备 5.9G->6.0G 的补偿能力。")

# ==============================================================================
# 4. 训练流程
# ==============================================================================
def train_model(sim):
    # --- 配置物理场景 ---
    # 通道0 (参考): 6.0GHz 带宽, 无失配, 无非线性 (理想目标)
    REF_BW = 6.0e9
    
    # 通道1 (待校正): 5.8GHz 带宽, 0.42 延时, 0.98 增益, 1% 三次谐波
    # 注意：这次我们要求模型消除 HD3，所以 Target 不加 HD3
    BAD_BW = 5.8e9
    BAD_DELAY = 0.42
    BAD_GAIN = 0.98
    BAD_HD3 = 0.01 
    
    # 准备数据
    raw_batch = sim.prepare_training_batch(N=16384)
    input_list = []
    target_list = []
    
    for i in range(raw_batch.shape[0]):
        sig = raw_batch[i]
        # Target: 理想通道 (模型要学会变成这样)
        target = sim.apply_channel_effect(sig, cutoff_freq=REF_BW, delay_samples=0.0, gain=1.0, hd3=0.0)
        # Input: 实际恶劣通道
        input_bad = sim.apply_channel_effect(sig, cutoff_freq=BAD_BW, delay_samples=BAD_DELAY, gain=BAD_GAIN, hd3=BAD_HD3)
        
        target_list.append(target)
        input_list.append(input_bad)
        
    in_t = numpy_to_torch(np.array(input_list), device)
    tg_t = numpy_to_torch(np.array(target_list), device)
    
    # 归一化幅度
    scale = torch.max(torch.abs(tg_t)).item()
    in_t /= scale
    tg_t /= scale
    
    # 实例化模型
    # lin_taps=511 (处理复杂的频响失配)
    # nonlin_taps=31 (处理 Volterra 记忆效应)
    model = HybridCalibrationModel(lin_taps=511, nonlin_taps=31).to(device)
    
    # --- 关键步骤：物理初始化 ---
    physics_aware_initialization(model, sim.fs, fc_ref=REF_BW, fc_bad=BAD_BW, taps=511)
    
    # Loss 函数 (简单的 MSE + 频域 Loss)
    mse_loss = nn.MSELoss()
    
    # 优化器配置
    # 技巧：给不同的层不同的学习率
    optimizer = optim.Adam([
        {'params': model.linear_fir.parameters(), 'lr': 1e-4}, # 已经初始化很好了，微调即可
        {'params': model.global_delay.parameters(), 'lr': 1e-2}, # Delay 需要快速收敛
        {'params': model.gain, 'lr': 1e-3},
        {'params': model.volterra.parameters(), 'lr': 1e-3} # 非线性层从零开始学
    ])
    
    # 学习率衰减
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000], gamma=0.1)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始训练 (3000 Epochs)...")
    loss_history = []
    
    for epoch in range(3001):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(in_t)
        
        # 计算 Loss
        # 1. 时域对齐
        l_time = mse_loss(pred, tg_t)
        # 2. 频域幅度对齐 (Log Domain)
        fft_p = torch.fft.rfft(pred, dim=-1).abs()
        fft_t = torch.fft.rfft(tg_t, dim=-1).abs()
        l_freq = torch.mean(torch.abs(torch.log10(fft_p+1e-8) - torch.log10(fft_t+1e-8)))
        
        total_loss = 100.0 * l_time + 10.0 * l_freq
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(total_loss.item())
        
        if epoch % 500 == 0:
            d = model.global_delay.delay.item()
            g = model.gain.item()
            # 这里的 Delay 目标应该是 -0.42，Gain 目标应该是 ~1.02
            print(f"Ep {epoch:04d} | Loss: {total_loss.item():.5f} | Delay: {d:.4f} | Gain: {g:.4f}")
            
    return model, scale, loss_history

# ==============================================================================
# 5. 验证与丰富可视化
# ==============================================================================
def verify_and_plot(sim, model, scale):
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始验证与绘图...")
    model.eval()
    
    # --- 场景设定 ---
    # 为了展示非线性消除效果，我们选一个较低的频率，让其3次谐波(HD3)落在带内
    # 比如 f_in = 1.0 GHz -> HD3 = 3.0 GHz (在 5.8G 带宽内)
    test_freq = 1.0e9 
    
    # 生成测试信号
    # 1. 理想参考 (Ref)
    src_clean = sim.generate_tone(test_freq, N=8192)
    ref_sig = sim.apply_channel_effect(src_clean, 6.0e9, 0.0, 1.0, hd3=0.0)
    
    # 2. 原始恶劣信号 (Bad)
    bad_sig = sim.apply_channel_effect(src_clean, 5.8e9, 0.42, 0.98, hd3=0.01)
    
    # 3. 模型校正 (Calibrated)
    with torch.no_grad():
        inp_t = numpy_to_torch(bad_sig/scale, device).reshape(1, 1, -1)
        cal_sig = model(inp_t).cpu().numpy().flatten() * scale
    
    # --- 绘图准备 ---
    # 为了频谱图清晰，加窗
    win = np.blackman(len(ref_sig))
    
    def get_spec(sig):
        spec = np.abs(np.fft.rfft(sig * win))
        spec_db = 20 * np.log10(spec + 1e-12)
        spec_db -= np.max(spec_db) # 归一化
        return spec_db
    
    spec_ref = get_spec(ref_sig)
    spec_bad = get_spec(bad_sig)
    spec_cal = get_spec(cal_sig)
    freqs = np.linspace(0, sim.fs/2, len(spec_ref))
    
    # --- 组合图表 ---
    plt.figure(figsize=(15, 10))
    
    # 1. 时域波形对比 (Zoom In)
    plt.subplot(2, 2, 1)
    zoom_slice = slice(4000, 4100)
    t_axis = np.arange(len(ref_sig)) / sim.fs * 1e9 # ns
    plt.plot(t_axis[zoom_slice], ref_sig[zoom_slice], 'k-', linewidth=2, label='Ideal Target', alpha=0.3)
    plt.plot(t_axis[zoom_slice], bad_sig[zoom_slice], 'r--', label='Uncalibrated')
    plt.plot(t_axis[zoom_slice], cal_sig[zoom_slice], 'g.-', label='Calibrated (Volterra)', markersize=5)
    plt.title("Time Domain Waveform (Zoom In)")
    plt.xlabel("Time (ns)"); plt.ylabel("Amplitude")
    plt.legend(); plt.grid(True)
    
    # 2. 频谱对比 (全频带)
    plt.subplot(2, 2, 2)
    plt.plot(freqs/1e9, spec_bad, 'r', label='Uncalibrated', alpha=0.5)
    plt.plot(freqs/1e9, spec_cal, 'g', label='Calibrated', alpha=0.8)
    # 标注 HD3
    hd3_freq = test_freq * 3
    plt.axvline(hd3_freq/1e9, color='b', linestyle=':', label='HD3 Frequency')
    plt.title(f"Spectrum (Input={test_freq/1e9}GHz)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Magnitude (dB)")
    plt.ylim(-100, 10)
    plt.legend(); plt.grid(True)
    
    # 3. 扫频 SFDR 曲线 (展示中间是否还有凹陷)
    plt.subplot(2, 1, 2)
    scan_freqs = np.linspace(0.1e9, 9.0e9, 40)
    sfdr_before = []
    sfdr_after = []
    
    print("正在计算扫频 SFDR 曲线...", end="")
    for f in scan_freqs:
        # 生成单音
        s_src = sim.generate_tone(f, N=4096)
        # 物理通道
        s_ref = sim.apply_channel_effect(s_src, 6.0e9, 0.0, 1.0, hd3=0.0) # 无失配参考
        s_bad = sim.apply_channel_effect(s_src, 5.8e9, 0.42, 0.98, hd3=0.01) # 待校正
        
        with torch.no_grad():
            inp_t = numpy_to_torch(s_bad/scale, device).reshape(1, 1, -1)
            s_cal = model(inp_t).cpu().numpy().flatten() * scale
            
        # 拼接 TIADC (偶数Ref, 奇数Bad/Cal)
        # 这样才能看出 mismatch spur
        L = min(len(s_ref), len(s_bad)); L-=L%2
        ti_bad = np.zeros(L); ti_bad[0::2]=s_ref[:L:2]; ti_bad[1::2]=s_bad[1:L:2]
        ti_cal = np.zeros(L); ti_cal[0::2]=s_ref[:L:2]; ti_cal[1::2]=s_cal[1:L:2]
        
        def get_sfdr_val(sig, fin):
            # Mismatch spur @ fs/2 - fin
            spur_f = sim.fs/2 - fin
            if abs(spur_f - fin) < 0.2e9: return np.nan # 忽略奇点
            
            spec = np.abs(np.fft.rfft(sig * np.blackman(len(sig))))
            spec_db = 20*np.log10(spec + 1e-12)
            spec_db -= np.max(spec_db)
            
            freqs_ax = np.linspace(0, sim.fs/2, len(spec))
            idx = np.argmin(np.abs(freqs_ax - spur_f))
            spur_pow = np.max(spec_db[max(0,idx-5):min(len(spec),idx+5)])
            return -spur_pow

        sfdr_before.append(get_sfdr_val(ti_bad, f))
        sfdr_after.append(get_sfdr_val(ti_cal, f))
        
    print("完成")
    
    plt.plot(scan_freqs/1e9, sfdr_before, 'r--o', label='Before Cal (Mismatch Only)', alpha=0.5)
    plt.plot(scan_freqs/1e9, sfdr_after, 'g-o', label='After Cal (Physics Init + Volterra)', linewidth=2)
    plt.title("TIADC SFDR Performance vs Frequency")
    plt.xlabel("Input Frequency (GHz)"); plt.ylabel("SFDR (dBc)")
    plt.grid(True); plt.legend()
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"=== TIADC Volterra Calibration (Device: {device}) ===")
    sim = TIADCSimulator(fs=20e9)
    
    # 1. 训练
    model, scale, loss_history = train_model(sim)
    
    # 2. 验证
    verify_and_plot(sim, model, scale)