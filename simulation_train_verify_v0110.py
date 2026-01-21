import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 锁定随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. 物理仿真引擎 (纯线性模式)
# ==============================================================================
class TIADCSimulator:
    def __init__(self, fs=20e9):
        self.fs = fs

    def fractional_delay(self, sig, delay):
        """利用频域相移实现高精度分数延时"""
        N = len(sig)
        X = np.fft.rfft(sig)
        k = np.arange(len(X))
        # Phase shift: e^(-j * 2pi * k * delay / N)
        phase_shift = np.exp(-1j * 2 * np.pi * k * delay / N)
        return np.fft.irfft(X * phase_shift, n=N)

    # def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0):
    #     """
    #     施加通道效应：
    #     1. 带宽限制 (Butterworth Lowpass)
    #     2. 纯线性增益 (Gain Mismatch)
    #     3. 时间偏差 (Timing Skew)
    #     4. 量化与噪声 (可选，这里为了验证算法极限暂时保留底噪，但去掉了强非线性)
    #     """
    #     nyquist = self.fs / 2
        
    #     # 1. 带宽限制
    #     # 注意：如果截止频率过低，高频信息丢失是无法找回的，这里设为 6G/5.9G 比较温和
    #     b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
    #     sig_bw = signal.lfilter(b, a, sig)
        
    #     # 2. 增益 (纯线性)
    #     sig_gain = sig_bw * gain
        
    #     # 3. 分数延时
    #     sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
    #     # 4. 极低底噪 (防止除零或完全过拟合，模拟理想环境)
    #     noise_floor = np.random.normal(0, 1e-6, len(sig_delayed)) 
    #     sig_out = sig_delayed + noise_floor
        
    #     return sig_out

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0, jitter_std=100e-15, n_bits=12):
        """
        施加通道效应：
        1. 带宽限制 (Butterworth Lowpass)
        2. 纯线性增益 (Gain Mismatch)
        3. 时间偏差 (Timing Skew)
        4. 时钟抖动 (Jitter) [新增] - 模拟采样时刻的随机不确定性
        5. 热噪声 (Thermal Noise)
        6. 量化 (Quantization) [新增] - 模拟 ADC 有效位数
        """
        nyquist = self.fs / 2
        
        # 1. 带宽限制
        # 注意：如果截止频率过低，高频信息丢失是无法找回的
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig)
        
        # 2. 增益 (纯线性)
        sig_gain = sig_bw * gain
        
        # 3. 分数延时 (Timing Skew) - 这是一个固定的时间偏差
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 4. 时钟抖动 (Aperture Jitter) [新增]
        # 原理：利用泰勒展开近似 y(t + dt) ≈ y(t) + y'(t)*dt
        # jitter_std 单位为秒，例如 100fs = 100e-15
        if jitter_std > 0:
            # 计算信号关于时间的斜率 (Derivative): dy/dt
            # np.gradient 计算的是每个采样点的差分，除以采样间隔 (1/fs) 得到斜率
            slope = np.gradient(sig_delayed) * self.fs
            
            # 生成随机抖动时间 delta_t (高斯分布)
            dt_noise = np.random.normal(0, jitter_std, len(sig_delayed))
            
            # 叠加抖动引起的幅度误差
            sig_jittered = sig_delayed + slope * dt_noise
        else:
            sig_jittered = sig_delayed
        
        # 5. 热噪声 (底噪)
        # 物理上热噪声存在于量化之前
        noise_floor = np.random.normal(0, 1e-4, len(sig_jittered)) # 稍微调大一点模拟真实环境 (-80dB)
        sig_noisy = sig_jittered + noise_floor
        
        # 6. ADC 量化 (Quantization) [新增]
        if n_bits is not None:
            # 假设信号归一化满量程 (Full Scale) 为 -1.0 到 +1.0 (Vpp=2.0)
            v_range = 2.0
            levels = 2**n_bits
            step = v_range / levels
            
            # 模拟 ADC 输入饱和 (Clipping)
            sig_clipped = np.clip(sig_noisy, -1.0, 1.0)
            
            # 均匀量化：Round 到最近的量化电平
            sig_out = np.round(sig_clipped / step) * step
        else:
            sig_out = sig_noisy
            
        return sig_out

    def prepare_training_batch(self, N=32768):
        """生成富含频率成分的宽带训练信号"""
        t = np.arange(N) / self.fs
        # 组合多种信号以覆盖全频段
        chirp_lin = signal.chirp(t, f0=1e6, t1=t[-1], f1=self.fs*0.45, method='linear')
        chirp_log = signal.chirp(t, f0=1e5, t1=t[-1], f1=self.fs*0.45, method='logarithmic')
        step = np.zeros(N); step[:N//4]=0.5; step[N//4:N//2]=-0.5; step[N//2:]=0.0
        # 粉红噪声 (模拟底噪较多的宽带信号)
        white = np.random.randn(N)
        b, a = signal.butter(1, 0.05, btype='low')
        pink = signal.lfilter(b, a, white)
        pink /= (np.std(pink) * 3)
        
        return np.stack([chirp_lin, chirp_log, step, pink], axis=0)

    def generate_tone_data(self, freq, N=8192):
        """生成单音信号用于扫频验证"""
        t = np.arange(N) / self.fs
        src = np.sin(2 * np.pi * freq * t)
        return src * 0.9

# ==============================================================================
# 2. 混合校准模型 (Hybrid V3 - Linear)
# ==============================================================================
class DifferentiableDelay(nn.Module):
    """可微分延时层：专门对付 Timing Skew"""
    def __init__(self, init_delay=0.0):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor([init_delay]))

    def forward(self, x):
        N = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        k = torch.arange(X.shape[-1], device=x.device)
        # 频域相移
        phase_shift = torch.exp(-1j * 2 * np.pi * k * self.delay / N)
        return torch.fft.irfft(X * phase_shift, n=N, dim=-1)

class HybridCalibrationModel_V3(nn.Module):
    def __init__(self, taps=511):
        super().__init__()
        
        # 1. Delay (全局时延补偿)
        self.global_delay = DifferentiableDelay(init_delay=0.0)
        
        # 2. Gain (全局增益补偿)
        self.gain = nn.Parameter(torch.tensor([1.0]))
        
        # 3. FIR (频响微调与残差补偿)
        self.taps = taps
        # padding='same' 保证输入输出长度一致
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding='same', bias=False)
        
        # 初始化 FIR 为直通 (Dirac Delta)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, taps // 2] = 1.0

    def forward(self, x):
        # 典型的线性反演流程
        x_d = self.global_delay(x)  # 先对齐时间
        x_g = x_d * self.gain       # 再对齐幅度
        out = self.conv(x_g)        # 最后修整频响形状
        return out

# ==============================================================================
# 3. 损失函数 (高精度频域 + 时域)
# ==============================================================================
class ComplexMSELoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        # 不需要任何超参数，回归本源
        pass

    def forward(self, y_pred, y_target, model):
        # 裁剪边缘
        crop = 300 
        y_p = y_pred[..., crop:-crop]
        y_t = y_target[..., crop:-crop]

        # 1. 频域复数距离 (Complex Distance)
        # 这是为了修相位和幅度
        fft_p = torch.fft.rfft(y_p, dim=-1, norm='ortho')
        fft_t = torch.fft.rfft(y_t, dim=-1, norm='ortho')
        
        # 直接算欧几里得距离，不除以任何东西，稳健！
        # 即使高频信号小，但因为我们用了"蓝噪声"数据，这里的 diff 依然会很大
        loss_freq = torch.mean(torch.abs(fft_p - fft_t)**2)
        
        # 2. 时域 MSE (保底)
        loss_time = torch.mean((y_p - y_t)**2)
        
        # 3. 正则化 (极小，防止震荡)
        w = model.conv.weight
        loss_reg = torch.mean((w[:,:,1:] - w[:,:,:-1])**2)
        
        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg
    

# ==============================================================================
# 4. 训练流程 (去除人为非线性)
# ==============================================================================
def train_linear_calibration_blue_noise(simulator, device='cpu'):
    N_train = 32768
    
    # --- A. 准备数据 ---
    
    # [Data 1] Stage 1 专用: 纯低频 (<100MHz)
    t = np.arange(N_train) / simulator.fs
    low_freq_sig = signal.chirp(t, f0=1e6, t1=t[-1], f1=100e6, method='linear')
    batch_low = np.stack([low_freq_sig, low_freq_sig*0.5], axis=0) 
    
    # [Data 2] Stage 2 专用: 蓝噪声 (Blue Noise) - 高频增强！
    # 相比白噪声，蓝噪声的能量随频率增加 (+3dB/octave 或更多)
    # 这会强迫 MSE Loss 关注高频
    raw_white = np.random.randn(10, N_train) # 10个样本
    
    # 设计一个高通滤波器来"染色"噪声
    # 截止频率 1GHz，让高频能量通过，低频衰减
    b_high, a_high = signal.butter(1, 1.0e9/(simulator.fs/2), btype='high')
    blue_noise = signal.lfilter(b_high, a_high, raw_white)
    
    # 再叠加一点点白噪声防止低频完全为0
    batch_blue = blue_noise * 5.0 + raw_white * 0.1 

    # 辅助函数: 归一化并转Tensor
    def make_dataset(batch_data):
        in_l, tgt_l = [], []
        for i in range(batch_data.shape[0]):
            raw = batch_data[i]
            tgt = simulator.apply_channel_effect(raw, 6.0e9, 0.0, 1.0)
            inp = simulator.apply_channel_effect(raw, 5.9e9, 0.42, 0.98) # 物理参数
            in_l.append(inp)
            tgt_l.append(tgt)
        
        t_arr = np.array(tgt_l)
        i_arr = np.array(in_l)
        s = np.max(np.abs(t_arr))
        return (torch.FloatTensor(i_arr/s).unsqueeze(1).to(device), 
                torch.FloatTensor(t_arr/s).unsqueeze(1).to(device), s)

    inp_low, tgt_low, _ = make_dataset(batch_low)
    inp_blue, tgt_blue, scale_blue = make_dataset(batch_blue) # 使用蓝噪声
    
    model = HybridCalibrationModel_V3(taps=511).to(device)
    loss_fn = ComplexMSELoss(device=device)
    
    # =========================================================
    # Stage 1: 低频锚定 (Low-Freq Anchoring)
    # =========================================================
    print("=== Stage 1: Anchoring Delay with Low Freq ===")
    opt_s1 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 1e-2},
        {'params': model.gain, 'lr': 1e-2},
        {'params': model.conv.parameters(), 'lr': 0.0}
    ], betas=(0.9, 0.999)) # 默认 Adam
    
    for ep in range(301):
        opt_s1.zero_grad()
        out = model(inp_low)
        loss = loss_fn(out, tgt_low, model)
        loss.backward()
        opt_s1.step()
        if ep % 100 == 0:
            print(f"S1 Ep {ep} | Delay: {model.global_delay.delay.item():.4f}")
    # =========================================================
    # Stage 2: 蓝噪声 FIR 训练 (Blue Noise Training)
    # =========================================================
    print("\n=== Stage 2: FIR Training with Blue Noise ===")
    
    # 彻底锁死 Delay/Gain
    opt_s2 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0},
        {'params': model.gain, 'lr': 0.0},
        {'params': model.conv.parameters(), 'lr': 5e-4} # FIR 学习率
    ], betas=(0.5, 0.9)) # 调整 beta1 降低动量，防止震荡
    
    scheduler = optim.lr_scheduler.StepLR(opt_s2, step_size=500, gamma=0.2)

    for ep in range(1501):
        opt_s2.zero_grad()
        out = model(inp_blue) # 喂蓝噪声！
        
        # 普通的 Loss 即可，因为输入数据本身就是高频很强的
        loss = loss_fn(out, tgt_blue, model)
        
        loss.backward()
        opt_s2.step()
        scheduler.step()
        
        if ep % 200 == 0:
            print(f"S2 Ep {ep} | Loss: {loss.item():.6f}")

    return model, scale_blue

def train_real_world_simulation(simulator, device='cpu'):
    N_train = 32768
    M_average = 16  # 平均次数
    
    # =========================================================
    # 修改点：使用"带限白噪声"代替"Chirp"
    # 目的：消除 Chirp 起始位置的边缘效应，让低频训练更充分
    # =========================================================
    print(f"正在生成训练数据 (White Noise, M={M_average})...")
    
    # 1. 生成基准信号：带限白噪声 (0 - 9GHz)
    # 原始白噪声
    np.random.seed(42) # 保证可复现
    white_noise = np.random.randn(N_train)
    
    # 施加一个宽松的低通滤波 (比如 9GHz)，模拟真实信号源的带宽限制
    # 这样可以避免奈奎斯特频率附近的混叠干扰
    b_src, a_src = signal.butter(5, 9.0e9/(simulator.fs/2), btype='low')
    base_sig = signal.lfilter(b_src, a_src, white_noise)
    
    # 归一化幅度，防止削波
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9

    # 2. 模拟"实采平均"过程 (不变)
    captured_noisy_list = []
    for _ in range(M_average):
        # 每次采集都有独立的随机 Jitter 和 量化
        noisy_sig = simulator.apply_channel_effect(
            base_sig, 
            cutoff_freq=5.9e9, delay_samples=0.42, gain=0.98,
            jitter_std=100e-15,  # 开启 Jitter
            n_bits=12            # 开启量化
        )
        captured_noisy_list.append(noisy_sig)
    
    # 平均去噪
    averaged_sig = np.mean(np.stack(captured_noisy_list), axis=0)
    
    # 3. 构造 Target (不变)
    target_sig = simulator.apply_channel_effect(
        base_sig, 6.0e9, 0.0, 1.0, 
        jitter_std=0.0, n_bits=None 
    )
    
    # 归一化 & 转 Tensor (不变)
    scale = np.max(np.abs(target_sig))
    inp_t = torch.FloatTensor(averaged_sig/scale).view(1, 1, -1).to(device)
    tgt_t = torch.FloatTensor(target_sig/scale).view(1, 1, -1).to(device)
    
    # 4. 模型初始化 (不变)
    model = HybridCalibrationModel_V3(taps=511).to(device)
    loss_fn = ComplexMSELoss(device=device)
    
    # ... (Stage 1 和 Stage 2 的训练循环代码完全保持不变) ...
    # 为了节省篇幅，这里省略后面的训练循环代码
    # 请直接复制之前的 Stage 1 和 Stage 2 代码块
    
    # Stage 1: Delay Anchoring
    print("=== Stage 1: Delay Anchoring ===")
    opt_s1 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 1e-2},
        {'params': model.gain, 'lr': 1e-2},
        {'params': model.conv.parameters(), 'lr': 0.0}
    ])
    for ep in range(301):
        opt_s1.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s1.step()
        if ep % 100 == 0:
            print(f"S1 Ep {ep} | Delay: {model.global_delay.delay.item():.4f}")

    # Stage 2: FIR Training
    print("\n=== Stage 2: FIR Training ===")
    opt_s2 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0},
        {'params': model.gain, 'lr': 0.0},
        {'params': model.conv.parameters(), 'lr': 5e-4}
    ])
    for ep in range(1001):
        opt_s2.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s2.step()
        if ep % 200 == 0:
            print(f"S2 Ep {ep} | Loss: {loss.item():.6f}")

    return model, scale
# ==============================================================================
# 5. 验证与绘图
# ==============================================================================
def calculate_sfdr_improvement(sim, model, scale, freq, device):
    # 1. 生成单频信号
    src = sim.generate_tone_data(freq)
    
    # 2. 生成 Ideal (Ch0) 和 Bad (Ch1)
    # 必须与训练时的参数完全一致
    ch0 = sim.apply_channel_effect(src, 6.0e9, 0.0, 1.0)
    ch1_bad = sim.apply_channel_effect(src, 5.9e9, 0.42, 0.98)
    
    # 3. 模型推理校准
    with torch.no_grad():
        inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
        ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
    
    # 4. 截取稳态区域
    margin = 1000 
    c0 = ch0[margin:-margin]
    c1_b = ch1_bad[margin:-margin]
    c1_c = ch1_cal[margin:-margin]
    
    # 5. 交织 (Interleave) 模拟 TIADC
    # Ch0 是参考，Ch1 是被测 (Bad 或 Calibrated)
    def interleave(c_even, c_odd):
        L = min(len(c_even), len(c_odd))
        L -= L % 2
        out = np.zeros(L)
        out[0::2] = c_even[:L:2]
        out[1::2] = c_odd[1:L:2] # 注意：Ch1 在奇数位置
        return out
    
    tiadc_bad = interleave(c0, c1_b)
    tiadc_cal = interleave(c0, c1_c)
    
    # 6. 计算杂散 (Spur)
    # 理想情况下，除了信号频率 fs/2 - fin (Nyquist Image) 之外不应该有大的杂散
    # 但由于 Mismatch，会在 fs/2 - fin 处产生 Image Spur
    def get_max_spur(sig, signal_freq):
        # 加窗防止频谱泄露
        win = np.blackman(len(sig))
        fft_vals = np.abs(np.fft.rfft(sig * win))
        spec_db = 20 * np.log10(fft_vals + 1e-12)
        spec_db -= np.max(spec_db) # 归一化到 0dBF
        
        freqs = np.linspace(0, sim.fs/2, len(spec_db))
        
        # 寻找 Image Spur 的位置: fs/2 - fin
        image_freq = sim.fs/2 - signal_freq
        
        # 在 Image Freq 附近搜索最大值
        idx = np.argmin(np.abs(freqs - image_freq))
        search_range = 20 # bins
        start = max(0, idx - search_range)
        end = min(len(spec_db), idx + search_range)
        
        spur_power = np.max(spec_db[start:end])
        return spur_power

    spur_bad = get_max_spur(tiadc_bad, freq)
    spur_cal = get_max_spur(tiadc_cal, freq)
    
    # 返回改善值 (正值代表变好了)
    # Spur 一般是负值 (e.g., -40dB), 越小越好
    # Improvement = (-90) - (-40) = -50? 不对
    # Improvement = Bad - Cal = (-40) - (-90) = 50dB
    return spur_bad - spur_cal

def calculate_comprehensive_metrics(sim, model, scale, device):
    print("\n=== 正在计算综合指标 (ENOB, SINAD, THD) ===")
    
    # 测试频点
    test_freqs = np.arange(0.1e9, 9.6e9, 0.25e9)
    enob_list = []
    sinad_list = []
    
    # 存储一个典型频率的时域波形用于画图 (比如 2.1GHz)
    plot_freq = 2.1e9
    wave_data = {}

    for f in test_freqs:
        # 1. 生成数据
        src = sim.generate_tone_data(f)
        ch0 = sim.apply_channel_effect(src, 6.0e9, 0.0, 1.0) # Ideal
        ch1_bad = sim.apply_channel_effect(src, 5.9e9, 0.42, 0.98) # Uncalibrated
        
        # 2. 校准
        with torch.no_grad():
            inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
            ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
            
        # 3. 截取稳态
        margin = 1000
        y_ideal = ch0[margin:-margin]
        y_bad   = ch1_bad[margin:-margin]
        y_cal   = ch1_cal[margin:-margin]
        
        # 保存画图数据
        if abs(f - plot_freq) < 0.1e9:
            wave_data = {'ideal': y_ideal, 'bad': y_bad, 'cal': y_cal}

        # 4. TIADC 交织 (Ideal vs Calibrated)
        # 我们这里计算 Calibrated 后的 TIADC 整体性能
        # 偶数点是 Ideal(Ch0), 奇数点是 Calibrated(Ch1)
        # 对比对象是：完美的单通道数据 (全是 Ideal)
        
        def interleave(c_even, c_odd):
            L = min(len(c_even), len(c_odd))
            L -= L % 2
            out = np.zeros(L)
            out[0::2] = c_even[:L:2]
            out[1::2] = c_odd[1:L:2]
            return out
            
        tiadc_cal = interleave(y_ideal, y_cal)
        tiadc_ref = interleave(y_ideal, y_ideal) # 完美的参考信号
        
        # 5. 计算 SINAD & ENOB
        # 误差信号 = 校准后信号 - 完美参考信号
        error_sig = tiadc_cal - tiadc_ref
        
        # 信号功率 (Signal Power)
        p_signal = np.mean(tiadc_ref**2)
        # 噪声+失真功率 (Noise + Distortion Power)
        p_nad = np.mean(error_sig**2)
        
        # 防止除零
        if p_nad < 1e-15: p_nad = 1e-15
            
        sinad = 10 * np.log10(p_signal / p_nad)
        enob = (sinad - 1.76) / 6.02
        
        sinad_list.append(sinad)
        enob_list.append(enob)
        
    return test_freqs, sinad_list, enob_list, wave_data

# ==============================================================================
# 2. 频响失配分析 (Mismatch Response Analysis)
# ==============================================================================
def analyze_mismatch_response(sim, model, scale, device):
    print("\n=== 正在分析频响失配 (Magnitude & Phase Mismatch) ===")
    
    # 使用脉冲响应法测量全频段特性
    # 生成一个完美的脉冲
    N = 8192
    impulse = np.zeros(N)
    impulse[100] = 1.0 # 稍微延后一点避免边缘
    
    # 1. 物理通道响应
    # Ideal Ch0
    resp_ideal = sim.apply_channel_effect(impulse, 6.0e9, 0.0, 1.0)
    # Bad Ch1
    resp_bad = sim.apply_channel_effect(impulse, 5.9e9, 0.42, 0.98)
    
    # 2. 模型校准后的 Ch1 响应
    with torch.no_grad():
        inp_t = torch.FloatTensor(resp_bad/scale).view(1,1,-1).to(device)
        # 模型处理
        resp_cal = model(inp_t).cpu().numpy().flatten() * scale
        
    # 3. 计算 FFT
    fft_ideal = np.fft.rfft(resp_ideal)
    fft_bad = np.fft.rfft(resp_bad)
    fft_cal = np.fft.rfft(resp_cal)
    
    freqs = np.linspace(0, sim.fs/2, len(fft_ideal))
    
    # 4. 计算失配 (Mismatch)
    # 理想情况下，H_ch1 / H_ch0 应该等于 1 (0dB, 0度)
    
    # 校准前失配
    H_mismatch_pre = fft_bad / (fft_ideal + 1e-12)
    mag_err_pre = 20 * np.log10(np.abs(H_mismatch_pre))
    phase_err_pre = np.angle(H_mismatch_pre, deg=True)
    
    # 校准后失配
    H_mismatch_post = fft_cal / (fft_ideal + 1e-12)
    mag_err_post = 20 * np.log10(np.abs(H_mismatch_post))
    phase_err_post = np.angle(H_mismatch_post, deg=True)
    
    return freqs, mag_err_pre, phase_err_pre, mag_err_post, phase_err_post


# ==============================================================================
# 运行绘图
# ==============================================================================
if __name__ == "__main__":
    # 自动选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备: {device}")
    
    # 1. 初始化仿真器与模型训练
    sim = TIADCSimulator(fs=20e9)
    # 使用之前定义的"白噪声/实采模拟"训练函数
    model, scale = train_real_world_simulation(sim, device)
    
    # ==================================================================
    # 2. 扫频验证 (SFDR)
    # ==================================================================
    print("\n=== 正在进行扫频验证 (SFDR) ===")
    
    # 扫频范围：从 100M 到 9.5G
    test_freqs = np.arange(0.1e9, 9.6e9, 0.2e9)
    improvements = []
    
    for f in test_freqs:
        # 跳过 Nyquist 点 (10GHz) 附近的频率，避免镜像重叠影响计算
        if abs(f - 10e9) < 0.5e9: 
            improvements.append(0)
            continue
            
        # 计算单个频点的 SFDR 改善值
        imp = calculate_sfdr_improvement(sim, model, scale, f, device)
        improvements.append(imp)
        print(f"Freq: {f/1e9:.2f} GHz | SFDR Improvement: {imp:.2f} dB")
    
    # --- [新增] 绘图 0: SFDR 改善曲线 ---
    plt.figure(figsize=(10, 5))
    plt.plot(test_freqs/1e9, improvements, 'b-o', linewidth=2, label='Calibration Improvement')
    plt.title("SFDR Improvement vs Input Frequency")
    plt.xlabel("Input Frequency (GHz)")
    plt.ylabel("SFDR Improvement (dB)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.axhline(0, color='r', linestyle='--', alpha=0.3, label='Baseline') # 0dB 参考线
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==================================================================
    # 3. 综合指标分析 (ENOB, SINAD, THD)
    # ==================================================================
    # 注意：确保 calculate_comprehensive_metrics 函数已定义
    freqs, sinad, enob, wave = calculate_comprehensive_metrics(sim, model, scale, device)
    
    # ==================================================================
    # 4. 频响失配分析 (Gain/Phase Mismatch)
    # ==================================================================
    # 注意：确保 analyze_mismatch_response 函数已定义
    f_resp, mag_pre, ph_pre, mag_post, ph_post = analyze_mismatch_response(sim, model, scale, device)
    
    # ==================================================================
    # 5. 绘制剩余图表
    # ==================================================================
    
    # --- 绘图 1: ENOB & SINAD ---
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(freqs/1e9, sinad, 'g-o', linewidth=2)
    plt.title("SINAD (Signal-to-Noise-and-Distortion Ratio)")
    plt.ylabel("dB"); plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(freqs/1e9, enob, 'm-o', linewidth=2)
    plt.title("ENOB (Effective Number of Bits)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Bits"); plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # --- 绘图 2: 时域残差 (Residual Error) ---
    plt.figure(figsize=(10, 4))
    # 取一小段波形展示 (假设 wave 字典里有数据)
    L = 100
    start = 5000
    t_axis = np.arange(L)
    
    # 计算误差
    err_pre = wave['bad'][start:start+L] - wave['ideal'][start:start+L]
    err_post = wave['cal'][start:start+L] - wave['ideal'][start:start+L]
    
    plt.plot(t_axis, err_pre, 'r--', label='Pre-Calib Error', alpha=0.5)
    plt.plot(t_axis, err_post, 'b.-', label='Post-Calib Error (Residual)', linewidth=2)
    plt.title(f"Time Domain Error Residuals @ 2.1GHz")
    plt.xlabel("Sample Index"); plt.ylabel("Amplitude Error"); plt.grid(True)
    plt.legend()
    plt.show()

    # --- 绘图 3: 频响失配分析 (核心原理图) ---
    plt.figure(figsize=(12, 5))
    
    # 幅度失配
    plt.subplot(1,2,1)
    plt.plot(f_resp/1e9, mag_pre, 'r', label='Before Calib', alpha=0.5)
    plt.plot(f_resp/1e9, mag_post, 'b', label='After Calib', linewidth=2)
    plt.title("Gain Mismatch (dB)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Mismatch (dB)")
    # 根据实际误差范围调整 ylim，通常校准后在 0 附近
    plt.ylim(-1.0, 1.0) 
    plt.grid(True)
    plt.legend()
    
    # 相位失配
    plt.subplot(1,2,2)
    plt.plot(f_resp/1e9, ph_pre, 'r', label='Before Calib', alpha=0.5)
    plt.plot(f_resp/1e9, ph_post, 'b', label='After Calib', linewidth=2)
    plt.title("Phase Mismatch (Degrees)")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Mismatch (deg)")
    plt.ylim(-5, 5) 
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
# if __name__ == "__main__":
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     sim = TIADCSimulator(fs=20e9)
#     model, scale = train_linear_calibration_blue_noise(sim, device)
    
#     print("\n=== 正在进行扫频验证 (SFDR) ===")
    
#     # 扫频范围：从 100M 到 9.5G
#     test_freqs = np.arange(0.1e9, 9.6e9, 0.2e9)
#     improvements = []
    
#     for f in test_freqs:
#         # 跳过 Nyquist 点附近的频率，因为那里 Image 和 Signal 重叠，没法算 SFDR
#         if abs(f - 10e9) < 0.5e9: 
#             improvements.append(0)
#             continue
            
#         imp = calculate_sfdr_improvement(sim, model, scale, f, device)
#         improvements.append(imp)
#         print(f"Freq: {f/1e9:.2f} GHz | SFDR Improvement: {imp:.2f} dB")

#     # 绘图
#     plt.figure(figsize=(10, 6))
#     plt.plot(test_freqs/1e9, improvements, 'b-o', linewidth=2, label='Linear DDSP Model')
#     plt.title("TIADC Mismatch Calibration Performance (Pure Linear Scenario)")
#     plt.xlabel("Input Frequency (GHz)")
#     plt.ylabel("SFDR Improvement (dB)")
#     plt.grid(True, which='both', linestyle='--', alpha=0.7)
#     plt.axhline(0, color='k', linestyle='-', alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()