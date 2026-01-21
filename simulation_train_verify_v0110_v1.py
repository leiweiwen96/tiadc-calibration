import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 锁定随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. 物理仿真引擎 (支持非线性、Jitter、量化)
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

    def generate_tone_data(self, freq, N=8192):
        """生成单音信号用于扫频验证"""
        t = np.arange(N) / self.fs
        src = np.sin(2 * np.pi * freq * t)
        return src * 0.9

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0, 
                             jitter_std=100e-15, n_bits=12, 
                             hd2=0.0, hd3=0.0): 
        """
        施加通道效应：
        1. 非线性失真 (HD2, HD3)
        2. 带宽限制 (Butterworth Lowpass)
        3. 纯线性增益 (Gain Mismatch)
        4. 时间偏差 (Timing Skew)
        5. 时钟抖动 (Jitter)
        6. 热噪声
        7. 量化
        """
        nyquist = self.fs / 2
        
        # 1. 非线性失真 (Non-linearity)
        # 模拟前端非线性: y = x + hd2*x^2 + hd3*x^3
        sig_nonlinear = sig + hd2 * (sig**2) + hd3 * (sig**3)
        
        # 2. 带宽限制 (Filtering)
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig_nonlinear)
        
        # 3. 增益 (Gain)
        sig_gain = sig_bw * gain
        
        # 4. 延时 (Delay)
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 5. Jitter (时钟抖动)
        if jitter_std > 0:
            # 利用斜率近似抖动带来的幅度误差: dy = y' * dt
            slope = np.gradient(sig_delayed) * self.fs
            dt_noise = np.random.normal(0, jitter_std, len(sig_delayed))
            sig_jittered = sig_delayed + slope * dt_noise
        else:
            sig_jittered = sig_delayed
            
        # 6. 热噪声 (底噪)
        noise_floor = np.random.normal(0, 1e-4, len(sig_jittered))
        sig_noisy = sig_jittered + noise_floor
        
        # 7. 量化 (Quantization)
        if n_bits is not None:
            v_range = 2.0 # 假设满量程 +/- 1.0
            levels = 2**n_bits
            step = v_range / levels
            sig_clipped = np.clip(sig_noisy, -1.0, 1.0)
            sig_out = np.round(sig_clipped / step) * step
        else:
            sig_out = sig_noisy
            
        return sig_out

# ==============================================================================
# 2. 混合校准模型 Components
# ==============================================================================
class PolynomialCorrection(nn.Module):
    """非线性校准模块: y = x + a*x^2 + b*x^3"""
    def __init__(self, order=3):
        super().__init__()
        # 初始化系数为 0 (假设一开始没有非线性)
        # poly_coeffs[0] 对应 x^2, poly_coeffs[1] 对应 x^3
        self.poly_coeffs = nn.Parameter(torch.zeros(order - 1)) 

    def forward(self, x):
        out = x.clone()
        # 叠加 x^2
        out = out + self.poly_coeffs[0] * (x**2)
        # 叠加 x^3
        if len(self.poly_coeffs) > 1:
            out = out + self.poly_coeffs[1] * (x**3)
        return out

class FarrowDelay(nn.Module):
    """
    可微分分数延时模块 (3阶 Farrow 结构)
    """
    def __init__(self):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor(0.0))
        # 定义 Farrow 结构的 4 个固定 FIR 子滤波器系数 (Cubic Lagrange)
        self.register_buffer('kernel_0', torch.tensor([[[0, 1, 0, 0]]], dtype=torch.float32))
        self.register_buffer('kernel_1', torch.tensor([[[-1/3, -1/2, 1, -1/6]]], dtype=torch.float32))
        self.register_buffer('kernel_2', torch.tensor([[[1/2, -1, 1/2, 0]]], dtype=torch.float32))
        self.register_buffer('kernel_3', torch.tensor([[[-1/6, 1/2, -1/2, 1/6]]], dtype=torch.float32))

    def forward(self, x):
        d = self.delay
        # Padding 策略: 左1 右2，配合长4的核
        padded_x = F.pad(x, (1, 2), mode='replicate') 
        
        c0 = F.conv1d(padded_x, self.kernel_0)
        c1 = F.conv1d(padded_x, self.kernel_1)
        c2 = F.conv1d(padded_x, self.kernel_2)
        c3 = F.conv1d(padded_x, self.kernel_3)
        
        # Horner's Method 求多项式值
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out

class HybridCalibrationModel(nn.Module): 
    """
    物理感知混合校准模型 (V4)
    结构: Polynomial(非线性) -> Gain -> Delay -> FIR(线性残差)
    """
    def __init__(self, taps=511):
        super().__init__()
        # 1. 非线性校准
        self.poly = PolynomialCorrection(order=3) 
        
        # 2. 增益校准
        self.gain = nn.Parameter(torch.tensor(1.0))

        # 3. 时延校准 (Farrow)
        self.global_delay = FarrowDelay()
        
        # 4. 频响校准 (FIR)
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        # 初始化 FIR 为直通
        with torch.no_grad():
            self.conv.weight.data.fill_(0.0)
            self.conv.weight.data[0, 0, taps//2] = 1.0

    def forward(self, x):
        # 逆过程顺序：先解非线性，再解线性
        x_poly = self.poly(x)
        x_gain = x_poly * self.gain
        x_delay = self.global_delay(x_gain)
        out = self.conv(x_delay)
        return out

# ==============================================================================
# 3. 损失函数
# ==============================================================================
class ComplexMSELoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

    def forward(self, y_pred, y_target, model):
        # 裁剪边缘，避免卷积瞬态影响
        crop = 300 
        y_p = y_pred[..., crop:-crop]
        y_t = y_target[..., crop:-crop]

        # 1. 频域复数距离 (关注相位和幅度)
        fft_p = torch.fft.rfft(y_p, dim=-1, norm='ortho')
        fft_t = torch.fft.rfft(y_t, dim=-1, norm='ortho')
        loss_freq = torch.mean(torch.abs(fft_p - fft_t)**2)
        
        # 2. 时域 MSE (保底)
        loss_time = torch.mean((y_p - y_t)**2)
        
        # 3. FIR 平滑正则化 (防止高频震荡)
        w = model.conv.weight
        loss_reg = torch.mean((w[:,:,1:] - w[:,:,:-1])**2)
        
        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg

# ==============================================================================
# 4. 训练流程 (相干平均 + 三阶段训练)
# ==============================================================================
def train_real_world_simulation(simulator, device='cpu'):
    """
    包含非线性校准的完整训练流程
    """
    N_train = 32768
    M_average = 16  # 相干平均次数
    
    # 设置非线性参数 (用于生成"坏"数据)
    HD2_TRAIN = 1e-3  # -60dBc
    HD3_TRAIN = 1e-3  # -60dBc
    
    # --- A. 准备数据 ---
    print(f"正在生成训练数据 (White Noise, Avg={M_average}, with HD2/HD3)...")
    
    # 1. 生成基准信号 (带限白噪声)
    white_noise = np.random.randn(N_train)
    b_src, a_src = signal.butter(6, 9.0e9/(simulator.fs/2), btype='low')
    base_sig = signal.lfilter(b_src, a_src, white_noise)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9

    # 2. 模拟 Input (含 Jitter/量化/非线性 -> 平均去噪)
    captured_noisy_list = []
    for _ in range(M_average):
        noisy_sig = simulator.apply_channel_effect(
            base_sig, 
            cutoff_freq=5.9e9, delay_samples=0.42, gain=0.98,
            jitter_std=100e-15,  # 开启物理抖动
            n_bits=12,           # 开启物理量化
            hd2=HD2_TRAIN,       # [关键] 注入非线性
            hd3=HD3_TRAIN
        )
        captured_noisy_list.append(noisy_sig)
    
    averaged_sig = np.mean(np.stack(captured_noisy_list), axis=0)
    
    # 3. 模拟 Target (完美参考)
    target_sig = simulator.apply_channel_effect(
        base_sig, 6.0e9, 0.0, 1.0, 
        jitter_std=0.0, n_bits=None, hd2=0.0, hd3=0.0
    )
    
    # 4. 归一化 & 转 Tensor
    scale = np.max(np.abs(target_sig))
    inp_t = torch.FloatTensor(averaged_sig/scale).view(1, 1, -1).to(device)
    tgt_t = torch.FloatTensor(target_sig/scale).view(1, 1, -1).to(device)
    
    # --- B. 模型初始化 ---
    model = HybridCalibrationModel(taps=511).to(device)
    loss_fn = ComplexMSELoss(device=device)
    
    # =========================================================
    # Stage 1: 线性锚定 (Delay & Gain)
    # =========================================================
    print("=== Stage 1: Delay & Gain Anchoring ===")
    opt_s1 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 1e-2},
        {'params': model.gain, 'lr': 1e-2},
        {'params': model.conv.parameters(), 'lr': 0.0}, # Lock
        {'params': model.poly.parameters(), 'lr': 0.0}  # Lock
    ], betas=(0.9, 0.999))
    
    for ep in range(301):
        opt_s1.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s1.step()
        if ep % 100 == 0:
            print(f"S1 Ep {ep} | Delay: {model.global_delay.delay.item():.4f}")

    # =========================================================
    # Stage 2: 线性宽带精调 (FIR)
    # =========================================================
    print("\n=== Stage 2: FIR Training ===")
    opt_s2 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0},
        {'params': model.gain, 'lr': 0.0},
        {'params': model.conv.parameters(), 'lr': 5e-4}, # Train FIR
        {'params': model.poly.parameters(), 'lr': 0.0}
    ], betas=(0.5, 0.9))
    
    scheduler = optim.lr_scheduler.StepLR(opt_s2, step_size=500, gamma=0.2)

    for ep in range(1001):
        opt_s2.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s2.step()
        scheduler.step()
        if ep % 200 == 0:
            print(f"S2 Ep {ep} | Loss: {loss.item():.6f}")

    # =========================================================
    # Stage 3: 非线性校准 (Nonlinear Tuning)
    # =========================================================
    print("\n=== Stage 3: Nonlinear Calibration (Poly) ===")
    # 此时线性部分已经很准了，锁死它们，专门修非线性
    opt_s3 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0},
        {'params': model.gain, 'lr': 0.0},
        {'params': model.conv.parameters(), 'lr': 0.0},
        {'params': model.poly.parameters(), 'lr': 1e-3} # Train Poly
    ])

    for ep in range(501):
        opt_s3.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s3.step()
        if ep % 100 == 0:
            coeffs = model.poly.poly_coeffs.detach().cpu().numpy()
            print(f"S3 Ep {ep} | Loss: {loss.item():.6f} | HD2: {coeffs[0]:.5f}, HD3: {coeffs[1]:.5f}")

    return model, scale, (HD2_TRAIN, HD3_TRAIN)

# ==============================================================================
# 5. 验证与绘图
# ==============================================================================
def calculate_sfdr_improvement(sim, model, scale, freq, device, hd_params):
    hd2, hd3 = hd_params
    
    # 1. 生成单频信号
    src = sim.generate_tone_data(freq)
    
    # 2. 生成 Ideal (Ch0) 和 Bad (Ch1)
    # 注意：验证时也要给 Bad Channel 加同样的非线性
    ch0 = sim.apply_channel_effect(src, 6.0e9, 0.0, 1.0, jitter_std=0, n_bits=None, hd2=0, hd3=0)
    ch1_bad = sim.apply_channel_effect(src, 5.9e9, 0.42, 0.98, jitter_std=0, n_bits=None, hd2=hd2, hd3=hd3)
    
    # 3. 模型推理校准
    with torch.no_grad():
        inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
        ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
    
    # 4. 截取稳态区域
    margin = 1000 
    c0 = ch0[margin:-margin]
    c1_b = ch1_bad[margin:-margin]
    c1_c = ch1_cal[margin:-margin]
    
    # 5. 交织模拟 TIADC
    def interleave(c_even, c_odd):
        L = min(len(c_even), len(c_odd))
        L -= L % 2
        out = np.zeros(L)
        out[0::2] = c_even[:L:2]
        out[1::2] = c_odd[1:L:2]
        return out
    
    tiadc_bad = interleave(c0, c1_b)
    tiadc_cal = interleave(c0, c1_c)
    
    # 6. 计算最大杂散 (Spur)
    def get_max_spur(sig, signal_freq):
        win = np.blackman(len(sig))
        fft_vals = np.abs(np.fft.rfft(sig * win))
        spec_db = 20 * np.log10(fft_vals + 1e-12)
        spec_db -= np.max(spec_db) # 归一化到 0dBF
        
        freqs = np.linspace(0, sim.fs/2, len(spec_db))
        
        # 排除信号本身和谐波
        # 只搜索主要的交织杂散 (Nyquist Image): fs/2 - fin
        image_freq = sim.fs/2 - signal_freq
        
        # 在 Image Freq 附近搜索
        idx = np.argmin(np.abs(freqs - image_freq))
        search_range = 20
        start = max(0, idx - search_range)
        end = min(len(spec_db), idx + search_range)
        
        spur_power = np.max(spec_db[start:end])
        return spur_power

    spur_bad = get_max_spur(tiadc_bad, freq)
    spur_cal = get_max_spur(tiadc_cal, freq)
    
    return spur_bad - spur_cal

def calculate_comprehensive_metrics(sim, model, scale, device, hd_params):
    print("\n=== 正在计算综合指标 (ENOB, SINAD) ===")
    hd2, hd3 = hd_params
    test_freqs = np.arange(0.1e9, 9.6e9, 0.0025e9)
    enob_list = []
    sinad_list = []
    plot_freq = 2.1e9
    wave_data = {}

    for f in test_freqs:
        src = sim.generate_tone_data(f)
        ch0 = sim.apply_channel_effect(src, 6.0e9, 0.0, 1.0, hd2=0, hd3=0)
        ch1_bad = sim.apply_channel_effect(src, 5.9e9, 0.42, 0.98, hd2=hd2, hd3=hd3)
        
        with torch.no_grad():
            inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
            ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
            
        margin = 1000
        y_ideal = ch0[margin:-margin]
        y_bad   = ch1_bad[margin:-margin]
        y_cal   = ch1_cal[margin:-margin]
        
        if abs(f - plot_freq) < 0.1e9:
            wave_data = {'ideal': y_ideal, 'bad': y_bad, 'cal': y_cal}

        def interleave(c_even, c_odd):
            L = min(len(c_even), len(c_odd))
            L -= L % 2
            out = np.zeros(L)
            out[0::2] = c_even[:L:2]
            out[1::2] = c_odd[1:L:2]
            return out
            
        tiadc_cal = interleave(y_ideal, y_cal)
        tiadc_ref = interleave(y_ideal, y_ideal)
        
        error_sig = tiadc_cal - tiadc_ref
        p_signal = np.mean(tiadc_ref**2)
        p_nad = np.mean(error_sig**2)
        if p_nad < 1e-15: p_nad = 1e-15
            
        sinad = 10 * np.log10(p_signal / p_nad)
        enob = (sinad - 1.76) / 6.02
        sinad_list.append(sinad)
        enob_list.append(enob)
        
    return test_freqs, sinad_list, enob_list, wave_data

def analyze_mismatch_response(sim, model, scale, device, hd_params):
    print("\n=== 正在分析频响失配 ===")
    hd2, hd3 = hd_params
    N = 8192
    impulse = np.zeros(N)
    impulse[100] = 1.0 
    
    # 脉冲响应分析通常只看线性部分，所以这里 HD 设为 0 比较直观
    # 或者如果想看大信号下的频率响应，可以用 chirp
    # 这里为了看 filter 匹配程度，暂时用线性模式
    resp_ideal = sim.apply_channel_effect(impulse, 6.0e9, 0.0, 1.0, hd2=0, hd3=0)
    resp_bad = sim.apply_channel_effect(impulse, 5.9e9, 0.42, 0.98, hd2=0, hd3=0)
    
    with torch.no_grad():
        inp_t = torch.FloatTensor(resp_bad/scale).view(1,1,-1).to(device)
        resp_cal = model(inp_t).cpu().numpy().flatten() * scale
        
    fft_ideal = np.fft.rfft(resp_ideal)
    fft_bad = np.fft.rfft(resp_bad)
    fft_cal = np.fft.rfft(resp_cal)
    
    freqs = np.linspace(0, sim.fs/2, len(fft_ideal))
    
    H_mismatch_pre = fft_bad / (fft_ideal + 1e-12)
    mag_err_pre = 20 * np.log10(np.abs(H_mismatch_pre))
    phase_err_pre = np.angle(H_mismatch_pre, deg=True)
    
    H_mismatch_post = fft_cal / (fft_ideal + 1e-12)
    mag_err_post = 20 * np.log10(np.abs(H_mismatch_post))
    phase_err_post = np.angle(H_mismatch_post, deg=True)
    
    return freqs, mag_err_pre, phase_err_pre, mag_err_post, phase_err_post

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备: {device}")
    
    sim = TIADCSimulator(fs=20e9)
    # 训练模型 (返回训练好的模型，归一化因子，以及训练时使用的 HD 参数)
    model, scale, hd_params = train_real_world_simulation(sim, device)
    
    # === 验证 ===
    print("\n=== 正在进行扫频验证 (SFDR) ===")
    test_freqs = np.arange(0.1e9, 9.6e9, 0.2e9)
    improvements = []
    
    for f in test_freqs:
        if abs(f - 10e9) < 0.5e9: 
            improvements.append(0)
            continue
        # 传入训练时的 hd_params，确保验证场景与问题场景一致
        imp = calculate_sfdr_improvement(sim, model, scale, f, device, hd_params)
        improvements.append(imp)
        print(f"Freq: {f/1e9:.2f} GHz | SFDR Improvement: {imp:.2f} dB")
    
    # 绘图: SFDR
    plt.figure(figsize=(10, 5))
    plt.plot(test_freqs/1e9, improvements, 'b-o', linewidth=2, label='Calibration Improvement')
    plt.title("SFDR Improvement vs Input Frequency")
    plt.xlabel("Input Frequency (GHz)")
    plt.ylabel("SFDR Improvement (dB)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.axhline(0, color='r', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 综合指标
    freqs, sinad, enob, wave = calculate_comprehensive_metrics(sim, model, scale, device, hd_params)
    
    # 频响分析
    f_resp, mag_pre, ph_pre, mag_post, ph_post = analyze_mismatch_response(sim, model, scale, device, hd_params)
    
    # 绘图: ENOB
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(freqs/1e9, sinad, 'g-o', linewidth=2)
    plt.title("SINAD")
    plt.ylabel("dB"); plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(freqs/1e9, enob, 'm-o', linewidth=2)
    plt.title("ENOB")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("Bits"); plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 绘图: 频响失配
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(f_resp/1e9, mag_pre, 'r', label='Pre-Calib', alpha=0.5)
    plt.plot(f_resp/1e9, mag_post, 'b', label='Post-Calib', linewidth=2)
    plt.title("Gain Mismatch (dB)"); plt.ylim(-1.0, 1.0); plt.grid(True); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(f_resp/1e9, ph_pre, 'r', label='Pre-Calib', alpha=0.5)
    plt.plot(f_resp/1e9, ph_post, 'b', label='Post-Calib', linewidth=2)
    plt.title("Phase Mismatch (deg)"); plt.ylim(-5, 5); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()