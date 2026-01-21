import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 锁定随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 0. FPGA 辅助工具 (新增：定点数量化模拟)
# ==============================================================================
class Quantizer(nn.Module):
    """
    模拟 FPGA 定点数量化效应
    forward 过程: Float -> Fixed(Int) -> Float (用于梯度回传和误差评估)
    """
    def __init__(self, bit_width=16, dynamic_range=2.0):
        super().__init__()
        self.bit_width = bit_width
        self.dynamic_range = dynamic_range
        self.scale = (2**(bit_width - 1) - 1) / (dynamic_range / 2)

    def forward(self, x):
        # 模拟量化：缩放 -> 取整 -> 截断 -> 还原
        x_int = torch.round(x * self.scale)
        # 饱和截断 (Saturate)
        max_val = 2**(self.bit_width - 1) - 1
        min_val = -2**(self.bit_width - 1)
        x_clamped = torch.clamp(x_int, min_val, max_val)
        return x_clamped / self.scale

# ==============================================================================
# 1. 物理仿真引擎
# ==============================================================================
class TIADCSimulator:
    def __init__(self, fs=20e9):
        self.fs = fs

    def fractional_delay(self, sig, delay):
        N = len(sig)
        X = np.fft.rfft(sig)
        k = np.arange(len(X))
        # 增加防止相位卷绕的保护，虽然对于纯延迟通常不需要，但处理大延迟时更稳健
        phase_shift = np.exp(-1j * 2 * np.pi * k * delay / N)
        return np.fft.irfft(X * phase_shift, n=N)

    def generate_tone_data(self, freq, N=8192):
        t = np.arange(N) / self.fs
        src = np.sin(2 * np.pi * freq * t)
        return src * 0.9

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0, 
                             jitter_std=100e-15, n_bits=12, 
                             hd2=0.0, hd3=0.0, snr_target=None): 
        nyquist = self.fs / 2
        
        # 1. 非线性失真
        sig_nonlinear = sig + hd2 * (sig**2) + hd3 * (sig**3)
        
        # 2. 带宽限制 (模拟模拟前端滤波器)
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig_nonlinear)
        
        # 3. 增益失配
        sig_gain = sig_bw * gain
        
        # 4. 时延失配
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 5. 时钟抖动 (Jitter)
        # Jitter 产生的噪声功率与信号频率和抖动值成正比
        if jitter_std > 0:
            slope = np.gradient(sig_delayed) * self.fs # 信号斜率
            dt_noise = np.random.normal(0, jitter_std, len(sig_delayed))
            sig_jittered = sig_delayed + slope * dt_noise
        else:
            sig_jittered = sig_delayed
            
        # 6. 热噪声 (Thermal Noise)
        # 如果指定了 n_bits (量化)，通常热噪声底应该低于量化噪声或与其相当
        # 这里改为更科学的 SNR 设置，或者基于位宽的本底噪声
        if snr_target is not None:
             # 根据目标 SNR 添加高斯白噪声
            sig_power = np.mean(sig_jittered**2)
            noise_power = sig_power / (10**(snr_target/10))
            thermal_noise = np.random.normal(0, np.sqrt(noise_power), len(sig_jittered))
        else:
            # 默认热噪声 (模拟约 14bit 系统的热噪声底)
            thermal_noise = np.random.normal(0, 1e-4, len(sig_jittered))
            
        sig_noisy = sig_jittered + thermal_noise
        
        # 7. 量化 (ADC Digitization)
        if n_bits is not None:
            v_range = 2.0 
            levels = 2**n_bits
            step = v_range / levels
            # 模拟 ADC 的输入截断 (Clipping)
            sig_clipped = np.clip(sig_noisy, -1.0, 1.0)
            # 模拟量化
            sig_out = np.round(sig_clipped / step) * step
        else:
            sig_out = sig_noisy
            
        return sig_out

# ==============================================================================
# 2. 混合校准模型
# ==============================================================================
class PolynomialCorrection(nn.Module):
    def __init__(self, order=3):
        super().__init__()
        self.poly_coeffs = nn.Parameter(torch.zeros(order - 1)) 

    def forward(self, x):
        out = x.clone()
        out = out + self.poly_coeffs[0] * (x**2)
        if len(self.poly_coeffs) > 1:
            out = out + self.poly_coeffs[1] * (x**3)
        return out

class FarrowDelay(nn.Module):
    def __init__(self):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor(0.0))
        # FPGA 部署提示: 这些系数在 FPGA 中需要预先计算并量化
        self.register_buffer('kernel_0', torch.tensor([[[0, 1, 0, 0]]], dtype=torch.float32))
        self.register_buffer('kernel_1', torch.tensor([[[-1/3, -1/2, 1, -1/6]]], dtype=torch.float32))
        self.register_buffer('kernel_2', torch.tensor([[[1/2, -1, 1/2, 0]]], dtype=torch.float32))
        self.register_buffer('kernel_3', torch.tensor([[[-1/6, 1/2, -1/2, 1/6]]], dtype=torch.float32))

    def forward(self, x):
        d = self.delay
        padded_x = F.pad(x, (1, 2), mode='replicate') 
        c0 = F.conv1d(padded_x, self.kernel_0)
        c1 = F.conv1d(padded_x, self.kernel_1)
        c2 = F.conv1d(padded_x, self.kernel_2)
        c3 = F.conv1d(padded_x, self.kernel_3)
        # Horner's method 结构，适合 FPGA DSP Slice 级联
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out

class HybridCalibrationModel(nn.Module): 
    def __init__(self, taps=511, fpga_simulation=False):
        super().__init__()
        self.poly = PolynomialCorrection(order=3) 
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        
        # FPGA 部署提示: 511 Taps 对于 20Gsps 来说非常巨大。
        # 实际部署可能需要截断系数或使用 FFT 域卷积。
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        with torch.no_grad():
            self.conv.weight.data.fill_(0.0)
            self.conv.weight.data[0, 0, taps//2] = 1.0
            
        self.fpga_simulation = fpga_simulation
        # 始终初始化量化器，以便随时开启仿真
        # 假设 FPGA 数据通路为 16bit，系数为 18bit
        # 数据量化范围设为 4.0 ([-2.0, 2.0]) 以防止信号过冲截断
        self.data_quant = Quantizer(bit_width=16, dynamic_range=4.0)
        # 权重量化范围设为 4.0 ([-2.0, 2.0]) 以容纳主抽头系数 1.0
        self.weight_quant = Quantizer(bit_width=18, dynamic_range=4.0)

    def forward(self, x):
        # 1. 非线性校正
        x_poly = self.poly(x)
        
        # 2. 增益校正
        x_gain = x_poly * self.gain
        
        # 3. 时延校正
        x_delay = self.global_delay(x_gain)
        
        # 4. 线性均衡 (FIR)
        if self.fpga_simulation:
            # 模拟 FPGA 定点运算：对输入和权重进行量化
            w_q = self.weight_quant(self.conv.weight)
            x_q = self.data_quant(x_delay)
            out = F.conv1d(x_q, w_q, padding=self.conv.padding, stride=self.conv.stride)
        else:
            out = self.conv(x_delay)
            
        return out

# ==============================================================================
# 3. 损失函数
# ==============================================================================
class ComplexMSELoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

    def forward(self, y_pred, y_target, model):
        crop = 300 
        y_p = y_pred[..., crop:-crop]
        y_t = y_target[..., crop:-crop]
        fft_p = torch.fft.rfft(y_p, dim=-1, norm='ortho')
        fft_t = torch.fft.rfft(y_t, dim=-1, norm='ortho')
        loss_freq = torch.mean(torch.abs(fft_p - fft_t)**2)
        loss_time = torch.mean((y_p - y_t)**2)
        w = model.conv.weight
        loss_reg = torch.mean((w[:,:,1:] - w[:,:,:-1])**2)
        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg

# ==============================================================================
# 4. 训练流程 (相对校准)
# ==============================================================================
def train_relative_calibration(simulator, device='cpu'):
    N_train = 32768
    M_average = 16 
    
    # 定义通道参数
    params_ch0 = {'cutoff_freq': 6.0e9, 'delay_samples': 0.0, 'gain': 1.0, 'hd2': 1e-3, 'hd3': 1e-3}
    params_ch1 = {'cutoff_freq': 5.9e9, 'delay_samples': 0.42, 'gain': 0.98, 'hd2': 2e-3, 'hd3': 2e-3}
    
    print(f"开始相对校准训练: Mapping Ch1 -> Ch0")

    np.random.seed(42)
    white_noise = np.random.randn(N_train)
    b_src, a_src = signal.butter(6, 9.0e9/(simulator.fs/2), btype='low')
    base_sig = signal.lfilter(b_src, a_src, white_noise)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9
    base_sig = base_sig + 0.5e-3 * base_sig**2 

    ch0_captures = []
    ch1_captures = []
    for _ in range(M_average):
        # 使用默认 n_bits=12 模拟 ADC
        sig0 = simulator.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch0)
        ch0_captures.append(sig0)
        sig1 = simulator.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch1)
        ch1_captures.append(sig1)
    
    avg_ch0 = np.mean(np.stack(ch0_captures), axis=0) # Target
    avg_ch1 = np.mean(np.stack(ch1_captures), axis=0) # Input
    
    scale = np.max(np.abs(avg_ch0))
    inp_t = torch.FloatTensor(avg_ch1/scale).view(1, 1, -1).to(device)
    tgt_t = torch.FloatTensor(avg_ch0/scale).view(1, 1, -1).to(device)
    
    # 训练时不开启 fpga_simulation，因为梯度需要高精度
    model = HybridCalibrationModel(taps=511, fpga_simulation=False).to(device)
    loss_fn = ComplexMSELoss(device=device)
    
    print("=== Stage 1: Relative Delay & Gain ===")
    opt_s1 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 1e-2},
        {'params': model.gain, 'lr': 1e-2},
        {'params': model.conv.parameters(), 'lr': 0.0},
        {'params': model.poly.parameters(), 'lr': 0.0}
    ])
    for ep in range(301):
        opt_s1.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s1.step()

    print("=== Stage 2: Relative FIR ===")
    opt_s2 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0},
        {'params': model.gain, 'lr': 0.0},
        {'params': model.conv.parameters(), 'lr': 5e-4},
        {'params': model.poly.parameters(), 'lr': 0.0}
    ], betas=(0.5, 0.9))
    for ep in range(1001):
        opt_s2.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s2.step()

    print("=== Stage 3: Relative Nonlinearity ===")
    opt_s3 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0},
        {'params': model.gain, 'lr': 0.0},
        {'params': model.conv.parameters(), 'lr': 0.0},
        {'params': model.poly.parameters(), 'lr': 1e-3}
    ])
    for ep in range(501):
        opt_s3.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s3.step()

    return model, scale, params_ch0, params_ch1

# ==============================================================================
# 5. 验证工具 (新增 THD 和 频谱分析)
# ==============================================================================
def calculate_metrics_detailed(sim, model, scale, device, p_ch0, p_ch1):
    """
    [Corrected] 计算 SINAD, ENOB, THD, SFDR 的对比数据 (Pre vs Post)
    基于 TIADC 交织后的全速率数据进行计算，正确反映失配带来的影响。
    """
    print("\n=== 正在计算综合指标对比 (基于 TIADC 交织输出) ===")
    # 开启 FPGA 仿真模式进行验证，查看定点化影响
    model.fpga_simulation = True 
    print(">> 注意: 验证阶段已开启 FPGA 定点化模拟 (Fixed-Point Simulation)")
    
    test_freqs = np.arange(0.1e9, 9.6e9, 0.5e9)
    
    # 存储结果的字典
    metrics = {
        'sinad_pre': [], 'sinad_post': [],
        'enob_pre': [],  'enob_post': [],
        'thd_pre': [],   'thd_post': [],
        'sfdr_pre': [],  'sfdr_post': []
    }
    
    # 辅助：交织函数 (修正版：包含抽取)
    def interleave(c0_full, c1_full):
        # 模拟 20GSps TIADC：
        # ADC0 取偶数点 (0, 2, 4...)
        # ADC1 取奇数点 (1, 3, 5...)
        
        s0 = c0_full[0::2] # 降采样到 10GSps
        s1 = c1_full[1::2] # 降采样到 10GSps
        
        L = min(len(s0), len(s1))
        out = np.zeros(2*L)
        out[0::2] = s0[:L]
        out[1::2] = s1[:L]
        return out

    # 辅助：基于频谱计算指标
    def calc_spectrum_metrics(sig, fs, input_freq):
        # 加窗 (Blackman)
        win = np.blackman(len(sig))
        # 归一化因子 (Coherent Gain)
        cg = np.mean(win)
        
        # FFT
        fft_spec = np.fft.rfft(sig * win)
        fft_mag = np.abs(fft_spec) / (len(sig)/2 * cg) # 归一化幅度
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0/fs)
        
        # 1. 寻找基波峰值
        idx_fund = np.argmin(np.abs(fft_freqs - input_freq))
        span = 5
        s = max(0, idx_fund - span)
        e = min(len(fft_mag), idx_fund + span)
        idx_peak = s + np.argmax(fft_mag[s:e])
        
        p_fund = fft_mag[idx_peak]**2
        
        # 2. 计算噪声+失真 (NAD)
        # 将基波及其附近的能量剔除
        mask = np.ones_like(fft_mag, dtype=bool)
        mask[:5] = False # DC
        mask[idx_peak-span : idx_peak+span+1] = False # Fund
        
        # 剩余所有频点能量和
        p_noise_dist = np.sum(fft_mag[mask]**2) + 1e-20
        
        # SINAD
        sinad = 10 * np.log10(p_fund / p_noise_dist)
        
        # ENOB
        enob = (sinad - 1.76) / 6.02
        
        # SFDR (最大杂散)
        p_spur_max = np.max(fft_mag[mask]**2) if np.any(mask) else 1e-20
        sfdr = 10 * np.log10(p_fund / p_spur_max)
        
        # THD (前5次谐波)
        p_harm_sum = 0
        for h in range(2, 6):
            h_freq = input_freq * h
            if h_freq < fs/2: # Nyquist内
                idx_h = np.argmin(np.abs(fft_freqs - h_freq))
                s_h = max(0, idx_h - span)
                e_h = min(len(fft_mag), idx_h + span)
                if e_h > s_h:
                    p_harm_sum += np.max(fft_mag[s_h:e_h])**2
        
        thd = 10 * np.log10(p_harm_sum / p_fund + 1e-20)
        
        return sinad, enob, thd, sfdr
    
    for f in test_freqs:
        # 生成稍长的数据
        src = sim.generate_tone_data(f, N=8192*2)
        
        # Ch0 (Ref)
        y_ch0 = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch0)
        # Ch1 (Bad)
        y_ch1_raw = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch1)
        
        # Ch1 (Cal)
        with torch.no_grad():
            inp_t = torch.FloatTensor(y_ch1_raw/scale).view(1,1,-1).to(device)
            y_ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
            
        # 截断稳定区
        margin = 500
        c0 = y_ch0[margin:-margin]
        c1_raw = y_ch1_raw[margin:-margin]
        c1_cal = y_ch1_cal[margin:-margin]
        
        # 构建 TIADC 全速率信号
        sig_pre = interleave(c0, c1_raw)
        sig_post = interleave(c0, c1_cal)
        
        # 计算指标
        s_pre, e_pre, t_pre, sf_pre = calc_spectrum_metrics(sig_pre, sim.fs, f)
        s_post, e_post, t_post, sf_post = calc_spectrum_metrics(sig_post, sim.fs, f)
        
        metrics['sinad_pre'].append(s_pre)
        metrics['sinad_post'].append(s_post)
        metrics['enob_pre'].append(e_pre)
        metrics['enob_post'].append(e_post)
        metrics['thd_pre'].append(t_pre)
        metrics['thd_post'].append(t_post)
        metrics['sfdr_pre'].append(sf_pre)
        metrics['sfdr_post'].append(sf_post)

    # 验证完成后关闭 FPGA 仿真模式 (或者保持开启取决于后续用途)
    model.fpga_simulation = False
    return test_freqs, metrics

def analyze_and_plot_spectrum(sim, model, scale, device, p_ch0, p_ch1, plot_freq=2.1e9):
    """
    [新增] 频谱对比分析
    画出 Reference, Uncalibrated, Calibrated 三者的频谱对比
    这是论文中用于展示"镜像杂散消除"最直观的图
    """
    print(f"\n=== 正在绘制 {plot_freq/1e9}GHz 处的频谱对比图 ===")
    src = sim.generate_tone_data(plot_freq, N=16384) # 稍微长一点以获得高频率分辨率
    
    # 生成数据 (模拟 TIADC 交织输出)
    # Ref: Ideal TIADC (Ch0 + Ch0) -> 代表校准的终极目标
    # Bad: Uncal TIADC (Ch0 + Ch1_Bad) -> 存在镜像杂散
    # Cal: Calib TIADC (Ch0 + Ch1_Cal) -> 杂散应消失
    
    ch0_real = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch0)
    ch1_bad  = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch1)
    
    with torch.no_grad():
        inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
        ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
        
    def interleave(c0, c1):
        L = min(len(c0), len(c1))
        L -= L % 2
        out = np.zeros(L)
        out[0::2] = c0[:L:2]
        out[1::2] = c1[1:L:2]
        return out
        
    tiadc_ref = interleave(ch0_real, ch0_real)
    tiadc_bad = interleave(ch0_real, ch1_bad)
    tiadc_cal = interleave(ch0_real, ch1_cal)
    
    # 计算 PSD
    def get_psd(sig):
        win = np.blackman(len(sig))
        freqs, psd = signal.periodogram(sig, sim.fs, window='blackman', scaling='spectrum')
        psd_db = 10 * np.log10(psd + 1e-15)
        return freqs, psd_db

    f, psd_ref = get_psd(tiadc_ref)
    _, psd_bad = get_psd(tiadc_bad)
    _, psd_cal = get_psd(tiadc_cal)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(f/1e9, psd_bad, 'r', alpha=0.5, label='Uncalibrated (Mismatch Spur)')
    plt.plot(f/1e9, psd_cal, 'b', alpha=0.8, label='Calibrated (Proposed)')
    # plt.plot(f/1e9, psd_ref, 'k--', alpha=0.3, label='Reference (Ch0 Baseline)') # 可选
    
    # 标注镜像频率
    image_freq = sim.fs/2 - plot_freq
    plt.axvline(image_freq/1e9, color='g', linestyle='--', alpha=0.5, label='Image Freq Location')
    
    plt.title(f"TIADC Spectrum Analysis @ {plot_freq/1e9} GHz Input")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power Spectrum (dB)")
    plt.ylim(-120, 0)
    plt.legend()
    plt.grid(True)
    plt.show()

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sim = TIADCSimulator(fs=20e9)
    model, scale, p_ch0, p_ch1 = train_relative_calibration(sim, device)
    
    # 计算对比指标
    test_freqs, m = calculate_metrics_detailed(sim, model, scale, device, p_ch0, p_ch1)
    
    # 绘图
    plt.figure(figsize=(10, 13))
    
    # 1. SINAD 对比
    plt.subplot(4,1,1)
    plt.plot(test_freqs/1e9, m['sinad_pre'], 'r--o', alpha=0.5, label='Pre-Calib')
    plt.plot(test_freqs/1e9, m['sinad_post'], 'g-o', linewidth=2, label='Post-Calib')
    plt.title("SINAD Improvement")
    plt.ylabel("dB"); plt.legend(); plt.grid(True)
    
    # 2. ENOB 对比
    plt.subplot(4,1,2)
    plt.plot(test_freqs/1e9, m['enob_pre'], 'r--o', alpha=0.5, label='Pre-Calib')
    plt.plot(test_freqs/1e9, m['enob_post'], 'm-o', linewidth=2, label='Post-Calib')
    plt.title("ENOB Improvement")
    plt.ylabel("Bits"); plt.legend(); plt.grid(True)
    
    # 3. THD 对比
    plt.subplot(4,1,3)
    plt.plot(test_freqs/1e9, m['thd_pre'], 'r--o', alpha=0.5, label='Pre-Calib (Ch1 Raw)')
    plt.plot(test_freqs/1e9, m['thd_post'], 'b-o', linewidth=2, label='Post-Calib (Ch1 Cal)')
    # 可选：画出 Ch0 的 THD 作为基准线 (Target)
    # plt.axhline(-60, color='k', linestyle='--', alpha=0.3, label='Reference (Ch0)')
    plt.title("THD Comparison")
    plt.ylabel("dBc"); plt.legend(); plt.grid(True)

    # 4. SFDR 对比
    plt.subplot(4,1,4)
    plt.plot(test_freqs/1e9, m['sfdr_pre'], 'r--o', alpha=0.5, label='Pre-Calib (Ch1 Raw)')
    plt.plot(test_freqs/1e9, m['sfdr_post'], 'c-o', linewidth=2, label='Post-Calib (Ch1 Cal)')
    plt.title("SFDR Improvement")
    plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.show()