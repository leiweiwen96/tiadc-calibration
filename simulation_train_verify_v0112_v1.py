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
# 1. 物理仿真引擎
# ==============================================================================
class TIADCSimulator:
    def __init__(self, fs=20e9):
        self.fs = fs

    def fractional_delay(self, sig, delay):
        N = len(sig)
        X = np.fft.rfft(sig)
        k = np.arange(len(X))
        phase_shift = np.exp(-1j * 2 * np.pi * k * delay / N)
        return np.fft.irfft(X * phase_shift, n=N)

    def generate_tone_data(self, freq, N=8192):
        t = np.arange(N) / self.fs
        src = np.sin(2 * np.pi * freq * t)
        return src * 0.9

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0, 
                             jitter_std=100e-15, n_bits=12, 
                             hd2=0.0, hd3=0.0): 
        nyquist = self.fs / 2
        sig_nonlinear = sig + hd2 * (sig**2) + hd3 * (sig**3)
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig_nonlinear)
        sig_gain = sig_bw * gain
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        if jitter_std > 0:
            slope = np.gradient(sig_delayed) * self.fs
            dt_noise = np.random.normal(0, jitter_std, len(sig_delayed))
            sig_jittered = sig_delayed + slope * dt_noise
        else:
            sig_jittered = sig_delayed
            
        noise_floor = np.random.normal(0, 1e-4, len(sig_jittered))
        sig_noisy = sig_jittered + noise_floor
        
        if n_bits is not None:
            v_range = 2.0 
            levels = 2**n_bits
            step = v_range / levels
            sig_clipped = np.clip(sig_noisy, -1.0, 1.0)
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
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out

class HybridCalibrationModel(nn.Module): 
    def __init__(self, taps=511):
        super().__init__()
        self.poly = PolynomialCorrection(order=3) 
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        with torch.no_grad():
            self.conv.weight.data.fill_(0.0)
            self.conv.weight.data[0, 0, taps//2] = 1.0

    def forward(self, x):
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
        sig0 = simulator.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch0)
        ch0_captures.append(sig0)
        sig1 = simulator.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch1)
        ch1_captures.append(sig1)
    
    avg_ch0 = np.mean(np.stack(ch0_captures), axis=0) # Target
    avg_ch1 = np.mean(np.stack(ch1_captures), axis=0) # Input
    
    scale = np.max(np.abs(avg_ch0))
    inp_t = torch.FloatTensor(avg_ch1/scale).view(1, 1, -1).to(device)
    tgt_t = torch.FloatTensor(avg_ch0/scale).view(1, 1, -1).to(device)
    
    model = HybridCalibrationModel(taps=511).to(device)
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
    计算 SINAD, ENOB, THD, SFDR 的对比数据 (Pre vs Post)
    """
    print("\n=== 正在计算综合指标对比 (Pre vs Post) ===")
    test_freqs = np.arange(0.1e9, 9.6e9, 0.5e9)
    
    # 存储结果的字典
    metrics = {
        'sinad_pre': [], 'sinad_post': [],
        'enob_pre': [],  'enob_post': [],
        'thd_pre': [],   'thd_post': [],
        'sfdr_pre': [],  'sfdr_post': []
    }
    
    for f in test_freqs:
        src = sim.generate_tone_data(f)
        
        # 1. 生成物理信号
        # Ch0 (Ref)
        y_ref_all = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch0)
        # Ch1 (Bad) - 校准前
        y_bad_all = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch1)
        
        # 2. 模型校准 -> Ch1 (Calibrated)
        with torch.no_grad():
            inp_t = torch.FloatTensor(y_bad_all/scale).view(1,1,-1).to(device)
            y_cal_all = model(inp_t).cpu().numpy().flatten() * scale
            
        # 3. 截取稳态数据
        margin = 1000
        y_ref = y_ref_all[margin:-margin]
        y_bad = y_bad_all[margin:-margin]
        y_cal = y_cal_all[margin:-margin]
        
        # --- 辅助函数：计算单组数据的指标 ---
        def get_metrics(y_meas, y_golden):
            # A. 计算 SINAD & ENOB (基于时域误差)
            # 误差 = 测量值 - 黄金标准 (Ch0)
            error = y_meas - y_golden
            p_sig = np.mean(y_golden**2)
            p_nd = np.mean(error**2) + 1e-20
            
            sinad = 10 * np.log10(p_sig / p_nd)
            enob = (sinad - 1.76) / 6.02
            
            # B. 计算 THD (基于自身频谱)
            # THD 衡量的是波形自身的非线性，不需要对比 Reference
            win = np.blackman(len(y_meas))
            fft_mag = np.abs(np.fft.rfft(y_meas * win))
            # rFFT 对应的真实频率轴
            freqs = np.fft.rfftfreq(len(y_meas), d=1.0 / sim.fs)
            
            # 基波能量
            idx_fund = np.argmin(np.abs(freqs - f))
            s0 = max(0, idx_fund - 5)
            e0 = min(len(fft_mag), idx_fund + 5)
            p_fund = np.sum(fft_mag[s0:e0]**2) + 1e-20
            
            # 谐波能量 (2-5次)
            p_harm = 0
            for h in range(2, 6):
                h_freq = f * h
                if h_freq < sim.fs/2:
                    idx_h = np.argmin(np.abs(freqs - h_freq))
                    # 简单防止越界
                    s = max(0, idx_h-5); e = min(len(fft_mag), idx_h+5)
                    p_harm += np.sum(fft_mag[s:e]**2)
            
            thd = 10 * np.log10(p_harm / p_fund + 1e-20)

            # C. 计算 SFDR (基于自身频谱)
            # SFDR = 基波功率 / (最大杂散功率)，单位 dBc
            p_spec = fft_mag**2
            mask = np.ones_like(p_spec, dtype=bool)
            # 排除 DC 附近
            dc_guard = 3
            mask[:dc_guard] = False
            # 排除基波邻域
            mask[s0:e0] = False
            # 取剩余频点中的最大杂散
            p_spur = np.max(p_spec[mask]) if np.any(mask) else 1e-20
            sfdr = 10 * np.log10(p_fund / (p_spur + 1e-20))

            return sinad, enob, thd, sfdr

        # 4. 分别计算 校准前 和 校准后
        s_pre, e_pre, t_pre, sf_pre = get_metrics(y_bad, y_ref) # Bad vs Ref
        s_post, e_post, t_post, sf_post = get_metrics(y_cal, y_ref) # Cal vs Ref
        
        metrics['sinad_pre'].append(s_pre)
        metrics['sinad_post'].append(s_post)
        metrics['enob_pre'].append(e_pre)
        metrics['enob_post'].append(e_post)
        metrics['thd_pre'].append(t_pre)
        metrics['thd_post'].append(t_post)
        metrics['sfdr_pre'].append(sf_pre)
        metrics['sfdr_post'].append(sf_post)

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
    plt.xlabel("Frequency (GHz)")额
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