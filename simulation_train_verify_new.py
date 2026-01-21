import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# 锁定随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. 物理仿真引擎 (保持不变)
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

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0):
        nyquist = self.fs / 2
        
        # 1. 带宽限制 (Linear Filter)
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig)
        
        # 2. 增益
        # 模拟放大器的非线性： y = ax + bx^3
        sig_gain = sig_bw * gain
        
        # 3. 分数延时 (Timing Skew)
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 4. 加入热噪声 (Additive White Gaussian Noise) [新增]
        noise_floor = np.random.normal(0, 1e-4, len(sig_delayed)) # -80dB 左右的底噪
        sig_noisy = sig_delayed + noise_floor
        
        # 5. ADC 量化 (比如 12-bit) [新增]
        # 假设信号范围是 -1 到 1
        bits = 12
        levels = 2**bits
        sig_quantized = np.round(sig_noisy * (levels/2)) / (levels/2)
        
        return sig_quantized

    def prepare_training_batch(self, N=32768):
        # 保持你刚才的黄金组合
        t = np.arange(N) / self.fs
        chirp_lin = signal.chirp(t, f0=1e6, t1=t[-1], f1=self.fs*0.48, method='linear')
        chirp_log = signal.chirp(t, f0=1e5, t1=t[-1], f1=self.fs*0.48, method='logarithmic')
        chirp_mid = signal.chirp(t, f0=1e7, t1=t[-1], f1=2.0e9, method='linear')
        step = np.zeros(N); step[:N//4]=0.5; step[N//4:N//2]=-0.5; step[N//2:3*N//4]=0.8; step[3*N//4:]=-0.2
        white = np.random.randn(N); b, a = signal.butter(1, 0.02, btype='low'); pink = signal.lfilter(b, a, white); pink /= np.std(pink) * 3
        return np.stack([chirp_lin, chirp_log, chirp_mid, step, pink], axis=0)

    def generate_tone_data(self, freq, N=8192):
        t = np.arange(N) / self.fs
        src = np.sin(2 * np.pi * freq * t)
        return src * 0.9

# ==============================================================================
# 升级版模型：Hybrid V2 (Non-linear + Linear)
# ==============================================================================
class HybridCalibrationModel_V3(nn.Module):
    def __init__(self, taps=511):
        super().__init__()
        
        # 1. 先修延时 (最关键，必须先对齐时间)
        self.global_delay = DifferentiableDelay(init_delay=0.0)
        
        # 2. 再修增益 (把幅度归一化)
        self.gain = nn.Parameter(torch.tensor([1.0]))
        
        # 4. 最后 FIR 扫尾 (修残差)
        self.taps = taps
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding='same', bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, taps // 2] = 1.0

    def forward(self, x):
        # 顺序变了！
        x_d = self.global_delay(x)  # Step 1: Undo Delay
        x_g = x_d * self.gain       # Step 2: Undo Gain
        out = self.conv(x_g)        # Step 4: Residual
        return out

class DifferentiableDelay(nn.Module):
    """可微分的纯延时层 (基于频域相位旋转)"""
    def __init__(self, init_delay=0.0):
        super().__init__()
        # 初始化为一个可学习的参数
        self.delay = nn.Parameter(torch.tensor([init_delay]))

    def forward(self, x):
        # x shape: [Batch, 1, N]
        N = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        k = torch.arange(X.shape[-1], device=x.device)
        
        # 核心：e^(-j * 2pi * k * delay / N)
        # 注意：这里我们是要“反向补偿”延时，所以如果是校准，模型应该学出 -0.42
        # 或者我们定义这个层是施加延时，模型学出 -0.42 来抵消物理延时
        phase_shift = torch.exp(-1j * 2 * np.pi * k * self.delay / N)
        
        return torch.fft.irfft(X * phase_shift, n=N, dim=-1)

# ==============================================================================
# 3. Loss (保持你刚才的优秀 Loss)
# ==============================================================================
class HighPrecisionLoss(nn.Module):
    def __init__(self, taps=511, device='cpu'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.freq_weights = None

    def forward(self, y_pred, y_target, model):
        crop = 300 
        y_p_val = y_pred[..., crop:-crop]
        y_t_val = y_target[..., crop:-crop]

        # 1. 时域 Loss
        l_time = self.mse(y_p_val, y_t_val)
         

        # 2. 频域 Loss
        fft_p = torch.fft.rfft(y_p_val, dim=-1, norm='ortho').abs()
        fft_t = torch.fft.rfft(y_t_val, dim=-1, norm='ortho').abs()
        
        if self.freq_weights is None or self.freq_weights.shape[0] != fft_p.shape[-1]:
            freq_bins = fft_p.shape[-1]
            w = torch.ones(freq_bins).to(y_p_val.device)
            w[:100] = 50.0  # DC-60M
            w[100:1600] = 20.0 # 60M-1G
            w[1600:3200] = 5.0 # 1G-2G
            self.freq_weights = w
        
        log_diff = torch.abs(torch.log10(fft_p + 1e-8) - torch.log10(fft_t + 1e-8))
        l_freq = torch.mean(log_diff * self.freq_weights)
        
        # 3. DC Loss
        l_dc_signal = self.mse(torch.mean(y_p_val, dim=-1), torch.mean(y_t_val, dim=-1))

        # 4. 正则化 (针对 FIR)
        # 因为我们有显式 Delay 层，FIR 的系数应该非常接近 Delta 函数
        # 所以我们可以加强 "Compactness" 约束，防止它乱跑
        w_coef = model.conv.weight
        l_smooth = torch.mean((w_coef[:,:,1:] - w_coef[:,:,:-1])**2)
        
        total = 100.0 * l_time + 20.0 * l_freq + 1000.0 * l_dc_signal + 1.0 * l_smooth
        return total

# ==============================================================================
# 4. 训练流程
# ==============================================================================
def train_ddsp(simulator, device='cpu'):
    N_train = 32768
    # 必须确保训练数据包含非线性！
    raw_batch = simulator.prepare_training_batch(N=N_train) 
    
    input_list = []
    target_list = []
    
    for i in range(raw_batch.shape[0]):
        raw_sig = raw_batch[i]
        # Target: 纯净通道
        ch0 = simulator.apply_channel_effect(raw_sig, cutoff_freq=6.0e9, delay_samples=0.0, gain=1.0)
        
        # Input: 恶劣通道 (必须使用 apply_channel_effect_realistic)
        # 假设你之前已经定义了 realistic 函数，包含 hd3=0.01
        # 如果没有，请确保这里产生的数据确实带有三次谐波
        # 这里模拟 HD3 = 1% (-40dB)
        ch1_nonlinear = raw_sig + 0.01 * (raw_sig**3) 
        ch1_bad = simulator.apply_channel_effect(ch1_nonlinear, cutoff_freq=5.9e9, delay_samples=0.42, gain=0.98)
        
        target_list.append(ch0)
        input_list.append(ch1_bad)
    
    target_arr = np.array(target_list)
    input_arr = np.array(input_list)
    
    scale = np.max(np.abs(target_arr[0]))
    target_arr /= scale
    input_arr /= scale
    
    inp_t = torch.FloatTensor(input_arr).unsqueeze(1).to(device)
    tgt_t = torch.FloatTensor(target_arr).unsqueeze(1).to(device)
    
    model = HybridCalibrationModel_V3(taps=511).to(device)
    loss_fn = HighPrecisionLoss(taps=511, device=device)

    optimizer_p1 = optim.Adam([
        {'params': model.conv.parameters(), 'lr': 1e-4},
        {'params': model.gain, 'lr': 1e-4},
        {'params': model.global_delay.parameters(), 'lr': 1e-4} 
    ])
    
    scheduler_p1 = optim.lr_scheduler.StepLR(optimizer_p1, step_size=200, gamma=0.1)
    
    for epoch in range(2001): # 稍微减少轮数，600轮足够了
        optimizer_p1.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        optimizer_p1.step()
        scheduler_p1.step() # 更新学习率
        
        if epoch % 100 == 0:
            current_lr = scheduler_p1.get_last_lr()[0] # 监控 LR
            print(f"P1 | Ep {epoch} | Loss: {loss.item():.4f} | Gain: {model.gain.item():.4f} | Delay: {model.global_delay.delay.item():.4f} | LR: {current_lr:.1e}")

    return model, scale
# ==============================================================================
# 5. 验证与绘图 (微调：验证时要把 Gain/Delay 都算上)
# ==============================================================================
def calculate_metrics(sim, model, scale, freq, device):
    src = sim.generate_tone_data(freq)
    ch0 = sim.apply_channel_effect(src, 6.0e9, 0.0, 1.0)
    ch1_bad = sim.apply_channel_effect(src, 5.9e9, 0.42, 0.98)
    
    with torch.no_grad():
        inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
        ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
    
    margin = 600 
    c0 = ch0[margin:-margin]
    c1_b = ch1_bad[margin:-margin]
    c1_c = ch1_cal[margin:-margin]
    
    def make_tiadc(c_even, c_odd):
        L = min(len(c_even), len(c_odd)); L -= L%2
        out = np.zeros(L); out[0::2] = c_even[:L:2]; out[1::2] = c_odd[1:L:2]
        return out
    
    tiadc_bad = make_tiadc(c0, c1_b)
    tiadc_cal = make_tiadc(c0, c1_c)
    
    def get_spur(sig):
        win = np.blackman(len(sig))
        spec = 20*np.log10(np.abs(np.fft.rfft(sig*win)) + 1e-12)
        spec -= np.max(spec)
        freqs = np.linspace(0, sim.fs/2, len(spec))
        spur_f = sim.fs/2 - freq
        idx = np.argmin(np.abs(freqs - spur_f))
        window = 10
        spur_amp = np.max(spec[max(0, idx-window):min(len(spec), idx+window)])
        return spur_amp, spec, freqs

    spur_bad, _, _ = get_spur(tiadc_bad)
    spur_cal, _, _ = get_spur(tiadc_cal)
    
    return spur_bad - spur_cal

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device} with Hybrid Architecture")
    
    sim = TIADCSimulator(fs=20e9)
    model, scale = train_ddsp(sim, device)
    
    print("\n=== 3. 运行扫频验证 ===")
    # 频率列表
    raw_freqs = np.concatenate([
        np.linspace(0.01e9, 1.0e9, 20),
        np.linspace(1.1e9, 3.0e9, 10),
        np.arange(3.1e9, 9.65e9, 0.25e9)
    ])
    raw_freqs.sort()
    safe_freqs = [f for f in raw_freqs if abs(f - 5.0e9) > 0.15e9]
    
    improvements = []
    print(f"扫描 {len(safe_freqs)} 个频点...", end="")
    for i, f in enumerate(safe_freqs):
        imp = calculate_metrics(sim, model, scale, f, device)
        improvements.append(imp)
        if i % 10 == 0: print(".", end="", flush=True)
    print(" 完成！")
    
    # 计算移动平均
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # 原始数据 (半透明)
    plt.plot(np.array(safe_freqs)/1e9, improvements, 'r.-', alpha=0.3, linewidth=1)
    # 平滑数据 (实线)
    window_size = 5
    smooth_data = moving_average(improvements, window_size)
    # 补齐坐标轴
    smooth_freqs = safe_freqs[window_size-1:]

    plt.figure(figsize=(10, 5))
    # plt.plot(np.array(safe_freqs)/1e9, improvements, 'r.-', linewidth=2, label='Hybrid Model')
    plt.plot(np.array(smooth_freqs)/1e9, smooth_data, 'r-', linewidth=2.5, label='Hybrid Model (Trend)')
    plt.title("SFDR Improvement (Structured DDSP: Gain+Delay+FIR)")
    plt.xlabel("GHz"); plt.ylabel("dB"); plt.grid(True)
    plt.ylim(0, 90) 
    plt.legend()
    plt.show()