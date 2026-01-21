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

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0):
        nyquist = self.fs / 2
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig)
        sig_gain = sig_bw * gain
        sig_out = self.fractional_delay(sig_gain, delay_samples)
        return sig_out

    def generate_chirp_data(self, N=16384):
        t = np.arange(N) / self.fs
        src = signal.chirp(t, f0=1e6, t1=t[-1], f1=self.fs*0.48, method='linear')
        return src * 0.95
    
    def generate_mixed_training_data(sim, N=32768, device='cpu'):
        # 1. Chirp 信号 (保持原有)
        # 为了覆盖更低频，建议 f0 设为 0 或者极小值，但在 log chirp 下 0 会报错，线性 chirp 可以设 0
        t = np.arange(N) / sim.fs
        chirp_sig = signal.chirp(t, f0=0, t1=t[-1], f1=sim.fs*0.48, method='linear') * 0.95
    
        # 2. DC 常数信号 (关键！解决低频失效)
        # 模拟几个不同的直流电平，比如 0.5, -0.5, 0.8
        # 这样模型必须学会：输入 0.5 -> 输出 0.5 * Gain_Correction
        dc_val = np.random.uniform(-0.9, 0.9)
        dc_sig = np.ones(N) * dc_val

        # 3. 宽带白噪声 (关键！解决纹波)
        noise_sig = np.random.normal(0, 0.3, N)

        # 组合成一个 Batch (Batch Size = 3)
        # 这样一次反向传播就能同时优化 DC、相位和全频段
        raw_batch = np.stack([chirp_sig, dc_sig, noise_sig], axis=0) # Shape: [3, N]
    
        # --- 物理仿真 ---
        # 对这三种信号都通过 "坏通道" 和 "好通道"
    
        # ch0 (Target): 纯净通道
        target_batch = []
        input_batch = []
    
        for i in range(3):
            sig = raw_batch[i]
            # Target
            t = sim.apply_channel_effect(sig, cutoff_freq=6.0e9, delay_samples=0.0, gain=1.0)
            target_batch.append(t)
        
            # Input (Bad Channel)
            # 注意：这里模拟了真实的 Gain Mismatch (0.98)
            inp = sim.apply_channel_effect(sig, cutoff_freq=5.9e9, delay_samples=0.42, gain=0.98)
            input_batch.append(inp)
        
        target_batch = np.array(target_batch)
        input_batch = np.array(input_batch)
    
        # 归一化 (可选，但建议统一 Scale)
        global_scale = np.max(np.abs(target_batch))
        target_batch /= global_scale
        input_batch /= global_scale
    
        # 转 Tensor
        inp_t = torch.FloatTensor(input_batch).unsqueeze(1).to(device) # [3, 1, N]
        tgt_t = torch.FloatTensor(target_batch).unsqueeze(1).to(device) # [3, 1, N]
    
        return inp_t, tgt_t

    def generate_tone_data(self, freq, N=8192):
        t = np.arange(N) / self.fs
        src = np.sin(2 * np.pi * freq * t)
        return src * 0.9

# ==============================================================================
# 2. 模型 (升级为 511 点)
# ==============================================================================
class DFS_FIR_Filter(nn.Module):
    def __init__(self, taps=511): # [Change 1] 阶数增加
        super().__init__()
        self.taps = taps
        # padding='same' 保证输出长度不变，但边缘是脏的
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding='same', bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            # 生成一个以 taps//2 为中心的理想低通滤波器作为初始值（例如截止频率 0.4 fs）
            n = torch.arange(taps) - (taps - 1) / 2
            f_c = 0.4 # 归一化截止频率
            h = 2 * f_c * torch.sinc(2 * f_c * n)
            # 加上汉宁窗平滑
            window = torch.hann_window(taps)
            h_windowed = h * window
            # 赋值给权重
            self.conv.weight[0, 0, :] = h_windowed

    def forward(self, x):
        return self.conv(x)


# ==============================================================================
# 3. Loss (适配长滤波器)
# ==============================================================================
class HighPrecisionLoss(nn.Module):
    def __init__(self, taps=511):
        super().__init__()
        self.mse = nn.MSELoss()
        self.taps = taps
        # 移除固定的线性权重，或者改为全 1
        self.freq_weights = None

    def forward(self, y_pred, y_target, model):
        crop = 300 
        y_p_val = y_pred[..., crop:-crop]
        y_t_val = y_target[..., crop:-crop]

        # 1. 时域 Loss (保持权重)
        l_time = self.mse(y_p_val, y_t_val)
        
        # 2. 频域 Loss (移除线性加权，改为对数幅度 Loss 更符合听感/射频指标)
        fft_p = torch.fft.rfft(y_p_val, dim=-1, norm='ortho').abs()
        fft_t = torch.fft.rfft(y_t_val, dim=-1, norm='ortho').abs()
        
        # 使用对数差值 (Log-Spectral Distance 变体)，这能均衡大小信号的贡献
        # 防止大信号掩盖小信号误差
        l_freq = torch.mean(torch.abs(torch.log10(fft_p + 1e-8) - torch.log10(fft_t + 1e-8)))
        
        # 3. 正则化 (大幅降低)
        w = model.conv.weight
        # 降低权重，给 511 个点自由度
        l_reg = torch.mean((w[:,:,1:] - w[:,:,:-1])**2)

        current_dc_gain = torch.sum(model.conv.weight)
        
        # 你的目标增益应该是 1.0 (因为你在数据预处理时已经把 ch0 和 ch1 的幅度归一化了)
        # 如果 ch1_bad 还是原始幅度，这里目标可能是 1.0/0.98，根据你的 scale 策略定
        # 这里假设输入数据已经做了幅度对齐，或者模型需要学出这个 scale
        
        # 简单粗暴的方法：直接计算输入输出的均值差
        # 或者更严谨地，假设目标就是让系数和维持在某个值（通常是 1.0）
        target_dc_gain = 1.0 
        l_dc = (current_dc_gain - target_dc_gain) ** 2
        
        # 修改总 Loss 权重组合
        # 强调频域对数 Loss，降低正则化
        w = model.conv.weight
        
        # 1. 平滑度 (Smoothness): 抑制高频噪声
        l_smooth = torch.mean((w[:,:,1:] - w[:,:,:-1])**2)
        
        # 2. 稀疏/衰减约束 (Decay): 强迫远离中心的系数衰减为0
        # 这能有效减少“截断效应”带来的频域纹波
        center = self.taps // 2
        indices = torch.arange(self.taps).to(w.device)
        # 距离中心越远，惩罚越大
        spatial_weight = ((indices - center).abs() / center) ** 2
        l_decay = torch.mean(w**2 * spatial_weight)

        total = 100.0 * l_time + 10.0 * l_freq + 10.0 * l_smooth + 10.0 * l_decay + 1000.0 * l_dc
        return total

# ==============================================================================
# 4. 训练流程
# ==============================================================================
def train_ddsp(simulator, device='cpu'):
    print("\n=== 1. 生成训练数据 (Chirp) ===")
    # 增加数据长度，喂饱 511 阶滤波器
    N_train = 32768 
    raw_src = simulator.generate_chirp_data(N=N_train)
    
    # Ch1 对齐 Ch0 (不延展带宽)
    ch0_real = simulator.apply_channel_effect(raw_src, cutoff_freq=6.0e9, delay_samples=0.0, gain=1.0)
    ch1_bad = simulator.apply_channel_effect(raw_src, cutoff_freq=5.9e9, delay_samples=0.42, gain=0.98)
    
    scale = np.max(np.abs(ch0_real))
    ch0_real /= scale
    ch1_bad /= scale
    
    inp = torch.FloatTensor(ch1_bad).view(1, 1, -1).to(device)
    tgt = torch.FloatTensor(ch0_real).view(1, 1, -1).to(device)
    
    model = DFS_FIR_Filter(taps=511).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = HighPrecisionLoss(taps=511)
    
    print("\n=== 2. 开始训练 (511 Taps) ===")
    # 511参数多，收敛可能稍微慢一点点，多跑几轮
    for epoch in range(3001): 
        optimizer.zero_grad()
        out = model(inp)
        loss = loss_fn(out, tgt, model)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")
            
    return model, scale

# ==============================================================================
# 5. 验证工具
# ==============================================================================
def calculate_metrics(sim, model, scale, freq, device):
    src = sim.generate_tone_data(freq)
    ch0 = sim.apply_channel_effect(src, 6.0e9, 0.0, 1.0)
    ch1_bad = sim.apply_channel_effect(src, 5.9e9, 0.42, 0.98)
    
    with torch.no_grad():
        inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
        ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
    
    # [Change 4] 验证时的 Margin 也要加大
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

    spur_bad, spec_bad, _ = get_spur(tiadc_bad)
    spur_cal, spec_cal, f_axis = get_spur(tiadc_cal)
    
    return spur_bad - spur_cal, spec_bad, spec_cal, f_axis

# ==============================================================================
# 6. 主程序
# ==============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device} with 511 Taps Filter")
    
    sim = TIADCSimulator(fs=20e9)
    model, scale = train_ddsp(sim, device)
    
    print("\n=== 3. 运行扫频验证 ===")
    
    raw_freqs = np.arange(0.1e9, 9.65e9, 0.137e9)
    safe_freqs = [f for f in raw_freqs if abs(f - 5.0e9) > 0.15e9]
    
    improvements = []
    print(f"扫描 {len(safe_freqs)} 个频点...", end="")
    for i, f in enumerate(safe_freqs):
        imp, _, _, _ = calculate_metrics(sim, model, scale, f, device)
        improvements.append(imp)
        if i % 10 == 0: print(".", end="", flush=True)
    print(" 完成！")
    
    # 绘图
    plt.figure(figsize=(12, 10))
    
    # SFDR
    plt.subplot(2, 1, 1)
    plt.plot(np.array(safe_freqs)/1e9, improvements, 'b-', linewidth=2)
    plt.title("SFDR Improvement (511 Taps High Precision)")
    plt.xlabel("GHz"); plt.ylabel("dB"); plt.grid(True)
    plt.ylim(0, 85) # 511点可能会刷出更高的分数
    
    # 系数形状
    coeffs = model.conv.weight.detach().cpu().numpy().flatten()
    plt.subplot(2, 1, 2)
    plt.plot(coeffs)
    plt.title(f"Learned FIR Coefficients (Taps={len(coeffs)})")
    plt.xlabel("Tap Index")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()