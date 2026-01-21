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

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0):
        """
        施加通道效应：
        1. 带宽限制 (Butterworth Lowpass)
        2. 纯线性增益 (Gain Mismatch)
        3. 时间偏差 (Timing Skew)
        4. 量化与噪声 (可选，这里为了验证算法极限暂时保留底噪，但去掉了强非线性)
        """
        nyquist = self.fs / 2
        
        # 1. 带宽限制
        # 注意：如果截止频率过低，高频信息丢失是无法找回的，这里设为 6G/5.9G 比较温和
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig)
        
        # 2. 增益 (纯线性)
        sig_gain = sig_bw * gain
        
        # 3. 分数延时
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 4. 极低底噪 (防止除零或完全过拟合，模拟理想环境)
        noise_floor = np.random.normal(0, 1e-6, len(sig_delayed)) 
        sig_out = sig_delayed + noise_floor
        
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
class HighPrecisionLoss(nn.Module):
    def __init__(self, fs=20e9, device='cpu'):
        super().__init__()
        self.mse = nn.MSELoss()
        # 预计算频率向量用于归一化
        self.freq_weights = None
        self.fs = fs

    def forward(self, y_pred, y_target, model):
        crop = 300 
        y_p_val = y_pred[..., crop:-crop]
        y_t_val = y_target[..., crop:-crop]

        # 1. 时域 Loss (基础保底)
        l_time = self.mse(y_p_val, y_t_val)
        
        # 2. 频域双重归一化 Loss (Dual-Normalized Loss)
        # ---------------------------------------------------------
        fft_p = torch.fft.rfft(y_p_val, dim=-1, norm='ortho')
        fft_t = torch.fft.rfft(y_t_val, dim=-1, norm='ortho')
        
        # [归一化 1]: 相对误差 (Relative Complex Error)
        # 解决带宽滚降导致的高频被忽略问题
        # 无论信号强弱，只关注“偏离了百分之多少”
        diff_complex = fft_p - fft_t
        mag_target = torch.abs(fft_t) + 1e-8 # 加上小量防止除零
        relative_error = torch.abs(diff_complex) / mag_target
        
        # [归一化 2]: 1/f 物理补偿 (Physics Compensation)
        # 解决相位杠杆导致的低频被忽略问题
        # 将“相位误差”还原为“时延误差”，实现全频段公平
        if self.freq_weights is None or self.freq_weights.shape[0] != fft_p.shape[-1]:
            freq_bins = fft_p.shape[-1]
            # 生成频率轴: 0 ~ fs/2
            f_axis = torch.linspace(0, self.fs/2, freq_bins, device=y_p_val.device)
            f_axis[0] = f_axis[1] # 避免 DC 除零
            
            # 权重 = 1 / f
            # 乘以一个常数 fs/2 是为了让权重数值不要太小，方便训练，不影响梯度方向
            self.freq_weights = (self.fs / 2) / f_axis
            
        # 组合：全频段归一化误差
        normalized_spectral_error = relative_error * self.freq_weights
        
        # 3. Top-K 策略 (依然保留，用于自动聚焦难点)
        # 这里的难点不再受信号强度和频率影响，而是真正的“拟合难点”
        diff_flat = normalized_spectral_error.view(-1)
        k = int(diff_flat.shape[0] * 0.25) # 关注最差的 25%
        top_k_error, _ = torch.topk(diff_flat, k)
        
        l_freq = torch.mean(top_k_error)
        
        # 4. 正则化 (通用场景建议给极小值或 0)
        # 因为相对误差对零点附近的扰动很敏感，如果正则化太强会限制 FIR 修复能力
        l_smooth = 0.0 
        if hasattr(model, 'conv'):
             w_coef = model.conv.weight
             l_smooth = torch.mean((w_coef[:,:,1:] - w_coef[:,:,:-1])**2)
        
        # 权重建议：l_freq 现在是相对值，数值通常在 0.01~1.0 之间
        # 给一个较大的系数让它主导梯度
        return 10.0 * l_time + 100.0 * l_freq + 0.0 * l_smooth
    
# 效果不错  
# class HighPrecisionLoss(nn.Module):
#     def __init__(self, taps=511, device='cpu'):
#         super().__init__()
#         self.mse = nn.MSELoss()

#     def forward(self, y_pred, y_target, model):
#         crop = 300 
#         y_p_val = y_pred[..., crop:-crop]
#         y_t_val = y_target[..., crop:-crop]

#         # 1. 时域 Loss (基础对齐，包含所有相位信息)
#         # 这一项其实就是全频段的 Complex Loss，但权重不好调，作为保底
#         l_time = self.mse(y_p_val, y_t_val)
        
#         # 2. 频域复数 Loss (Complex Frequency Loss)
#         # 关键：这里不要加 .abs()，我们要保留复数形式！
#         fft_p = torch.fft.rfft(y_p_val, dim=-1, norm='ortho')
#         fft_t = torch.fft.rfft(y_t_val, dim=-1, norm='ortho')
        
#         # 计算复数距离 (Complex Distance)
#         # error_spectrum 包含了 幅度误差 和 相位误差
#         # diff = a+bj, abs(diff) = sqrt(a^2 + b^2)
#         diff_complex = fft_p - fft_t
#         error_spectrum = torch.abs(diff_complex) 
        
#         # --- Top-K 聚焦策略 (依然保留) ---
#         # 找出模长误差最大的那些频点 (无论是相位歪了还是幅度歪了)
#         diff_flat = error_spectrum.view(-1)
#         k = int(diff_flat.shape[0] * 0.25) # 关注最差的 25%
#         top_k_error, _ = torch.topk(diff_flat, k)
        
#         l_freq = torch.mean(top_k_error)
        
#         # 3. 正则化 (保持微小)
#         w_coef = model.conv.weight
#         l_smooth = torch.mean((w_coef[:,:,1:] - w_coef[:,:,:-1])**2)
        
#         # 权重分配：
#         # 复数 Loss 的数值通常比 Log Magnitude Loss 小很多 (因为没有 log)
#         # 所以这里给 l_freq 的权重需要非常大，比如 2000.0 或 5000.0
#         return 10.0 * l_time + 5000.0 * l_freq + 0.001 * l_smooth
    
# class HighPrecisionLoss(nn.Module):
#     def __init__(self, taps=511, device='cpu'):
#         super().__init__()
#         self.mse = nn.MSELoss()

#     def forward(self, y_pred, y_target, model):
#         crop = 300 
#         y_p_val = y_pred[..., crop:-crop]
#         y_t_val = y_target[..., crop:-crop]

#         # 1. 时域 Loss (基础对齐)
#         l_time = self.mse(y_p_val, y_t_val)
        
#         # 2. 频域 Top-K Loss (核心修改)
#         fft_p = torch.fft.rfft(y_p_val, dim=-1, norm='ortho').abs()
#         fft_t = torch.fft.rfft(y_t_val, dim=-1, norm='ortho').abs()
        
#         # 计算对数谱差异 (Log Spectral Distance)
#         log_diff = torch.abs(torch.log10(fft_p + 1e-9) - torch.log10(fft_t + 1e-9))
        
#         # --- 自动聚焦策略 ---
#         # 展平所有频点的误差
#         diff_flat = log_diff.view(-1)
        
#         # 动态挖掘：只优化误差最大的 25% 频点
#         # 无论凹陷出现在中频还是高频，它们都会被 Top-K 选中
#         k = int(diff_flat.shape[0] * 0.25)
#         top_k_error, _ = torch.topk(diff_flat, k)
        
#         l_freq = torch.mean(top_k_error)
        
#         # 3. 正则化 
#         # 在 Stage 2 我们希望 FIR 能够为了填坑而自由弯曲，所以正则化系数给极小
#         w_coef = model.conv.weight
#         l_smooth = torch.mean((w_coef[:,:,1:] - w_coef[:,:,:-1])**2)
        
#         # 组合 Loss (大幅提高频域权重)
#         total = 10.0 * l_time + 1000.0 * l_freq + 0.001 * l_smooth
#         return total

# class HighPrecisionLoss(nn.Module):
#     def __init__(self, taps=511, device='cpu'):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.freq_weights = None

#     def forward(self, y_pred, y_target, model):
#         # 去除边缘效应
#         crop = 300 
#         y_p_val = y_pred[..., crop:-crop]
#         y_t_val = y_target[..., crop:-crop]

#         # 1. 时域 Loss (基础对齐)
#         l_time = self.mse(y_p_val, y_t_val)
        
#         # 2. 频域 Loss (幅频响应一致性) 
#         fft_p = torch.fft.rfft(y_p_val, dim=-1, norm='ortho').abs()
#         fft_t = torch.fft.rfft(y_t_val, dim=-1, norm='ortho').abs()
        
#         # 动态初始化频率权重
#         if self.freq_weights is None or self.freq_weights.shape[0] != fft_p.shape[-1]:
#             freq_bins = fft_p.shape[-1]
#             w = torch.ones(freq_bins, device=y_p_val.device)
#             # 这里的权重策略可以根据实际凹陷调整，但在纯线性模式下，平坦权重通常也工作得很好
#             # 这里稍微加强中低频，保证基础准确
#             w[:] = 20.0  

#             # k_start = int(4.0e9 / (10e9 / freq_bins))
#             # w[k_start:] = 10.0  # 给高频 20倍权重
            
#             self.freq_weights = w
        
#         # 对数谱差异 (Log Spectral Distance)
#         # 加上 1e-9 防止 log(0)
#         log_diff = torch.abs(torch.log10(fft_p + 1e-9) - torch.log10(fft_t + 1e-9))
#         l_freq = torch.mean(log_diff * self.freq_weights)
        
#         # 3. DC Loss (直流分量)
#         l_dc = self.mse(torch.mean(y_p_val, dim=-1), torch.mean(y_t_val, dim=-1))

#         # 4. 正则化 (让 FIR 保持平滑，不要产生剧烈的梳状滤波)
#         w_coef = model.conv.weight
#         l_smooth = torch.mean((w_coef[:,:,1:] - w_coef[:,:,:-1])**2)
        
#         # 组合 Loss
#         total = 100.0 * l_time + 50.0 * l_freq + 100.0 * l_dc + 0.1 * l_smooth
#         return total

# ==============================================================================
# 4. 训练流程 (去除人为非线性)
# ==============================================================================
def train_linear_calibration(simulator, device='cpu'):
    N_train = 32768
    
    # --- A. 准备数据 ---
    # [数据组 1]: 纯低频 (<100MHz) -> 用于锁 Delay/Gain
    t = np.arange(N_train) / simulator.fs
    low_freq_sig = signal.chirp(t, f0=1e6, t1=t[-1], f1=100e6, method='linear')
    batch_low = np.stack([low_freq_sig, low_freq_sig * 0.5, -low_freq_sig], axis=0) 
    
    # [数据组 2]: 宽带噪声 -> 用于修 FIR
    raw_batch_high = simulator.prepare_training_batch(N=N_train)
    noise = np.random.randn(*raw_batch_high.shape)
    b, a = signal.butter(1, 0.8, btype='high') 
    noise = signal.lfilter(b, a, noise) * 0.5
    batch_high = raw_batch_high + noise

    # 辅助函数
    def make_pairs(batch_data):
        in_l, tgt_l = [], []
        for i in range(batch_data.shape[0]):
            raw = batch_data[i]
            # Target: 6.0G
            tgt = simulator.apply_channel_effect(raw, 6.0e9, 0.0, 1.0)
            # Input: 5.9G, Delay 0.42 (物理值)
            inp = simulator.apply_channel_effect(raw, 5.9e9, 0.42, 0.98)
            in_l.append(inp)
            tgt_l.append(tgt)
        
        t_arr = np.array(tgt_l)
        i_arr = np.array(in_l)
        s = np.max(np.abs(t_arr))
        return (torch.FloatTensor(i_arr/s).unsqueeze(1).to(device), 
                torch.FloatTensor(t_arr/s).unsqueeze(1).to(device), s)

    inp_low, tgt_low, _ = make_pairs(batch_low)
    inp_high, tgt_high, scale_high = make_pairs(batch_high)
    
    model = HybridCalibrationModel_V3(taps=511).to(device)
    loss_fn = HighPrecisionLoss(device=device) # 使用新的 Top-K Loss
    mse_loss = nn.MSELoss()
    
    # =========================================================
    # Stage 1: 低频锚定 (Low-Freq Anchoring)
    # 目标：利用无群延时效应的低频数据，锁定物理 Delay (-0.46左右)
    # =========================================================
    print("=== Stage 1: Low-Freq Anchoring (<100MHz) ===")
    
    optimizer_s1 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 1e-2},
        {'params': model.gain, 'lr': 1e-2},
        {'params': model.conv.parameters(), 'lr': 0.0} # 锁死 FIR
    ])
    
    for epoch in range(301):
        optimizer_s1.zero_grad()
        output = model(inp_low)     # 只喂低频
        loss = mse_loss(output, tgt_low) # 用简单 MSE
        loss.backward()
        optimizer_s1.step()
        
        if epoch % 50 == 0:
            print(f"S1 Ep {epoch} | MSE: {loss.item():.6f} | "
                  f"Delay: {model.global_delay.delay.item():.4f} | "
                  f"Gain: {model.gain.item():.4f}")

    # =========================================================
    # Stage 2: Top-K 自动修补 (Top-K Auto Filling)
    # 目标：FIR 自动弯曲去填补误差最大的频段 (中频凹陷/高频滚降)
    # =========================================================
    print("\n=== Stage 2: Top-K FIR Fine Tuning ===")
    
    optimizer_s2 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0}, # 彻底锁死
        {'params': model.gain, 'lr': 0.0},                      # 彻底锁死
        {'params': model.conv.parameters(), 'lr': 5e-4}         # FIR 学习率
    ])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer_s2, step_size=500, gamma=0.5)

    for epoch in range(1501):
        optimizer_s2.zero_grad()
        output = model(inp_high)    # 喂宽带数据
        loss = loss_fn(output, tgt_high, model) # 使用 Top-K Loss
        loss.backward()
        optimizer_s2.step()
        scheduler.step()
        
        if epoch % 200 == 0:
            print(f"S2 Ep {epoch} | Top-K Loss: {loss.item():.4f} | "
                  f"Delay: {model.global_delay.delay.item():.4f} (Locked)")

    return model, scale_high

# def train_linear_calibration(simulator, device='cpu'):
#     N_train = 32768
#     # 1. 准备数据 (增加高频噪声，保持不变)
#     raw_batch = simulator.prepare_training_batch(N=N_train) 
#     high_freq_noise = np.random.randn(*raw_batch.shape)
#     b, a = signal.butter(1, 0.8, btype='high') 
#     high_freq_noise = signal.lfilter(b, a, high_freq_noise) * 0.5
#     raw_batch = raw_batch + high_freq_noise

#     # 归一化输入/输出
#     target_list = []
#     input_list = []
#     for i in range(raw_batch.shape[0]):
#         raw_sig = raw_batch[i]
#         target_sig = simulator.apply_channel_effect(raw_sig, 6.0e9, 0.0, 1.0)
#         input_sig = simulator.apply_channel_effect(raw_sig, 5.9e9, 0.42, 0.98)
#         target_list.append(target_sig)
#         input_list.append(input_sig)
    
#     target_arr = np.array(target_list)
#     input_arr = np.array(input_list)
#     scale = np.max(np.abs(target_arr))
#     target_arr /= scale
#     input_arr /= scale
    
#     inp_t = torch.FloatTensor(input_arr).unsqueeze(1).to(device)
#     tgt_t = torch.FloatTensor(target_arr).unsqueeze(1).to(device)
    
#     model = HybridCalibrationModel_V3(taps=511).to(device)
#     # Loss 先不用，Stage 1 我们手动算简单的 MSE
#     loss_fn = HighPrecisionLoss(taps=511, device=device)
    
#     # =========================================================
#     # Stage 1: 时域强行对齐 (Time Alignment)
#     # 目的：利用 MSE 对时延的极高敏感度，精准锁定 -0.42
#     # =========================================================
#     print("=== Stage 1: Time Domain Alignment (Delay/Gain Only) ===")
    
#     optimizer_stage1 = optim.Adam([
#         {'params': model.global_delay.parameters(), 'lr': 5e-3}, # 调小一点，防止过冲
#         {'params': model.gain, 'lr': 5e-3},
#         {'params': model.conv.parameters(), 'lr': 0.0}           # 锁死 FIR
#     ])
    
#     mse_loss = nn.MSELoss()

#     for epoch in range(501):
#         optimizer_stage1.zero_grad()
#         output = model(inp_t)
        
#         # 关键：只看中间部分的时域波形差异
#         # 纯时域 MSE 对 Delay 的梯度非常干净
#         crop = 500
#         loss = mse_loss(output[..., crop:-crop], tgt_t[..., crop:-crop])
        
#         loss.backward()
#         optimizer_stage1.step()
        
#         if epoch % 100 == 0:
#             print(f"S1 Ep {epoch} | MSE: {loss.item():.6f} | "
#                   f"Delay: {model.global_delay.delay.item():.4f} (Target:-0.42) | "
#                   f"Gain: {model.gain.item():.4f}")

#     # =========================================================
#     # Stage 2: 频域精细修补 (Frequency Equalization)
#     # 目的：FIR 接管剩余的频响误差
#     # =========================================================
#     print("\n=== Stage 2: FIR Fine Tuning (Fixed Delay/Gain) ===")
    
#     # 重置频率权重，全频段均衡
#     # 稍微给高频一点点优待，但不要太偏科
#     with torch.no_grad():
#         fft_len = inp_t.shape[-1] // 2 + 1
#         w = torch.ones(fft_len, device=device) * 10.0
#         # 3GHz - 9GHz 区域加权 (补偿带宽失配)
#         idx_start = int(3.0e9 / (10e9/fft_len))
#         idx_end = int(9.5e9 / (10e9/fft_len))
#         w[idx_start:idx_end] = 30.0 
#         loss_fn.freq_weights = w

#     optimizer_stage2 = optim.Adam([
#         {'params': model.global_delay.parameters(), 'lr': 0.0}, # 彻底锁死
#         {'params': model.gain, 'lr': 0.0},                      # 彻底锁死
#         {'params': model.conv.parameters(), 'lr': 2e-4}         # 细工出慢活
#     ])
    
#     scheduler = optim.lr_scheduler.StepLR(optimizer_stage2, step_size=500, gamma=0.5)

#     for epoch in range(1501):
#         optimizer_stage2.zero_grad()
#         output = model(inp_t)
#         loss = loss_fn(output, tgt_t, model)
#         loss.backward()
#         optimizer_stage2.step()
#         scheduler.step()
        
#         if epoch % 200 == 0:
#             print(f"S2 Ep {epoch} | Loss: {loss.item():.4f} | "
#                   f"Delay: {model.global_delay.delay.item():.4f} | "
#                   f"Gain: {model.gain.item():.4f}")

#     return model, scale

# def train_linear_calibration(simulator, device='cpu'):
#     N_train = 32768
#     # 1. 准备数据
#     raw_batch = simulator.prepare_training_batch(N=N_train) 
    
#     # 增加高频噪声注入，强迫模型关注 8GHz+
#     high_freq_noise = np.random.randn(*raw_batch.shape)
#     b, a = signal.butter(1, 0.8, btype='high') # 0.8*Nyquist = 8GHz
#     high_freq_noise = signal.lfilter(b, a, high_freq_noise) * 0.5
#     raw_batch = raw_batch + high_freq_noise

#     input_list = []
#     target_list = []
    
#     for i in range(raw_batch.shape[0]):
#         raw_sig = raw_batch[i]
#         # Target: 6.0G, 0.0 Delay, 1.0 Gain
#         target_sig = simulator.apply_channel_effect(raw_sig, 6.0e9, 0.0, 1.0)
#         # Input: 5.9G, 0.42 Delay, 0.98 Gain
#         input_sig = simulator.apply_channel_effect(raw_sig, 5.9e9, 0.42, 0.98)
#         target_list.append(target_sig)
#         input_list.append(input_sig)
    
#     target_arr = np.array(target_list)
#     input_arr = np.array(input_list)
    
#     scale = np.max(np.abs(target_arr))
#     target_arr /= scale
#     input_arr /= scale
    
#     inp_t = torch.FloatTensor(input_arr).unsqueeze(1).to(device)
#     tgt_t = torch.FloatTensor(target_arr).unsqueeze(1).to(device)
    
#     model = HybridCalibrationModel_V3(taps=511).to(device)
#     loss_fn = HighPrecisionLoss(taps=511, device=device)
    
#     # === 阶段一：粗调 (Coarse Tuning) ===
#     # 目的：锁定 Global Gain 和 Delay，暂时不让 FIR 乱动
#     print("=== Stage 1: Global Parameter Locking ===")
#     optimizer_stage1 = optim.Adam([
#         {'params': model.global_delay.parameters(), 'lr': 1e-2}, # 大火收汁
#         {'params': model.gain, 'lr': 1e-2},
#         {'params': model.conv.parameters(), 'lr': 0.0}           # 锁死 FIR
#     ])
    
#     for epoch in range(500):
#         optimizer_stage1.zero_grad()
#         output = model(inp_t)
#         # 阶段一主要看时域和 DC，不用太关注高频细节
#         loss = loss_fn(output, tgt_t, model)
#         loss.backward()
#         optimizer_stage1.step()
        
#         if epoch % 100 == 0:
#             print(f"S1 Ep {epoch} | Loss: {loss.item():.4f} | "
#                   f"Delay: {model.global_delay.delay.item():.4f} | "
#                   f"Gain: {model.gain.item():.4f}")

#     # === 阶段二：精调 (Fine Tuning) ===
#     # 目的：Global 参数微调，释放 FIR 全力修补频响
#     print("\n=== Stage 2: FIR Fine Tuning ===")
#     optimizer_stage2 = optim.Adam([
#         {'params': model.global_delay.parameters(), 'lr': 1e-5}, # 极小 LR，防止漂移
#         {'params': model.gain, 'lr': 1e-5},                      # 极小 LR
#         {'params': model.conv.parameters(), 'lr': 1e-4}          # 正常 LR 训练 FIR
#     ])
#     scheduler = optim.lr_scheduler.StepLR(optimizer_stage2, step_size=500, gamma=0.5)

#     # 这里的 Loss 权重必须拉满，特别是高频
#     loss_fn.freq_weights = None # 重置权重让它重新初始化
#     # 如果你在 Loss 类里改了权重，确保现在使用的是“高频加强版”权重

#     for epoch in range(1501):
#         optimizer_stage2.zero_grad()
#         output = model(inp_t)
#         loss = loss_fn(output, tgt_t, model)
#         loss.backward()
#         optimizer_stage2.step()
#         scheduler.step()
        
#         if epoch % 100 == 0:
#             print(f"S2 Ep {epoch} | Loss: {loss.item():.4f} | "
#                   f"Delay: {model.global_delay.delay.item():.4f} | "
#                   f"Gain: {model.gain.item():.4f}")

#     return model, scale

# def train_linear_calibration(simulator, device='cpu'):
#     N_train = 32768
#     raw_batch = simulator.prepare_training_batch(N=N_train) 
    
#     input_list = []
#     target_list = []
    
#     # 构造训练对
#     for i in range(raw_batch.shape[0]):
#         raw_sig = raw_batch[i]
        
#         # Target: 理想通道 (6.0G BW, 0 Delay, 1.0 Gain)
#         target_sig = simulator.apply_channel_effect(
#             raw_sig, cutoff_freq=6.0e9, delay_samples=0.0, gain=1.0
#         )
        
#         # Input: 恶劣通道 (5.9G BW, 0.42 Delay, 0.98 Gain)
#         # 注意：这里直接使用 apply_channel_effect，不再手动加 raw_sig**3
#         input_sig = simulator.apply_channel_effect(
#             raw_sig, cutoff_freq=5.9e9, delay_samples=0.42, gain=0.98
#         )
        
#         target_list.append(target_sig)
#         input_list.append(input_sig)
    
#     # 转为 Tensor
#     target_arr = np.array(target_list)
#     input_arr = np.array(input_list)
    
#     # 归一化 (很重要，利于梯度下降)
#     scale = np.max(np.abs(target_arr))
#     target_arr /= scale
#     input_arr /= scale
    
#     inp_t = torch.FloatTensor(input_arr).unsqueeze(1).to(device)
#     tgt_t = torch.FloatTensor(target_arr).unsqueeze(1).to(device)
    
#     # 初始化模型与优化器
#     model = HybridCalibrationModel_V3(taps=511).to(device)
#     loss_fn = HighPrecisionLoss(taps=511, device=device)

#     optimizer = optim.Adam([
#         {'params': model.conv.parameters(), 'lr': 1e-4},
#         {'params': model.gain, 'lr': 1e-3},           # 增益参数可以用大一点的学习率
#         {'params': model.global_delay.parameters(), 'lr': 1e-3} # 延时参数同理
#     ])
    
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    
#     print(f"开始训练... (设备: {device})")
#     epochs = 1500
#     for epoch in range(epochs + 1):
#         optimizer.zero_grad()
        
#         # 前向传播
#         output = model(inp_t)
        
#         # 计算 Loss
#         loss = loss_fn(output, tgt_t, model)
        
#         # 反向传播
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         if epoch % 100 == 0:
#             d_val = model.global_delay.delay.item()
#             g_val = model.gain.item()
#             print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f} | "
#                   f"Learned Delay: {d_val:.4f} (Target:-0.42) | "
#                   f"Learned Gain: {g_val:.4f} (Target:~1.02)")

#     return model, scale

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

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sim = TIADCSimulator(fs=20e9)
    model, scale = train_linear_calibration(sim, device)
    
    print("\n=== 正在进行扫频验证 (SFDR) ===")
    
    # 扫频范围：从 100M 到 9.5G
    test_freqs = np.arange(0.1e9, 9.6e9, 0.2e9)
    improvements = []
    
    for f in test_freqs:
        # 跳过 Nyquist 点附近的频率，因为那里 Image 和 Signal 重叠，没法算 SFDR
        if abs(f - 10e9) < 0.5e9: 
            improvements.append(0)
            continue
            
        imp = calculate_sfdr_improvement(sim, model, scale, f, device)
        improvements.append(imp)
        print(f"Freq: {f/1e9:.2f} GHz | SFDR Improvement: {imp:.2f} dB")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(test_freqs/1e9, improvements, 'b-o', linewidth=2, label='Linear DDSP Model')
    plt.title("TIADC Mismatch Calibration Performance (Pure Linear Scenario)")
    plt.xlabel("Input Frequency (GHz)")
    plt.ylabel("SFDR Improvement (dB)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.axhline(0, color='k', linestyle='-', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()