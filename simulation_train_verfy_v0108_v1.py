import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import time

# ==============================================================================
# 0. 全局配置
# ==============================================================================
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def numpy_to_torch(arr, device):
    return torch.FloatTensor(arr).unsqueeze(1).to(device)

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

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0, hd3=0.0):
        # 物理通道模型
        nyquist = self.fs / 2
        
        # 1. 非线性 (本方案不修它，但仿真里要有，以测试模型抗干扰能力)
        sig_nonlinear = sig + hd3 * (sig**3)
        
        # 2. 带宽限制
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig_nonlinear)
        
        # 3. 增益 & 延时
        sig_gain = sig_bw * gain
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 4. 底噪
        return sig_delayed + np.random.normal(0, 1e-5, len(sig_delayed))

    def prepare_training_batch(self, N=32768):
        t = np.arange(N) / self.fs
        # 训练信号：Chirp + Step + Noise
        chirp = signal.chirp(t, f0=1e6, t1=t[-1], f1=self.fs*0.48, method='linear')
        step = np.zeros(N); step[:N//2]=0.5; step[N//2:]=-0.5
        pink = np.random.randn(N)
        return np.stack([chirp, step, pink], axis=0)

    def generate_tone(self, freq, N=8192):
        t = np.arange(N) / self.fs
        return np.sin(2 * np.pi * freq * t) * 0.9

# ==============================================================================
# 2. 模型定义 (V3: Linear FIR -> Delay -> Gain)
# ==============================================================================
class DifferentiableDelay(nn.Module):
    def __init__(self, init_delay=0.0):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor([init_delay]))

    def forward(self, x):
        N = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        k = torch.arange(X.shape[-1], device=x.device)
        phase = torch.exp(-1j * 2 * np.pi * k * self.delay / N)
        return torch.fft.irfft(X * phase, n=N, dim=-1)

class HybridCalibrationModel_V3(nn.Module):
    def __init__(self, taps=511):
        super().__init__()
        # 1. FIR (Eq)
        self.linear_fir = nn.Conv1d(1, 1, kernel_size=taps, padding='same', bias=False)
        # 2. Delay
        self.global_delay = DifferentiableDelay(init_delay=0.0)
        # 3. Gain
        self.gain = nn.Parameter(torch.tensor([1.0]))
        
        with torch.no_grad():
            self.linear_fir.weight.zero_()
            self.linear_fir.weight[0, 0, taps // 2] = 1.0

    def forward(self, x):
        x = self.linear_fir(x)
        x = self.global_delay(x)
        x = x * self.gain
        return x

# ==============================================================================
# 3. 物理感知初始化 (修复切片版)
# ==============================================================================
def physics_aware_initialization(model, fs, fc_ref, fc_bad, taps):
    print(f"[{time.strftime('%H:%M:%S')}] 计算物理逆滤波器系数...")
    nyquist = fs / 2
    b_ref, a_ref = signal.butter(5, fc_ref / nyquist, btype='low')
    b_bad, a_bad = signal.butter(5, fc_bad / nyquist, btype='low')
    
    # 使用大量点数计算频响
    N_FFT = 16384
    w, h_ref = signal.freqz(b_ref, a_ref, worN=N_FFT//2 + 1)
    _, h_bad = signal.freqz(b_bad, a_bad, worN=N_FFT//2 + 1)
    
    h_comp = h_ref / (h_bad + 1e-12)
    h_abs = np.clip(np.abs(h_comp), 0, 20.0) # 限幅
    h_comp = h_abs * np.exp(1j * np.angle(h_comp))
    
    # IRFFT 转时域
    impulse = np.fft.irfft(h_comp, n=N_FFT)
    impulse = np.roll(impulse, N_FFT // 2)
    
    # 强制切片长度为 taps
    center = N_FFT // 2
    start = center - taps // 2
    end = start + taps
    fir_coeffs = impulse[start:end] * np.blackman(taps)
    
    with torch.no_grad():
        model.linear_fir.weight.data.copy_(
            torch.from_numpy(fir_coeffs).float().view(1, 1, -1).to(device)
        )
    print("  -> 初始化完成。")

# ==============================================================================
# 4. 训练流程
# ==============================================================================
def train_model(sim):
    # 场景：参考6.0G，输入5.8G+误差
    REF_BW, BAD_BW = 6.0e9, 5.8e9
    
    raw_batch = sim.prepare_training_batch(N=16384)
    input_list, target_list = [], []
    
    for i in range(raw_batch.shape[0]):
        sig = raw_batch[i]
        # Target: 理想 (无HD3)
        target = sim.apply_channel_effect(sig, REF_BW, 0.0, 1.0, hd3=0.0)
        # Input: 恶劣 (含HD3, 但V3模型只修线性部分)
        input_bad = sim.apply_channel_effect(sig, BAD_BW, 0.42, 0.98, hd3=0.01)
        target_list.append(target)
        input_list.append(input_bad)
        
    in_t = numpy_to_torch(np.array(input_list), device)
    tg_t = numpy_to_torch(np.array(target_list), device)
    scale = torch.max(torch.abs(tg_t)).item()
    in_t /= scale; tg_t /= scale
    
    # 初始化 V3
    model = HybridCalibrationModel_V3(taps=511).to(device)
    physics_aware_initialization(model, sim.fs, REF_BW, BAD_BW, 511)
    
    optimizer = optim.Adam([
        {'params': model.linear_fir.parameters(), 'lr': 1e-4},
        {'params': model.global_delay.parameters(), 'lr': 1e-2},
        {'params': model.gain, 'lr': 1e-3}
    ])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1500], gamma=0.1)
    loss_fn = nn.MSELoss()
    
    print(f"[{time.strftime('%H:%M:%S')}] 开始训练 (V3 纯线性)...")
    loss_history = []
    
    for epoch in range(2001):
        model.train()
        optimizer.zero_grad()
        pred = model(in_t)
        
        # Loss
        l_time = loss_fn(pred, tg_t)
        fft_p = torch.fft.rfft(pred, dim=-1).abs()
        fft_t = torch.fft.rfft(tg_t, dim=-1).abs()
        l_freq = torch.mean(torch.abs(torch.log10(fft_p+1e-9) - torch.log10(fft_t+1e-9)))
        
        loss = 100.0 * l_time + 10.0 * l_freq
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"Ep {epoch:04d} | Loss: {loss.item():.5f} | Delay: {model.global_delay.delay.item():.4f}")
            
    return model, scale, loss_history

# ==============================================================================
# 5. 验证与SFDR绘图 (绘制正值的SFDR)
# ==============================================================================
def verify_sfdr_plot(sim, model, scale):
    print(f"\n[{time.strftime('%H:%M:%S')}] 计算全频段 SFDR...")
    model.eval()
    
    freqs = np.linspace(0.1e9, 9.5e9, 50)
    sfdr_before = []
    sfdr_after = []
    
    for f in freqs:
        # 生成测试信号
        s_src = sim.generate_tone(f, N=4096)
        s_ref = sim.apply_channel_effect(s_src, 6.0e9, 0.0, 1.0, hd3=0.0)
        s_bad = sim.apply_channel_effect(s_src, 5.8e9, 0.42, 0.98, hd3=0.01)
        
        with torch.no_grad():
            inp = numpy_to_torch(s_bad/scale, device).reshape(1, 1, -1)
            s_cal = model(inp).cpu().numpy().flatten() * scale
            
        # 拼接 TIADC
        L = min(len(s_ref), len(s_bad)); L-=L%2
        ti_bad = np.zeros(L); ti_bad[0::2]=s_ref[:L:2]; ti_bad[1::2]=s_bad[1:L:2]
        ti_cal = np.zeros(L); ti_cal[0::2]=s_ref[:L:2]; ti_cal[1::2]=s_cal[1:L:2]
        
        # === SFDR 计算核心逻辑 ===
        def calculate_mismatch_sfdr(sig, f_in):
            # 1. 计算频谱 (dB)
            win = np.blackman(len(sig))
            spec = np.abs(np.fft.rfft(sig * win))
            spec_db = 20 * np.log10(spec + 1e-12)
            # 归一化：让信号峰值为 0dB
            peak_val = np.max(spec_db)
            spec_db -= peak_val
            
            # 2. 确定频率轴
            freqs_axis = np.linspace(0, sim.fs/2, len(spec))
            
            # 3. 找到 Mismatch Spur 的位置 (fs/2 - fin)
            spur_freq = abs(sim.fs/2 - f_in)
            
            # 4. 剔除奇点 (当 Spur 和 Signal 重叠时，SFDR 无意义)
            if abs(spur_freq - f_in) < 0.2e9: 
                return np.nan
            
            # 5. 测量该位置的杂散功率
            idx = np.argmin(np.abs(freqs_axis - spur_freq))
            # 在附近取最大值防止频率对不准
            search_window = 5
            spur_power = np.max(spec_db[max(0, idx-search_window):min(len(spec), idx+search_window)])
            
            # 6. 计算 SFDR (信号 - 杂散)
            # 信号是 0dB，杂散是负数(例如 -80)，SFDR = 0 - (-80) = 80
            return -spur_power 

        sfdr_before.append(calculate_mismatch_sfdr(ti_bad, f))
        sfdr_after.append(calculate_mismatch_sfdr(ti_cal, f))

    # === 绘图 ===
    plt.figure(figsize=(10, 6))
    
    plt.plot(freqs/1e9, sfdr_before, 'r--o', label='Before Calibration', alpha=0.5)
    plt.plot(freqs/1e9, sfdr_after, 'g-^', label='After Calibration (Mismatch SFDR)', linewidth=2)
    
    # 装饰
    plt.title("TIADC Mismatch-Limited SFDR Performance")
    plt.xlabel("Input Frequency (GHz)")
    plt.ylabel("SFDR (dB)") # 正值，越高越好
    plt.legend(loc='lower right')
    plt.grid(True, which='both', alpha=0.6)
    plt.ylim(20, 110) # 设置范围，让高分更明显
    
    # 标注奇点
    plt.annotate('Singularity Region\n(fs/4 = 5GHz)', xy=(5, 25), xytext=(5, 40),
                 arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计
    avg_imp = np.nanmean(sfdr_after) - np.nanmean(sfdr_before)
    print(f"\n平均 SFDR 提升: +{avg_imp:.2f} dB")
    print("注：该图展示的是'失配杂散抑制能力'。非线性谐波不计入此指标。")

if __name__ == "__main__":
    print(f"Running V3 Simulation on {device}...")
    sim = TIADCSimulator(fs=20e9)
    model, scale, _ = train_model(sim)
    verify_sfdr_plot(sim, model, scale)