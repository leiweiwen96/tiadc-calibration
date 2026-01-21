import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# ==============================================================================
# 1. 物理环境 (保持 0.5 强耦合)
# ==============================================================================
def get_transmission_response(fs, cutoff_freq, length_factor):
    fc = cutoff_freq / (1.0 + 0.5 * length_factor) 
    sos = signal.bessel(5, fc, 'low', fs=fs, output='sos')
    return sos

def generate_data(total_samples=50000, symbol_rate=2e9, fs=40e9):
    t = np.arange(total_samples) / fs
    sym_len = int(fs / symbol_rate)
    levels = np.array([-3, -1, 1, 3])
    
    sos_a_p1 = get_transmission_response(fs, 12e9, 0.2) 
    sos_a_p2 = get_transmission_response(fs, 10e9, 1.0) 
    sos_b_p2 = get_transmission_response(fs, 12e9, 0.2) 
    sos_b_p1 = get_transmission_response(fs, 10e9, 1.0) 
    prop_delay = 12 

    def make_signal(mode, noise_std=0.01):
        if mode == 'A_only':
            bits_a = np.random.choice(levels, size=total_samples // sym_len + 200)
            bits_b = np.zeros_like(bits_a)
        elif mode == 'B_only':
            bits_a = np.zeros(total_samples // sym_len + 200)
            bits_b = np.random.choice(levels, size=total_samples // sym_len + 200)
        else: # Mix
            bits_a = np.random.choice(levels, size=total_samples // sym_len + 200)
            bits_b = np.random.choice(levels, size=total_samples // sym_len + 200)
            
        raw_a = np.repeat(bits_a, sym_len)[:total_samples + 2000]
        raw_b = np.repeat(bits_b, sym_len)[:total_samples + 2000]
        
        sa_p1 = signal.sosfilt(sos_a_p1, raw_a)
        sa_p2 = signal.sosfilt(sos_a_p2, raw_a)
        sb_p2 = signal.sosfilt(sos_b_p2, raw_b)
        sb_p1 = signal.sosfilt(sos_b_p1, raw_b)
        
        p1 = sa_p1[:-2000] + 0.5 * np.roll(sb_p1, prop_delay)[:-2000]
        p2 = np.roll(sa_p2, prop_delay)[:-2000] + 0.5 * sb_p2[:-2000]
        
        p1 += np.random.normal(0, noise_std, len(p1))
        p2 += np.random.normal(0, noise_std, len(p2))
        
        return p1[:total_samples], p2[:total_samples], raw_a[:total_samples], raw_b[:total_samples]

    data = {}
    data['A_only'] = make_signal('A_only', noise_std=1e-5)
    data['B_only'] = make_signal('B_only', noise_std=1e-5)
    data['Mix']    = make_signal('Mix', noise_std=0.02)
    return data

# ==============================================================================
# 2. 归一化 LMS (NLMS)
# ==============================================================================
class NLMSFilter:
    def __init__(self, taps=201, step_size=0.1, eps=1e-6):
        self.taps = taps
        self.mu = step_size
        self.eps = eps
        self.weights = np.zeros(taps)
        self.weights[taps//2] = 1.0 # Center Init

    def update(self, x, e):
        norm = np.dot(x, x) + self.eps
        self.weights += (self.mu / norm) * e * x
        
    # 为了验证方便，保留 filter 函数，但在训练循环中我们手动计算点积
    def filter(self, signal):
        return np.convolve(signal, self.weights, mode='same')

# ==============================================================================
# 3. 辅助函数
# ==============================================================================
def prepare_target(raw_bits, sigma=2.0):
    return gaussian_filter1d(raw_bits, sigma=sigma)

def align_data(measured, target):
    corr = signal.correlate(measured[5000:15000], target[5000:15000], mode='full')
    lag = np.argmax(np.abs(corr))
    shift = lag - (10000 - 1)
    target_aligned = np.roll(target, shift)
    if shift > 0: target_aligned[:shift] = 0
    else: target_aligned[shift:] = 0
    return target_aligned

# ==============================================================================
# 4. 修正后的状态机校准 (Corrected State Machine)
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    print(">>> Generating Data...")
    data = generate_data()
    
    # 归一化
    p1_a, p2_a, bits_a, _ = data['A_only']
    p1_b, p2_b, _, bits_b = data['B_only']
    
    norm_p1 = np.std(p1_a)
    norm_p2 = np.std(p2_b)
    p1_a /= norm_p1; p2_a /= norm_p2
    p1_b /= norm_p1; p2_b /= norm_p2
    
    tgt_a = align_data(p1_a, prepare_target(bits_a, 2.0))
    tgt_b = align_data(p2_b, prepare_target(bits_b, 2.0))
    
    # 初始化
    w11 = NLMSFilter(201, step_size=0.05)
    w12 = NLMSFilter(201, step_size=0.05); w12.weights[:] = 0 
    w21 = NLMSFilter(201, step_size=0.05); w21.weights[:] = 0
    w22 = NLMSFilter(201, step_size=0.05)
    
    L = 201
    mid = L // 2
    
    # --- Phase 1: Main Path (A->P1, B->P2) ---
    print(">>> Phase 1: Training Main Paths...")
    # W11 (A only)
    for n in range(L, 20000):
        x = p1_a[n-L:n][::-1]
        y = np.dot(w11.weights, x)
        e = tgt_a[n-1-mid] - y
        w11.update(x, e)
        
    # W22 (B only)
    for n in range(L, 20000):
        x = p2_b[n-L:n][::-1]
        y = np.dot(w22.weights, x)
        e = tgt_b[n-1-mid] - y
        w22.update(x, e)
    
    # --- Phase 2: Crosstalk (P2->P1, P1->P2) ---
    print(">>> Phase 2: Training Cancellers (Real-time Calculation)...")
    
    # Train W12 (Cancel B in P1)
    # 关键修改：我们在循环内实时计算 y_main，保证延时一致！
    for n in range(L, 20000):
        x_p1 = p1_b[n-L:n][::-1] # B signal in P1 (Leakage)
        x_p2 = p2_b[n-L:n][::-1] # B signal in P2 (Source)
        
        # 1. 用已经锁定的 W11 计算当前的"漏过来"的主信号
        y_main = np.dot(w11.weights, x_p1) 
        
        # 2. 用正在训练的 W12 计算抵消信号
        y_cancel = np.dot(w12.weights, x_p2)
        
        # 3. 目标是两者相加为0
        e = 0 - (y_main + y_cancel)
        w12.update(x_p2, e) # 只更新 W12
        
    # Train W21 (Cancel A in P2)
    for n in range(L, 20000):
        x_p1 = p1_a[n-L:n][::-1] # A signal in P1 (Source)
        x_p2 = p2_a[n-L:n][::-1] # A signal in P2 (Leakage)
        
        y_main = np.dot(w22.weights, x_p2) # W22 is locked
        y_cancel = np.dot(w21.weights, x_p1)
        
        e = 0 - (y_main + y_cancel)
        w21.update(x_p1, e) # 只更新 W21
        
    print("    Done. Filters locked.")

    # --- Verify ---
    print(">>> Running Test...")
    p1_mix, p2_mix, bits_a_mix, _ = data['Mix']
    p1_mix /= norm_p1
    p2_mix /= norm_p2
    
    # 验证时可以使用 convolve，因为是对整段信号处理，mode='same' 是合适的
    # 或者为了严谨，我们也可以写个循环，但 convolve 比较快且在这里只影响边缘
    y1 = w11.filter(p1_mix) + w12.filter(p2_mix)
    
    gain = 3.0 / np.max(np.abs(y1))
    y1 *= gain
    
    start, end = 10000, 25000
    sig = y1[start:end]
    ref = bits_a_mix[start:end]
    
    # Align & SER
    corr = signal.correlate(ref[::4], sig[::4], mode='full')
    shift = (np.argmax(np.abs(corr)) - (len(sig[::4])-1)) * 4
    if shift > 0: s=sig[:-shift]; r=ref[shift:]
    else: s=sig[-shift:]; r=ref[:shift]
    L = min(len(s), len(r))
    s = s[:L]; r = r[:L]
    
    sps = 20
    n = L // sps
    s_mat = s[:n*sps].reshape(n, sps)
    r_mat = r[:n*sps].reshape(n, sps)
    
    best_ser = 1.0
    best_ph = 0
    for ph in range(sps):
        samp = s_mat[:, ph]
        truth = r_mat[:, 10]
        dec = np.zeros_like(samp)
        dec[samp>2]=3; dec[(samp>0)&(samp<=2)]=1
        dec[(samp>-2)&(samp<=0)]=-1; dec[samp<=-2]=-3
        ser = np.mean(dec != truth)
        if ser < best_ser: best_ser=ser; best_ph=ph
            
    print(f"\n>>> Final NLMS Corrected SER: {best_ser:.6f}")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(w11.weights, color='yellow', label='Main (W11)')
    ax[0].plot(w12.weights, color='#00ffcc', label='Cancel (W12)')
    ax[0].set_title("Learned Filters (Causal & Correct)")
    ax[0].legend()
    
    per = 40
    d = s[2000:6000]
    d = d[:(len(d)//per)*per].reshape(-1, per)
    ax[1].plot(d.T, color='#00ffcc', alpha=0.5, lw=1.0)
    ax[1].axvline(best_ph, c='r', ls=':')
    ax[1].set_title(f"Perfect Eye (SER={best_ser:.1e})")
    plt.show()