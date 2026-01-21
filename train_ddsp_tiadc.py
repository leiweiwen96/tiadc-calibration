import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==============================================================================
# 1. 配置类 (Configuration)
#    将所有超参数集中管理，方便复用和修改
# ==============================================================================
@dataclass
class TrainingConfig:
    # --- 硬件与信号参数 ---
    fs_system: float = 20e9       # 系统总采样率 (20 GSPS)
    num_channels: int = 2         # TIADC 通道数
    channel_id: int = 1           # 当前要校准的通道 ID (Ch0 是参考，我们校准 Ch1)
    
    # --- 模型参数 ---
    fir_taps: int = 65            # FIR 滤波器阶数 (推荐 65)
    
    # --- 训练参数 ---
    lr: float = 1e-3              # 学习率
    epochs: int = 2000            # 训练轮数
    
    # --- Loss 权重 ---
    alpha_time: float = 1.0       # 时域 MSE 权重
    beta_freq: float = 0.5        # 频域幅度谱权重 (关键: 修带宽)
    gamma_reg: float = 0.001      # 正则化权重 (关键: 硬件平滑)
    
    # --- 路径 ---
    mat_file_path: str = 'tiadc_serdes_data.mat'
    output_coeff_path: str = 'fir_coeffs_float.txt'

# ==============================================================================
# 2. DSP 工具类 (DSP Utils)
#    封装了信号生成、对齐、理想目标构建等核心数学逻辑
# ==============================================================================
class DSPUtils:
    @staticmethod
    def fractional_delay(sig, delay_samples):
        """
        利用 FFT 实现高精度的分数延时 (Fractional Delay)。
        这是构建"理想目标"的核心。
        
        Args:
            sig: 输入信号 (1D numpy array)
            delay_samples: 需要延迟的采样点数 (可以是小数，如 0.5)
        """
        N = len(sig)
        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(N)
        
        # 频域相移定理: x(t - tau) <-> X(f) * exp(-j * 2pi * f * tau)
        # 注意: 这里的 freqs 是归一化频率 [0, 0.5]
        # 相位旋转需要乘以 2pi 和 索引范围
        # 修正: rfftfreq 返回的是 [0, 1/N, ... 0.5], 对应的 index 是 [0, 1, ... N/2]
        # 直接用 complex exponential 构建相移向量
        omega = 2 * np.pi * np.arange(len(X)) / N 
        # 但因为是 rfft，频率轴长度是 N//2 + 1
        # 让我们用更稳健的方式：
        
        # 重新生成对应的角频率 k
        k = np.arange(len(X))
        phase_shift = np.exp(-1j * 2 * np.pi * k * delay_samples / N)
        
        X_shifted = X * phase_shift
        return np.fft.irfft(X_shifted, n=N)

    @staticmethod
    def coarse_align(ref, dist):
        """
        计算并修正整数倍的粗延时 (Coarse Delay)。
        解决线缆长度差异导致的几十个点的偏差。
        """
        correlation = signal.correlate(ref, dist, mode='full')
        lags = signal.correlation_lags(len(ref), len(dist), mode='full')
        lag = lags[np.argmax(correlation)]
        
        # 对齐操作
        if lag > 0:
            # dist 滞后了，取 dist 后半段
            dist_aligned = dist[lag:]
            ref_aligned = ref[:len(dist_aligned)]
        elif lag < 0:
            # dist 超前了
            ref_aligned = ref[abs(lag):]
            dist_aligned = dist[:len(ref_aligned)]
        else:
            ref_aligned = ref
            dist_aligned = dist
            
        # 截断到相同长度
        min_len = min(len(ref_aligned), len(dist_aligned))
        return ref_aligned[:min_len], dist_aligned[:min_len], lag

    @staticmethod
    def generate_demo_data(fs=20e9, N=8192):
        """
        生成模拟数据 (Demo Mode)，防止没有 .mat 文件时报错。
        模拟了:
        1. 物理延时 (Coarse Delay)
        2. 带宽失配 (Bandwidth Mismatch, 低通滤波)
        3. 理想相位差 (Interleaving Phase)
        """
        print("[Demo] 正在生成合成数据...")
        t = np.arange(N) / fs
        # 原信号: 宽带 Chirp
        signal_src = signal.chirp(t, f0=10e6, t1=t[-1], f1=fs*0.45)
        
        # --- Channel 0 (Reference) ---
        # 假设 Ch0 也有轻微带宽限制
        b0, a0 = signal.butter(4, 0.8)
        ch0 = signal.lfilter(b0, a0, signal_src)
        
        # --- Channel 1 (Distorted) ---
        # 1. 物理延时 (模拟线长差异, 比如 20 个点)
        coarse_delay = 20
        # 2. 理想交替相位 (0.5 个子采样点)
        ideal_interleave_delay = 0.5 
        # 3. 带宽更窄 (模拟高频衰减)
        b1, a1 = signal.butter(4, 0.6) # 截止频率更低
        
        ch1_temp = signal.lfilter(b1, a1, signal_src)
        # 施加总延时
        total_delay = coarse_delay + ideal_interleave_delay
        ch1 = DSPUtils.fractional_delay(ch1_temp, total_delay)
        
        # 添加噪声
        ch0 += np.random.normal(0, 0.001, N)
        ch1 += np.random.normal(0, 0.001, N)
        
        return ch1, ch0, fs

# ==============================================================================
# 3. 数据集类 (Dataset)
#    负责加载数据、对齐、并生成"理想相移目标"
# ==============================================================================
class TIADCDataset:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.input_tensor = None
        self.target_tensor = None
        self._load_data()

    def _load_data(self):
        try:
            # 尝试加载 MATLAB 文件
            mat = sio.loadmat(self.cfg.mat_file_path)
            struct = mat['data_train']
            raw_input = struct['input'][0][0].flatten()  # Ch1
            raw_ref = struct['target'][0][0].flatten()   # Ch0
            print(f"[Data] 成功加载 {self.cfg.mat_file_path}")
        except FileNotFoundError:
            # 如果没文件，生成 Demo 数据
            print(f"[Data] 未找到文件，使用合成数据演示。")
            raw_input, raw_ref, _ = DSPUtils.generate_demo_data()

        # 1. 粗对齐 (Coarse Alignment)
        # 去除物理线长差异，只保留 TIADC 固有的相位差和微小误差
        ref_aligned, input_aligned, lag = DSPUtils.coarse_align(raw_ref, raw_input)
        print(f"[Data] 粗延时校正: {lag} 采样点")

        # 2. 生成理想目标 (Ideal Target Generation) - 核心步骤！
        # Ch1 应该比 Ch0 晚 k/N 个周期 (对于2通道，是0.5个周期)
        # 我们对 Ref 进行理想相移，作为 Ch1 的学习目标
        ideal_shift = self.cfg.channel_id / self.cfg.num_channels
        print(f"[Data] 生成理想目标，相移: {ideal_shift} samples")
        target_ideal = DSPUtils.fractional_delay(ref_aligned, ideal_shift)

        # 3. 掐头去尾 (去除 FFT 和滤波的边缘效应)
        margin = 200
        x = input_aligned[margin:-margin]
        y = target_ideal[margin:-margin]

        # 4. 归一化
        scale = np.max(np.abs(y))
        x = x / scale
        y = y / scale

        # 5. 转 Tensor [Batch=1, Channel=1, Length]
        self.input_tensor = torch.FloatTensor(x).view(1, 1, -1)
        self.target_tensor = torch.FloatTensor(y).view(1, 1, -1)

    def get_data(self):
        return self.input_tensor, self.target_tensor

# ==============================================================================
# 4. 模型定义 (Model)
#    基于 1D 卷积的 FIR 滤波器，特殊的初始化
# ==============================================================================
class DFS_FIR_Filter(nn.Module):
    def __init__(self, num_taps=65):
        super().__init__()
        # Padding='same' 保证输出长度不变
        self.conv = nn.Conv1d(1, 1, kernel_size=num_taps, padding='same', bias=False)
        self._init_weights(num_taps)

    def _init_weights(self, num_taps):
        # 初始化为 Delta 函数 (直通)，而不是随机噪声
        # 这样训练是从"不做处理"开始微调，收敛更快更稳
        with torch.no_grad():
            self.conv.weight.zero_()
            center = num_taps // 2
            self.conv.weight[0, 0, center] = 1.0

    def forward(self, x):
        return self.conv(x)

    @property
    def coeffs(self):
        return self.conv.weight.detach().cpu().numpy().flatten()

# ==============================================================================
# 5. 混合损失函数 (Loss Function)
# ==============================================================================
class HybridLoss(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.cfg = config
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_target, model):
        # A. 时域损失 (对齐相位和延时)
        loss_time = self.mse(y_pred, y_target)

        # B. 频域损失 (修复带宽失配)
        # 使用 FFT 幅度谱的 L1 距离
        spec_pred = torch.fft.rfft(y_pred, dim=-1).abs()
        spec_target = torch.fft.rfft(y_target, dim=-1).abs()
        # 加上 Log 可以让低能量部分的误差也被重视 (可选)
        loss_freq = torch.mean(torch.abs(spec_pred - spec_target))

        # C. 正则化 (平滑系数)
        coeffs = model.conv.weight.flatten()
        # 惩罚相邻系数的差值，使滤波器平滑，减少高频噪声增益
        loss_reg = torch.mean((coeffs[1:] - coeffs[:-1])**2)

        total = (self.cfg.alpha_time * loss_time) + \
                (self.cfg.beta_freq * loss_freq) + \
                (self.cfg.gamma_reg * loss_reg)
        
        return total, loss_time, loss_freq

# ==============================================================================
# 6. 训练器与可视化 (Trainer)
# ==============================================================================
class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.dataset = TIADCDataset(config)
        self.model = DFS_FIR_Filter(num_taps=config.fir_taps)
        self.loss_fn = HybridLoss(config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        
    def train(self):
        print(f"\n=== 开始训练 (FIR Taps: {self.cfg.fir_taps}) ===")
        input_data, target_data = self.dataset.get_data()
        
        history = []
        
        for epoch in range(self.cfg.epochs):
            self.optimizer.zero_grad()
            
            # Forward
            y_pred = self.model(input_data)
            
            # Loss
            loss, l_t, l_f = self.loss_fn(y_pred, target_data, self.model)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            history.append(loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:04d} | Total: {loss.item():.6f} | Time: {l_t.item():.6f} | Freq: {l_f.item():.6f}")
                
        self.plot_results(input_data, target_data, y_pred)
        self.save_coeffs()

    def save_coeffs(self):
        coeffs = self.model.coeffs
        np.savetxt(self.cfg.output_coeff_path, coeffs)
        print(f"\n[Success] 系数已保存至: {self.cfg.output_coeff_path}")
        print(f"系数范围: Max={coeffs.max():.4f}, Min={coeffs.min():.4f}")

    def plot_results(self, x_in, y_tgt, y_pred):
        # 转换为 Numpy 方便绘图
        x = x_in.detach().numpy().flatten()
        tgt = y_tgt.detach().numpy().flatten()
        pred = y_pred.detach().numpy().flatten()
        coeffs = self.model.coeffs
        
        plt.figure(figsize=(14, 10))
        
        # 1. 时域对比 (Zoom In)
        plt.subplot(2, 2, 1)
        # 找个中间位置画
        mid = len(x) // 2
        rng = slice(mid, mid + 100)
        plt.plot(tgt[rng], 'k-', linewidth=2, label='Target (Ideal Ch1)', alpha=0.6)
        plt.plot(x[rng], 'r--', label='Input (Raw Ch1)')
        plt.plot(pred[rng], 'b-.', label='Calibrated')
        plt.title('Time Domain Alignment (Zoom)')
        plt.legend()
        plt.grid(True)
        
        # 2. 频域对比 (PSD)
        plt.subplot(2, 2, 2)
        f, P_tgt = signal.welch(tgt, fs=self.cfg.fs_system, nperseg=1024)
        _, P_in = signal.welch(x, fs=self.cfg.fs_system, nperseg=1024)
        _, P_pred = signal.welch(pred, fs=self.cfg.fs_system, nperseg=1024)
        
        plt.semilogy(f/1e9, P_tgt, 'k', label='Target')
        plt.semilogy(f/1e9, P_in, 'r', label='Input (Bandwidth Loss)')
        plt.semilogy(f/1e9, P_pred, 'b', label='Calibrated')
        plt.title('Frequency Response (Bandwidth Check)')
        plt.xlabel('Freq (GHz)')
        plt.ylabel('PSD')
        plt.legend()
        plt.grid(True)
        
        # 3. 滤波器系数
        plt.subplot(2, 2, 3)
        plt.stem(coeffs)
        plt.title(f'Learned FIR Coefficients (N={len(coeffs)})')
        plt.grid(True)
        
        # 4. 误差残留
        plt.subplot(2, 2, 4)
        plt.plot(pred - tgt, 'g')
        plt.title('Residual Error (Calibrated - Target)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 7. 程序入口 (Main)
# ==============================================================================
if __name__ == "__main__":
    # 实例化配置
    config = TrainingConfig()
    
    # 可以在这里覆盖默认配置
    # config.fir_taps = 129
    # config.mat_file_path = "my_new_data.mat"
    
    # 初始化训练器并运行
    trainer = ModelTrainer(config)
    trainer.train()