import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
class Config:
    M = 4              # TIADC 通道数
    fs = 10e9          # 总采样率 10 GS/s
    n_points = 4096    # 样本点数
    batch_size = 32
    epochs = 1000      # 训练轮数
    lr = 1e-3          # 学习率
    
    # 滤波器设置
    filter_tap = 41    # 每个通道校正滤波器的抽头数 (最好是奇数)
    
    # 模拟失配参数 (Ground Truth Mismatches)
    # 通道:          [0,      1,      2,      3]
    gains_gt =      [1.0,    1.02,   0.98,   1.01]    # 增益失配
    offsets_gt =    [0.0,    0.05,  -0.03,   0.02]    # 偏置失配
    # 时间偏差 (单位: 秒). 10GSps -> Ts=100ps. 
    # 模拟几皮秒的误差
    time_skew_gt =  [0.0,    5e-12, -3e-12,  2e-12]   

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==========================================
# 2. 模拟 TIADC 数据生成器 (Simulation)
# ==========================================
def generate_synthetic_data(batch_size, n_points, config, is_training=True):
    """
    生成模拟TIADC采样数据，包含Gain, Offset, Timing Skew失配。
    """
    t_base = np.arange(n_points) / config.fs
    
    # 为了模拟Time Skew，我们需要生成"连续"信号，然后在偏移后的时刻采样
    # 信号源：混合正弦波 (Chirp 或 Multi-tone)
    # 这里使用随机频率的正弦波组合来覆盖频带
    
    batch_data = []
    batch_target = []
    
    for _ in range(batch_size):
        # 随机生成频率 (0 ~ Nyquist)
        if is_training:
            freqs = np.random.uniform(0.1*config.fs/2, 0.9*config.fs/2, 3) # 3个随机频点
            amps = np.random.uniform(0.5, 1.0, 3)
        else:
            freqs = [config.fs/4 + 1.23e6] # 测试时用单音信号便于观察频谱
            amps = [1.0]

        # 初始化通道数据容器
        interleaved_signal = np.zeros(n_points)
        clean_signal = np.zeros(n_points)
        
        # 对每个子ADC进行采样
        for m in range(config.M):
            # 子ADC的采样时刻
            # 理想时刻: t = (m + k*M) * Ts
            # 实际时刻: t' = t + skew[m]
            indices = np.arange(m, n_points, config.M)
            t_ideal = indices / config.fs
            t_actual = t_ideal + config.time_skew_gt[m]
            
            # 生成该通道的模拟值
            val = np.zeros_like(t_actual)
            val_clean = np.zeros_like(t_ideal)
            
            for f, a in zip(freqs, amps):
                # 模拟失配信号
                signal_component = a * np.sin(2 * np.pi * f * t_actual)
                val += signal_component
                
                # 理想参考信号 (Ground Truth)
                clean_component = a * np.sin(2 * np.pi * f * t_ideal)
                val_clean += clean_component
            
            # 应用增益和偏置失配
            val = val * config.gains_gt[m] + config.offsets_gt[m]
            
            # 填入交织序列
            interleaved_signal[indices] = val
            clean_signal[indices] = val_clean
            
        batch_data.append(interleaved_signal)
        batch_target.append(clean_signal)
        
    return torch.tensor(np.array(batch_data), dtype=torch.float32).to(device), \
           torch.tensor(np.array(batch_target), dtype=torch.float32).to(device)

# ==========================================
# 3. 可微分校正模型 (Differentiable Model)
# ==========================================
class TIADCCorrector(nn.Module):
    def __init__(self, config):
        super(TIADCCorrector, self).__init__()
        self.M = config.M
        self.filter_tap = config.filter_tap
        self.padding = (config.filter_tap - 1) // 2
        
        # 1. 可学习的 Gain 和 Offset (每个通道一个参数)
        # 初始化为 Gain=1, Offset=0
        self.gains = nn.Parameter(torch.ones(1, self.M, 1))
        self.offsets = nn.Parameter(torch.zeros(1, self.M, 1))
        
        # 2. 可学习的 FIR 滤波器 (用于校正 Time Skew 和频响)
        # 形状: (out_channels, in_channels/groups, kernel_size)
        # 这里使用 Group Conv，每个通道独立滤波
        self.filters = nn.Conv1d(
            in_channels=self.M, 
            out_channels=self.M, 
            kernel_size=self.filter_tap, 
            padding=self.padding, 
            groups=self.M, 
            bias=False
        )
        
        self._init_filters()

    def _init_filters(self):
        # 关键步骤：初始化为 Dirac Delta (直通)，否则训练极难收敛
        with torch.no_grad():
            self.filters.weight.zero_()
            center_idx = self.filter_tap // 2
            for i in range(self.M):
                self.filters.weight[i, 0, center_idx] = 1.0

    def forward(self, x):
        # Input x shape: [Batch, Length]
        B, L = x.shape
        
        # 1. 模拟解交织 (De-interleave) -> [Batch, M, L/M]
        # 注意：这里假设 L 是 M 的整数倍
        x_poly = x.view(B, L // self.M, self.M).permute(0, 2, 1)
        
        # 2. 逆向校正 Offset 和 Gain
        # y = (x - offset) / gain
        x_corr = (x_poly - self.offsets) / (self.gains + 1e-8)
        
        # 3. 时间/频率校正 (FIR Filter)
        # 输入: [Batch, M, Sub_Len], 输出: [Batch, M, Sub_Len]
        x_corr = self.filters(x_corr)
        
        # 4. 重新交织 (Re-interleave) -> [Batch, Length]
        x_out = x_corr.permute(0, 2, 1).contiguous().view(B, L)
        
        return x_out

# ==========================================
# 4. 训练与评估 (Training & Eval)
# ==========================================
def train_model():
    model = TIADCCorrector(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    print("开始训练...")
    model.train()
    for epoch in range(cfg.epochs):
        # 生成动态数据
        inputs, targets = generate_synthetic_data(cfg.batch_size, cfg.n_points, cfg, is_training=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 损失函数：时域 MSE
        loss = criterion(outputs, targets)
        
        # 可选：频域损失 (Spectral Loss) 可以加速高频收敛，这里为了简洁仅用MSE
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{cfg.epochs}], Loss: {loss.item():.6e}")
            
    return model, loss_history

def evaluate_and_plot(model):
    model.eval()
    # 生成测试数据（单音信号，便于观察杂散）
    inputs, targets = generate_synthetic_data(1, cfg.n_points, cfg, is_training=False)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # 转换为 Numpy
    x_in = inputs.cpu().numpy().flatten()
    x_out = outputs.cpu().numpy().flatten()
    x_gt = targets.cpu().numpy().flatten()
    
    # 1. 时域波形 (取前200点)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_in[:100], label='Distorted Input', alpha=0.7)
    plt.plot(x_gt[:100], label='Ideal Target', linestyle='--')
    plt.plot(x_out[:100], label='Corrected Output')
    plt.title("Time Domain Waveform (Zoomed)")
    plt.legend()
    plt.grid(True)
    
    # 2. 功率谱密度 (PSD) 对比
    f, Pxx_in = welch(x_in, fs=cfg.fs, nperseg=1024, window='hann')
    _, Pxx_out = welch(x_out, fs=cfg.fs, nperseg=1024, window='hann')
    
    plt.subplot(2, 1, 2)
    plt.semilogy(f/1e9, Pxx_in, label='Input PSD (Distorted)', color='red', alpha=0.5)
    plt.semilogy(f/1e9, Pxx_out, label='Output PSD (Corrected)', color='green', alpha=0.8)
    plt.title("Power Spectral Density (PSD)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("PSD (V**2/Hz)")
    plt.legend()
    plt.grid(True, which='both')
    
    # 3. 显示学到的参数
    print("\n=== 校正参数分析 ===")
    print("True Gains: ", cfg.gains_gt)
    print("Learned Gains: ", model.gains.detach().cpu().numpy().flatten())
    print("-" * 30)
    print("True Offsets: ", cfg.offsets_gt)
    print("Learned Offsets: ", model.offsets.detach().cpu().numpy().flatten())
    print("-" * 30)
    print("Learned Filters (Center Taps):")
    print(model.filters.weight.data[:, 0, cfg.filter_tap//2].cpu().numpy())
    
    plt.tight_layout()
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 训练模型
    trained_model, loss_hist = train_model()
    
    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(loss_hist)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.show()
    
    # 验证效果
    evaluate_and_plot(trained_model)