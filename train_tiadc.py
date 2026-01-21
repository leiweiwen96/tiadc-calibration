import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. 数据加载与预处理 (Data Engineering)
# ==============================================================================
def load_and_align_data(mat_file):
    print(f"正在加载数据: {mat_file} ...")
    try:
        mat = sio.loadmat(mat_file)
        # 注意：MATLAB结构体在Python中读取层级较深，需根据实际情况调整索引
        # 假设结构是 data_train.input 和 data_train.target
        struct = mat['data_train']
        
        # 提取数据并展平
        raw_input = struct['input'][0][0].flatten()  # Ch1 (待校准)
        raw_target = struct['target'][0][0].flatten() # Ch0 (参考标准)
        fs = struct['fs'][0][0][0][0]
        
    except Exception as e:
        print(f"数据读取错误: {e}")
        print("请检查 .mat 文件结构。正在使用合成数据进行演示...")
        # Fallback: 生成模拟数据以防读取失败
        t = np.linspace(0, 1e-6, 4096)
        raw_target = np.sin(2*np.pi*1e9*t)
        # 模拟延时和增益误差
        raw_input = 0.9 * np.sin(2*np.pi*1e9*t - 0.5) 
        fs = 20e9

    print("执行粗延时对齐 (Coarse Alignment)...")
    # 计算互相关
    correlation = signal.correlate(raw_target, raw_input, mode='full')
    lags = signal.correlation_lags(len(raw_target), len(raw_input), mode='full')
    lag = lags[np.argmax(correlation)]
    
    print(f"检测到物理延时: {lag} 个采样点")
    
    # 移位对齐
    if lag > 0:
        aligned_input = np.roll(raw_input, lag)
        # 掐头去尾，避免边缘效应
        aligned_input[:lag] = 0 
    elif lag < 0:
        aligned_input = np.roll(raw_input, lag)
        aligned_input[lag:] = 0
    else:
        aligned_input = raw_input

    # 归一化 (这一步对神经网络收敛很重要)
    scale = np.max(np.abs(raw_target))
    input_norm = aligned_input / scale
    target_norm = raw_target / scale
    
    # 截取中间稳定段用于训练
    margin = 100
    x = input_norm[margin:-margin]
    y = target_norm[margin:-margin]

    # 转为 Tensor: [Batch, Channel, Time]
    x_tensor = torch.FloatTensor(x).view(1, 1, -1)
    y_tensor = torch.FloatTensor(y).view(1, 1, -1)
    
    return x_tensor, y_tensor, fs

# ==============================================================================
# 2. DDSP 模型定义 (Model Architecture)
# ==============================================================================
class DFS_FIR_Filter(nn.Module):
    def __init__(self, num_taps=65):
        super().__init__()
        self.num_taps = num_taps
        
        # 定义卷积层，padding='same' 保证输入输出长度一致
        self.conv = nn.Conv1d(1, 1, kernel_size=num_taps, padding='same', bias=False)
        
        # === 关键初始化技巧 ===
        # 不要随机初始化！要初始化为"直通 (Identity)"或"微弱低通"
        # 这样模型是从"不校准"开始学习，而不是从"乱码"开始
        with torch.no_grad():
            self.conv.weight.zero_()
            center_idx = num_taps // 2
            self.conv.weight[0, 0, center_idx] = 1.0

    def forward(self, x):
        return self.conv(x)

    def get_coeffs(self):
        return self.conv.weight.detach().cpu().numpy().flatten()

# ==============================================================================
# 3. 混合损失函数 (Hybrid Loss Function)
# ==============================================================================
def hybrid_loss(y_pred, y_true, model):
    # A. 时域损失 (MSE): 负责对齐波形，修相位和延时
    loss_time = torch.mean((y_pred - y_true) ** 2)
    
    # B. 频域损失 (Spectral Loss): 负责修带宽失配
    # 使用 FFT 幅度谱的差异
    fft_pred = torch.fft.rfft(y_pred, dim=-1)
    fft_true = torch.fft.rfft(y_true, dim=-1)
    loss_freq = torch.mean(torch.abs(torch.abs(fft_pred) - torch.abs(fft_true)))
    
    # C. 正则化 (Regularization): 保证系数平滑，适合硬件实现
    coeffs = model.conv.weight.flatten()
    loss_reg = torch.mean((coeffs[1:] - coeffs[:-1])**2)
    
    # 权重分配 (可以根据实验调整)
    return loss_time + 0.1 * loss_freq + 0.01 * loss_reg

# ==============================================================================
# 4. 主训练流程 (Main Loop)
# ==============================================================================
def train():
    # A. 加载数据
    mat_path = 'tiadc_serdes_data.mat'
    input_data, target_data, fs = load_and_align_data(mat_path)
    
    # B. 初始化
    taps = 65  # 推荐 65 阶
    model = DFS_FIR_Filter(num_taps=taps)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n=== 开始训练 (FIR Taps: {taps}) ===")
    
    # C. 迭代
    loss_history = []
    epochs = 1500
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward
        y_pred = model(input_data)
        
        # Loss
        loss = hybrid_loss(y_pred, target_data, model)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f}")

    # ==============================================================================
    # 5. 结果分析与绘图
    # ==============================================================================
    print("\n=== 训练完成，正在绘图... ===")
    coeffs = model.get_coeffs()
    
    # 推理一次获取最终波形
    with torch.no_grad():
        calibrated_output = model(input_data).numpy().flatten()
    
    input_np = input_data.numpy().flatten()
    target_np = target_data.numpy().flatten()
    
    plt.figure(figsize=(14, 8))
    
    # 图1: 时域对比 (局部放大)
    plt.subplot(2, 2, 1)
    zoom_range = slice(1000, 1200) # 任意选取一段
    plt.plot(target_np[zoom_range], 'k', label='Target (Ch0)', linewidth=2, alpha=0.3)
    plt.plot(input_np[zoom_range], 'r--', label='Input (Ch1 Uncal)')
    plt.plot(calibrated_output[zoom_range], 'b-.', label='Calibrated (Ch1)')
    plt.title("Time Domain Alignment")
    plt.legend()
    plt.grid(True)
    
    # 图2: 学习到的 FIR 系数
    plt.subplot(2, 2, 2)
    plt.stem(coeffs)
    plt.title(f"Learned FIR Coefficients (N={taps})")
    plt.grid(True)
    
    # 图3: 频域响应 (验证带宽补偿)
    plt.subplot(2, 1, 2)
    
    # 计算 PSD (功率谱密度)
    f, Pxx_in = signal.welch(input_np, fs, nperseg=1024)
    _, Pxx_tgt = signal.welch(target_np, fs, nperseg=1024)
    _, Pxx_cal = signal.welch(calibrated_output, fs, nperseg=1024)
    
    plt.semilogy(f/1e9, Pxx_tgt, 'k', label='Target (Ref)', linewidth=2, alpha=0.3)
    plt.semilogy(f/1e9, Pxx_in, 'r', label='Input (Bandwidth Loss)')
    plt.semilogy(f/1e9, Pxx_cal, 'b', label='Calibrated (Restored)')
    plt.title("Frequency Spectrum (Bandwidth Mismatch Correction)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("PSD (V**2/Hz)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 保存系数供 FPGA 使用
    np.savetxt("fir_coeffs_float.txt", coeffs)
    print("系数已保存至 fir_coeffs_float.txt")

if __name__ == "__main__":
    train()