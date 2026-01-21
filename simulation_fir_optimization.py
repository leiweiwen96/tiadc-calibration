import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from simulation_train_verify_v0119_v1 import TIADCSimulator, HybridCalibrationModel, ComplexMSELoss, calculate_metrics_detailed

# 锁定随机种子
torch.manual_seed(42)
np.random.seed(42)

def run_fir_optimization_experiment():
    print("=== 开始 FIR 阶数优化实验 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sim = TIADCSimulator(fs=20e9)
    
    # 候选阶数：从极简到原版 (注意必须是奇数)
    taps_candidates = [21, 41, 63, 127, 255, 511]
    results_sinad = []
    
    # 定义通道参数 (保持与主程序一致)
    params_ch0 = {'cutoff_freq': 6.0e9, 'delay_samples': 0.0, 'gain': 1.0, 'hd2': 1e-3, 'hd3': 1e-3}
    params_ch1 = {'cutoff_freq': 5.9e9, 'delay_samples': 0.42, 'gain': 0.98, 'hd2': 2e-3, 'hd3': 2e-3}

    # 准备训练数据
    N_train = 32768
    white_noise = np.random.randn(N_train)
    b_src, a_src = signal.butter(6, 9.0e9/(sim.fs/2), btype='low')
    base_sig = signal.lfilter(b_src, a_src, white_noise)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9
    base_sig = base_sig + 0.5e-3 * base_sig**2 
    
    sig0 = sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch0)
    sig1 = sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch1)
    
    scale = np.max(np.abs(sig0))
    inp_t = torch.FloatTensor(sig1/scale).view(1, 1, -1).to(device)
    tgt_t = torch.FloatTensor(sig0/scale).view(1, 1, -1).to(device)
    
    loss_fn = ComplexMSELoss(device=device)

    for taps in taps_candidates:
        print(f"\n>> 测试 FIR Taps = {taps}")
        
        # 初始化模型
        model = HybridCalibrationModel(taps=taps, fpga_simulation=False).to(device)
        
        # Stage 1: Delay & Gain (快速收敛)
        opt_s1 = optim.Adam([
            {'params': model.global_delay.parameters(), 'lr': 1e-2},
            {'params': model.gain, 'lr': 1e-2}
        ])
        for _ in range(200):
            opt_s1.zero_grad()
            loss = loss_fn(model(inp_t), tgt_t, model)
            loss.backward()
            opt_s1.step()
            
        # Stage 2: FIR (快速收敛)
        opt_s2 = optim.Adam([
            {'params': model.conv.parameters(), 'lr': 1e-3}
        ])
        for _ in range(500):
            opt_s2.zero_grad()
            loss = loss_fn(model(inp_t), tgt_t, model)
            loss.backward()
            opt_s2.step()
            
        # 验证性能
        # 选取 2.1GHz 点的 SINAD 作为代表
        target_freq = 2.1e9
        
        # 复用 calculate_metrics_detailed 函数
        # 注意：这里会打印很多日志，我们暂时忍受一下，或者修改函数
        test_freqs, metrics = calculate_metrics_detailed(sim, model, scale, device, params_ch0, params_ch1)
        
        # 找到 2.1GHz 对应的索引
        idx = np.argmin(np.abs(test_freqs - target_freq))
        sinad_val = metrics['sinad_post'][idx]
        
        results_sinad.append(sinad_val)
        print(f"   Taps={taps} -> SINAD @ 2.1GHz = {sinad_val:.2f} dB")

    print("\n=== 实验结果汇总 ===")
    print("FIR阶数 | SINAD (dB) | 资源消耗评估")
    print("--------|------------|-------------")
    for t, s in zip(taps_candidates, results_sinad):
        resource = "极低" if t < 30 else "低" if t < 60 else "中" if t < 150 else "高"
        print(f"{t:7d} | {s:10.2f} | {resource}")
        
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(taps_candidates, results_sinad, 'o-', linewidth=2)
    plt.xlabel("FIR Taps")
    plt.ylabel("SINAD @ 2.1GHz (dB)")
    plt.title("FPGA Feasibility: FIR Taps vs Performance")
    plt.grid(True)
    for x, y in zip(taps_candidates, results_sinad):
        plt.text(x, y+1, f"{y:.1f}")
    plt.show()

if __name__ == "__main__":
    run_fir_optimization_experiment()
