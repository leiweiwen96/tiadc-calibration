import numpy as np
import matplotlib.pyplot as plt

def fractional_delay(sig, delay_samples):
    """
    核心算法：利用 FFT 实现高精度分数延时
    原理：时域延时 delta_t <==> 频域乘以 exp(-j * 2*pi * f * delta_t)
    """
    N = len(sig)
    
    # 1. 变换到频域
    X = np.fft.rfft(sig)
    
    # 2. 生成频率轴 (归一化频率 f = k/N)
    # rfftfreq 返回的是 [0, 1/N, 2/N, ... 0.5]
    freqs = np.fft.rfftfreq(N)
    
    # 3. 施加线性相位旋转 (Linear Phase Shift)
    # 相位因子 = exp(-j * 2π * f * delay)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_samples)
    
    # 4. 变换回时域
    X_shifted = X * phase_shift
    return np.fft.irfft(X_shifted, n=N)

def run_demo():
    # ==========================================
    # 1. 生成测试信号 (Sinc 脉冲)
    # ==========================================
    N = 100
    t = np.arange(N)
    # 在第 20 个点生成一个 Sinc 脉冲，带宽有限，非常适合测试 FFT
    # 为什么不用方波？因为方波有无穷高频，FFT移动会有吉布斯现象(振铃)
    center = 20.0
    sig_origin = np.sinc(t - center)
    
    # ==========================================
    # 2. 执行分数延时
    # ==========================================
    delay_1 = 0.5   # 移动半个点 (最难的情况)
    delay_2 = 10.3  # 任意小数移动
    
    sig_shift_0p5 = fractional_delay(sig_origin, delay_1)
    sig_shift_10p3 = fractional_delay(sig_origin, delay_2)
    
    # ==========================================
    # 3. 硬核指标验证 (Validation)
    # ==========================================
    print("=== 算法可靠性验证报告 ===")
    
    # 验证 A: 能量守恒 (Parseval定理)
    # 延时只是改变相位，不应该改变信号的总能量/幅度
    energy_in = np.sum(sig_origin**2)
    energy_out = np.sum(sig_shift_10p3**2)
    print(f"1. 能量守恒检查: {'通过' if np.isclose(energy_in, energy_out) else '失败'}")
    print(f"   输入能量: {energy_in:.6f}, 输出能量: {energy_out:.6f}")
    
    # 验证 B: 整数延时精度
    # 如果移动 5.0 个点，应该和 numpy.roll (硬数组移位) 一模一样
    sig_shift_int = fractional_delay(sig_origin, 5.0)
    sig_roll_int = np.roll(sig_origin, 5)
    # 掐头去尾比较(消除FFT循环卷积边缘影响)
    err = np.max(np.abs(sig_shift_int[10:-10] - sig_roll_int[10:-10]))
    print(f"2. 整数移动精度: {'完美' if err < 1e-10 else '有误差'}")
    print(f"   最大误差: {err:.2e}")

    # ==========================================
    # 4. 可视化绘图
    # ==========================================
    plt.figure(figsize=(12, 8))
    
    # 图1: 时域细节 (验证 0.5 移动)
    plt.subplot(2, 1, 1)
    plt.plot(t, sig_origin, 'ko-', label='Original (Peak @ 20.0)', alpha=0.5, markersize=6)
    plt.plot(t, sig_shift_0p5, 'r*-', label='Shift +0.5 (Peak should be @ 20.5)')
    
    # 辅助线：标出原来的点和新的点
    plt.axvline(20, color='k', linestyle=':', alpha=0.3)
    plt.axvline(20.5, color='r', linestyle=':', alpha=0.3)
    
    plt.xlim(15, 25) # 放大看局部
    plt.title(f"Time Domain: Sub-sample Shift (+{delay_1} samples)")
    plt.xlabel("Sample Index")
    plt.grid(True)
    plt.legend()
    
    # 图2: 长距离移动对比
    plt.subplot(2, 2, 3)
    plt.plot(t, sig_origin, 'k', label='Original')
    plt.plot(t, sig_shift_10p3, 'g', label=f'Shift +{delay_2}')
    plt.title(f"Time Domain: Long Shift (+{delay_2} samples)")
    plt.legend()
    plt.grid(True)
    
    # 图3: 频域幅度谱 (验证无损)
    plt.subplot(2, 2, 4)
    # 计算幅度谱
    spec_origin = np.abs(np.fft.rfft(sig_origin))
    spec_shift = np.abs(np.fft.rfft(sig_shift_10p3))
    
    plt.plot(spec_origin, 'k', linewidth=3, label='Original Spectrum')
    plt.plot(spec_shift, 'y--', linewidth=2, label='Shifted Spectrum')
    plt.title("Frequency Domain: Magnitude Spectrum")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()