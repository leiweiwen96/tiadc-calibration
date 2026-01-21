function generate_serdes_data()
    % 使用 SerDes/RF 原理生成高保真 TIADC 训练数据
    % 目标：利用 S 参数或传输线模型产生真实的"模拟带宽失配"
    
    clc; clear; close all;
    
    fprintf('=== SerDes Physics-Based Data Generation ===\n');
    
    % ==========================================
    % 1. 基础设置
    % ==========================================
    fs_sys = 20e9;              % 系统总采样率 20 Gsps
    ts = 1/fs_sys;
    N_samples = 16384;          % 样本数
    t = (0:N_samples-1) * ts;
    
    % ==========================================
    % 2. 构建输入信号 (Chirp 信号)
    % ==========================================
    % 为了训练全频段，必须使用 Chirp (DC -> Nyquist)
    % 模拟源信号 (假设源本身是理想的)
    f_start = 10e6;             % 10 MHz
    f_stop = 9.5e9;             % 9.5 GHz (接近 Nyquist)
    
    % 生成 Chirp
    tx_signal = chirp(t, f_start, t(end), f_stop, 'linear');
    
    % 加一点源端的随机抖动 (Tx Jitter) - 可选
    % 这里先保持源纯净，主要看通道差异
    
    % ==========================================
    % 3. 构建物理通道模型 (关键步骤)
    % ==========================================
    % 我们使用 rationalfit 或滤波器来近似 PCB 走线的 S21 参数
    % 如果你有真实的 .s4p 文件最好，没有的话我们用传输线方程模拟
    
    fprintf('Modeling PCB Trace Mismatch...\n');
    
    % --- Channel 0 (参考通道): 短走线，低损耗 ---
    % 物理参数: 2英寸走线, 介质损耗角正切 0.01
    model_ch0 = create_transmission_line_model(fs_sys, 2.0, 0.01); 
    
    % --- Channel 1 (待校通道): 长走线，高损耗 (制造带宽失配) ---
    % 物理参数: 2.5英寸走线 (引入延时), 介质损耗 0.02 (引入高频衰减)
    % 这种差异就是现实中 PCB 布局不对称导致的
    model_ch1 = create_transmission_line_model(fs_sys, 2.5, 0.025);
    
    % ==========================================
    % 4. 信号通过通道 (Time Domain Simulation)
    % ==========================================
    % 使用线性系统仿真 (lsim 或 filter)
    
    fprintf('Simulating Signal Propagation...\n');
    rx_ch0 = lsim(model_ch0, tx_signal, t);
    rx_ch1 = lsim(model_ch1, tx_signal, t);
    
    % ==========================================
    % 5. 模拟 ADC 采样与量化
    % ==========================================
    % 这里模拟 TIADC 的动作：
    % Ch0 和 Ch1 是交替工作的。但在训练时，我们需要它们"对齐"的数据
    % 来学习差异。
    
    % 叠加一些热噪声 (Thermal Noise)
    noise_floor = -60; % dB
    rx_ch0 = add_noise(rx_ch0, noise_floor);
    rx_ch1 = add_noise(rx_ch1, noise_floor);
    
    % ==========================================
    % 6. 保存数据用于 Python/DDSP 训练
    % ==========================================
    % 注意：rx_ch1 会因为走线长而有一个物理延时。
    % 在送入 DDSP 之前，我们通常会在 Python 里做"整数位对齐"
    % 这里我们直接保存原始数据。
    
    data_train.input = rx_ch1;   % 烂的数据
    data_train.target = rx_ch0;  % 好的数据 (Golden Ref)
    data_train.fs = fs_sys;
    
    save('tiadc_serdes_data.mat', 'data_train');
    fprintf('Data saved to tiadc_serdes_data.mat\n');
    
    % ==========================================
    % 7. 绘图验证 (Paper Quality Plots)
    % ==========================================
    figure('Position', [100, 100, 1000, 600]);
    
    % 时域对比 (放大看延时)
    subplot(2,2,1);
    plot(t(1:100)*1e9, rx_ch0(1:100), 'b', 'LineWidth', 1.5); hold on;
    plot(t(1:100)*1e9, rx_ch1(1:100), 'r--');
    title('Time Domain (Zoomed)');
    xlabel('Time (ns)'); ylabel('Amplitude');
    legend('Ch0 (Ref)', 'Ch1 (Distorted)');
    grid on;
    
    % 频域对比 (看带宽失配)
    subplot(2,2,2);
    [H0, f] = freqz(model_ch0.num{1}, model_ch0.den{1}, 1024, fs_sys);
    [H1, ~] = freqz(model_ch1.num{1}, model_ch1.den{1}, 1024, fs_sys);
    
    plot(f/1e9, 20*log10(abs(H0)), 'b', 'LineWidth', 1.5); hold on;
    plot(f/1e9, 20*log10(abs(H1)), 'r', 'LineWidth', 1.5);
    title('S21 Insertion Loss (Bandwidth Mismatch)');
    xlabel('Frequency (GHz)'); ylabel('Magnitude (dB)');
    legend('Ch0 (Short Trace)', 'Ch1 (Long Trace)');
    grid on;
    
    % 冲激响应对比
    subplot(2,1,2);
    impulse(model_ch0, 2e-9); hold on;
    impulse(model_ch1, 2e-9);
    title('Channel Impulse Response (Physical Delay)');
    legend('Ch0', 'Ch1');
    grid on;
end

% =========================================================
% 辅助函数：构建物理传输线模型
% =========================================================
function sys = create_transmission_line_model(fs, length_inch, loss_tangent)
    % 物理常数
    c = 3e8;
    er = 4.2; % FR4 介电常数
    v_prop = c / sqrt(er);
    
    % 1. 计算物理延时 (Time Delay)
    len_m = length_inch * 0.0254;
    delay = len_m / v_prop;
    
    % 2. 模拟高频损耗 (Skin effect + Dielectric loss)
    % 经验公式：越长带宽越低
    % 这里调整一下系数，让衰减更明显一点，同时避免超宽带报错
    % 假设 5 inch 对应 3GHz 带宽
    ref_len = 5; 
    ref_bw = 3e9;
    fc = ref_bw * (ref_len / length_inch); 
    
    % === [关键修复] 奈奎斯特钳位 ===
    nyquist = fs / 2;
    if fc >= nyquist * 0.95
        % 如果物理带宽超过了奈奎斯特频率，说明在当前采样率下
        % 这个通道几乎是全通的。我们将其限制在边界内。
        fc = nyquist * 0.95; 
    end
    
    % 构建模拟滤波器
    [b, a] = butter(2, fc / nyquist);
    
    % 转换为 tf 对象 (仅用于存储系数，不用于 lsim 直接调用)
    sys_filter = tf(b, a, 1/fs);
    
    % 返回结构体或对象
    sys = sys_filter;
    sys.UserData.delay_sec = delay;
end

function y = add_noise(x, snr_db)
    signal_power = mean(x.^2);
    noise_power = signal_power / (10^(snr_db/10));
    noise = sqrt(noise_power) * randn(size(x));
    y = x + noise;
end

% 重写 lsim 包装器以处理手动延时
function y = lsim(sys, u, t)
    % 1. 滤波 (模拟频率响应损耗)
    y_filtered = filter(sys.num{1}, sys.den{1}, u);
    
    % 2. 延时 (模拟物理长度)
    dt = t(2) - t(1);
    delay_sec = sys.UserData.delay_sec;
    delay_samples = delay_sec / dt;
    
    % 使用分数延时滤波器或简单插值
    % 这里为了演示原理，使用频域移相实现高精度延时
    N = length(y_filtered);
    X = fft(y_filtered);
    f = (0:N-1)' * (1/(t(end)-t(1))); % 频率轴
    phase_shift = exp(-1j * 2 * pi * f * delay_sec);
    y = real(ifft(X .* phase_shift));
end