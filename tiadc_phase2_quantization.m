function tiadc_phase2_quantization()
    % Phase 2: Fixed-Point Analysis & Resource Estimation
    % 目标：生成 IEEE TIM 论文中的 "Hardware Feasibility Analysis" 图表
    
    clc; clear; close all;
    
    % ==========================================
    % 1. 基础设置与浮点训练 (复用阶段1)
    % ==========================================
    fs = 20e9;              
    num_taps = 65;          % 固定为 65 阶 (根据阶段1的最佳结果)
    n_fft = 2^nextpow2(num_taps) * 4; 
    
    fprintf('=== Phase 2: Quantization Analysis (FIR Taps: %d) ===\n', num_taps);

    % --- 生成数据 ---
    t = (0:4095)/fs;
    % 训练集 (多音)
    [x_ref_train, x_dist_train] = generate_tiadc_data(t, linspace(0.5e9, 8.5e9, 10), ones(1,10), fs);
    % 测试集 (单音 3.7GHz)
    [x_ref_test, x_dist_test] = generate_tiadc_data(t, 3.7e9, 0.9, fs);
    
    % --- 浮点模型训练 (Golden Reference) ---
    fprintf('Step 1: Training Floating-point Model (Golden Reference)...\n');
    % 快速训练设置
    x0 = zeros((n_fft/2 + 1) * 2, 1);
    cost_func = @(x) ddsp_cost_function(x, x_dist_train, x_ref_train, num_taps, n_fft);
    options = optimoptions('fminunc', 'Display', 'none', 'MaxIterations', 200, 'Algorithm', 'quasi-newton');
    x_opt = fminunc(cost_func, x0, options);
    
    % 获取浮点系数
    h_float = get_fir_from_params(x_opt, num_taps, n_fft);
    
    % 计算浮点基准 SFDR
    y_float = conv(x_dist_test, h_float, 'same');
    tiadc_float = interleave(x_ref_test, y_float);
    sfdr_float = calc_sfdr(tiadc_float, fs*2);
    fprintf('Golden Reference SFDR: %.2f dB\n', sfdr_float);

    % ==========================================
    % 2. 定点化扫描实验 (Bit-width Sweep)
    % ==========================================
    fprintf('\nStep 2: Running Quantization Sweep...\n');
    
    bit_widths = 8:1:18; % 扫描 8 到 18 bit
    sfdr_results = zeros(size(bit_widths));
    
    for i = 1:length(bit_widths)
        bits = bit_widths(i);
        
        % --- 量化核心逻辑 ---
        % 假设 FPGA 采用定点小数表示。
        % 首先归一化系数，使其最大值适配量化范围，然后量化
        % FIR 系数通常在 [-1, 1] 之间，量化为 Q(bits).
        
        h_fixed = fixed_point_quantize(h_float, bits);
        
        % 使用量化后的系数进行滤波
        y_fixed = conv(x_dist_test, h_fixed, 'same');
        
        % 组合 TIADC 并计算 SFDR
        tiadc_fixed = interleave(x_ref_test, y_fixed);
        sfdr_curr = calc_sfdr(tiadc_fixed, fs*2);
        
        sfdr_results(i) = sfdr_curr;
        fprintf('  Bits: %d | SFDR: %.2f dB\n', bits, sfdr_curr);
    end
    
    % ==========================================
    % 3. 资源估算 (Resource Estimation)
    % ==========================================
    % 假设目标 FPGA: Xilinx Kintex-7 或 UltraScale
    % FIR 结构: Direct Form or Transpose Form
    % DSP Slices: 乘法器数量
    % Block RAM: 如果系数多可能用到
    
    % 简单估算模型：
    % 我们的滤波器是非对称的 (用于校正相位)，无法利用对称性减少乘法器。
    % 每个 Tap 需要 1 个乘法器 (DSP Slice)。
    dsp_usage = num_taps; 
    
    % 假设输入数据 8-bit (ADC), 系数 N-bit
    % 估算逻辑资源 (LUTs/FFs) - 粗略经验公式
    % LUTs ≈ Taps * (BitWidth * 2)
    
    fprintf('\n=== FPGA Resource Estimation (Theoretical) ===\n');
    fprintf('Target: Xilinx Kintex-7 (Example)\n');
    fprintf('Filter Order: %d\n', num_taps);
    fprintf('DSP48 Slices (Multipliers): %d\n', dsp_usage);
    fprintf('Note: Traditional method requires 3 filters (FIR+Allpass+Farrow)\n');
    
    % ==========================================
    % 4. 绘图 (Paper Ready)
    % ==========================================
    figure('Color', 'w', 'Position', [100, 100, 800, 500]);
    
    % 绘制 SFDR vs Bit-width 曲线
    plot(bit_widths, sfdr_results, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    hold on;
    yline(sfdr_float, 'g--', 'Floating Point Baseline', 'LineWidth', 2);
    
    % 标注推荐点 (比如 14 bit)
    rec_idx = find(sfdr_results >= sfdr_float - 1.0, 1); % 找到第一个接近浮点的位宽
    if ~isempty(rec_idx)
        rec_bit = bit_widths(rec_idx);
        rec_sfdr = sfdr_results(rec_idx);
        plot(rec_bit, rec_sfdr, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
        text(rec_bit, rec_sfdr-5, sprintf('Recommended: %d-bit\n(Trade-off Point)', rec_bit), ...
            'HorizontalAlignment', 'center', 'Color', 'r', 'FontWeight', 'bold');
    end
    
    grid on;
    xlabel('Quantization Bit Width (bits)', 'FontSize', 12);
    ylabel('SFDR (dB)', 'FontSize', 12);
    title('Impact of Coefficient Quantization on Calibration Performance', 'FontSize', 14);
    legend('Quantized Performance', 'Floating Point Baseline', 'Location', 'SouthEast');
    ylim([20, 90]);
    
    fprintf('Done. Please save the plot for your paper.\n');
end

% =========================================================================
% 辅助函数
% =========================================================================

function h_q = fixed_point_quantize(h, bits)
    % 模拟定点量化
    % bits: 总位宽 (包含符号位)
    % 假设采用 Q1.X 格式 (因为 FIR 系数通常 < 1)
    
    max_val = max(abs(h));
    
    % 动态缩放：找到最合适的量化范围
    % 实际 FPGA 中，这对应调整小数点位置
    % 这里我们简单地将 +/- 1 范围映射到整数
    
    scale = 2^(bits-1) - 1;
    
    % 量化过程：缩放 -> 取整 -> 还原
    h_int = round(h * scale);
    
    % 饱和截断 (防止溢出)
    h_int(h_int > scale) = scale;
    h_int(h_int < -scale) = -scale;
    
    h_q = h_int / scale;
end

function y = interleave(ch0, ch1)
    % 交替拼接
    len = min(length(ch0), length(ch1));
    y = zeros(1, len*2);
    y(1:2:end) = ch0(1:len);
    y(2:2:end) = ch1(1:len);
end

% =========================================================================
% 辅助函数定义
% =========================================================================

function [ref, dist] = generate_tiadc_data(t, freqs, amps, fs)
    % 模拟 TIADC 数据生成
    % ref: 参考通道 (Channel 0) - 假设接近理想，略带带宽限制
    % dist: 待校正通道 (Channel 1) - 包含增益、延时、带宽失配
    
    % 1. 生成源信号
    sig_src = zeros(size(t));
    for i = 1:length(freqs)
        sig_src = sig_src + amps(i) * sin(2*pi*freqs(i)*t);
    end
    sig_src = sig_src / max(abs(sig_src)) * 0.9; % 归一化
    
    % 2. 通道 0 (参考): 4.5GHz 带宽
    [b0, a0] = butter(5, 4.5e9/(fs/2));
    ref = filter(b0, a0, sig_src);
    
    % 3. 通道 1 (失真): 
    %   - 带宽失配: 4.0GHz (比 Ch0 窄，高频衰减大)
    %   - 增益误差: 1.05
    %   - 时间偏斜: 0.2 个采样周期 (Fractional Delay)
    
    % A. 带宽限制
    [b1, a1] = butter(5, 4.0e9/(fs/2));
    sig_temp = filter(b1, a1, sig_src);
    
    % B. 分数延时 (频域相移法)
    delay_samples = 0.2; 
    spec = fft(sig_temp);
    N = length(sig_temp);
    f_idx = [0:N/2, -N/2+1:-1]; % FFT 频率索引
    phase_shift = exp(-1j * 2 * pi * f_idx * delay_samples / N);
    sig_delayed = real(ifft(spec .* phase_shift));
    
    % C. 增益误差
    dist = sig_delayed * 1.05;
end

function cost = ddsp_cost_function(params, input, target, num_taps, n_fft)
    % 损失函数：计算 FIR 滤波后的 MSE
    % params: 优化变量 [log_mags; phases]
    
    % 1. 还原 FIR 系数
    h = get_fir_from_params(params, num_taps, n_fft);
    
    % 2. 时域卷积 (使用 filter 或 conv)
    % 使用 conv 'same' 保持长度一致
    y_est = conv(input, h, 'same');
    
    % 3. 计算 MSE
    % 去除边缘效应 (Padding 导致的误差)
    valid_mask = false(size(y_est));
    margin = num_taps;
    valid_mask(margin:end-margin) = true;
    
    err = y_est(valid_mask) - target(valid_mask);
    cost = mean(err.^2);
end

function h = get_fir_from_params(params, num_taps, n_fft)
    % 核心：从频域参数重构时域 FIR (可微过程)
    
    num_points = n_fft/2 + 1;
    log_mag = params(1:num_points);
    phase = params(num_points+1:end);
    
    % 1. 构建正半轴复数谱
    mag = exp(log_mag);
    H_half = mag .* exp(1j * phase);
    
    % 2. 构建全频谱 (Hermitian Symmetric 保证 IFFT 为实数)
    % H = [DC, PosFreq, Nyquist, NegFreq]
    H_full = zeros(n_fft, 1);
    H_full(1:num_points) = H_half;
    % 负频率部分是正频率的共轭倒序 (不含 DC 和 Nyquist)
    H_full(num_points+1:end) = conj(flipud(H_half(2:end-1)));
    
    % 3. IDFT 转时域
    h_temp = real(ifft(H_full));
    
    % 4. 循环移位 (Centering)
    h_centered = circshift(h_temp, n_fft/2);
    
    % 5. 截断 (Truncate)
    start_idx = floor((n_fft - num_taps)/2) + 1;
    h_trunc = h_centered(start_idx : start_idx + num_taps - 1);
    
    % 6. 加窗 (Windowing) - Blackman 窗
    w = blackman(num_taps);
    h = h_trunc .* w;
    
    % 归一化 (可选，这里不归一化让网络学习增益)
end

function [sfdr, f, Y_db] = calc_sfdr(sig, fs)
    % 计算 SFDR
    L = length(sig);
    w = blackman(L)';
    Y = abs(fft(sig .* w));
    Y = Y(1:floor(L/2)+1);
    f = fs * (0:(L/2)) / L;
    
    % 转 dB 并归一化
    Y_db = 20*log10(Y + 1e-12);
    Y_db = Y_db - max(Y_db);
    
    % 找峰值
    [~, locs] = findpeaks(Y_db, 'MinPeakHeight', -100, 'MinPeakDistance', 10);
    if isempty(locs)
        sfdr = 0; return;
    end
    
    % 排序峰值  
    peaks = Y_db(locs);
    sorted_peaks = sort(peaks, 'descend');
    
    if length(sorted_peaks) < 2


        
        sfdr = 80; % 只有一个主峰，性能极好
    else
        % 主峰是 0dB (已归一化)，第二峰即杂散
        spur = sorted_peaks(2);
        sfdr = 0 - spur;
    end
end