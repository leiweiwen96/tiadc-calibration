function tiadc_ddsp_demo()
    % 基于 DDSP (可微频域采样) 的 TIADC 校准 MATLAB 演示
    
    clc; clear; close all;
    
    % ==========================================
    % 1. 仿真参数设置
    % ==========================================
    fs = 20e9;              % 采样率 20 Gsps
    N_sim = 4096;           % 仿真点数
    t = (0:N_sim-1) / fs;
    
    num_taps = 65;          % FIR 滤波器阶数 (建议 65 或 129)
    % FFT 点数，设大一些以获得更好的频域分辨率
    n_fft = 2^nextpow2(num_taps) * 4; 
    
    fprintf('=== TIADC Calibration Simulation (MATLAB) ===\n');
    fprintf('Sampling Rate: %.1f Gsps, FIR Taps: %d\n', fs/1e9, num_taps);

    % ==========================================
    % 2. 生成数据 (Simulator)
    % ==========================================
    % A. 生成训练信号 (多音信号，覆盖全频段)
    freqs_train = linspace(0.5e9, 8.5e9, 15); 
    amps_train = rand(1, length(freqs_train)) * 0.5 + 0.5;
    [x_ref_train, x_dist_train] = generate_tiadc_data(t, freqs_train, amps_train, fs);
    
    % B. 生成测试信号 (单音信号，不在训练集中，用于验证泛化性)
    freq_test = 3.7e9; % 3.7 GHz
    [x_ref_test, x_dist_test] = generate_tiadc_data(t, freq_test, 0.9, fs);

    % ==========================================
    % 3. DDSP 优化核心 (Training)
    % ==========================================
    % 优化变量：[Log_Magnitude; Phase]
    % 频域点数 (DC 到 Nyquist)
    num_freq_points = n_fft/2 + 1;
    
    % 初始化：
    % Log Magnitude = 0 (线性增益=1)
    % Phase = 0 (无延时)
    x0 = zeros(num_freq_points * 2, 1);
    
    % 定义损失函数 (MSE)
    % fun 输入: 优化变量 x
    % fun 输出: MSE 误差
    cost_func = @(x) ddsp_cost_function(x, x_dist_train, x_ref_train, num_taps, n_fft);
    
    % 优化选项
    options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'Display', 'iter-detailed', ...
        'MaxFunctionEvaluations', 20000, ...
        'MaxIterations', 500, ...
        'OptimalityTolerance', 1e-7, ...
        'StepTolerance', 1e-7);
    
    fprintf('\nStarting Optimization (Training)...\n');
    tic;
    [x_opt, final_loss] = fminunc(cost_func, x0, options);
    train_time = toc;
    fprintf('Training finished in %.2f seconds. Final MSE: %.2e\n', train_time, final_loss);

    % ==========================================
    % 4. 验证与评估 (Evaluation)
    % ==========================================
    % 从优化结果中提取 FIR 系数
    h_opt = get_fir_from_params(x_opt, num_taps, n_fft);
    
    % 应用校正到测试集
    % 注意：MATLAB conv 会改变长度，取 'same'
    y_cal_test = conv(x_dist_test, h_opt, 'same');
    
    % 组合 TIADC 输出 (Interleaving)
    % 我们假设 Ch0 是参考通道，Ch1 是校正后通道
    % 理想 Ch0 数据
    ch0_data = x_ref_test; 
    
    % 未校正 Ch1 数据
    ch1_raw = x_dist_test;
    
    % 校正后 Ch1 数据
    ch1_cal = y_cal_test;
    
    % 构造交替采样序列
    tiadc_uncal = zeros(1, N_sim * 2);
    tiadc_uncal(1:2:end) = ch0_data; % 奇数点 Ch0
    tiadc_uncal(2:2:end) = ch1_raw;  % 偶数点 Ch1 (Uncalibrated)
    
    tiadc_cal = zeros(1, N_sim * 2);
    tiadc_cal(1:2:end) = ch0_data;   % 奇数点 Ch0
    tiadc_cal(2:2:end) = ch1_cal;    % 偶数点 Ch1 (Calibrated)
    
    % 计算 SFDR
    [sfdr_pre, f_axis, spec_pre] = calc_sfdr(tiadc_uncal, fs*2);
    [sfdr_post, ~, spec_post] = calc_sfdr(tiadc_cal, fs*2);
    
    fprintf('\n=== Results ===\n');
    fprintf('SFDR Before Calibration: %.2f dB\n', sfdr_pre);
    fprintf('SFDR After Calibration:  %.2f dB\n', sfdr_post);
    fprintf('Improvement:             %.2f dB\n', sfdr_post - sfdr_pre);
    
    % ==========================================
    % 5. 绘图 (Paper Ready Plots)
    % ==========================================
    figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
    
    % 子图 1: 滤波器冲激响应 (Impulse Response)
    subplot(2, 2, 1);
    stem(h_opt, 'filled', 'Color', [0, 0.4470, 0.7410]);
    title(sprintf('Learned FIR Coefficients (N=%d)', num_taps));
    xlabel('Tap Index'); grid on;
    axis tight;
    
    % 子图 2: 频响特性 (学习到了什么？)
    subplot(2, 2, 2);
    [H_resp, w] = freqz(h_opt, 1, 1024, fs);
    yyaxis left
    plot(w/1e9, 20*log10(abs(H_resp)));
    ylabel('Magnitude (dB)');
    yyaxis right
    plot(w/1e9, unwrap(angle(H_resp)) * 180/pi);
    ylabel('Phase (deg)');
    title('Learned Filter Frequency Response');
    xlabel('Frequency (GHz)'); grid on;
    legend('Mag', 'Phase');
    
    % 子图 3: 频谱对比 (验证 SFDR)
    subplot(2, 1, 2);
    plot(f_axis/1e9, spec_pre, 'r', 'LineWidth', 1.0); hold on;
    plot(f_axis/1e9, spec_post, 'b', 'LineWidth', 1.2);
    yline(-sfdr_pre, 'r--', sprintf('Pre: %.1f dB', sfdr_pre));
    yline(-sfdr_post, 'b--', sprintf('Post: %.1f dB', sfdr_post));
    title(sprintf('Spectrum Comparison (Test Tone @ %.1f GHz)', freq_test/1e9));
    xlabel('Frequency (GHz)'); ylabel('Amplitude (dB)');
    legend('Uncalibrated', 'Calibrated');
    ylim([-100, 5]); grid on;
    
    fprintf('Done.\n');
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