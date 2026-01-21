function verify_result()
    clc; clear; close all;
    
    % 1. 加载 Python 训练好的系数
    try
        coeffs = load('fir_coeffs_float.txt');
        coeffs = coeffs(:); % [关键] 强制转为列向量
        fprintf('成功加载系数，阶数: %d\n', length(coeffs));
    catch
        error('找不到 fir_coeffs_float.txt，请先运行 Python 训练脚本！');
    end
    
    % 2. 加载测试数据 (单音)
    try
        load('tiadc_test_tone.mat'); 
    catch
        error('找不到 tiadc_test_tone.mat，请先运行 MATLAB 生成测试数据脚本！');
    end
    
    % [核心修复在这里]：加载后立即转置为列向量
    x_dist = data_test.input(:);  % <--- 加上 (:), 变成 N x 1
    x_ref  = data_test.target(:); % <--- 加上 (:), 变成 N x 1
    fs     = data_test.fs;
    
    % ==========================================
    % 3. 关键步骤：对齐与滤波
    % ==========================================
    
    % A. 粗延时对齐
    [c, lags] = xcorr(x_ref, x_dist); % 现在它们都是列向量了，不需要这里再加 (:)
    [~, I] = max(c);
    lag = lags(I);
    fprintf('检测到物理粗延时: %d 采样点\n', lag);
    
    % 对齐 Ch1 (现在的拼接操作就不会报错了)
    if lag > 0
        % [列向量; 列向量] -> OK
        x_dist_aligned = [x_dist(lag+1:end); zeros(lag,1)];
    elseif lag < 0
        x_dist_aligned = [zeros(abs(lag),1); x_dist(1:end+lag)];
    else
        x_dist_aligned = x_dist; 
    end
    
    % B. 应用 Python 训练的 FIR 滤波器
    y_cal = conv(x_dist_aligned, coeffs, 'same');
    
    % ==========================================
    % 4. 模拟 TIADC 交替采样 (Interleaving)
    % ==========================================
    N = 4096;
    start_idx = 1000;
    
    % 确保取出的片段转置为 行向量 (1xN) 以便进行交替拼接
    % (tiadc_uncal 是 1x2N 的行向量)
    ch0_segment = x_ref(start_idx : start_idx+N-1)'; 
    ch1_raw_segment = x_dist_aligned(start_idx : start_idx+N-1)';
    ch1_cal_segment = y_cal(start_idx : start_idx+N-1)';
    
    % --- 场景 1: 校准前 (Uncalibrated TIADC) ---
    tiadc_uncal = zeros(1, 2*N);
    tiadc_uncal(1:2:end) = ch0_segment;     
    tiadc_uncal(2:2:end) = ch1_raw_segment; 
    
    % --- 场景 2: 校准后 (Calibrated TIADC) ---
    tiadc_cal = zeros(1, 2*N);
    tiadc_cal(1:2:end) = ch0_segment;       
    tiadc_cal(2:2:end) = ch1_cal_segment;   
    
    % ==========================================
    % 5. 计算 SFDR 并绘图
    % ==========================================
    [sfdr_pre, f_axis, spec_pre] = calc_sfdr_spectrum(tiadc_uncal, fs);
    [sfdr_post, ~, spec_post]    = calc_sfdr_spectrum(tiadc_cal, fs);
    
    fprintf('\n=== 验证结果 (Test Tone: %.1f GHz) ===\n', data_test.freq/1e9);
    fprintf('校准前 SFDR: %.2f dB\n', sfdr_pre);
    fprintf('校准后 SFDR: %.2f dB\n', sfdr_post);
    fprintf('提升幅度:   %.2f dB\n', sfdr_post - sfdr_pre);
    
    % 绘图
    figure('Color', 'w', 'Position', [200, 200, 900, 700]);
    
    subplot(2,1,1);
    plot(f_axis/1e9, spec_pre, 'r', 'LineWidth', 1); hold on;
    plot(f_axis/1e9, spec_post, 'b', 'LineWidth', 1);
    legend('Uncalibrated', 'Calibrated');
    title(sprintf('Spectrum Comparison (SFDR: %.1f -> %.1f dB)', sfdr_pre, sfdr_post));
    xlabel('Frequency (GHz)'); ylabel('Magnitude (dB)');
    ylim([-100, 0]); grid on;
    
    subplot(2,1,2);
    % 这里的减法也安全了，因为都是转置后的行向量
    plot(ch1_raw_segment - ch0_segment, 'r'); hold on;
    plot(ch1_cal_segment - ch0_segment, 'b');
    title('Error Residual (Time Domain)');
    legend('Error Before', 'Error After');
    grid on;
end

function [sfdr, f, Y_db] = calc_sfdr_spectrum(sig, fs)
    L = length(sig);
    w = blackman(L)'; % 窗函数
    Y = fft(sig .* w);
    
    P2 = abs(Y/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    Y_db = 20*log10(P1 + 1e-12);
    Y_db = Y_db - max(Y_db); 
    f = fs * (0:(L/2)) / L;
    
    [~, idx] = max(Y_db);
    
    mask_width = 10; 
    search_region = Y_db;
    search_region(max(1, idx-mask_width) : min(length(Y_db), idx+mask_width)) = -200;
    
    max_spur = max(search_region);
    sfdr = 0 - max_spur;
end