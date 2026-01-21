function TIADC_Calibration_Final()
    % =========================================================================
    % TIADC 宽带失配校准仿真 (MATLAB 终极稳定版)
    % 修复方案: 移除 FFT Loss，改用时域高阶差分 Loss (纯实数运算，防止崩溃)
    % =========================================================================
    
    clc; clear; close all;
    
    % --- 0. 全局设置 ---
    fs = 20e9;              % 采样率 20 GSPS
    Taps = 129;             % FIR 滤波器阶数
    Epochs = 3000;          % 训练轮数
    LearnRate = 0.001;      % 学习率
    
    % 检查工具箱
    if isempty(which('dlgradient'))
        error('错误: 需要 Deep Learning Toolbox 才能运行此程序。');
    end

    disp('=== 1. 生成训练数据 (Chirp) ===');
    N_train = 16384;
    raw_src = generate_chirp(N_train, fs);
    
    % --- 物理损伤设定 ---
    ch0_ref = apply_channel_effect(raw_src, fs, 9.5e9, 0.0, 1.0);
    ch1_bad = apply_channel_effect(raw_src, fs, 6.0e9, 0.42, 0.88);
    
    % 归一化
    scale_factor = max(abs(ch0_ref));
    ch0_ref = ch0_ref / scale_factor;
    ch1_bad = ch1_bad / scale_factor;
    
    % 转换为 dlarray (SSB 格式: Spatial, Spatial, Batch)
    X_train = dlarray(reshape(ch1_bad, [1, length(ch1_bad), 1]), 'SSB'); 
    Y_target = dlarray(reshape(ch0_ref, [1, length(ch0_ref), 1]), 'SSB');
    
    % --- 2. 初始化 FIR 模型 ---
    weights = zeros(1, Taps, 1);
    weights(1, floor(Taps/2)+1, 1) = 1.0;
    W = dlarray(weights); 
    
    % Adam 状态
    avgGrad = [];
    avgSqGrad = [];
    
    disp(['=== 2. 开始训练 (', num2str(Epochs), ' Epochs) - 使用时域高频 Loss ===']);
    
    start_time = tic;
    
    for epoch = 1:Epochs
        % 计算梯度
        [loss, grads, mse_val, hf_val] = dlfeval(@model_loss_real, W, X_train, Y_target, epoch);
        
        % 更新权重
        [W, avgGrad, avgSqGrad] = adamupdate(W, grads, avgGrad, avgSqGrad, epoch, LearnRate);
        
        % 打印进度
        if mod(epoch, 500) == 0 || epoch == 1
            stage = "Phase 1";
            if epoch > 1000, stage = "Phase 2"; end
            
            fprintf('Epoch %04d [%s] | Total: %.4f | Time: %.6f | HighFreq: %.6f\n', ...
                epoch, stage, extractdata(loss), extractdata(mse_val), extractdata(hf_val));
        end
    end
    toc(start_time);
    
    W_final = extractdata(W); 
    
    % =====================================================================
    % 3. 扫频验证
    % =====================================================================
    disp('=== 3. 运行高密度扫频验证 (避开 5GHz 盲区) ===');
    
    freqs = 0.1e9 : 0.137e9 : 9.65e9;
    freqs(abs(freqs - 5.0e9) < 0.15e9) = []; % 避坑
    
    improvements = zeros(size(freqs));
    
    fprintf('正在扫描 %d 个频点...', length(freqs));
    
    for i = 1:length(freqs)
        f = freqs(i);
        N_test = 8192;
        src_tone = generate_tone(f, N_test, fs);
        
        ref_sig = apply_channel_effect(src_tone, fs, 9.5e9, 0.0, 1.0);
        bad_sig = apply_channel_effect(src_tone, fs, 6.0e9, 0.42, 0.88);
        
        % 推理
        cal_sig = conv2(bad_sig, W_final, 'same') * scale_factor;
        
        % TIADC 拼接
        margin = 500;
        valid_idx = margin : (N_test - margin);
        tiadc_bad = interleave_signals(ref_sig(valid_idx), bad_sig(valid_idx));
        tiadc_cal = interleave_signals(ref_sig(valid_idx), cal_sig(valid_idx));
        
        improvements(i) = calculate_sfdr(tiadc_cal, fs, f) - calculate_sfdr(tiadc_bad, fs, f);
        
        if mod(i, 10) == 0, fprintf('.'); end
    end
    fprintf(' 完成！\n');
    
    % =====================================================================
    % 4. 绘图
    % =====================================================================
    figure('Position', [100, 100, 1000, 800], 'Color', 'w');
    
    % 图 1: SFDR 改善
    subplot(2, 1, 1);
    plot(freqs/1e9, improvements, 'b-', 'LineWidth', 2);
    hold on;
    area(freqs/1e9, improvements, 'FaceColor', 'b', 'FaceAlpha', 0.1);
    title('SFDR Improvement (MATLAB Robust Version)');
    xlabel('Frequency (GHz)'); ylabel('dB');
    grid on; ylim([0, 70]);
    
    % 图 2: 典型频谱 (7.2G)
    [~, idx] = min(abs(freqs - 7.2e9));z
    f_chk = freqs(idx);
    src_chk = generate_tone(f_chk, 8192, fs);
    c0 = apply_channel_effect(src_chk, fs, 9.5e9, 0.0, 1.0);
    c1 = apply_channel_effect(src_chk, fs, 6.0e9, 0.42, 0.88);
    c1_cal = conv2(c1/scale_factor, W_final, 'same') * scale_factor;
    
    tiadc_bad = interleave_signals(c0, c1);
    tiadc_cal = interleave_signals(c0, c1_cal);
    
    subplot(2, 1, 2);
    [p_b, fa] = calculate_psd(tiadc_bad, fs);
    [p_c, ~] = calculate_psd(tiadc_cal, fs);
    plot(fa/1e9, p_b, 'r', 'Color', [1 0 0 0.5]); hold on;
    plot(fa/1e9, p_c, 'b', 'LineWidth', 1.5);
    title(['Spectrum @ ', num2str(f_chk/1e9, '%.2f'), ' GHz']);
    ylim([-100, 5]); grid on; legend('Before', 'After');
    
end

% =========================================================================
% 核心修正: 纯实数 Loss 函数 (无 FFT)
% =========================================================================
function [total_loss, grads, l_time, l_high_freq] = model_loss_real(W, X, Target, epoch)
    % 1. 卷积 (Forward)
    W_reshaped = reshape(W, [1, length(W), 1, 1]);
    Y_pred = dlconv(X, W_reshaped, 0, 'Padding', 'same');
    
    % 2. 基础时域 Loss (MSE)
    l_time = mean((Y_pred - Target).^2, 'all');
    
    % 3. 高频 Loss (替代 FFT Loss)
    % 原理: 信号的一阶差分对应高频分量。
    % 计算差分后的 MSE，等效于在频域加权优化高频。
    
    % 一阶差分 (High Pass)
    % dlarray 支持索引操作，结果依然保持实数
    Y_diff1 = Y_pred(1, 2:end, :) - Y_pred(1, 1:end-1, :);
    T_diff1 = Target(1, 2:end, :) - Target(1, 1:end-1, :);
    l_diff1 = mean((Y_diff1 - T_diff1).^2, 'all');
    
    % 二阶差分 (Ultra High Pass, 相当于 f^2 加权)
    Y_diff2 = Y_diff1(1, 2:end, :) - Y_diff1(1, 1:end-1, :);
    T_diff2 = T_diff1(1, 2:end, :) - T_diff1(1, 1:end-1, :);
    l_diff2 = mean((Y_diff2 - T_diff2).^2, 'all');
    
    l_high_freq = l_diff1 + l_diff2;
    
    % 4. 正则化
    w_diff = W(1, 2:end) - W(1, 1:end-1);
    l_reg = mean(w_diff.^2, 'all');
    
    % 5. 课程学习权重
    if epoch < 1000
        % Phase 1: 专注对齐
        total_loss = 100 * l_time + 0.1 * l_high_freq + 1e-4 * l_reg;
    else
        % Phase 2: 猛攻高频 (差分 Loss 权重加大)
        total_loss = 200 * l_time + 5.0 * l_high_freq + 1e-6 * l_reg;
    end
    
    grads = dlgradient(total_loss, W);
end

% --- 辅助函数 (保持不变) ---
function sig = generate_chirp(N, fs)
    t = (0:N-1) / fs;
    sig = chirp(t, 1e6, t(end), 0.48*fs, 'linear') * 0.95;
end
function sig = generate_tone(f, N, fs)
    t = (0:N-1) / fs;
    sig = 0.9 * sin(2*pi*f*t);
end
function sig_out = apply_channel_effect(sig, fs, cut_freq, delay, gain)
    nyquist = fs / 2;
    [b, a] = butter(5, cut_freq / nyquist, 'low');
    sig_bw = filter(b, a, sig);
    sig_out = fft_delay(sig_bw * gain, delay);
end
function y = fft_delay(x, delay_samples)
    N = length(x);
    X = fft(x);
    k = [0 : floor(N/2)-1, -floor(N/2) : -1];
    phase_shift = exp(-1j * 2 * pi * k * delay_samples / N);
    y = real(ifft(X .* phase_shift));
end
function out = interleave_signals(sig_even, sig_odd)
    L = min(length(sig_even), length(sig_odd)); L = L - mod(L, 2);
    out = zeros(1, L);
    out(1:2:end) = sig_even(1:2:L); out(2:2:end) = sig_odd(2:2:L);
end
function [P_db, f_axis] = calculate_psd(sig, fs)
    L = length(sig); w = blackman(L)'; spec = fft(sig .* w);
    P2 = abs(spec/L); P1 = P2(1:floor(L/2)+1); P1(2:end-1) = 2*P1(2:end-1);
    P_db = 20*log10(P1 + 1e-12); P_db = P_db - max(P_db);
    f_axis = fs * (0:(L/2)) / L;
end
function sfdr_val = calculate_sfdr(sig, fs, fin)
    [P_db, f_axis] = calculate_psd(sig, fs);
    spur_freq = fs/2 - fin;
    [~, idx] = min(abs(f_axis - spur_freq));
    window = 10;
    spur_amp = max(P_db(max(1, idx-window):min(length(P_db), idx+window)));
    sfdr_val = -spur_amp; 
end