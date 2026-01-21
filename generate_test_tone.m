function generate_test_tone()
    % 生成一个未见过的单音信号用于验证
    fs = 20e9;
    t = (0:8191)/fs;
    
    % === 关键设置：测试频率 ===
    % 选一个非整数倍的频率，避免被 FFT 掩盖
    % 比如 3.7 GHz (在 Channel 1 带宽衰减区)
    f_test = 3.7e9; 
    
    % 生成源信号
    tx_signal = sin(2*pi*f_test*t);
    
    % === 这里的模型必须和训练时的一模一样！===
    % (直接复制 generate_serdes_data.m 里的 create_transmission_line_model)
    model_ch0 = create_transmission_line_model(fs, 2.0, 0.01); 
    model_ch1 = create_transmission_line_model(fs, 2.5, 0.025);
    
    % 通过物理通道
    rx_ch0 = lsim(model_ch0, tx_signal, t);
    rx_ch1 = lsim(model_ch1, tx_signal, t);
    
    % 添加噪声
    rx_ch0 = add_noise(rx_ch0, -60);
    rx_ch1 = add_noise(rx_ch1, -60);
    
    % 保存
    data_test.input = rx_ch1;  % 待校准
    data_test.target = rx_ch0; % 参考
    data_test.fs = fs;
    data_test.freq = f_test;
    
    save('tiadc_test_tone.mat', 'data_test');
    fprintf('测试数据已生成：3.7 GHz 单音\n');
end

% ... (请把 create_transmission_line_model 等辅助函数也复制在下面)
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