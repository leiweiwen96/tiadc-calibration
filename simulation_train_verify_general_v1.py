"""
更通用、更能落地的 TIADC 校准仿真/验证脚本（v1）

核心思路（相对 v0112_v1 更工程化）：
- 训练目标放在 TIADC 交织输出层面：让 (Ch0 + Cal(Ch1)) 的交织输出尽量逼近 (Ch0 + Ch0)
- 训练数据做 domain randomization：多类型激励（tone / two-tone / 带限噪声），随机幅度/随机失配
- 校准模型加入“落地友好”约束：
  - 分数延时参数限制在 [0, 1)（避免外推不稳定）
  - FIR 采用对称（线性相位）结构，乘法器可减半；并用因果形式实现（靠固定群时延落地）
  - 可选：系数量化仿真（STE），提前评估定点化风险
- 训练策略改进：采用分阶段训练（Staged Training）以获得类似 v0112 的高精度收敛

依赖：torch, numpy, scipy, matplotlib
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ------------------------------------------------------------------------------
# 0. 随机种子
# ------------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ------------------------------------------------------------------------------
# 1. 物理仿真引擎（保持与原脚本一致的抽象，但训练/验证可统一噪声与量化）
# ------------------------------------------------------------------------------
class TIADCSimulator:
    def __init__(self, fs: float = 20e9):
        self.fs = fs

    def fractional_delay(self, sig: np.ndarray, delay: float) -> np.ndarray:
        """
        频域分数延时（周期延拓）。
        注：用于生成“物理通道”数据的近似模型；训练/验证时会裁剪边界避免绕回影响。
        """
        n = len(sig)
        x = np.fft.rfft(sig)
        k = np.arange(len(x))
        phase = np.exp(-1j * 2 * np.pi * k * delay / n)
        return np.fft.irfft(x * phase, n=n)

    def generate_tone(self, freq: float, n: int) -> np.ndarray:
        t = np.arange(n) / self.fs
        return np.sin(2 * np.pi * freq * t)

    def generate_two_tone(self, f1: float, f2: float, n: int) -> np.ndarray:
        t = np.arange(n) / self.fs
        return 0.5 * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))

    def generate_bandlimited_noise(self, n: int, cutoff: float) -> np.ndarray:
        wn = np.random.randn(n)
        b, a = signal.butter(6, cutoff / (self.fs / 2), btype="low")
        return signal.lfilter(b, a, wn)

    def apply_channel_effect(
        self,
        sig: np.ndarray,
        *,
        cutoff_freq: float,
        delay_samples: float,
        gain: float = 1.0,
        offset: float = 0.0,
        jitter_std: float = 100e-15,
        n_bits: int | None = 12,
        noise_std: float = 1e-4,
        hd2: float = 0.0,
        hd3: float = 0.0,
    ) -> np.ndarray:
        nyq = self.fs / 2

        # 非线性
        sig_nl = sig + hd2 * (sig**2) + hd3 * (sig**3)

        # 带宽与相位（IIR）
        b, a = signal.butter(5, cutoff_freq / nyq, btype="low")
        sig_bw = signal.lfilter(b, a, sig_nl)

        # 增益与延时
        sig_g = sig_bw * gain
        sig_d = self.fractional_delay(sig_g, delay_samples)

        # 抖动
        if jitter_std > 0:
            slope = np.gradient(sig_d) * self.fs
            dt = np.random.normal(0, jitter_std, len(sig_d))
            sig_j = sig_d + slope * dt
        else:
            sig_j = sig_d

        # 噪声底
        if noise_std > 0:
            sig_n = sig_j + np.random.normal(0, noise_std, len(sig_j))
        else:
            sig_n = sig_j

        # DC Offset
        sig_n = sig_n + offset

        # 量化
        if n_bits is None:
            return sig_n

        v_range = 2.0
        levels = 2**n_bits
        step = v_range / levels
        sig_clip = np.clip(sig_n, -1.0, 1.0)
        return np.round(sig_clip / step) * step


# ------------------------------------------------------------------------------
# 2. 更落地的校准器（约束 delay + 对称 FIR + 可选量化）
# ------------------------------------------------------------------------------
def delay_line_np(x: np.ndarray, d: int) -> np.ndarray:
    """纯整数延时（用首样本 replicate 填充），保持长度不变。"""
    if d <= 0:
        return x
    return np.concatenate([np.full(d, x[0], dtype=x.dtype), x[:-d]])


def delay_line_torch(x: torch.Tensor, d: int) -> torch.Tensor:
    """纯整数延时（replicate pad），保持长度不变。x: [B,1,N]"""
    if d <= 0:
        return x
    n = x.shape[-1]
    return F.pad(x, (d, 0), mode="replicate")[..., :n]


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for round()."""
    return (x - x.detach()) + torch.round(x.detach())


def quantize_symmetric_ste(x: torch.Tensor, bits: int, clip: float = 1.0) -> torch.Tensor:
    """
    对称定点量化仿真（STE）：
    - 量化到 [-clip, clip]
    - 量化级数约为 2^(bits-1)（含符号）
    """
    if bits is None:
        return x
    qmax = (2 ** (bits - 1)) - 1
    x_c = torch.clamp(x / clip, -1.0, 1.0)
    x_q = _ste_round(x_c * qmax) / qmax
    return x_q * clip


class PolynomialCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.hd2 = nn.Parameter(torch.tensor(0.0))
        self.hd3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.hd2 * (x**2) + self.hd3 * (x**3)


class ConstrainedFarrowDelay(nn.Module):
    """
    4-tap 三次 Farrow（Lagrange 形式），delay 限制在 [0, 1)。
    这样更符合硬件分数延时常用的工作区间。
    """

    def __init__(self):
        super().__init__()
        self.delay_raw = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("k0", torch.tensor([[[0, 1, 0, 0]]], dtype=torch.float32))
        self.register_buffer("k1", torch.tensor([[[-1 / 3, -1 / 2, 1, -1 / 6]]], dtype=torch.float32))
        self.register_buffer("k2", torch.tensor([[[1 / 2, -1, 1 / 2, 0]]], dtype=torch.float32))
        self.register_buffer("k3", torch.tensor([[[-1 / 6, 1 / 2, -1 / 2, 1 / 6]]], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # d in [0, 1)
        d = torch.sigmoid(self.delay_raw) * (1.0 - 1e-6)

        # replicate pad: 4-tap 需要历史/未来；这里用于离线训练，落地时通常用寄存器/边界处理
        px = F.pad(x, (1, 2), mode="replicate")
        c0 = F.conv1d(px, self.k0)
        c1 = F.conv1d(px, self.k1)
        c2 = F.conv1d(px, self.k2)
        c3 = F.conv1d(px, self.k3)
        return c0 + d * (c1 + d * (c2 + d * c3))


class GeneralFIR(nn.Module):
    """
    通用 FIR 滤波器（非对称）。
    虽然比对称 FIR 多消耗一倍乘法器，但对于校准非线性相位失配（如模拟带宽差异）是必须的。
    """

    def __init__(self, taps: int = 127):
        super().__init__()
        self.taps = taps
        # padding 策略：为了让 center tap 位于中间，通常 padding = taps // 2
        # 我们这里手动 padding 以便更灵活控制因果性，但在 forward 里实现
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=0, bias=False)
        
        # 初始化为 Delta 函数 (直通)
        with torch.no_grad():
            self.conv.weight.zero_()
            center = taps // 2
            self.conv.weight[0, 0, center] = 1.0

    def forward(self, x: torch.Tensor, *, quant_bits: int | None = None) -> torch.Tensor:
        w = self.conv.weight
        if quant_bits is not None:
            # 量化权重
            w_q = quantize_symmetric_ste(w, quant_bits, clip=1.0)
            # 使用量化后的权重进行卷积
            # F.conv1d(input, weight, ...)
            # 需要注意 padding：我们希望输出长度与输入一致（或通过裁剪对齐）
            # 这里采用 replicate padding 保持长度
            pad_left = self.taps // 2
            pad_right = self.taps - 1 - pad_left
            px = F.pad(x, (pad_left, pad_right), mode="replicate")
            return F.conv1d(px, w_q)
        else:
            # 浮点卷积
            pad_left = self.taps // 2
            pad_right = self.taps - 1 - pad_left
            px = F.pad(x, (pad_left, pad_right), mode="replicate")
            return self.conv(px)
    
    def full_weights(self):
        return self.conv.weight.reshape(-1)


class DeployableCalibrator(nn.Module):
    def __init__(self, taps: int = 127, quant_bits: int | None = None):
        super().__init__()
        self.quant_bits = quant_bits
        self.poly = PolynomialCorrection()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))
        # 召回 Farrow Delay 用于粗调
        self.delay = ConstrainedFarrowDelay() 
        self.fir = GeneralFIR(taps=taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,N]
        y = self.poly(x)
        y = y * self.gain + self.offset
        y = self.delay(y) # Farrow 先粗调时间
        y = self.fir(y, quant_bits=self.quant_bits) # FIR 再精调频响
        return y


# ------------------------------------------------------------------------------
# 3. TIADC 交织与指标（以交织输出为主）
# ------------------------------------------------------------------------------
def interleave_tiadc(ch0: np.ndarray, ch1: np.ndarray) -> np.ndarray:
    """
    简化 TIADC：偶采样来自 ch0，奇采样来自 ch1。
    """
    L = min(len(ch0), len(ch1))
    L -= L % 2
    out = np.zeros(L, dtype=np.float64)
    out[0::2] = ch0[:L:2]
    out[1::2] = ch1[1:L:2]
    return out


def spectrum_metrics(sig: np.ndarray, fs: float, fund_freq: float, guard_bins: int = 6) -> Dict[str, float]:
    """
    基于单边谱的简化指标：
    - THD：2~5次谐波功率 / 基波功率（dBc）
    - SFDR：最大杂散功率 / 基波功率（dBc，返回为正数）
    注：这里是“工程口径”的快速估计，若要严格 ADC 指标，需要相干采样/更严谨的bin合并策略。
    """
    x = sig - np.mean(sig)
    win = np.blackman(len(x))
    X = np.fft.rfft(x * win)
    mag2 = (np.abs(X) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    idx_f = int(np.argmin(np.abs(freqs - fund_freq)))
    s0 = max(0, idx_f - guard_bins)
    e0 = min(len(mag2), idx_f + guard_bins)
    p_fund = float(np.sum(mag2[s0:e0]) + 1e-20)

    # THD 2~5
    p_harm = 0.0
    for h in range(2, 6):
        hf_raw = fund_freq * h
        # 计算混叠频率（Fold back to [0, fs/2]）
        hf = hf_raw % fs
        if hf > fs / 2:
            hf = fs - hf

        # 如果谐波混叠到基波附近（例如 fs/2 附近的信号），可能需要排除
        # 这里简单处理：只要不在 DC 或 Nyquist 极边缘
        if hf < 1e6 or hf > fs / 2 - 1e6:
            continue

        idx_h = int(np.argmin(np.abs(freqs - hf)))
        s = max(0, idx_h - guard_bins)
        e = min(len(mag2), idx_h + guard_bins)
        p_harm += float(np.sum(mag2[s:e]))
    thd = 10 * np.log10((p_harm + 1e-20) / p_fund)

    # SFDR：最大 spur（排除 DC/基波）
    mask = np.ones_like(mag2, dtype=bool)
    mask[:3] = False
    mask[s0:e0] = False
    p_spur = float(np.max(mag2[mask])) if np.any(mask) else 1e-20
    sfdr = 10 * np.log10(p_fund / (p_spur + 1e-20))
    return {"thd_db": thd, "sfdr_db": sfdr}


# ------------------------------------------------------------------------------
# 4. 训练数据配置（domain randomization）
# ------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    fs: float = 20e9
    n: int = 16384
    batch_size: int = 8
    steps: int = 800
    lr: float = 2e-3
    taps: int = 127
    crop: int = 1024

    # 噪声/量化（训练与验证保持一致更真实）
    jitter_std: float = 100e-15
    noise_std: float = 1e-4
    n_bits: int | None = 12

    # 随机激励
    fmin: float = 0.2e9
    fmax: float = 7.0e9  # 降到 7GHz (0.35fs)，避开 Farrow 高频衰减区
    noise_cutoff: float = 7.0e9

    # 参数随机范围（ch1 相对 ch0 的失配）
    gain_range: Tuple[float, float] = (0.95, 1.05)
    offset_range: Tuple[float, float] = (-0.05, 0.05)
    delay_range: Tuple[float, float] = (0.0, 0.8)  # samples
    cutoff_range: Tuple[float, float] = (5.6e9, 6.2e9)
    hd2_range: Tuple[float, float] = (0.0, 3e-3)
    hd3_range: Tuple[float, float] = (0.0, 3e-3)


def _rand_uniform(a: float, b: float) -> float:
    return a + (b - a) * random.random()


def build_random_signal(sim: TIADCSimulator, cfg: TrainConfig) -> Tuple[np.ndarray, float]:
    """
    返回 (signal, fund_freq)
    - 对 noise/two-tone，fund_freq 取主 tone 频率（用于评估时的指标定义）
    """
    n = cfg.n
    kind = random.choice(["tone", "two_tone", "noise"])
    amp = _rand_uniform(0.2, 0.9)

    if kind == "tone":
        f = _rand_uniform(cfg.fmin, cfg.fmax)
        x = sim.generate_tone(f, n)
        return amp * x, f

    if kind == "two_tone":
        f1 = _rand_uniform(cfg.fmin, cfg.fmax * 0.8)
        f2 = min(cfg.fmax, f1 + _rand_uniform(0.2e9, 1.2e9))
        x = sim.generate_two_tone(f1, f2, n)
        return amp * x, f1

    # noise
    x = sim.generate_bandlimited_noise(n, cutoff=cfg.noise_cutoff)
    x = x / (np.max(np.abs(x)) + 1e-12)
    # 为了避免完全“无基波”的指标尴尬，叠加一个弱 tone
    f = _rand_uniform(cfg.fmin, cfg.fmax)
    x = 0.9 * x + 0.1 * sim.generate_tone(f, n)
    return amp * x, f


def build_channel_params(cfg: TrainConfig) -> Tuple[Dict, Dict]:
    # ch0 设为“参考”，也允许它有小非线性/带宽限制（更接近现实）
    p0 = {
        "cutoff_freq": 6.0e9,
        "delay_samples": 0.0,
        "gain": 1.0,
        "offset": 0.0,
        "hd2": 1e-3,
        "hd3": 1e-3,
    }

    p1 = {
        "cutoff_freq": _rand_uniform(*cfg.cutoff_range),
        "delay_samples": _rand_uniform(*cfg.delay_range),
        "gain": _rand_uniform(*cfg.gain_range),
        "offset": _rand_uniform(*cfg.offset_range),
        "hd2": _rand_uniform(*cfg.hd2_range),
        "hd3": _rand_uniform(*cfg.hd3_range),
    }
    return p0, p1


# ------------------------------------------------------------------------------
# 5. 训练：直接对齐 TIADC 交织输出
# ------------------------------------------------------------------------------
def train_general(sim: TIADCSimulator, cfg: TrainConfig, device: str = "cpu") -> DeployableCalibrator:
    model = DeployableCalibrator(taps=cfg.taps, quant_bits=None).to(device)

    # 因果对称 FIR 的固定群时延（落地时通常在另一通道加同样的纯延时）
    gd = (cfg.taps - 1) // 2
    # 为了能校准“滞后”的通道（delay > 0），参考通道需要多延时一点（padding），
    # 这样校准通道只需产生正延时（delay < 1）即可对齐。
    padding = 1
    ref_delay = gd + padding

    # ==============================================================================
    # 策略调整：分阶段训练 (Staged Training)
    # 借鉴 v0112_v1 的成功经验，分步对齐参数，避免相互干扰
    # ==============================================================================

    # 定义优化器组
    def get_opt(stage: str):
        if stage == "coarse": # Stage 1: 粗调 (Delay, Gain, Offset)
            return optim.Adam([
                {'params': model.delay.parameters(), 'lr': 1e-2}, # Farrow 主力
                {'params': model.gain, 'lr': 1e-2}, 
                {'params': model.offset, 'lr': 1e-2},
                {'params': model.fir.parameters(), 'lr': 0.0}, # 冻结 FIR
                {'params': model.poly.parameters(), 'lr': 0.0}
            ])
        elif stage == "fine": # Stage 2: 精调 (FIR)
            return optim.Adam([
                {'params': model.delay.parameters(), 'lr': 1e-4}, # 微调
                {'params': model.gain, 'lr': 1e-4},
                {'params': model.offset, 'lr': 1e-4},
                {'params': model.fir.parameters(), 'lr': 1e-3}, # FIR 接管
                {'params': model.poly.parameters(), 'lr': 0.0}
            ])
        elif stage == "nonlinear": # Stage 3: 非线性 (Poly)
            return optim.Adam([
                {'params': model.delay.parameters(), 'lr': 0.0},
                {'params': model.gain, 'lr': 0.0},
                {'params': model.offset, 'lr': 0.0},
                {'params': model.fir.parameters(), 'lr': 1e-5},
                {'params': model.poly.parameters(), 'lr': 1e-3}
            ])
        return None

    stages = [
        ("coarse", 300),  # 300 steps 快速对齐时延/增益/偏置
        ("fine", 600),    # 600 steps 学习频响
        ("nonlinear", 300)  # 300 steps 拟合非线性
    ]

    total_step = 0
    # 训练时降低随机噪声，帮助模型更容易捕捉静态特征 (模拟 v0112 的平均去噪效果)
    train_jitter = cfg.jitter_std * 0.1
    train_noise = cfg.noise_std * 0.1

    for stage_name, steps in stages:
        print(f"\\n=== Training Stage: {stage_name} ({steps} steps) ===")
        opt = get_opt(stage_name)

        for i in range(steps):
            total_step += 1

            # 组 batch
            ch0_list, ch1_list, ref_list = [], [], []
            for _ in range(cfg.batch_size):
                src, _ = build_random_signal(sim, cfg)
                p0, p1 = build_channel_params(cfg)

                y0 = sim.apply_channel_effect(
                    src,
                    jitter_std=train_jitter,
                    noise_std=train_noise,
                    n_bits=cfg.n_bits,
                    **p0,
                )
                y1 = sim.apply_channel_effect(
                    src,
                    jitter_std=train_jitter,
                    noise_std=train_noise,
                    n_bits=cfg.n_bits,
                    **p1,
                )
                # 对齐：参考支路使用 ref_delay (gd + padding)
                y0d = delay_line_np(y0, ref_delay)
                ref = interleave_tiadc(y0d, y0d)
                ch0_list.append(y0d)
                ch1_list.append(y1)
                ref_list.append(ref)

            # 转 torch（统一归一化）
            ch0 = np.stack(ch0_list, axis=0)
            ch1 = np.stack(ch1_list, axis=0)
            ref = np.stack(ref_list, axis=0)
            scale = np.max(np.abs(ref), axis=1, keepdims=True) + 1e-12
            ch0_t = torch.tensor(ch0 / scale, dtype=torch.float32, device=device).unsqueeze(1)
            ch1_t = torch.tensor(ch1 / scale, dtype=torch.float32, device=device).unsqueeze(1)
            ref_t = torch.tensor(ref / scale, dtype=torch.float32, device=device).unsqueeze(1)

            # 校准 ch1，然后交织
            ch1_cal = model(ch1_t)
            ch0_aligned = ch0_t  # 已经在 numpy 侧加了 gd 纯延时
            # 交织（torch 版）
            L = min(ch0_aligned.shape[-1], ch1_cal.shape[-1])
            L = L - (L % 2)
            out = torch.zeros((cfg.batch_size, 1, L), device=device)
            out[..., 0::2] = ch0_aligned[..., :L:2]
            out[..., 1::2] = ch1_cal[..., 1:L:2]

            # 裁剪稳态
            c = cfg.crop
            out_c = out[..., c:-c]
            ref_c = ref_t[..., c:-c]

            # 损失：时域 + 频域
            loss_time = torch.mean((out_c - ref_c) ** 2)
            fft_o = torch.fft.rfft(out_c, dim=-1, norm="ortho")
            fft_r = torch.fft.rfft(ref_c, dim=-1, norm="ortho")
            loss_freq = torch.mean(torch.abs(fft_o - fft_r) ** 2)

            w = model.fir.full_weights()
            loss_smooth = torch.mean((w[1:] - w[:-1]) ** 2)
            loss_coef = 1e-3 * (model.poly.hd2**2 + model.poly.hd3**2)

            loss = 100.0 * loss_time + 100.0 * loss_freq + 1e-3 * loss_smooth + loss_coef

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0 or i == steps - 1:
                with torch.no_grad():
                    d = torch.sigmoid(model.delay.delay_raw).item()
                    print(
                        f"[{stage_name} s{i:03d}] loss={loss.item():.4e} "
                        f"gain={model.gain.item():.4f} off={model.offset.item():.4f} delay={d:.4f} " 
                        f"hd2={model.poly.hd2.item():.2e}"
                    )

    return model


# ------------------------------------------------------------------------------
# 6. 验证：对 TIADC 交织输出扫频并画 SFDR/THD（Pre vs Post）
# ------------------------------------------------------------------------------
def evaluate_sweep(
    sim: TIADCSimulator,
    model: DeployableCalibrator,
    *,
    device: str,
    cfg: TrainConfig,
    p0: Dict,
    p1: Dict,
    freqs: np.ndarray,
    quant_bits_eval: int | None = None,
) -> Dict[str, list]:
    out = {"sfdr_pre": [], "sfdr_post": [], "thd_pre": [], "thd_post": []}
    model.eval()
    gd = (cfg.taps - 1) // 2
    # 验证时同样保持 ref_delay = gd + 1
    padding = 1
    ref_delay = gd + padding

    for f in freqs:
        src = 0.9 * sim.generate_tone(float(f), cfg.n)
        y0 = sim.apply_channel_effect(
            src,
            jitter_std=cfg.jitter_std,
            noise_std=cfg.noise_std,
            n_bits=cfg.n_bits,
            **p0,
        )
        y1 = sim.apply_channel_effect(
            src,
            jitter_std=cfg.jitter_std,
            noise_std=cfg.noise_std,
            n_bits=cfg.n_bits,
            **p1,
        )
        # 为了与“部署形态”一致：
        # Pre-Calib: 两路都加同样的 ref_delay (模拟 Bypass 下的对齐状态，含 delta 失配)
        y0d = delay_line_np(y0, ref_delay)
        y1d = delay_line_np(y1, ref_delay)
        tiadc_pre = interleave_tiadc(y0d, y1d)

        with torch.no_grad():
            scale = np.max(np.abs(tiadc_pre)) + 1e-12
            x1 = torch.tensor(y1 / scale, dtype=torch.float32, device=device).view(1, 1, -1)
            # 临时切换量化位宽（仅用于验证定点化敏感性）
            old_q = model.quant_bits
            model.quant_bits = quant_bits_eval
            y1c = model(x1).cpu().numpy().reshape(-1) * scale
            model.quant_bits = old_q

        # y1c 已含 FIR 群时延 + Farrow 延时，目标是对齐 y0d
        tiadc_post = interleave_tiadc(y0d, y1c)

        # 裁剪稳态
        c = cfg.crop
        pre_c = tiadc_pre[c:-c]
        post_c = tiadc_post[c:-c]

        m_pre = spectrum_metrics(pre_c, sim.fs, float(f))
        m_post = spectrum_metrics(post_c, sim.fs, float(f))
        out["sfdr_pre"].append(m_pre["sfdr_db"])
        out["sfdr_post"].append(m_post["sfdr_db"])
        out["thd_pre"].append(m_pre["thd_db"])
        out["thd_post"].append(m_post["thd_db"])

    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig()
    sim = TIADCSimulator(fs=cfg.fs)

    print("=== 训练：更通用/更落地的交织输出对齐方案 ===")
    model = train_general(sim, cfg, device=device)

    # 选择一个“固定失配”进行扫频展示（也可以改为随机失配重复多次统计）
    p0 = {"cutoff_freq": 6.0e9, "delay_samples": 0.0, "gain": 1.0, "offset": 0.0, "hd2": 1e-3, "hd3": 1e-3}
    p1 = {"cutoff_freq": 5.9e9, "delay_samples": 0.42, "gain": 0.98, "offset": 0.02, "hd2": 2e-3, "hd3": 2e-3}

    freqs = np.arange(0.5e9, 7.1e9, 0.5e9)  # 扫频范围也限制在 7GHz
    print("\\n=== 验证：TIADC 交织输出扫频（Pre vs Post）===")
    m = evaluate_sweep(sim, model, device=device, cfg=cfg, p0=p0, p1=p1, freqs=freqs, quant_bits_eval=None)

    # 可选：验证“系数量化”对性能的影响（例如 12bit 系数）
    m_q12 = evaluate_sweep(sim, model, device=device, cfg=cfg, p0=p0, p1=p1, freqs=freqs, quant_bits_eval=12)

    # 绘图
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(freqs / 1e9, m["sfdr_pre"], "r--o", alpha=0.5, label="Pre-Calib (TIADC)")
    plt.plot(freqs / 1e9, m["sfdr_post"], "b-o", linewidth=2, label="Post-Calib (TIADC)")
    plt.plot(freqs / 1e9, m_q12["sfdr_post"], "c-o", linewidth=2, alpha=0.8, label="Post-Calib (Coeff Q=12b)")
    plt.title("SFDR vs Input Frequency (TIADC Output)")
    plt.ylabel("dBc")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(freqs / 1e9, m["thd_pre"], "r--o", alpha=0.5, label="Pre-Calib (TIADC)")
    plt.plot(freqs / 1e9, m["thd_post"], "b-o", linewidth=2, label="Post-Calib (TIADC)")
    plt.plot(freqs / 1e9, m_q12["thd_post"], "c-o", linewidth=2, alpha=0.8, label="Post-Calib (Coeff Q=12b)")
    plt.title("THD vs Input Frequency (TIADC Output)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("dBc")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()