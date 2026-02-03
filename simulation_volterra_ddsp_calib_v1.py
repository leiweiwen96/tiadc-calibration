"""
Volterra-DDSP TIADC 校准方案（v1）
================================

你给的截图公式核心是：
- **交织(Interleaving) => 频谱搬移项**：ω - 2πk/M（产生镜像 spur）
- **非线性(Volterra) => 多重卷积**：p 阶核 H_{m,p}(·) 产生谐波/互调(IMD)

本脚本给出一个“按公式搭出来”的新方案（不要求和现有方案一致）：
1) **按相位(m=0..M-1)建模**：每相一个“线性逆” + 一个“Volterra 近似（记忆多项式）”
2) **DDSP 端到端可微训练**：用频域自监督 loss 直接压制
   - 镜像 spur（来自 ω-2πk/M）
   - 两音 IMD3（来自 p 阶卷积项）
3) **可选有监督参考**：仿真/或实测有参考 ADC 时，用时域 MSE 进一步约束

运行（默认 CPU）：
    python -u simulation_volterra_ddsp_calib_v1.py

常用环境变量：
- M=2/4/...                    交织路数
- SUPERVISED_REF=0/1           是否使用参考输出监督
- TRAIN_STEPS / TRAIN_N / TRAIN_LR
- LIN_TAPS / VOL_K / VOL_P     线性 FIR tap、记忆深度K、非线性阶数P
- TRAIN_FMIN_HZ / TRAIN_FMAX_HZ训练频段（建议覆盖高频，否则会出现“4G后没效果”）
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


ADC_JITTER_STD = float(os.getenv("ADC_JITTER_STD", "100e-15"))
_nb = os.getenv("ADC_NBITS", "12")
ADC_NBITS = None if _nb.strip().lower() in ("none", "") else int(_nb)


_PLOT_DIR: Optional[Path] = None


def _plot_dir() -> Path:
    global _PLOT_DIR
    if _PLOT_DIR is not None:
        return _PLOT_DIR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(__file__).resolve().parent / "plots" / ts
    d.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR = d
    return d


def savefig(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out = _plot_dir() / f"{ts}_{name}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {out}")
    return out


def _frac_delay_np(sig: np.ndarray, delay: float) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    d = float(delay) % 1.0
    if abs(d) < 1e-12:
        return sig
    k0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    k1 = np.array([-1 / 3, -1 / 2, 1.0, -1 / 6], dtype=np.float64)
    k2 = np.array([1 / 2, -1.0, 1 / 2, 0.0], dtype=np.float64)
    k3 = np.array([-1 / 6, 1 / 2, -1 / 2, 1 / 6], dtype=np.float64)
    xpad = np.pad(sig, (1, 2), mode="edge")
    c0 = np.convolve(xpad, k0[::-1], mode="valid")
    c1 = np.convolve(xpad, k1[::-1], mode="valid")
    c2 = np.convolve(xpad, k2[::-1], mode="valid")
    c3 = np.convolve(xpad, k3[::-1], mode="valid")
    return (c0 + d * (c1 + d * (c2 + d * c3))).astype(np.float64, copy=False)


@dataclass
class ChannelParam:
    cutoff_hz: float
    delay_samp: float
    gain: float
    hd2: float = 0.0
    hd3: float = 0.0


class TIADCSimM:
    """
    M 路 TIADC 简化仿真：
    - 用全速 fs 网格生成 x[n]
    - 每路对 x 做：静态非线性 + 低通 + 分数延时 + 抖动 + 量化 + 小噪声
    - 最后只取本路相位 m::M 的样本，交织回全速 y_full
    """

    def __init__(self, *, fs: float, m: int, params: list[ChannelParam]):
        self.fs = float(fs)
        self.M = int(m)
        if len(params) != self.M:
            raise ValueError("len(params) must equal M")
        self.params = list(params)

    def tone(self, fin: float, n: int, amp: float = 0.9) -> np.ndarray:
        t = np.arange(int(n), dtype=np.float64) / self.fs
        return (np.sin(2 * np.pi * float(fin) * t) * float(amp)).astype(np.float64)

    def two_tone(self, f1: float, f2: float, n: int, amp: float = 0.6) -> np.ndarray:
        t = np.arange(int(n), dtype=np.float64) / self.fs
        x = (np.sin(2 * np.pi * float(f1) * t) + np.sin(2 * np.pi * float(f2) * t)) * float(amp)
        return np.clip(x, -0.999, 0.999).astype(np.float64)

    def _apply_ch(self, x: np.ndarray, p: ChannelParam) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        nyq = self.fs / 2.0
        y = x + float(p.hd2) * (x**2) + float(p.hd3) * (x**3)
        b, a = signal.butter(5, float(p.cutoff_hz) / nyq, btype="low")
        y = signal.lfilter(b, a, y)
        y = _frac_delay_np(y * float(p.gain), float(p.delay_samp))
        if ADC_JITTER_STD and float(ADC_JITTER_STD) > 0:
            slope = np.gradient(y) * self.fs
            dt = np.random.normal(0.0, float(ADC_JITTER_STD), len(y))
            y = y + slope * dt
        if ADC_NBITS is not None:
            levels = 2 ** int(ADC_NBITS)
            step = 2.0 / levels
            y = np.clip(y, -1.0, 1.0)
            y = np.round(y / step) * step
        y = y + np.random.normal(0.0, 1e-4, len(y))
        return y.astype(np.float64, copy=False)

    def capture(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x, dtype=np.float64)
        for m, p in enumerate(self.params):
            y_m_full = self._apply_ch(x, p)
            y[m :: self.M] = y_m_full[m :: self.M]
        return y


class FarrowDelay(nn.Module):
    def __init__(self):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor(0.0))
        k0 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        k1 = torch.tensor([-1 / 3, -1 / 2, 1.0, -1 / 6], dtype=torch.float32)
        k2 = torch.tensor([1 / 2, -1.0, 1 / 2, 0.0], dtype=torch.float32)
        k3 = torch.tensor([-1 / 6, 1 / 2, -1 / 2, 1 / 6], dtype=torch.float32)
        self.register_buffer("k0", k0.view(1, 1, -1))
        self.register_buffer("k1", k1.view(1, 1, -1))
        self.register_buffer("k2", k2.view(1, 1, -1))
        self.register_buffer("k3", k3.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.remainder(self.delay, 1.0)
        if torch.all(torch.abs(d) < 1e-9):
            return x
        xp = F.pad(x, (1, 2), mode="replicate")
        c0 = F.conv1d(xp, self.k0)
        c1 = F.conv1d(xp, self.k1)
        c2 = F.conv1d(xp, self.k2)
        c3 = F.conv1d(xp, self.k3)
        return c0 + d * (c1 + d * (c2 + d * c3))


class PerPhaseLinear(nn.Module):
    """每相线性校准（H_{m,1} 的逆近似）：gain + frac-delay + FIR。"""

    def __init__(self, taps: int):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.delay = FarrowDelay()
        self.fir = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        with torch.no_grad():
            self.fir.weight.zero_()
            self.fir.weight[0, 0, taps // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        x = self.delay(x)
        return self.fir(x)


class MemoryPolynomial(nn.Module):
    """记忆多项式：Volterra 的可实现近似（对齐 H_{m,p} 里的 p 阶项）。"""

    def __init__(self, k: int, p: int):
        super().__init__()
        self.K = int(k)
        self.P = int(p)
        if self.K < 1 or self.P < 2:
            raise ValueError("K>=1, P>=2 required")
        self.alpha = float(os.getenv("VOL_ALPHA", "0.8"))
        self.res_clip = float(os.getenv("VOL_RES_CLIP", "0.2"))
        max_p2 = float(os.getenv("VOL_MAX_P2", "5e-2"))
        max_p3 = float(os.getenv("VOL_MAX_P3", "3e-2"))
        max_hi = float(os.getenv("VOL_MAX_PHI", "1e-2"))
        scales = []
        for pp in range(2, self.P + 1):
            scales.append(max_p2 if pp == 2 else max_p3 if pp == 3 else max_hi)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.raw = nn.Parameter(torch.zeros(self.K, self.P - 1))

    @staticmethod
    def _delay(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return x
        xp = F.pad(x, (k, 0), mode="constant", value=0.0)
        return xp[..., :-k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        scales = self.scales.to(x.device, x.dtype)
        raw = self.raw.to(x.device, x.dtype)
        res = torch.zeros_like(x)
        for kk in range(self.K):
            xd = self._delay(x, kk)
            a = scales * torch.tanh(raw[kk, :])
            for i, pp in enumerate(range(2, self.P + 1)):
                res = res + a[i] * (xd**pp)
        res = torch.clamp(res, -float(self.res_clip), float(self.res_clip))
        return x + float(self.alpha) * res


class VolterraDDSPCalibrator(nn.Module):
    """总校准器：按相拆子序列 -> 线性 -> Volterra近似 -> 写回全速。"""

    def __init__(self, *, m: int, lin_taps: int, vol_k: int, vol_p: int):
        super().__init__()
        self.M = int(m)
        self.lin = nn.ModuleList([PerPhaseLinear(lin_taps) for _ in range(self.M)])
        self.vol = nn.ModuleList([MemoryPolynomial(vol_k, vol_p) for _ in range(self.M)])

    def forward(self, y_full: torch.Tensor) -> torch.Tensor:
        B, C, N = y_full.shape
        out = torch.zeros_like(y_full)
        for m in range(self.M):
            sub = y_full[..., m:: self.M]
            sub = self.lin[m](sub)
            sub = self.vol[m](sub)
            out[..., m:: self.M] = sub
        return out


def _fold_bin(k: int, n: int) -> int:
    kk = int(k) % int(n)
    return kk if kk <= n // 2 else n - kk


def image_bins_for_fund(*, k0: int, n: int, m: int) -> list[int]:
    """
    与 ω-2πk/M 对齐的离散 bin 近似：
        k_img(k) = fold( k0 + k*(N/M) ), k=1..M-1
    """
    M = int(m)
    step = int(round(int(n) / M))
    bins = [_fold_bin(int(k0) + kk * step, int(n)) for kk in range(1, M)]
    bins = [b for b in bins if 1 <= b <= n // 2 - 1]
    return sorted(set(bins))


def imd3_bins_two_tone(*, k1: int, k2: int, n: int) -> list[int]:
    b1 = _fold_bin(2 * int(k1) - int(k2), int(n))
    b2 = _fold_bin(2 * int(k2) - int(k1), int(n))
    out = [b for b in (b1, b2) if 1 <= b <= n // 2 - 1]
    return sorted(set(out))


def spec_power(sig: torch.Tensor) -> torch.Tensor:
    n = int(sig.shape[-1])
    win = torch.hann_window(n, device=sig.device, dtype=sig.dtype).view(1, 1, -1)
    S = torch.fft.rfft((sig - torch.mean(sig, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    return S.real**2 + S.imag**2


def band_energy(P: torch.Tensor, k: int, guard: int) -> torch.Tensor:
    s = max(0, int(k) - int(guard))
    e = min(P.shape[-1], int(k) + int(guard) + 1)
    return torch.sum(P[..., s:e], dim=-1)


def loss_image_imd(
    y: torch.Tensor,
    *,
    k_funds: list[int],
    k_imgs: list[int],
    k_imd: list[int],
    guard: int,
    w_img: float,
    w_imd: float,
) -> torch.Tensor:
    P = spec_power(y)
    eps = 1e-12
    p_fund = 0.0
    for k in k_funds:
        p_fund = p_fund + band_energy(P, k, guard)
    p_fund = p_fund + eps
    p_img = 0.0
    for k in k_imgs:
        p_img = p_img + band_energy(P, k, guard)
    p_imd = 0.0
    for k in k_imd:
        p_imd = p_imd + band_energy(P, k, guard)
    return torch.mean(float(w_img) * (p_img / p_fund) + float(w_imd) * (p_imd / p_fund))


def loss_fund_hold(yhat: torch.Tensor, yin: torch.Tensor, *, k_funds: list[int], guard: int) -> torch.Tensor:
    """
    基波保持（很关键）：只压 spur/IMD 容易把基波也“顺手压掉/拉爆”，导致 SINAD/SFDR 变差。
    这里用 log 幅度对齐，让校准更稳、更工程。
    """
    Ph = spec_power(yhat)
    Pi = spec_power(yin)
    eps = 1e-12
    lh = 0.0
    for k in k_funds:
        ph = band_energy(Ph, k, guard) + eps
        pi = band_energy(Pi, k, guard) + eps
        lh = lh + torch.mean((torch.log10(ph) - torch.log10(pi)) ** 2)
    return lh / float(max(len(k_funds), 1))


def loss_spec_mse(yhat: torch.Tensor, yref: torch.Tensor) -> torch.Tensor:
    """频域参考对齐：让整体频谱形状一致（比只盯若干 bin 更稳）。"""
    Ph = spec_power(yhat)
    Pr = spec_power(yref)
    return torch.mean((torch.log10(Ph + 1e-12) - torch.log10(Pr + 1e-12)) ** 2)


def _sample_k_biased(k_min: int, k_max: int, *, high_bias: float) -> int:
    """
    high_bias<1 会偏向高频（u**high_bias 更靠近 1）。
    用于解决“4GHz 后没效果”常见问题：训练频点太平均导致高频学不到。
    """
    u = float(np.random.rand())
    u = u ** float(max(1e-3, high_bias))
    return int(k_min + round((k_max - k_min) * u))


def mse_time(yhat: torch.Tensor, yref: torch.Tensor, crop: int = 256) -> torch.Tensor:
    if crop <= 0:
        return torch.mean((yhat - yref) ** 2)
    return torch.mean((yhat[..., crop:-crop] - yref[..., crop:-crop]) ** 2)


def calc_metrics(sig: np.ndarray, fs: float, fin: float) -> tuple[float, float, float, float]:
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    win = np.blackman(n)
    cg = float(np.mean(win))
    S = np.fft.rfft((sig - np.mean(sig)) * win)
    mag = np.abs(S) / (n / 2 * cg + 1e-20)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - fin)))
    span = 5
    s0 = max(0, idx - span)
    e0 = min(len(mag), idx + span + 1)
    idxp = s0 + int(np.argmax(mag[s0:e0]))
    p_fund = float(mag[idxp] ** 2) + 1e-30
    mask = np.ones_like(mag, dtype=bool)
    mask[:5] = False
    mask[max(0, idxp - span) : min(len(mask), idxp + span + 1)] = False
    p_nd = float(np.sum(mag[mask] ** 2)) + 1e-30
    sinad = 10.0 * np.log10(p_fund / p_nd)
    enob = (sinad - 1.76) / 6.02
    p_spur = float(np.max(mag[mask] ** 2)) if np.any(mask) else 1e-30
    sfdr = 10.0 * np.log10(p_fund / (p_spur + 1e-30))
    p_h = 0.0
    for h in range(2, 6):
        hf = (fin * h) % fs
        if hf > fs / 2:
            hf = fs - hf
        if hf < 1e6 or hf > fs / 2 - 1e6:
            continue
        k = int(np.argmin(np.abs(freqs - hf)))
        if abs(k - idxp) <= span:
            continue
        ss = max(0, k - span)
        ee = min(len(mag), k + span + 1)
        kk = ss + int(np.argmax(mag[ss:ee]))
        if abs(kk - idxp) <= span:
            continue
        p_h += float(mag[kk] ** 2)
    thd = 10.0 * np.log10((p_h + 1e-30) / p_fund)
    return float(sinad), float(enob), float(thd), float(sfdr)


def train(
    *,
    sim: TIADCSimM,
    sim_ref: Optional[TIADCSimM],
    calibrator: VolterraDDSPCalibrator,
    device: str,
) -> VolterraDDSPCalibrator:
    np.random.seed(42)
    torch.manual_seed(42)

    fs = float(sim.fs)
    M = int(sim.M)

    steps = _env_int("TRAIN_STEPS", 1400)
    lr = _env_float("TRAIN_LR", 6e-4)
    n = _env_int("TRAIN_N", 32768)
    guard = _env_int("LOSS_GUARD", 3)
    w_img = _env_float("W_IMG", 2500.0)
    w_imd = _env_float("W_IMD", 1800.0)
    w_fund = _env_float("W_FUND", 3.0)
    w_delta = _env_float("W_DELTA", 0.8)
    # 监督参考时，时域对齐应该是主损失（否则仍会出现“投机解”）
    w_time = _env_float("W_TIME", 20.0)
    w_spec = _env_float("W_SPEC", 4.0)
    w_reg = _env_float("W_REG", 2e-5)
    supervised = _env_bool("SUPERVISED_REF", True) and (sim_ref is not None)

    fmin = _env_float("TRAIN_FMIN_HZ", 0.5e9)
    fmax = _env_float("TRAIN_FMAX_HZ", 8.0e9)
    k_min = max(int(np.ceil(fmin * n / fs)), 8)
    k_max = min(int(np.floor(fmax * n / fs)), n // 2 - 8)
    high_bias = _env_float("TRAIN_HIGH_BIAS", 0.35)

    # 分阶段：先把线性（H_{m,1}^{-1}）学稳，再放开 Volterra（避免一开始就“乱拧”导致指标崩）
    lin_only = _env_int("LIN_ONLY_STEPS", 700)
    lin_params = list(calibrator.lin.parameters())
    vol_params = list(calibrator.vol.parameters())
    opt_lin = optim.Adam(lin_params, lr=float(lr), betas=(0.5, 0.9))
    opt_all = optim.Adam(lin_params + vol_params, lr=float(lr), betas=(0.5, 0.9))

    print(f">> train | device={device} | M={M} | supervised={supervised}")
    print(
        f"   N={n} steps={steps} lr={lr:g} f=[{fmin/1e9:.2f},{fmax/1e9:.2f}]GHz | "
        f"LIN_TAPS={len(calibrator.lin[0].fir.weight.view(-1))} | LIN_ONLY_STEPS={lin_only} | HIGH_BIAS={high_bias:g}"
    )

    for step in range(int(steps)):
        is_two_tone = (np.random.rand() < 0.55)
        if not is_two_tone:
            k0 = _sample_k_biased(k_min, k_max, high_bias=high_bias)
            fin = float(k0) * fs / float(n)
            amp = float(np.random.uniform(0.45, 0.95))
            x = sim.tone(fin, n, amp)
            y = sim.capture(x)
            yref = sim_ref.capture(x) if supervised and sim_ref is not None else None
            k_funds = [k0]
            k_imgs = image_bins_for_fund(k0=k0, n=n, m=M)
            k_imd: list[int] = []
        else:
            k1 = _sample_k_biased(k_min, k_max, high_bias=high_bias)
            k2 = _sample_k_biased(k_min, k_max, high_bias=high_bias)
            if abs(k2 - k1) < 20:
                k2 = k1 + 37
            k2 = int(np.clip(k2, k_min, k_max))
            f1 = float(k1) * fs / float(n)
            f2 = float(k2) * fs / float(n)
            amp = float(np.random.uniform(0.35, 0.75))
            x = sim.two_tone(f1, f2, n, amp)
            y = sim.capture(x)
            yref = sim_ref.capture(x) if supervised and sim_ref is not None else None
            k_funds = sorted({_fold_bin(k1, n), _fold_bin(k2, n)})
            k_imgs = sorted(set(image_bins_for_fund(k0=k1, n=n, m=M) + image_bins_for_fund(k0=k2, n=n, m=M)))
            k_imd = imd3_bins_two_tone(k1=k1, k2=k2, n=n)

        s = max(float(np.max(np.abs(y))), 1e-12)
        yt = torch.tensor(y / s, dtype=torch.float32, device=device).view(1, 1, -1)
        yhat = calibrator(yt)

        loss = loss_image_imd(
            yhat,
            k_funds=k_funds,
            k_imgs=k_imgs,
            k_imd=k_imd,
            guard=guard,
            w_img=w_img,
            w_imd=w_imd,
        )
        if supervised and yref is not None:
            yr = torch.tensor(yref / s, dtype=torch.float32, device=device).view(1, 1, -1)
            # 监督参考：主约束（时域 + 频域）
            loss = loss + float(w_time) * mse_time(yhat, yr, crop=256) + float(w_spec) * loss_spec_mse(yhat, yr)
            # 自监督项退化为“正则”，防止过拟合到参考而忽略镜像/IMD结构
            loss = loss + 0.2 * float(w_fund) * loss_fund_hold(yhat, yt, k_funds=k_funds, guard=guard)
            loss = loss + 0.2 * float(w_delta) * torch.mean((yhat - yt) ** 2)
        else:
            # 无监督场景：必须约束“别把基波搞坏 + 别乱改波形”
            loss = loss + float(w_fund) * loss_fund_hold(yhat, yt, k_funds=k_funds, guard=guard)
            loss = loss + float(w_delta) * torch.mean((yhat - yt) ** 2)

        reg = 0.0
        for mod in calibrator.modules():
            if isinstance(mod, nn.Conv1d):
                w = mod.weight.view(-1)
                reg = reg + torch.mean((w[1:] - w[:-1]) ** 2)
            if isinstance(mod, MemoryPolynomial):
                reg = reg + torch.mean(mod.raw**2)
        loss = loss + float(w_reg) * reg

        opt = opt_lin if step < int(lin_only) else opt_all
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=1.0)
        opt.step()

        if step % 175 == 0:
            print(f"step {step:4d}/{steps} | loss={loss.item():.3e} | two_tone={is_two_tone}")

    return calibrator


def evaluate_sweep(*, sim: TIADCSimM, calibrator: VolterraDDSPCalibrator, device: str) -> None:
    fs = float(sim.fs)
    n = _env_int("EVAL_N", 16384)
    # 评估一定要“bin 对齐”，否则谱泄漏会把 spur 能量摊平，导致看起来“没效果”
    f_grid = np.arange(0.2e9, 8.2e9, 0.2e9)
    freqs = []
    for f in f_grid:
        k = int(np.clip(round(float(f) * n / fs), 8, n // 2 - 8))
        freqs.append(float(k) * fs / float(n))
    freqs = np.array(freqs, dtype=np.float64)
    m = {k: [] for k in ["sinad_pre", "sinad_post", "enob_pre", "enob_post", "thd_pre", "thd_post", "sfdr_pre", "sfdr_post"]}
    for f in freqs:
        x = sim.tone(float(f), n, 0.9)
        y = sim.capture(x)
        s = max(float(np.max(np.abs(y))), 1e-12)
        with torch.no_grad():
            yt = torch.tensor(y / s, dtype=torch.float32, device=device).view(1, 1, -1)
            yhat = calibrator(yt).cpu().numpy().flatten() * s
        a0 = calc_metrics(y, fs, float(f))
        a1 = calc_metrics(yhat, fs, float(f))
        m["sinad_pre"].append(a0[0]); m["sinad_post"].append(a1[0])
        m["enob_pre"].append(a0[1]); m["enob_post"].append(a1[1])
        m["thd_pre"].append(a0[2]); m["thd_post"].append(a1[2])
        m["sfdr_pre"].append(a0[3]); m["sfdr_post"].append(a1[3])

    fghz = freqs / 1e9
    plt.figure(figsize=(10, 12))
    plt.subplot(4, 1, 1)
    plt.plot(fghz, m["sinad_pre"], "r--", alpha=0.6, label="Pre")
    plt.plot(fghz, m["sinad_post"], "b-", linewidth=2, label="Post")
    plt.grid(True); plt.legend(); plt.ylabel("dB"); plt.title("SINAD")
    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], "r--", alpha=0.6, label="Pre")
    plt.plot(fghz, m["enob_post"], "b-", linewidth=2, label="Post")
    plt.grid(True); plt.legend(); plt.ylabel("bits"); plt.title("ENOB")
    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], "r--", alpha=0.6, label="Pre")
    plt.plot(fghz, m["thd_post"], "b-", linewidth=2, label="Post")
    plt.grid(True); plt.legend(); plt.ylabel("dBc"); plt.title("THD")
    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], "r--", alpha=0.6, label="Pre")
    plt.plot(fghz, m["sfdr_post"], "b-", linewidth=2, label="Post")
    plt.grid(True); plt.legend(); plt.ylabel("dBc"); plt.xlabel("GHz"); plt.title("SFDR")
    plt.tight_layout()
    savefig("volterra_ddsp_metrics_vs_freq")

    print("\n=== 关键频点（Pre -> Post）===")
    for g in [0.5, 2.0, 4.0, 5.5, 7.5]:
        i = int(np.argmin(np.abs(freqs - g * 1e9)))
        print(
            f"f={freqs[i]/1e9:>4.1f}GHz | "
            f"SINAD {m['sinad_pre'][i]:>6.2f}->{m['sinad_post'][i]:>6.2f} dB | "
            f"THD {m['thd_pre'][i]:>7.2f}->{m['thd_post'][i]:>7.2f} dBc | "
            f"SFDR {m['sfdr_pre'][i]:>6.2f}->{m['sfdr_post'][i]:>6.2f} dBc"
        )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = _env_float("FS_HZ", 20e9)
    M = _env_int("M", 2)

    # 示例：你可改成更贴你系统的失配/非线性参数
    base_cut = _env_float("BASE_CUTOFF_HZ", 8.0e9)
    params: list[ChannelParam] = []
    for m in range(M):
        # 让每相有轻微差异：带宽/增益/延时/非线性
        cut = base_cut * (1.0 - 0.015 * m)
        delay = 0.18 * m
        gain = 1.0 - 0.01 * m
        hd2 = 1e-3 + 2e-3 * m
        hd3 = 5e-4 + 1e-3 * m
        params.append(ChannelParam(cutoff_hz=cut, delay_samp=delay, gain=gain, hd2=hd2, hd3=hd3))

    sim = TIADCSimM(fs=fs, m=M, params=params)

    # 参考（可选）：所有相一致 + 无非线性（用于 SUPERVISED_REF=1）
    pref = [ChannelParam(cutoff_hz=base_cut, delay_samp=0.0, gain=1.0, hd2=0.0, hd3=0.0) for _ in range(M)]
    sim_ref = TIADCSimM(fs=fs, m=M, params=pref)

    lin_taps = _env_int("LIN_TAPS", 129)
    vol_k = _env_int("VOL_K", 21)
    vol_p = _env_int("VOL_P", 3)

    print("\n=== Volterra-DDSP-Calib v1 ===")
    print(f"device={device} | fs={fs/1e9:.2f}GSa/s | M={M} | SUPERVISED_REF={_env_bool('SUPERVISED_REF', True)}")
    print(f"LIN_TAPS={lin_taps} | VOL_K={vol_k} | VOL_P={vol_p}")

    calibrator = VolterraDDSPCalibrator(m=M, lin_taps=lin_taps, vol_k=vol_k, vol_p=vol_p).to(device)
    calibrator = train(sim=sim, sim_ref=sim_ref, calibrator=calibrator, device=device)
    evaluate_sweep(sim=sim, calibrator=calibrator, device=device)


if __name__ == "__main__":
    main()

