import os
import sys
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


# Windows 控制台编码（避免乱码）
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


# plots
_PLOT_DIR: Optional[Path] = None


def _get_plot_dir() -> Path:
    global _PLOT_DIR
    if _PLOT_DIR is not None:
        return _PLOT_DIR
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "plots" / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR = out_dir
    return out_dir


def save_current_figure(name: str, *, dpi: int = 200) -> Path:
    ts_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out = _get_plot_dir() / f"{ts_ms}_{name}.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {out}")
    return out


# 真实 ADC 口径
ADC_JITTER_STD = float(os.getenv("ADC_JITTER_STD", "100e-15"))
ADC_NBITS = int(os.getenv("ADC_NBITS", "12"))


class TIADCSimulator:
    def __init__(self, fs: float = 20e9):
        self.fs = float(fs)

    def generate_tone(self, fin: float, *, n: int) -> np.ndarray:
        t = np.arange(int(n), dtype=np.float64) / self.fs
        return np.sin(2 * np.pi * float(fin) * t)

    def fractional_delay(self, sig: np.ndarray, delay: float) -> np.ndarray:
        d = float(delay) % 1.0
        if abs(d) < 1e-12:
            return sig
        k0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        k1 = np.array([-1 / 3, -1 / 2, 1.0, -1 / 6], dtype=np.float64)
        k2 = np.array([1 / 2, -1.0, 1 / 2, 0.0], dtype=np.float64)
        k3 = np.array([-1 / 6, 1 / 2, -1 / 2, 1 / 6], dtype=np.float64)
        xpad = np.pad(sig.astype(np.float64), (1, 2), mode="edge")
        c0 = np.convolve(xpad, k0[::-1], mode="valid")
        c1 = np.convolve(xpad, k1[::-1], mode="valid")
        c2 = np.convolve(xpad, k2[::-1], mode="valid")
        c3 = np.convolve(xpad, k3[::-1], mode="valid")
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out.astype(sig.dtype, copy=False)

    def apply_channel(
        self,
        sig: np.ndarray,
        *,
        cutoff_freq: float,
        delay_samples: float,
        gain: float = 1.0,
        jitter_std: float = 100e-15,
        n_bits: Optional[int] = 12,
        hd2: float = 0.0,
        hd3: float = 0.0,
        snr_target: Optional[float] = None,
    ) -> np.ndarray:
        sig = np.asarray(sig, dtype=np.float64)
        nyq = self.fs / 2.0

        # 静态非线性（简化）
        sig = sig + float(hd2) * (sig**2) + float(hd3) * (sig**3)

        # 带宽
        b, a = signal.butter(5, float(cutoff_freq) / nyq, btype="low")
        sig = signal.lfilter(b, a, sig)

        # 增益+延时
        sig = sig * float(gain)
        sig = self.fractional_delay(sig, float(delay_samples))

        # 抖动（斜率近似）
        if jitter_std and float(jitter_std) > 0:
            slope = np.gradient(sig) * self.fs
            dt = np.random.normal(0.0, float(jitter_std), len(sig))
            sig = sig + slope * dt

        # 噪声
        if snr_target is not None:
            p = float(np.mean(sig**2))
            n_p = p / (10 ** (float(snr_target) / 10.0))
            sig = sig + np.random.normal(0.0, np.sqrt(n_p), len(sig))
        else:
            sig = sig + np.random.normal(0.0, 1e-4, len(sig))

        # 量化
        if n_bits is None:
            return sig.astype(np.float64, copy=False)
        v_range = 2.0
        levels = 2 ** int(n_bits)
        step = v_range / levels
        sig = np.clip(sig, -1.0, 1.0)
        return (np.round(sig / step) * step).astype(np.float64, copy=False)


def interleave_fullrate(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
    s0 = c0_full[0::2]
    s1 = c1_full[1::2]
    L = min(len(s0), len(s1))
    out = np.zeros(2 * L, dtype=np.float64)
    out[0::2] = s0[:L]
    out[1::2] = s1[:L]
    return out


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


class HybridCalibrationModel(nn.Module):
    def __init__(self, taps: int = 63):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, taps // 2] = 1.0
        self.post_qe = None
        self.post_nl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        x = self.global_delay(x)
        return self.conv(x)


class ComplexMSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, model: HybridCalibrationModel) -> torch.Tensor:
        crop = 300
        yp = y_pred[..., crop:-crop]
        yt = y_target[..., crop:-crop]
        lt = torch.mean((yp - yt) ** 2)
        Yp = torch.fft.rfft(yp, dim=-1, norm="ortho")
        Yt = torch.fft.rfft(yt, dim=-1, norm="ortho")
        lf = torch.mean(torch.abs(Yp - Yt) ** 2)
        w = model.conv.weight.view(-1)
        lr = torch.mean((w[1:] - w[:-1]) ** 2)
        return 100.0 * lt + 100.0 * lf + 0.001 * lr


class PhysicalNonlinearityLayer(nn.Module):
    def __init__(self, *, order: int = 3):
        super().__init__()
        self.order = int(order)
        if self.order < 2:
            raise ValueError("order must be >=2")
        self.alpha = float(os.getenv("POST_QE_ALPHA", "0.25"))
        max_p2 = float(os.getenv("POST_QE_MAX_P2", "8e-3"))
        max_p3 = float(os.getenv("POST_QE_MAX_P3", "6e-3"))
        max_hi = float(os.getenv("POST_QE_MAX_PHI", "2e-3"))
        scales = []
        for p in range(2, self.order + 1):
            scales.append(max_p2 if p == 2 else max_p3 if p == 3 else max_hi)
        self.register_buffer("_scales", torch.tensor(scales, dtype=torch.float32))
        self.raw_even = nn.Parameter(torch.zeros(self.order - 1, dtype=torch.float32))
        self.raw_odd = nn.Parameter(torch.zeros(self.order - 1, dtype=torch.float32))

    def _res(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        a = self._scales.to(x.device, x.dtype) * torch.tanh(raw.to(x.device, x.dtype))
        y = torch.zeros_like(x)
        for i, p in enumerate(range(2, self.order + 1)):
            y = y + a[i] * (x**p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        y = x.clone()
        y[..., 0::2] = xe + float(self.alpha) * self._res(xe, self.raw_even)
        y[..., 1::2] = xo + float(self.alpha) * self._res(xo, self.raw_odd)
        return y


class DifferentiableMemoryPolynomial(nn.Module):
    def __init__(self, *, memory_depth: int = 21, nonlinear_order: int = 3):
        super().__init__()
        self.K = int(memory_depth)
        self.P = int(nonlinear_order)
        if self.K < 1 or self.P < 2:
            raise ValueError("memory_depth>=1, nonlinear_order>=2 required")
        self.alpha = float(os.getenv("POST_NL_ALPHA", "0.25"))
        max_p2 = float(os.getenv("POST_NL_MAX_P2", "8e-3"))
        max_p3 = float(os.getenv("POST_NL_MAX_P3", "6e-3"))
        max_hi = float(os.getenv("POST_NL_MAX_PHI", "2e-3"))
        scales = []
        for p in range(2, self.P + 1):
            scales.append(max_p2 if p == 2 else max_p3 if p == 3 else max_hi)
        self.register_buffer("_scales", torch.tensor(scales, dtype=torch.float32))
        self.raw_even = nn.Parameter(torch.zeros(self.K, self.P - 1, dtype=torch.float32))
        self.raw_odd = nn.Parameter(torch.zeros(self.K, self.P - 1, dtype=torch.float32))

    @staticmethod
    def _delay(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return x
        xp = F.pad(x, (k, 0), mode="constant", value=0.0)
        return xp[..., :-k]

    def _res(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        scales = self._scales.to(x.device, x.dtype)
        raw_t = raw.to(x.device, x.dtype)
        y = torch.zeros_like(x)
        for k in range(self.K):
            xd = self._delay(x, k)
            a = scales * torch.tanh(raw_t[k, :])
            for i, p in enumerate(range(2, self.P + 1)):
                y = y + a[i] * (xd**p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        y = x.clone()
        y[..., 0::2] = xe + float(self.alpha) * self._res(xe, self.raw_even)
        y[..., 1::2] = xo + float(self.alpha) * self._res(xo, self.raw_odd)
        return y


def _harm_bins_folded(k: int, n: int, max_h: int = 5) -> list[int]:
    bins: list[int] = []
    for h in range(2, max_h + 1):
        kk = (h * int(k)) % int(n)
        if kk > n // 2:
            kk = n - kk
        if 1 <= kk <= (n // 2 - 1):
            bins.append(int(kk))
    return sorted(set(bins))


def residual_harmonic_loss(residual: torch.Tensor, target: torch.Tensor, fund_bin: int, *, guard: int = 3) -> torch.Tensor:
    n = residual.shape[-1]
    win = torch.hann_window(n, device=residual.device, dtype=residual.dtype).view(1, 1, -1)
    R = torch.fft.rfft(residual * win, dim=-1, norm="ortho")
    T = torch.fft.rfft(target * win, dim=-1, norm="ortho")
    Pr = (R.real**2 + R.imag**2)
    Pt = (T.real**2 + T.imag**2)
    eps = 1e-12
    s0 = max(0, int(fund_bin) - int(guard))
    e0 = min(Pr.shape[-1], int(fund_bin) + int(guard) + 1)
    p_fund = torch.sum(Pt[..., s0:e0], dim=-1) + eps
    p_h = 0.0
    for kh in _harm_bins_folded(int(fund_bin), int(n), max_h=5):
        ss = max(0, kh - int(guard))
        ee = min(Pr.shape[-1], kh + int(guard) + 1)
        p_h = p_h + torch.sum(Pr[..., ss:ee], dim=-1)
    p_h = p_h + eps
    return torch.mean(p_h / p_fund)


def train_post_qe(sim: TIADCSimulator, model: HybridCalibrationModel, p0: dict, p1: dict, pref: dict, device: str) -> PhysicalNonlinearityLayer:
    order = _env_int("POST_QE_ORDER", 3)
    steps = _env_int("POST_QE_STEPS", 450)
    lr = _env_float("POST_QE_LR", 5e-4)
    batch = _env_int("POST_QE_BATCH", 6)
    n = _env_int("POST_QE_TONE_N", 8192)
    fmin = _env_float("POST_QE_FMIN_HZ", 0.2e9)
    fmax = _env_float("POST_QE_FMAX_HZ", 6.2e9)
    guard = _env_int("POST_QE_GUARD", 3)
    w_time = _env_float("POST_QE_W_TIME", 10.0)
    w_harm = _env_float("POST_QE_W_HARM", 450.0)
    w_fund = _env_float("POST_QE_W_FUND", 8.0)
    w_delta = _env_float("POST_QE_W_DELTA", 25.0)
    ridge = _env_float("POST_QE_RIDGE", 2e-4)

    post_qe = PhysicalNonlinearityLayer(order=order).to(device)
    opt = optim.Adam(post_qe.parameters(), lr=float(lr), betas=(0.5, 0.9))
    fs = float(sim.fs)
    t = torch.arange(n, device=device, dtype=torch.float32) / fs
    k_min = max(int(np.ceil(fmin * n / fs)), 8)
    k_max = min(int(np.floor(fmax * n / fs)), n // 2 - 8)
    print(f">> Post-QE: order={order} steps={steps} lr={lr:g} batch={batch} N={n}")

    for step in range(int(steps)):
        loss_acc = 0.0
        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(n)
            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            y0 = sim.apply_channel(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0)
            y1 = sim.apply_channel(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1)
            yr = sim.apply_channel(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **pref)

            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1c = model(x1).detach().cpu().numpy().flatten() * s12

            xin = interleave_fullrate(y0, y1c)
            yt = yr[: len(xin)]
            s0 = max(float(np.max(np.abs(xin))), float(np.max(np.abs(yt))), 1e-12)
            x = torch.FloatTensor(xin / s0).view(1, 1, -1).to(device)
            y = torch.FloatTensor(yt / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.clamp(y, -1.0, 1.0)

            y_hat = post_qe(x)
            l_h = residual_harmonic_loss(y_hat - y, y, int(k), guard=guard)
            win = torch.hann_window(n, device=device, dtype=x.dtype).view(1, 1, -1)
            Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
            Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
            l_f = torch.mean((torch.log10(torch.abs(Yh[..., int(k)]) + 1e-12) - torch.log10(torch.abs(Yr[..., int(k)]) + 1e-12)) ** 2)
            l_d = torch.mean((y_hat - x) ** 2)
            l_t = torch.mean((y_hat - y) ** 2)
            reg = sum(torch.mean(p**2) for p in post_qe.parameters())
            loss = w_time * l_t + w_harm * l_h + w_fund * l_f + w_delta * l_d + ridge * reg
            loss_acc = loss_acc + loss

        loss_acc = loss_acc / float(max(int(batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_qe.parameters(), max_norm=1.0)
        opt.step()
        if step % 150 == 0:
            print(f"Post-QE step {step:4d}/{steps} | loss={loss_acc.item():.3e}")

    return post_qe


def train_post_nl(sim: TIADCSimulator, model: HybridCalibrationModel, p0: dict, p1: dict, pref: dict, device: str) -> DifferentiableMemoryPolynomial:
    taps = _env_int("STAGE3_TAPS", 21)
    order = _env_int("STAGE3_ORDER", 3)
    steps = _env_int("STAGE3_STEPS", 1200)
    lr = _env_float("STAGE3_LR", 4e-4)
    batch = _env_int("STAGE3_BATCH", 9)
    n = _env_int("STAGE3_TONE_N", 16384)
    fmin = _env_float("STAGE3_FMIN_HZ", 0.2e9)
    fmax = _env_float("STAGE3_FMAX_HZ", 6.2e9)
    guard = _env_int("STAGE3_HARM_GUARD", 3)
    w_time = _env_float("STAGE3_W_TIME", 40.0)
    w_harm = _env_float("STAGE3_W_HARM", 600.0)
    w_fund = _env_float("STAGE3_W_FUND", 6.0)
    w_delta = _env_float("STAGE3_W_DELTA", 8.0)
    w_reg = _env_float("STAGE3_W_REG", 2e-4)

    post_nl = DifferentiableMemoryPolynomial(memory_depth=taps, nonlinear_order=order).to(device)
    opt = optim.Adam(post_nl.parameters(), lr=float(lr), betas=(0.5, 0.9))
    fs = float(sim.fs)
    t = torch.arange(n, device=device, dtype=torch.float32) / fs
    k_min = max(int(np.ceil(fmin * n / fs)), 8)
    k_max = min(int(np.floor(fmax * n / fs)), n // 2 - 8)
    print(f">> Post-NL: taps={taps} order={order} steps={steps} lr={lr:g} batch={batch} N={n}")

    for step in range(int(steps)):
        loss_acc = 0.0
        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(n)
            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            y0 = sim.apply_channel(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0)
            y1 = sim.apply_channel(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1)
            yr = sim.apply_channel(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **pref)

            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1c = model(x1).detach().cpu().numpy().flatten() * s12

            xin = interleave_fullrate(y0, y1c)
            yt = yr[: len(xin)]
            s0 = max(float(np.max(np.abs(xin))), float(np.max(np.abs(yt))), 1e-12)
            x = torch.FloatTensor(xin / s0).view(1, 1, -1).to(device)
            y = torch.FloatTensor(yt / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.clamp(y, -1.0, 1.0)

            if model.post_qe is not None:
                with torch.no_grad():
                    x = model.post_qe(x)

            y_hat = post_nl(x)
            l_h = residual_harmonic_loss(y_hat - y, y, int(k), guard=guard)
            win = torch.hann_window(n, device=device, dtype=x.dtype).view(1, 1, -1)
            Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
            Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
            l_f = torch.mean((torch.log10(torch.abs(Yh[..., int(k)]) + 1e-12) - torch.log10(torch.abs(Yr[..., int(k)]) + 1e-12)) ** 2)
            l_d = torch.mean((y_hat - x) ** 2)
            l_t = torch.mean((y_hat - y) ** 2)
            reg = sum(torch.mean(p**2) for p in post_nl.parameters())
            loss = w_time * l_t + w_harm * l_h + w_fund * l_f + w_delta * l_d + w_reg * reg
            loss_acc = loss_acc + loss

        loss_acc = loss_acc / float(max(int(batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_nl.parameters(), max_norm=1.0)
        opt.step()
        if step % 150 == 0:
            w_mean = float(torch.mean(torch.stack([p.abs().mean() for p in post_nl.parameters()])).detach().cpu())
            print(f"Post-NL step {step:4d}/{steps} | loss={loss_acc.item():.3e} | w_mean~{w_mean:.2e}")

    return post_nl


def train_all(sim: TIADCSimulator, *, device: str):
    np.random.seed(42)
    torch.manual_seed(42)

    p0 = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 1e-3, "hd3": 0.5e-3}
    p1 = {"cutoff_freq": 7.8e9, "delay_samples": 0.25, "gain": 0.98, "hd2": 5e-3, "hd3": 3e-3}
    pref = {"cutoff_freq": p0["cutoff_freq"], "delay_samples": 0.0, "gain": 1.0, "hd2": 0.0, "hd3": 0.0, "snr_target": None}

    # Stage1/2 训练：带限噪声 + 平均
    n_train = 32768
    m_avg = 16
    white = np.random.randn(n_train)
    b, a = signal.butter(6, 7.0e9 / (sim.fs / 2), btype="low")
    base = signal.lfilter(b, a, white)
    base = base / np.max(np.abs(base)) * 0.9

    caps0, caps1 = [], []
    for _ in range(m_avg):
        caps0.append(sim.apply_channel(base, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0))
        caps1.append(sim.apply_channel(base, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1))
    avg0 = np.mean(np.stack(caps0), axis=0)
    avg1 = np.mean(np.stack(caps1), axis=0)

    scale = max(float(np.max(np.abs(avg0))), 1e-12)
    inp = torch.FloatTensor(avg1 / scale).view(1, 1, -1).to(device)
    tgt = torch.FloatTensor(avg0 / scale).view(1, 1, -1).to(device)

    model = HybridCalibrationModel(taps=63).to(device)
    loss_fn = ComplexMSELoss()

    print("=== Stage 1: Relative Delay & Gain ===")
    opt1 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 1e-2},
            {"params": model.gain, "lr": 1e-2},
            {"params": model.conv.parameters(), "lr": 0.0},
        ]
    )
    for _ in range(301):
        opt1.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt1.step()

    print("=== Stage 2: Relative FIR ===")
    opt2 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 0.0},
            {"params": model.gain, "lr": 0.0},
            {"params": model.conv.parameters(), "lr": 5e-4},
        ],
        betas=(0.5, 0.9),
    )
    for _ in range(1001):
        opt2.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt2.step()

    if _env_bool("ENABLE_POST_QE", True):
        model.post_qe = train_post_qe(sim, model, p0, p1, pref, device)
    if _env_bool("ENABLE_STAGE3_NONLINEAR", True):
        model.post_nl = train_post_nl(sim, model, p0, p1, pref, device)
    return model, scale, p0, p1


def calc_metrics(sig: np.ndarray, fs: float, fin: float):
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    win = np.blackman(n)
    cg = float(np.mean(win))
    S = np.fft.rfft(sig * win)
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
        ss = max(0, k - span)
        ee = min(len(mag), k + span + 1)
        kk = ss + int(np.argmax(mag[ss:ee]))
        p_h += float(mag[kk] ** 2)
    thd = 10.0 * np.log10((p_h + 1e-30) / p_fund)
    return float(sinad), float(enob), float(thd), float(sfdr)


def evaluate(sim: TIADCSimulator, model: HybridCalibrationModel, scale: float, device: str, p0: dict, p1: dict):
    freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    m = {k: [] for k in ["sinad_pre", "sinad_post", "sinad_qe", "sinad_nl",
                         "enob_pre", "enob_post", "enob_qe", "enob_nl",
                         "thd_pre", "thd_post", "thd_qe", "thd_nl",
                         "sfdr_pre", "sfdr_post", "sfdr_qe", "sfdr_nl"]}
    for f in freqs:
        src = sim.generate_tone(float(f), n=16384) * 0.9
        y0 = sim.apply_channel(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0)
        y1 = sim.apply_channel(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1)
        with torch.no_grad():
            x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
            y1c = model(x1).cpu().numpy().flatten() * float(scale)
        margin = 500
        c0 = y0[margin:-margin]
        c1r = y1[margin:-margin]
        c1c = y1c[margin:-margin]
        pre = interleave_fullrate(c0, c1r)
        post = interleave_fullrate(c0, c1c)
        qe = post
        if model.post_qe is not None:
            s = max(float(np.max(np.abs(post))), 1e-12)
            with torch.no_grad():
                x = torch.FloatTensor(post / s).view(1, 1, -1).to(device)
                qe = model.post_qe(x).cpu().numpy().flatten() * s
        nl = qe
        if model.post_nl is not None:
            s = max(float(np.max(np.abs(qe))), 1e-12)
            with torch.no_grad():
                x = torch.FloatTensor(qe / s).view(1, 1, -1).to(device)
                nl = model.post_nl(x).cpu().numpy().flatten() * s
        s0, e0, t0, sf0 = calc_metrics(pre, sim.fs, float(f))
        s1, e1, t1, sf1 = calc_metrics(post, sim.fs, float(f))
        s2, e2, t2, sf2 = calc_metrics(qe, sim.fs, float(f))
        s3, e3, t3, sf3 = calc_metrics(nl, sim.fs, float(f))
        m["sinad_pre"].append(s0); m["sinad_post"].append(s1); m["sinad_qe"].append(s2); m["sinad_nl"].append(s3)
        m["enob_pre"].append(e0); m["enob_post"].append(e1); m["enob_qe"].append(e2); m["enob_nl"].append(e3)
        m["thd_pre"].append(t0); m["thd_post"].append(t1); m["thd_qe"].append(t2); m["thd_nl"].append(t3)
        m["sfdr_pre"].append(sf0); m["sfdr_post"].append(sf1); m["sfdr_qe"].append(sf2); m["sfdr_nl"].append(sf3)
    return freqs, m


def plot_metrics(freqs: np.ndarray, m: dict):
    fghz = freqs / 1e9
    plt.figure(figsize=(10, 13))
    plt.subplot(4, 1, 1)
    plt.plot(fghz, m["sinad_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sinad_post"], "g-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sinad_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["sinad_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("SINAD Improvement"); plt.ylabel("dB"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["enob_post"], "m-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["enob_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["enob_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("ENOB Improvement"); plt.ylabel("Bits"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["thd_post"], "b-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["thd_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["thd_nl"], "k--o", alpha=0.85, label="Post-NL")
    plt.title("THD Comparison"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sfdr_post"], "c-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sfdr_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["sfdr_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("SFDR Improvement"); plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    save_current_figure("metrics_vs_freq")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== 运行配置摘要 ===")
    print(f"device={device} | ENABLE_POST_QE={_env_bool('ENABLE_POST_QE', True)} | ENABLE_STAGE3_NONLINEAR={_env_bool('ENABLE_STAGE3_NONLINEAR', True)}")
    sim = TIADCSimulator(fs=20e9)
    model, scale, p0, p1 = train_all(sim, device=device)
    freqs, m = evaluate(sim, model, scale, device, p0, p1)
    print("\n=== 关键频点（Pre -> Linear -> Post-QE -> Post-NL）===")
    for g in [0.1, 1.5, 3.0, 4.5, 5.5, 6.0, 6.5]:
        i = int(np.argmin(np.abs(freqs - g * 1e9)))
        print(
            f"f={freqs[i]/1e9:>4.1f}GHz | "
            f"SINAD {m['sinad_pre'][i]:>6.2f}->{m['sinad_post'][i]:>6.2f}->{m['sinad_qe'][i]:>6.2f}->{m['sinad_nl'][i]:>6.2f} dB | "
            f"THD {m['thd_pre'][i]:>7.2f}->{m['thd_post'][i]:>7.2f}->{m['thd_qe'][i]:>7.2f}->{m['thd_nl'][i]:>7.2f} dBc | "
            f"SFDR {m['sfdr_pre'][i]:>6.2f}->{m['sfdr_post'][i]:>6.2f}->{m['sfdr_qe'][i]:>6.2f}->{m['sfdr_nl'][i]:>6.2f} dBc"
        )
    plot_metrics(freqs, m)
    raise SystemExit(0)

import os
import sys
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


# ----------------------------
# 控制台编码（Windows 防乱码）
# ----------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ----------------------------
# env helpers
# ----------------------------
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


# ----------------------------
# plots
# ----------------------------
_PLOT_DIR: Optional[Path] = None


def _get_plot_dir() -> Path:
    global _PLOT_DIR
    if _PLOT_DIR is not None:
        return _PLOT_DIR
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "plots" / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR = out_dir
    return out_dir


def save_current_figure(name: str, *, dpi: int = 200) -> Path:
    ts_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out = _get_plot_dir() / f"{ts_ms}_{name}.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {out}")
    return out


# ----------------------------
# 真实口径
# ----------------------------
ADC_JITTER_STD = float(os.getenv("ADC_JITTER_STD", "100e-15"))
ADC_NBITS = int(os.getenv("ADC_NBITS", "12"))


# ==============================================================================
# Simulator
# ==============================================================================
class TIADCSimulator:
    def __init__(self, fs: float = 20e9):
        self.fs = float(fs)

    def generate_tone_data(self, f_in: float, *, N: int) -> np.ndarray:
        n = np.arange(int(N), dtype=np.float64)
        return np.sin(2.0 * np.pi * float(f_in) * n / self.fs).astype(np.float64)

    def fractional_delay(self, sig: np.ndarray, delay: float) -> np.ndarray:
        d = float(delay) % 1.0
        if abs(d) < 1e-12:
            return sig
        k0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        k1 = np.array([-1 / 3, -1 / 2, 1.0, -1 / 6], dtype=np.float64)
        k2 = np.array([1 / 2, -1.0, 1 / 2, 0.0], dtype=np.float64)
        k3 = np.array([-1 / 6, 1 / 2, -1 / 2, 1 / 6], dtype=np.float64)
        xpad = np.pad(sig.astype(np.float64), (1, 2), mode="edge")
        c0 = np.convolve(xpad, k0[::-1], mode="valid")
        c1 = np.convolve(xpad, k1[::-1], mode="valid")
        c2 = np.convolve(xpad, k2[::-1], mode="valid")
        c3 = np.convolve(xpad, k3[::-1], mode="valid")
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out.astype(sig.dtype, copy=False)

    def apply_channel_effect(
        self,
        sig: np.ndarray,
        *,
        cutoff_freq: float,
        delay_samples: float,
        gain: float = 1.0,
        jitter_std: float = 100e-15,
        n_bits: Optional[int] = 12,
        hd2: float = 0.0,
        hd3: float = 0.0,
        snr_target: Optional[float] = None,
    ) -> np.ndarray:
        sig = np.asarray(sig, dtype=np.float64)
        nyq = self.fs / 2.0

        sig_nl = sig + float(hd2) * (sig**2) + float(hd3) * (sig**3)
        b, a = signal.butter(5, float(cutoff_freq) / nyq, btype="low")
        sig_bw = signal.lfilter(b, a, sig_nl)
        sig_g = sig_bw * float(gain)
        sig_d = self.fractional_delay(sig_g, float(delay_samples))

        if jitter_std and float(jitter_std) > 0:
            slope = np.gradient(sig_d) * self.fs
            dt = np.random.normal(0.0, float(jitter_std), len(sig_d))
            sig_j = sig_d + slope * dt
        else:
            sig_j = sig_d

        if snr_target is not None:
            p = float(np.mean(sig_j**2))
            n_p = p / (10 ** (float(snr_target) / 10.0))
            thermal = np.random.normal(0.0, np.sqrt(n_p), len(sig_j))
        else:
            thermal = np.random.normal(0.0, 1e-4, len(sig_j))
        sig_n = sig_j + thermal

        if n_bits is None:
            return sig_n.astype(np.float64, copy=False)
        v_range = 2.0
        levels = 2 ** int(n_bits)
        step = v_range / levels
        sig_clip = np.clip(sig_n, -1.0, 1.0)
        return (np.round(sig_clip / step) * step).astype(np.float64, copy=False)


# ==============================================================================
# Stage1/2 model
# ==============================================================================
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


class HybridCalibrationModel(nn.Module):
    def __init__(self, taps: int = 63):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, taps // 2] = 1.0

        self.post_qe = None
        self.post_nl = None
        self.reference_params = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        x = self.global_delay(x)
        return self.conv(x)


class ComplexMSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, model: HybridCalibrationModel) -> torch.Tensor:
        crop = 300
        yp = y_pred[..., crop:-crop]
        yt = y_target[..., crop:-crop]
        loss_time = torch.mean((yp - yt) ** 2)
        Yp = torch.fft.rfft(yp, dim=-1, norm="ortho")
        Yt = torch.fft.rfft(yt, dim=-1, norm="ortho")
        loss_freq = torch.mean(torch.abs(Yp - Yt) ** 2)
        w = model.conv.weight.view(-1)
        loss_reg = torch.mean((w[1:] - w[:-1]) ** 2)
        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg


def interleave_fullrate(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
    s0 = c0_full[0::2]
    s1 = c1_full[1::2]
    L = min(len(s0), len(s1))
    out = np.zeros(2 * L, dtype=np.float64)
    out[0::2] = s0[:L]
    out[1::2] = s1[:L]
    return out


# ==============================================================================
# Post-QE / Post-NL layers (residual + bounded)
# ==============================================================================
class PhysicalNonlinearityLayer(nn.Module):
    def __init__(self, *, order: int = 3):
        super().__init__()
        self.order = int(order)
        if self.order < 2:
            raise ValueError("order must be >= 2")
        self.alpha = float(os.getenv("POST_QE_ALPHA", "0.25"))
        max_p2 = float(os.getenv("POST_QE_MAX_P2", "8e-3"))
        max_p3 = float(os.getenv("POST_QE_MAX_P3", "6e-3"))
        max_hi = float(os.getenv("POST_QE_MAX_PHI", "2e-3"))
        scales = []
        for p in range(2, self.order + 1):
            scales.append(max_p2 if p == 2 else max_p3 if p == 3 else max_hi)
        self.register_buffer("_scales", torch.tensor(scales, dtype=torch.float32))
        self.raw_even = nn.Parameter(torch.zeros(self.order - 1, dtype=torch.float32))
        self.raw_odd = nn.Parameter(torch.zeros(self.order - 1, dtype=torch.float32))

    def _res(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        a = self._scales.to(x.device, x.dtype) * torch.tanh(raw.to(x.device, x.dtype))
        y = torch.zeros_like(x)
        for i, p in enumerate(range(2, self.order + 1)):
            y = y + a[i] * (x ** p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        y = x.clone()
        y[..., 0::2] = xe + float(self.alpha) * self._res(xe, self.raw_even)
        y[..., 1::2] = xo + float(self.alpha) * self._res(xo, self.raw_odd)
        return y


class DifferentiableMemoryPolynomial(nn.Module):
    def __init__(self, *, memory_depth: int = 21, nonlinear_order: int = 3):
        super().__init__()
        self.K = int(memory_depth)
        self.P = int(nonlinear_order)
        if self.K < 1 or self.P < 2:
            raise ValueError("memory_depth>=1 and nonlinear_order>=2 required")
        self.alpha = float(os.getenv("POST_NL_ALPHA", "0.25"))
        max_p2 = float(os.getenv("POST_NL_MAX_P2", "8e-3"))
        max_p3 = float(os.getenv("POST_NL_MAX_P3", "6e-3"))
        max_hi = float(os.getenv("POST_NL_MAX_PHI", "2e-3"))
        scales = []
        for p in range(2, self.P + 1):
            scales.append(max_p2 if p == 2 else max_p3 if p == 3 else max_hi)
        self.register_buffer("_scales", torch.tensor(scales, dtype=torch.float32))  # (P-1,)
        self.raw_even = nn.Parameter(torch.zeros(self.K, self.P - 1, dtype=torch.float32))
        self.raw_odd = nn.Parameter(torch.zeros(self.K, self.P - 1, dtype=torch.float32))

    @staticmethod
    def _delay(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return x
        xp = F.pad(x, (k, 0), mode="constant", value=0.0)
        return xp[..., :-k]

    def _res(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        scales = self._scales.to(x.device, x.dtype)
        raw_t = raw.to(x.device, x.dtype)
        y = torch.zeros_like(x)
        for k in range(self.K):
            xd = self._delay(x, k)
            a = scales * torch.tanh(raw_t[k, :])
            for i, p in enumerate(range(2, self.P + 1)):
                y = y + a[i] * (xd ** p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        y = x.clone()
        y[..., 0::2] = xe + float(self.alpha) * self._res(xe, self.raw_even)
        y[..., 1::2] = xo + float(self.alpha) * self._res(xo, self.raw_odd)
        return y


# ==============================================================================
# Loss helpers
# ==============================================================================
def _harm_bins_folded(k: int, N: int, max_h: int = 5) -> list[int]:
    bins: list[int] = []
    for h in range(2, max_h + 1):
        kk = (h * int(k)) % int(N)
        if kk > N // 2:
            kk = N - kk
        if 1 <= kk <= (N // 2 - 1):
            bins.append(int(kk))
    return sorted(set(bins))


def residual_harmonic_loss_torch(residual: torch.Tensor, target: torch.Tensor, fund_bin: int, *, max_h: int = 5, guard: int = 3) -> torch.Tensor:
    n = residual.shape[-1]
    win = torch.hann_window(n, device=residual.device, dtype=residual.dtype).view(1, 1, -1)
    R = torch.fft.rfft(residual * win, dim=-1, norm="ortho")
    T = torch.fft.rfft(target * win, dim=-1, norm="ortho")
    Pr = (R.real**2 + R.imag**2)
    Pt = (T.real**2 + T.imag**2)
    eps = 1e-12
    s0 = max(0, int(fund_bin) - int(guard))
    e0 = min(Pr.shape[-1], int(fund_bin) + int(guard) + 1)
    p_fund = torch.sum(Pt[..., s0:e0], dim=-1) + eps
    p_h = 0.0
    for kh in _harm_bins_folded(int(fund_bin), int(n), max_h=int(max_h)):
        ss = max(0, kh - int(guard))
        ee = min(Pr.shape[-1], kh + int(guard) + 1)
        p_h = p_h + torch.sum(Pr[..., ss:ee], dim=-1)
    p_h = p_h + eps
    return torch.mean(p_h / p_fund)


# ==============================================================================
# Training
# ==============================================================================
def train_post_qe(
    *,
    sim: TIADCSimulator,
    model: HybridCalibrationModel,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    scale_ref: float,
    device: str,
) -> PhysicalNonlinearityLayer:
    qe_order = _env_int("POST_QE_ORDER", 3)
    steps = _env_int("POST_QE_STEPS", 450)
    lr = _env_float("POST_QE_LR", 5e-4)
    batch = _env_int("POST_QE_BATCH", 6)
    N = _env_int("POST_QE_TONE_N", 8192)
    fmin = _env_float("POST_QE_FMIN_HZ", 0.2e9)
    fmax = _env_float("POST_QE_FMAX_HZ", 6.2e9)
    guard = _env_int("POST_QE_GUARD", 3)
    w_time = _env_float("POST_QE_W_TIME", 10.0)
    w_harm = _env_float("POST_QE_W_HARM", 450.0)
    w_fund = _env_float("POST_QE_W_FUND", 8.0)
    w_delta = _env_float("POST_QE_W_DELTA", 25.0)
    ridge = _env_float("POST_QE_RIDGE", 2e-4)

    post_qe = PhysicalNonlinearityLayer(order=qe_order).to(device)
    opt = optim.Adam(post_qe.parameters(), lr=float(lr), betas=(0.5, 0.9))
    fs = float(sim.fs)
    t = torch.arange(N, device=device, dtype=torch.float32) / fs
    k_min = max(int(np.ceil(fmin * N / fs)), 8)
    k_max = min(int(np.floor(fmax * N / fs)), N // 2 - 8)

    print(f">> Post-QE: order={qe_order} steps={steps} lr={lr:g} batch={batch} N={N}")
    for step in range(int(steps)):
        loss_acc = 0.0
        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(N)
            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            y0 = sim.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0)
            y1 = sim.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1)
            yr = sim.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ref)

            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1c = model(x1).detach().cpu().numpy().flatten() * s12

            tiadc_in = interleave_fullrate(y0, y1c)
            target = yr[: len(tiadc_in)]
            s0 = max(float(np.max(np.abs(tiadc_in))), float(np.max(np.abs(target))), 1e-12)
            x = torch.FloatTensor(tiadc_in / s0).view(1, 1, -1).to(device)
            y = torch.FloatTensor(target / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.clamp(y, -1.0, 1.0)

            y_hat = post_qe(x)
            loss_h = residual_harmonic_loss_torch(y_hat - y, y, int(k), guard=guard)
            win = torch.hann_window(N, device=device, dtype=x.dtype).view(1, 1, -1)
            Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
            Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
            loss_f = torch.mean((torch.log10(torch.abs(Yh[..., int(k)]) + 1e-12) - torch.log10(torch.abs(Yr[..., int(k)]) + 1e-12)) ** 2)
            loss_d = torch.mean((y_hat - x) ** 2)
            loss_t = torch.mean((y_hat - y) ** 2)
            reg = sum(torch.mean(p**2) for p in post_qe.parameters())

            loss = w_time * loss_t + w_harm * loss_h + w_fund * loss_f + w_delta * loss_d + ridge * reg
            loss_acc = loss_acc + loss

        loss_acc = loss_acc / float(max(int(batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_qe.parameters(), max_norm=1.0)
        opt.step()
        if step % 150 == 0:
            print(f"Post-QE step {step:4d}/{steps} | loss={loss_acc.item():.3e}")

    return post_qe


def train_post_nl(
    *,
    sim: TIADCSimulator,
    model: HybridCalibrationModel,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    scale_ref: float,
    device: str,
) -> DifferentiableMemoryPolynomial:
    taps = _env_int("STAGE3_TAPS", 21)
    order = _env_int("STAGE3_ORDER", 3)
    steps = _env_int("STAGE3_STEPS", 1200)
    lr = _env_float("STAGE3_LR", 4e-4)
    batch = _env_int("STAGE3_BATCH", 9)
    N = _env_int("STAGE3_TONE_N", 16384)
    fmin = _env_float("STAGE3_FMIN_HZ", 0.2e9)
    fmax = _env_float("STAGE3_FMAX_HZ", 6.2e9)
    guard = _env_int("STAGE3_HARM_GUARD", 3)
    w_time = _env_float("STAGE3_W_TIME", 40.0)
    w_harm = _env_float("STAGE3_W_HARM", 600.0)
    w_fund = _env_float("STAGE3_W_FUND", 6.0)
    w_delta = _env_float("STAGE3_W_DELTA", 8.0)
    w_reg = _env_float("STAGE3_W_REG", 2e-4)

    post_nl = DifferentiableMemoryPolynomial(memory_depth=taps, nonlinear_order=order).to(device)
    opt = optim.Adam(post_nl.parameters(), lr=float(lr), betas=(0.5, 0.9))
    fs = float(sim.fs)
    t = torch.arange(N, device=device, dtype=torch.float32) / fs
    k_min = max(int(np.ceil(fmin * N / fs)), 8)
    k_max = min(int(np.floor(fmax * N / fs)), N // 2 - 8)

    print(f">> Post-NL: taps={taps} order={order} steps={steps} lr={lr:g} batch={batch} N={N}")
    for step in range(int(steps)):
        loss_acc = 0.0
        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(N)
            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            y0 = sim.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0)
            y1 = sim.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1)
            yr = sim.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ref)

            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1c = model(x1).detach().cpu().numpy().flatten() * s12

            tiadc_in = interleave_fullrate(y0, y1c)
            target = yr[: len(tiadc_in)]
            s0 = max(float(np.max(np.abs(tiadc_in))), float(np.max(np.abs(target))), 1e-12)
            x = torch.FloatTensor(tiadc_in / s0).view(1, 1, -1).to(device)
            y = torch.FloatTensor(target / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.clamp(y, -1.0, 1.0)

            if model.post_qe is not None:
                with torch.no_grad():
                    x = model.post_qe(x)

            y_hat = post_nl(x)
            loss_h = residual_harmonic_loss_torch(y_hat - y, y, int(k), guard=guard)
            win = torch.hann_window(N, device=device, dtype=x.dtype).view(1, 1, -1)
            Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
            Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
            loss_f = torch.mean((torch.log10(torch.abs(Yh[..., int(k)]) + 1e-12) - torch.log10(torch.abs(Yr[..., int(k)]) + 1e-12)) ** 2)
            loss_d = torch.mean((y_hat - x) ** 2)
            loss_t = torch.mean((y_hat - y) ** 2)
            reg = sum(torch.mean(p**2) for p in post_nl.parameters())

            loss = w_time * loss_t + w_harm * loss_h + w_fund * loss_f + w_delta * loss_d + w_reg * reg
            loss_acc = loss_acc + loss

        loss_acc = loss_acc / float(max(int(batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_nl.parameters(), max_norm=1.0)
        opt.step()
        if step % 150 == 0:
            w_mean = float(torch.mean(torch.stack([p.abs().mean() for p in post_nl.parameters()])).detach().cpu())
            print(f"Post-NL step {step:4d}/{steps} | loss={loss_acc.item():.3e} | w_mean~{w_mean:.2e}")

    return post_nl


def train_all(sim: TIADCSimulator, *, device: str):
    np.random.seed(42)
    torch.manual_seed(42)

    params_ch0 = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 1e-3, "hd3": 0.5e-3}
    params_ch1 = {"cutoff_freq": 7.8e9, "delay_samples": 0.25, "gain": 0.98, "hd2": 5e-3, "hd3": 3e-3}
    params_ref = {"cutoff_freq": params_ch0["cutoff_freq"], "delay_samples": 0.0, "gain": 1.0, "hd2": 0.0, "hd3": 0.0, "snr_target": None}

    N_train = 32768
    M_average = 16
    white = np.random.randn(N_train)
    b, a = signal.butter(6, 7.0e9 / (sim.fs / 2), btype="low")
    base_sig = signal.lfilter(b, a, white)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9

    ch0_caps, ch1_caps = [], []
    for _ in range(M_average):
        ch0_caps.append(sim.apply_channel_effect(base_sig, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0))
        ch1_caps.append(sim.apply_channel_effect(base_sig, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1))
    avg_ch0 = np.mean(np.stack(ch0_caps), axis=0)
    avg_ch1 = np.mean(np.stack(ch1_caps), axis=0)

    scale = max(float(np.max(np.abs(avg_ch0))), 1e-12)
    inp = torch.FloatTensor(avg_ch1 / scale).view(1, 1, -1).to(device)
    tgt = torch.FloatTensor(avg_ch0 / scale).view(1, 1, -1).to(device)

    model = HybridCalibrationModel(taps=63).to(device)
    loss_fn = ComplexMSELoss()

    print("=== Stage 1: Relative Delay & Gain ===")
    opt1 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 1e-2},
            {"params": model.gain, "lr": 1e-2},
            {"params": model.conv.parameters(), "lr": 0.0},
        ]
    )
    for _ in range(301):
        opt1.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt1.step()

    print("=== Stage 2: Relative FIR ===")
    opt2 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 0.0},
            {"params": model.gain, "lr": 0.0},
            {"params": model.conv.parameters(), "lr": 5e-4},
        ],
        betas=(0.5, 0.9),
    )
    for _ in range(1001):
        opt2.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt2.step()

    if _env_bool("ENABLE_POST_QE", True):
        model.post_qe = train_post_qe(sim=sim, model=model, params_ch0=params_ch0, params_ch1=params_ch1, params_ref=params_ref, scale_ref=scale, device=device)
    else:
        model.post_qe = None

    if _env_bool("ENABLE_STAGE3_NONLINEAR", True):
        model.post_nl = train_post_nl(sim=sim, model=model, params_ch0=params_ch0, params_ch1=params_ch1, params_ref=params_ref, scale_ref=scale, device=device)
    else:
        model.post_nl = None

    model.reference_params = params_ref
    return model, scale, params_ch0, params_ch1


# ==============================================================================
# Metrics
# ==============================================================================
def calc_spectrum_metrics(sig: np.ndarray, fs: float, fin: float):
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    win = np.blackman(n)
    cg = float(np.mean(win))
    S = np.fft.rfft(sig * win)
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
        ss = max(0, k - span)
        ee = min(len(mag), k + span + 1)
        kk = ss + int(np.argmax(mag[ss:ee]))
        p_h += float(mag[kk] ** 2)
    thd = 10.0 * np.log10((p_h + 1e-30) / p_fund)
    return float(sinad), float(enob), float(thd), float(sfdr)


def evaluate(sim: TIADCSimulator, model: HybridCalibrationModel, scale: float, device: str, p_ch0: dict, p_ch1: dict):
    test_freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    m = {k: [] for k in ["sinad_pre", "sinad_post", "sinad_post_qe", "sinad_post_nl",
                         "enob_pre", "enob_post", "enob_post_qe", "enob_post_nl",
                         "thd_pre", "thd_post", "thd_post_qe", "thd_post_nl",
                         "sfdr_pre", "sfdr_post", "sfdr_post_qe", "sfdr_post_nl"]}

    for f in test_freqs:
        src = sim.generate_tone_data(float(f), N=8192 * 2) * 0.9
        y0 = sim.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p_ch0)
        y1 = sim.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p_ch1)

        with torch.no_grad():
            x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
            y1c = model(x1).cpu().numpy().flatten() * float(scale)

        margin = 500
        c0 = y0[margin:-margin]
        c1r = y1[margin:-margin]
        c1c = y1c[margin:-margin]

        pre = interleave_fullrate(c0, c1r)
        post = interleave_fullrate(c0, c1c)

        post_qe = post
        if model.post_qe is not None:
            s = max(float(np.max(np.abs(post))), 1e-12)
            with torch.no_grad():
                x = torch.FloatTensor(post / s).view(1, 1, -1).to(device)
                y = model.post_qe(x).cpu().numpy().flatten() * s
            post_qe = y

        post_nl = post_qe
        if model.post_nl is not None:
            s = max(float(np.max(np.abs(post_qe))), 1e-12)
            with torch.no_grad():
                x = torch.FloatTensor(post_qe / s).view(1, 1, -1).to(device)
                y = model.post_nl(x).cpu().numpy().flatten() * s
            post_nl = y

        s0, e0, t0, sf0 = calc_spectrum_metrics(pre, sim.fs, float(f))
        s1, e1, t1, sf1 = calc_spectrum_metrics(post, sim.fs, float(f))
        s2, e2, t2, sf2 = calc_spectrum_metrics(post_qe, sim.fs, float(f))
        s3, e3, t3, sf3 = calc_spectrum_metrics(post_nl, sim.fs, float(f))

        m["sinad_pre"].append(s0); m["sinad_post"].append(s1); m["sinad_post_qe"].append(s2); m["sinad_post_nl"].append(s3)
        m["enob_pre"].append(e0); m["enob_post"].append(e1); m["enob_post_qe"].append(e2); m["enob_post_nl"].append(e3)
        m["thd_pre"].append(t0); m["thd_post"].append(t1); m["thd_post_qe"].append(t2); m["thd_post_nl"].append(t3)
        m["sfdr_pre"].append(sf0); m["sfdr_post"].append(sf1); m["sfdr_post_qe"].append(sf2); m["sfdr_post_nl"].append(sf3)

    return test_freqs, m


def plot_metrics(freqs: np.ndarray, m: dict):
    fghz = freqs / 1e9
    plt.figure(figsize=(10, 13))

    plt.subplot(4, 1, 1)
    plt.plot(fghz, m["sinad_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sinad_post"], "g-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sinad_post_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["sinad_post_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("SINAD Improvement"); plt.ylabel("dB"); plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["enob_post"], "m-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["enob_post_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["enob_post_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("ENOB Improvement"); plt.ylabel("Bits"); plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["thd_post"], "b-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["thd_post_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["thd_post_nl"], "k--o", alpha=0.85, label="Post-NL")
    plt.title("THD Comparison"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sfdr_post"], "c-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sfdr_post_qe"], color="#ff7f0e", marker="o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["sfdr_post_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("SFDR Improvement"); plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    save_current_figure("metrics_vs_freq")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== 运行配置摘要（SLIM）===")
    print(f"device={device} | ENABLE_POST_QE={_env_bool('ENABLE_POST_QE', True)} | ENABLE_STAGE3_NONLINEAR={_env_bool('ENABLE_STAGE3_NONLINEAR', True)}")

    sim = TIADCSimulator(fs=20e9)
    model, scale, p0, p1 = train_all(sim, device=device)
    freqs, m = evaluate(sim, model, scale, device, p0, p1)

    print("\n=== 关键频点（Pre -> Post-Linear -> Post-QE -> Post-NL）===")
    pick = [0.1, 1.5, 3.0, 4.5, 5.5, 6.0, 6.5]
    for g in pick:
        i = int(np.argmin(np.abs(freqs - g * 1e9)))
        print(
            f"f={freqs[i]/1e9:>4.1f}GHz | "
            f"SINAD {m['sinad_pre'][i]:>6.2f}->{m['sinad_post'][i]:>6.2f}->{m['sinad_post_qe'][i]:>6.2f}->{m['sinad_post_nl'][i]:>6.2f} dB | "
            f"THD {m['thd_pre'][i]:>7.2f}->{m['thd_post'][i]:>7.2f}->{m['thd_post_qe'][i]:>7.2f}->{m['thd_post_nl'][i]:>7.2f} dBc | "
            f"SFDR {m['sfdr_pre'][i]:>6.2f}->{m['sfdr_post'][i]:>6.2f}->{m['sfdr_post_qe'][i]:>6.2f}->{m['sfdr_post_nl'][i]:>6.2f} dBc"
        )

    plot_metrics(freqs, m)
# *** End of File
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 保存图片到本地（兼容无显示环境）
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==============================================================================
# 精简版：不影响 Stage1/2，仅调 Post‑EQ（全 Nyquist 指标）
# ==============================================================================

_PLOT_DIR: Optional[Path] = None


def _get_plot_dir() -> Path:
    global _PLOT_DIR
    if _PLOT_DIR is not None:
        return _PLOT_DIR
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(__file__).resolve().parent / "plots" / run_ts
    base.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR = base
    return _PLOT_DIR


def _safe_stem(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^0-9a-zA-Z._-]+", "_", name)
    return name.strip("_") or "figure"


def save_current_figure(name: str, *, dpi: int = 200, ext: str = "png") -> Path:
    out_dir = _get_plot_dir()
    ts_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_path = out_dir / f"{ts_ms}_{_safe_stem(name)}.{ext}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {out_path}")
    return out_path


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


# ==============================================================================
# 全局“真实 ADC 条件”口径（用于训练/验证一致性）
# ==============================================================================
ADC_JITTER_STD = float(os.getenv("ADC_JITTER_STD", "100e-15"))
ADC_NBITS = int(os.getenv("ADC_NBITS", "12"))
REF_JITTER_STD = float(os.getenv("REF_JITTER_STD", str(ADC_JITTER_STD)))
REF_NBITS = int(os.getenv("REF_NBITS", str(ADC_NBITS)))


class TIADCSimulator:
    def __init__(self, fs: float = 20e9):
        self.fs = float(fs)

    def fractional_delay(self, sig: np.ndarray, delay: float) -> np.ndarray:
        d = float(delay) % 1.0
        if abs(d) < 1e-12:
            return sig
        k0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        k1 = np.array([-1 / 3, -1 / 2, 1.0, -1 / 6], dtype=np.float64)
        k2 = np.array([1 / 2, -1.0, 1 / 2, 0.0], dtype=np.float64)
        k3 = np.array([-1 / 6, 1 / 2, -1 / 2, 1 / 6], dtype=np.float64)
        xpad = np.pad(sig.astype(np.float64), (1, 2), mode="edge")
        c0 = np.convolve(xpad, k0[::-1], mode="valid")
        c1 = np.convolve(xpad, k1[::-1], mode="valid")
        c2 = np.convolve(xpad, k2[::-1], mode="valid")
        c3 = np.convolve(xpad, k3[::-1], mode="valid")
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out.astype(sig.dtype, copy=False)

    def apply_channel_effect(
        self,
        sig: np.ndarray,
        *,
        cutoff_freq: float,
        delay_samples: float,
        gain: float = 1.0,
        jitter_std: float = 100e-15,
        n_bits: Optional[int] = 12,
        hd2: float = 0.0,
        hd3: float = 0.0,
        snr_target: Optional[float] = None,
    ) -> np.ndarray:
        nyquist = self.fs / 2
        sig_nl = sig + hd2 * (sig**2) + hd3 * (sig**3)
        b, a = signal.butter(5, float(cutoff_freq) / nyquist, btype="low")
        sig_bw = signal.lfilter(b, a, sig_nl)
        sig_gain = sig_bw * float(gain)
        sig_delayed = self.fractional_delay(sig_gain, float(delay_samples))

        if jitter_std and jitter_std > 0:
            slope = np.gradient(sig_delayed) * self.fs
            dt_noise = np.random.normal(0, float(jitter_std), len(sig_delayed))
            sig_j = sig_delayed + slope * dt_noise
        else:
            sig_j = sig_delayed

        if snr_target is not None:
            sig_power = float(np.mean(sig_j**2))
            noise_power = sig_power / (10 ** (float(snr_target) / 10))
            thermal = np.random.normal(0, np.sqrt(noise_power), len(sig_j))
        else:
            thermal = np.random.normal(0, 1e-4, len(sig_j))
        sig_noisy = sig_j + thermal

        if n_bits is not None:
            v_range = 2.0
            levels = 2 ** int(n_bits)
            step = v_range / levels
            sig_clip = np.clip(sig_noisy, -1.0, 1.0)
            sig_out = np.round(sig_clip / step) * step
        else:
            sig_out = sig_noisy
        return sig_out


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


class HybridCalibrationModel(nn.Module):
    def __init__(self, taps: int = 63):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, taps // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        x = self.global_delay(x)
        return self.conv(x)


class ComplexMSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, model: HybridCalibrationModel) -> torch.Tensor:
        crop = 300
        yp = y_pred[..., crop:-crop]
        yt = y_target[..., crop:-crop]
        loss_time = torch.mean((yp - yt) ** 2)
        Yp = torch.fft.rfft(yp, dim=-1, norm="ortho")
        Yt = torch.fft.rfft(yt, dim=-1, norm="ortho")
        loss_freq = torch.mean(torch.abs(Yp - Yt) ** 2)
        w = model.conv.weight.view(-1)
        loss_reg = torch.mean((w[1:] - w[:-1]) ** 2)
        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg


def tiadc_interleave_from_fullrate(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
    s0 = c0_full[0::2]
    s1 = c1_full[1::2]
    L = min(len(s0), len(s1))
    out = np.zeros(2 * L, dtype=np.float64)
    out[0::2] = s0[:L]
    out[1::2] = s1[:L]
    return out


def _ptv2_init_identity(conv: nn.Conv1d) -> None:
    taps = int(conv.weight.shape[-1])
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, taps // 2] = 1.0


def _apply_ptv2(x: torch.Tensor, *, post_even: nn.Conv1d, post_odd: nn.Conv1d) -> torch.Tensor:
    ye = post_even(x)
    yo = post_odd(x)
    out = ye.clone()
    out[..., 1::2] = yo[..., 1::2]
    return out


def _mask_exclude_band(mask: torch.Tensor, k0: int, guard: int) -> None:
    s0 = max(0, k0 - int(guard))
    e0 = min(mask.shape[-1], k0 + int(guard) + 1)
    mask[..., s0:e0] = False


def _harm_bins_folded(k: int, N: int, max_h: int = 5) -> list[int]:
    bins: list[int] = []
    for h in range(2, max_h + 1):
        kk = (h * int(k)) % int(N)
        if kk > N // 2:
            kk = N - kk
        if 1 <= kk <= (N // 2 - 1):
            bins.append(int(kk))
    return sorted(set(bins))


def _make_weight_full_nyq(freqs_r: torch.Tensor, *, fs: float, band_hz: float, fmin_hz: float, fmax_hz: float, hf_weight: float) -> torch.Tensor:
    fs = float(fs)
    nyq = fs / 2.0
    band_hz = float(band_hz)
    fmin_hz = float(fmin_hz)
    fmax_hz = float(fmax_hz)

    m_in = freqs_r <= band_hz
    img_lo = max(0.0, nyq - fmax_hz)
    img_hi = min(nyq, nyq - fmin_hz)
    m_img = (freqs_r >= img_lo) & (freqs_r <= img_hi)

    wf = torch.full_like(freqs_r, 0.12)
    if torch.any(m_in):
        u = torch.clamp(freqs_r / max(band_hz, 1.0), 0.0, 1.0)
        wf = torch.where(m_in, (0.2 + 0.8 * u) ** float(hf_weight), wf)
    if img_hi > img_lo and torch.any(m_img):
        u2 = torch.clamp((freqs_r - img_lo) / max(img_hi - img_lo, 1.0), 0.0, 1.0)
        wf = torch.where(m_img, (0.35 + 0.65 * u2) ** float(hf_weight), wf)
    wf = torch.where(freqs_r < 1e6, torch.zeros_like(wf), wf)
    return wf.view(1, 1, -1)


def train_post_eq_ptv2_ddsp_multi_tone(
    *,
    simulator: TIADCSimulator,
    model_stage12: HybridCalibrationModel,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    device: str,
    taps: int,
    band_hz: float,
    hf_weight: float,
    ridge: float,
    diff_reg: float,
    steps: int,
    lr: float,
    batch: int,
    tone_n: int,
    fmin_hz: float,
    fmax_hz: float,
    guard: int,
    w_time: float,
    w_freq: float,
    w_delta: float,
    w_img: float,
    w_spurmax: float,
    w_fund: float,
    w_smooth: float,
) -> Tuple[nn.Conv1d, nn.Conv1d]:
    taps = int(taps)
    if taps % 2 == 0:
        taps += 1

    post_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    post_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    _ptv2_init_identity(post_even)
    _ptv2_init_identity(post_odd)

    opt = optim.Adam(list(post_even.parameters()) + list(post_odd.parameters()), lr=float(lr), betas=(0.5, 0.9))

    fs = float(simulator.fs)
    N = int(tone_n)
    t = torch.arange(N, device=device, dtype=torch.float32) / float(fs)

    k_min = max(int(np.ceil(float(fmin_hz) * N / fs)), 8)
    k_max = min(int(np.floor(float(fmax_hz) * N / fs)), N // 2 - 8)

    freqs_r = torch.fft.rfftfreq(N, d=1.0 / fs).to(device)
    wf = _make_weight_full_nyq(freqs_r, fs=fs, band_hz=float(band_hz), fmin_hz=float(fmin_hz), fmax_hz=float(fmax_hz), hf_weight=float(hf_weight))

    eps = 1e-12
    neg_large = torch.tensor(-1e30, device=device, dtype=torch.float32)

    for step in range(int(steps)):
        opt.zero_grad()
        loss_acc = 0.0

        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(N)

            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            # 重要：训练时加入噪声和量化，模拟真实ADC条件
            # 这样Post-EQ才能学到有意义的校正（否则在干净数据上无残差可优化）
            y0 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch0)
            y1 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch1)
            # 关键修复：REF也应该有噪声+量化！训练目标应该是realistic的
            yr = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ref)

            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1_cal = model_stage12(x1).detach().cpu().numpy().flatten() * s12

            y_in = tiadc_interleave_from_fullrate(y0, y1_cal)
            y_tgt = yr[: len(y_in)]

            s = max(float(np.max(np.abs(y_tgt))), 1e-12)
            x_in = torch.FloatTensor(y_in / s).view(1, 1, -1).to(device)
            x_tgt = torch.FloatTensor(y_tgt / s).view(1, 1, -1).to(device)

            y_hat = _apply_ptv2(x_in, post_even=post_even, post_odd=post_odd)

            loss_time = torch.mean((y_hat - x_tgt) ** 2)

            Yh = torch.fft.rfft(y_hat, dim=-1, norm="ortho")
            Yt = torch.fft.rfft(x_tgt, dim=-1, norm="ortho")
            loss_freq = torch.mean(torch.abs((Yh - Yt) * wf) ** 2)

            loss_delta = torch.mean((y_hat - x_in) ** 2)

            P = (Yh.real**2 + Yh.imag**2)
            Pt = (Yt.real**2 + Yt.imag**2)

            s0 = max(0, k - int(guard))
            e0 = min(P.shape[-1], k + int(guard) + 1)
            p_fund = torch.sum(P[..., s0:e0]) + eps
            p_fund_t = torch.sum(Pt[..., s0:e0]) + eps

            k_img = int(N // 2 - k)
            p_img = eps
            if 1 <= k_img <= (N // 2 - 1):
                si = max(0, k_img - int(guard))
                ei = min(P.shape[-1], k_img + int(guard) + 1)
                p_img = torch.sum(P[..., si:ei]) + eps
            loss_img = p_img / p_fund

            mask = torch.ones_like(P, dtype=torch.bool)
            mask[..., :5] = False
            _mask_exclude_band(mask, k, guard)
            if 1 <= k_img <= (N // 2 - 1):
                _mask_exclude_band(mask, k_img, guard)
            for kh in _harm_bins_folded(k, N, max_h=5):
                _mask_exclude_band(mask, kh, guard)

            Pm = torch.where(mask, P, neg_large)
            p_spur_max = torch.max(Pm)
            loss_spurmax = p_spur_max / p_fund

            loss_fund_energy = torch.mean((torch.log(p_fund) - torch.log(p_fund_t)) ** 2)
            loss_fund_cplx = torch.mean(torch.abs(Yh[..., k] - Yt[..., k]) ** 2)
            loss_fund = loss_fund_energy + 2.0 * loss_fund_cplx

            we = post_even.weight.view(-1)
            wo = post_odd.weight.view(-1)
            loss_smooth = torch.mean((we[1:] - we[:-1]) ** 2) + torch.mean((wo[1:] - wo[:-1]) ** 2)
            loss_diff = torch.mean((we - wo) ** 2)
            loss_ridge = torch.mean(we**2) + torch.mean(wo**2)

            loss_one = (
                float(w_time) * loss_time
                + float(w_freq) * loss_freq
                + float(w_delta) * loss_delta
                + float(w_img) * loss_img
                + float(w_spurmax) * loss_spurmax
                + float(w_fund) * loss_fund
                + float(w_smooth) * loss_smooth
                + float(diff_reg) * loss_diff
                + float(ridge) * loss_ridge
            )
            loss_acc = loss_acc + loss_one

        loss = loss_acc / float(max(int(batch), 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(post_even.parameters()) + list(post_odd.parameters()), max_norm=1.0)
        opt.step()

        if step % 100 == 0:
            print(f"PostEQ(PTV2-DDSP) step {step:4d}/{int(steps)} | loss={loss.item():.3e}")

    return post_even, post_odd


def apply_post_eq_fir_ptv2(sig: np.ndarray, *, post_fir_even: nn.Module, post_fir_odd: nn.Module, device: str) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    s = max(float(np.max(np.abs(sig))), 1e-12)
    with torch.no_grad():
        x = torch.FloatTensor(sig / s).view(1, 1, -1).to(device)
        ye = post_fir_even(x).cpu().numpy().flatten()
        yo = post_fir_odd(x).cpu().numpy().flatten()
    out = ye.copy()
    out[1::2] = yo[1::2]
    return out * s


# ==============================================================================
# 3.5) Post‑NL：交织后非线性校正（验证 THD 可提升性）
# - 这里用最轻量的 PTV2 memoryless Poly23（even/odd 各一套 a2/a3）
# - 目标仍然贴参考 REF（理想/高性能仪器）
# - loss 专门压 2~5 次谐波，同时强保护基波与“最小改动”
# ==============================================================================
class PostNLPTV2Poly23(nn.Module):
    def __init__(self):
        super().__init__()
        # 直接学 a2/a3 很容易“过冲”引入新 spur；这里用 tanh 做限幅（可用环境变量调上限）
        self.raw_a2_even = nn.Parameter(torch.tensor(0.0))
        self.raw_a3_even = nn.Parameter(torch.tensor(0.0))
        self.raw_a2_odd = nn.Parameter(torch.tensor(0.0))
        self.raw_a3_odd = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("alpha", torch.tensor(1.0))

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(float(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1,1,N)
        alpha_t = self.alpha.to(dtype=x.dtype, device=x.device)

        a2_max = float(os.getenv("POST_NL_A2_MAX", "8e-3"))
        a3_max = float(os.getenv("POST_NL_A3_MAX", "6e-3"))
        a2e = a2_max * torch.tanh(self.raw_a2_even)
        a3e = a3_max * torch.tanh(self.raw_a3_even)
        a2o = a2_max * torch.tanh(self.raw_a2_odd)
        a3o = a3_max * torch.tanh(self.raw_a3_odd)
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        ye = xe + a2e * (xe**2) + a3e * (xe**3)
        yo = xo + a2o * (xo**2) + a3o * (xo**3)
        y_full = x.clone()
        y_full[..., 0::2] = ye
        y_full[..., 1::2] = yo

        # y = x + alpha*(y_full-x)
        return x + alpha_t * (y_full - x)


# ==============================================================================
# [线路A] Post‑NL：轻量 PTV2‑Hammerstein（带记忆 taps）
# - 每个相位(even/odd)各两条支路：x^2 与 x^3，再各自过一个小 FIR（taps 一般 7~15）
# - 相比 memoryless poly23，更容易处理“带宽(记忆)+非线性”耦合，THD/IMD 也更可控
# ==============================================================================
class PostNLPTV2Hammerstein(nn.Module):
    def __init__(self, *, taps: int = 11):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.taps = taps

        # x^2 分支 FIR
        self.h2_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        self.h2_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        # x^3 分支 FIR
        self.h3_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        self.h3_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)

        with torch.no_grad():
            self.h2_even.weight.zero_()
            self.h2_odd.weight.zero_()
            self.h3_even.weight.zero_()
            self.h3_odd.weight.zero_()

        self.register_buffer("alpha", torch.tensor(1.0))

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(float(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1,1,N) 交织输出
        alpha_t = self.alpha.to(dtype=x.dtype, device=x.device)

        # 分相位处理（在 half-rate 上卷积，最后写回到 full-rate）
        xe = x[..., 0::2]
        xo = x[..., 1::2]

        # 分支输入
        xe2 = xe**2
        xo2 = xo**2
        xe3 = xe**3
        xo3 = xo**3

        # 系数限幅（防止过冲引入新 spur）
        # 注意：Hammerstein 的 FIR 权重本身就是系数；这里做整体缩放限幅（可调）
        h2_scale = float(os.getenv("POST_NL_H2_SCALE", "8e-3"))
        h3_scale = float(os.getenv("POST_NL_H3_SCALE", "6e-3"))

        # 用 tanh 对权重做软限幅：w = scale * tanh(raw_w/scale)
        def _tanh_weight(conv: nn.Conv1d, scale: float) -> torch.Tensor:
            w = conv.weight
            return scale * torch.tanh(w / max(scale, 1e-12))

        # 用 functional conv1d 以便用限幅后的权重参与前向
        h2e = _tanh_weight(self.h2_even, h2_scale)
        h2o = _tanh_weight(self.h2_odd, h2_scale)
        h3e = _tanh_weight(self.h3_even, h3_scale)
        h3o = _tanh_weight(self.h3_odd, h3_scale)

        ce = F.conv1d(xe2, h2e, padding=self.taps // 2) + F.conv1d(xe3, h3e, padding=self.taps // 2)
        co = F.conv1d(xo2, h2o, padding=self.taps // 2) + F.conv1d(xo3, h3o, padding=self.taps // 2)

        y_full = x.clone()
        y_full[..., 0::2] = xe + ce
        y_full[..., 1::2] = xo + co

        return x + alpha_t * (y_full - x)


def apply_post_nl(sig: np.ndarray, *, post_nl: nn.Module, device: str) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    s = max(float(np.max(np.abs(sig))), 1e-12)
    with torch.no_grad():
        x = torch.FloatTensor(sig / s).view(1, 1, -1).to(device)
        y = post_nl(x).cpu().numpy().flatten()
    return y * s


def train_post_nl_ptv2_poly23_multi_tone(
    *,
    simulator: TIADCSimulator,
    model_stage12: HybridCalibrationModel,
    post_even: nn.Module,
    post_odd: nn.Module,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    device: str,
    steps: int,
    lr: float,
    batch: int,
    tone_n: int,
    fmin_hz: float,
    fmax_hz: float,
    guard: int,
    w_time: float,
    w_freq: float,
    w_harm: float,
    w_spurmax: float,
    w_fund: float,
    w_delta: float,
    ridge: float,
) -> PostNLPTV2Poly23:
    """
    训练一个非常轻量的交织后 Post‑NL（PTV2 Poly23），用于验证 THD 是否可继续提升。
    - 训练数据：多单音 + 随机幅度
    - 目标：参考 REF（默认 hd2=hd3=0）
    - 约束：谐波能量比 + 基波保护 + 最小改动
    """
    fs = float(simulator.fs)
    N = int(tone_n)
    t = torch.arange(N, device=device, dtype=torch.float32) / float(fs)

    k_min = max(int(np.ceil(float(fmin_hz) * N / fs)), 8)
    k_max = min(int(np.floor(float(fmax_hz) * N / fs)), N // 2 - 8)

    post_nl = PostNLPTV2Poly23().to(device)
    opt = optim.Adam(post_nl.parameters(), lr=float(lr), betas=(0.5, 0.9))
    eps = 1e-12
    neg_large = torch.tensor(-1e30, device=device, dtype=torch.float32)

    # 小权重的全 Nyquist 复谱贴合，避免 Post‑NL 把谱形“搞花”
    freqs_r = torch.fft.rfftfreq(N, d=1.0 / fs).to(device)
    wf = _make_weight_full_nyq(freqs_r, fs=fs, band_hz=6.2e9, fmin_hz=float(fmin_hz), fmax_hz=float(fmax_hz), hf_weight=1.4)

    for step in range(int(steps)):
        opt.zero_grad()
        loss_acc = 0.0

        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(N)

            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            # 重要：训练 Post-NL 时也必须使用真实 ADC 条件（噪声+量化）
            # 同时 REF 目标必须与输入处于同等口径（见 REF 定义处的说明）
            y0 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch0)
            y1 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch1)
            yr = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ref)

            # Stage1/2 校正 Ch1（不动）
            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1_cal = model_stage12(x1).detach().cpu().numpy().flatten() * s12

            # interleave -> Post‑EQ
            y_in = tiadc_interleave_from_fullrate(y0, y1_cal)
            y_tgt = yr[: len(y_in)]

            s = max(float(np.max(np.abs(y_tgt))), 1e-12)
            x_in = torch.FloatTensor(y_in / s).view(1, 1, -1).to(device)
            x_tgt = torch.FloatTensor(y_tgt / s).view(1, 1, -1).to(device)

            with torch.no_grad():
                y_eq = _apply_ptv2(x_in, post_even=post_even, post_odd=post_odd)

            y_hat = post_nl(y_eq)

            # 1) 最小改动（防止引入新 spur）
            loss_delta = torch.mean((y_hat - y_eq) ** 2)

            # 2) 时域贴合（小权重，仅用于稳定）
            loss_time = torch.mean((y_hat - x_tgt) ** 2)

            # 3) 频域谐波惩罚（2~5次，折叠到 Nyquist）
            Y = torch.fft.rfft(y_hat, dim=-1, norm="ortho")
            P = (Y.real**2 + Y.imag**2)
            s0 = max(0, k - int(guard))
            e0 = min(P.shape[-1], k + int(guard) + 1)
            p_fund = torch.sum(P[..., s0:e0]) + eps

            p_harm = 0.0
            for kh in _harm_bins_folded(k, N, max_h=5):
                hs = max(0, kh - int(guard))
                he = min(P.shape[-1], kh + int(guard) + 1)
                p_harm = p_harm + torch.sum(P[..., hs:he])
            loss_harm = p_harm / p_fund

            # 4) 基波保护（幅相 + 能量）：避免“压基波换 THD”
            Yt = torch.fft.rfft(x_tgt, dim=-1, norm="ortho")
            Pt = (Yt.real**2 + Yt.imag**2)
            p_fund_t = torch.sum(Pt[..., s0:e0]) + eps
            loss_fund_energy = torch.mean((torch.log(p_fund) - torch.log(p_fund_t)) ** 2)
            loss_fund_cplx = torch.mean(torch.abs(Y[..., k] - Yt[..., k]) ** 2)
            loss_fund = loss_fund_energy + 2.0 * loss_fund_cplx

            # 4.5) 频域整体贴合（很小权重）：约束全 Nyquist 的谱形不被 Post‑NL 破坏
            loss_freq = torch.mean(torch.abs((Y - Yt) * wf) ** 2)

            # 4.6) 非谐波最大 spur 抑制：避免 Post‑NL 引入新杂散，拖累 SFDR/SINAD
            k_img = int(N // 2 - k)
            mask = torch.ones_like(P, dtype=torch.bool)
            mask[..., :5] = False
            _mask_exclude_band(mask, k, guard)
            if 1 <= k_img <= (N // 2 - 1):
                _mask_exclude_band(mask, k_img, guard)
            for kh in _harm_bins_folded(k, N, max_h=5):
                _mask_exclude_band(mask, kh, guard)
            Pm = torch.where(mask, P, neg_large)
            p_spur_max = torch.max(Pm)
            loss_spurmax = p_spur_max / p_fund

            # 5) 系数正则（更保守）
            loss_ridge = (
                post_nl.raw_a2_even**2
                + post_nl.raw_a3_even**2
                + post_nl.raw_a2_odd**2
                + post_nl.raw_a3_odd**2
            )

            loss_one = (
                float(w_time) * loss_time
                + float(w_freq) * loss_freq
                + float(w_harm) * loss_harm
                + float(w_spurmax) * loss_spurmax
                + float(w_fund) * loss_fund
                + float(w_delta) * loss_delta
                + float(ridge) * loss_ridge
            )
            loss_acc = loss_acc + loss_one

        loss = loss_acc / float(max(int(batch), 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(post_nl.parameters(), max_norm=2.0)
        opt.step()

        if step % 100 == 0:
            # 打印“实际系数”（tanh 限幅后）
            a2_max = float(os.getenv("POST_NL_A2_MAX", "8e-3"))
            a3_max = float(os.getenv("POST_NL_A3_MAX", "6e-3"))
            a2e = float((a2_max * torch.tanh(post_nl.raw_a2_even)).detach().cpu())
            a3e = float((a3_max * torch.tanh(post_nl.raw_a3_even)).detach().cpu())
            a2o = float((a2_max * torch.tanh(post_nl.raw_a2_odd)).detach().cpu())
            a3o = float((a3_max * torch.tanh(post_nl.raw_a3_odd)).detach().cpu())
            print(f"PostNL(PTV2-Poly23) step {step:4d}/{int(steps)} | loss={loss.item():.3e} | a2e={a2e:+.2e} a3e={a3e:+.2e} a2o={a2o:+.2e} a3o={a3o:+.2e}")

    return post_nl


def train_post_nl_ptv2_hammerstein_multi_tone(
    *,
    simulator: TIADCSimulator,
    model_stage12: HybridCalibrationModel,
    post_even: nn.Module,
    post_odd: nn.Module,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    device: str,
    taps: int,
    steps: int,
    lr: float,
    batch: int,
    tone_n: int,
    fmin_hz: float,
    fmax_hz: float,
    guard: int,
    w_time: float,
    w_freq: float,
    w_harm: float,
    w_spurmax: float,
    w_fund: float,
    w_delta: float,
    ridge: float,
    w_smooth: float,
) -> PostNLPTV2Hammerstein:
    """
    线路A：PTV2‑Hammerstein（带记忆 taps），目标是“能提升 THD 且不顾此失彼”。
    """
    fs = float(simulator.fs)
    N = int(tone_n)
    t = torch.arange(N, device=device, dtype=torch.float32) / float(fs)

    k_min = max(int(np.ceil(float(fmin_hz) * N / fs)), 8)
    k_max = min(int(np.floor(float(fmax_hz) * N / fs)), N // 2 - 8)

    post_nl = PostNLPTV2Hammerstein(taps=int(taps)).to(device)
    # 训练阶段固定 alpha=1，训练完再扫 alpha
    post_nl.set_alpha(1.0)

    opt = optim.Adam(post_nl.parameters(), lr=float(lr), betas=(0.5, 0.9))
    eps = 1e-12
    neg_large = torch.tensor(-1e30, device=device, dtype=torch.float32)

    freqs_r = torch.fft.rfftfreq(N, d=1.0 / fs).to(device)
    wf = _make_weight_full_nyq(freqs_r, fs=fs, band_hz=6.2e9, fmin_hz=float(fmin_hz), fmax_hz=float(fmax_hz), hf_weight=1.4)

    for step in range(int(steps)):
        opt.zero_grad()
        loss_acc = 0.0

        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(N)

            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            # 重要：训练时加入噪声和量化，模拟真实ADC条件
            y0 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch0)
            y1 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch1)
            # 关键修复：REF也应该有噪声+量化，这样训练目标才realistic！
            # Post-NL的目标是"校正失配"，而不是"去噪"
            yr = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ref)

            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1_cal = model_stage12(x1).detach().cpu().numpy().flatten() * s12

            y_in = tiadc_interleave_from_fullrate(y0, y1_cal)
            y_tgt = yr[: len(y_in)]

            s = max(float(np.max(np.abs(y_tgt))), 1e-12)
            x_in = torch.FloatTensor(y_in / s).view(1, 1, -1).to(device)
            x_tgt = torch.FloatTensor(y_tgt / s).view(1, 1, -1).to(device)

            with torch.no_grad():
                y_eq = _apply_ptv2(x_in, post_even=post_even, post_odd=post_odd)

            y_hat = post_nl(y_eq)

            loss_delta = torch.mean((y_hat - y_eq) ** 2)
            loss_time = torch.mean((y_hat - x_tgt) ** 2)

            Y = torch.fft.rfft(y_hat, dim=-1, norm="ortho")
            Yt = torch.fft.rfft(x_tgt, dim=-1, norm="ortho")
            P = (Y.real**2 + Y.imag**2)
            Pt = (Yt.real**2 + Yt.imag**2)

            s0 = max(0, k - int(guard))
            e0 = min(P.shape[-1], k + int(guard) + 1)
            p_fund = torch.sum(P[..., s0:e0]) + eps
            p_fund_t = torch.sum(Pt[..., s0:e0]) + eps

            # 谐波惩罚（2~5）
            p_harm = 0.0
            for kh in _harm_bins_folded(k, N, max_h=5):
                hs = max(0, kh - int(guard))
                he = min(P.shape[-1], kh + int(guard) + 1)
                p_harm = p_harm + torch.sum(P[..., hs:he])
            loss_harm = p_harm / p_fund

            # 基波保护
            loss_fund_energy = torch.mean((torch.log(p_fund) - torch.log(p_fund_t)) ** 2)
            loss_fund_cplx = torch.mean(torch.abs(Y[..., k] - Yt[..., k]) ** 2)
            loss_fund = loss_fund_energy + 2.0 * loss_fund_cplx

            # 全频复谱贴合（小权重）
            loss_freq = torch.mean(torch.abs((Y - Yt) * wf) ** 2)

            # 非谐波最大 spur
            k_img = int(N // 2 - k)
            mask = torch.ones_like(P, dtype=torch.bool)
            mask[..., :5] = False
            _mask_exclude_band(mask, k, guard)
            if 1 <= k_img <= (N // 2 - 1):
                _mask_exclude_band(mask, k_img, guard)
            for kh in _harm_bins_folded(k, N, max_h=5):
                _mask_exclude_band(mask, kh, guard)
            Pm = torch.where(mask, P, neg_large)
            p_spur_max = torch.max(Pm)
            loss_spurmax = p_spur_max / p_fund

            # 正则：L2 + 平滑
            weights = [
                post_nl.h2_even.weight,
                post_nl.h2_odd.weight,
                post_nl.h3_even.weight,
                post_nl.h3_odd.weight,
            ]
            loss_ridge = sum(torch.mean(w**2) for w in weights)
            loss_smooth = 0.0
            for w in weights:
                wv = w.view(-1)
                loss_smooth = loss_smooth + torch.mean((wv[1:] - wv[:-1]) ** 2)

            loss_one = (
                float(w_time) * loss_time
                + float(w_freq) * loss_freq
                + float(w_harm) * loss_harm
                + float(w_spurmax) * loss_spurmax
                + float(w_fund) * loss_fund
                + float(w_delta) * loss_delta
                + float(ridge) * loss_ridge
                + float(w_smooth) * loss_smooth
            )
            loss_acc = loss_acc + loss_one

        loss = loss_acc / float(max(int(batch), 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(post_nl.parameters(), max_norm=2.0)
        opt.step()

        if step % 100 == 0:
            # 打印权重范数，监控是否“爆”
            n2e = float(torch.norm(post_nl.h2_even.weight.detach()).cpu())
            n2o = float(torch.norm(post_nl.h2_odd.weight.detach()).cpu())
            n3e = float(torch.norm(post_nl.h3_even.weight.detach()).cpu())
            n3o = float(torch.norm(post_nl.h3_odd.weight.detach()).cpu())
            print(
                f"PostNL(PTV2-Hammerstein) step {step:4d}/{int(steps)} | loss={loss.item():.3e} | "
                f"||h2e||={n2e:.2e} ||h2o||={n2o:.2e} ||h3e||={n3e:.2e} ||h3o||={n3o:.2e}"
            )

    return post_nl


# 更贴近实采数据的通道参数设置
# - 带宽：8.0GHz/7.8GHz（明显高于测试上限6.5GHz，避免滤波器衰减）
# - 延迟失配：0.25采样周期（符合实际器件规格）
# - 增益失配：2%（典型值）
# - **非线性失配加强**：让Post-NL/Post-EQ有明显改善空间
# 通道参数：更符合实际采集的情况
# **关键修改**：两个通道都有适度非线性，差异合理（不极端）
# 这样Post-Linear只能校正线性失配（延迟、增益、带宽）
# 非线性失配需要Post-NL来校正
CH0 = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 3e-3, "hd3": 2e-3}
CH1 = {"cutoff_freq": 7.8e9, "delay_samples": 0.25, "gain": 0.98, "hd2": 8e-3, "hd3": 6e-3}  # 2.7x/3x差异
# **关键修复**：REF使用与CH0相同的参数！
# Post-EQ/NL的目标是：把interleave(CH0,CH1)校正到接近CH0的水平
# 而不是校正到"完美信号"（那是不可能的）
#
# 关键：REF 作为 Post-EQ/Post-NL 的训练目标时，必须与输入处于“同等噪声/量化口径”
# 否则模型会被迫学习“去噪/反量化”（不可逆的 ill-posed 问题），表现为 Post-EQ/Post-NL 反而变差。
# 因此这里 **不使用 snr_target**（保持与 CH0/CH1 相同的底噪模型）。
REF = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 3e-3, "hd3": 2e-3, "snr_target": None}  # 使用CH0参数（噪声口径一致）

POST_EQ_TAPS = _env_int("POST_EQ_TAPS", 127)
POST_EQ_BAND_HZ = _env_float("POST_EQ_BAND_HZ", 6.2e9)
POST_EQ_HF_WEIGHT = _env_float("POST_EQ_HF_WEIGHT", 1.8)
POST_EQ_RIDGE = _env_float("POST_EQ_RIDGE", 1e-2)          # 更保守：避免学出“新 spur”
POST_EQ_PTV2_DIFF = _env_float("POST_EQ_PTV2_DIFF", 1e-1)  # even/odd 更相似：降低 PTV2 带外搬运

POST_EQ_TRAIN_STEPS = _env_int("POST_EQ_TRAIN_STEPS", 2400)  # 增加训练步数
POST_EQ_TRAIN_LR = _env_float("POST_EQ_TRAIN_LR", 4e-4)    # 提高学习率
POST_EQ_BATCH = _env_int("POST_EQ_BATCH", 6)
POST_EQ_TONE_N = _env_int("POST_EQ_TONE_N", 8192)
POST_EQ_FMIN_HZ = _env_float("POST_EQ_FMIN_HZ", 0.2e9)
POST_EQ_FMAX_HZ = _env_float("POST_EQ_FMAX_HZ", 6.2e9)
POST_EQ_GUARD = _env_int("POST_EQ_GUARD", 3)

POST_EQ_W_TIME = _env_float("POST_EQ_W_TIME", 2.0)
POST_EQ_W_FREQ = _env_float("POST_EQ_W_FREQ", 30.0)
POST_EQ_W_DELTA = _env_float("POST_EQ_W_DELTA", 80.0)
POST_EQ_W_IMG = _env_float("POST_EQ_W_IMG", 8.0)
POST_EQ_W_SPURMAX = _env_float("POST_EQ_W_SPURMAX", 4.0)
POST_EQ_W_FUND = _env_float("POST_EQ_W_FUND", 80.0)        # 强保护基波，避免“压基波换 spur/fund 比”
POST_EQ_W_SMOOTH = _env_float("POST_EQ_W_SMOOTH", 1e-3)

# Post‑NL 配置（用于验证 THD 可提升性；默认开启，但你可以用环境变量关闭）
POST_NL_ENABLE = _env_int("POST_NL_ENABLE", 1) != 0
POST_NL_ALPHA = _env_float("POST_NL_ALPHA", 0.35)  # “小修强度”：0=不做，1=全量非线性校正
POST_NL_MAX_DROP_DB = _env_float("POST_NL_MAX_DROP_DB", 1.5)  # 任何指标允许的最大退化（dB/dBc）- 放宽到1.5dB
POST_NL_EVAL_N = _env_int("POST_NL_EVAL_N", 8192)            # alpha 搜索用的短序列长度
# Post‑NL 类型：poly23 / hammerstein（线路A 默认 hammerstein）
POST_NL_MODE = os.getenv("POST_NL_MODE", "hammerstein").strip().lower()
POST_NL_TAPS = _env_int("POST_NL_TAPS", 11)  # Hammerstein taps（建议 7~15）
POST_NL_STEPS = _env_int("POST_NL_STEPS", 2400)  # 增加训练步数
POST_NL_LR = _env_float("POST_NL_LR", 2e-3)  # 提高学习率
POST_NL_BATCH = _env_int("POST_NL_BATCH", 8)
POST_NL_TONE_N = _env_int("POST_NL_TONE_N", POST_EQ_TONE_N)
POST_NL_FMIN_HZ = _env_float("POST_NL_FMIN_HZ", POST_EQ_FMIN_HZ)
POST_NL_FMAX_HZ = _env_float("POST_NL_FMAX_HZ", POST_EQ_FMAX_HZ)
POST_NL_GUARD = _env_int("POST_NL_GUARD", 2)

POST_NL_W_TIME = _env_float("POST_NL_W_TIME", 1.0)
POST_NL_W_FREQ = _env_float("POST_NL_W_FREQ", 1.5)
POST_NL_W_HARM = _env_float("POST_NL_W_HARM", 220.0)
POST_NL_W_SPURMAX = _env_float("POST_NL_W_SPURMAX", 10.0)
POST_NL_W_FUND = _env_float("POST_NL_W_FUND", 120.0)
POST_NL_W_DELTA = _env_float("POST_NL_W_DELTA", 120.0)
POST_NL_RIDGE = _env_float("POST_NL_RIDGE", 8e-3)
POST_NL_W_SMOOTH = _env_float("POST_NL_W_SMOOTH", 3e-3)


def train_relative_calibration(sim: TIADCSimulator, *, device: str) -> tuple[HybridCalibrationModel, float]:
    np.random.seed(42)
    torch.manual_seed(42)

    N_train = 32768
    M_average = 16
    white = np.random.randn(N_train)
    # 训练信号带宽限制在7GHz（略高于测试范围6.5GHz）
    b_src, a_src = signal.butter(6, 7.0e9 / (sim.fs / 2), btype="low")
    base_sig = signal.lfilter(b_src, a_src, white)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9

    ch0_caps = []
    ch1_caps = []
    for _ in range(M_average):
        ch0_caps.append(sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **CH0))
        ch1_caps.append(sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **CH1))
    avg_ch0 = np.mean(np.stack(ch0_caps), axis=0)
    avg_ch1 = np.mean(np.stack(ch1_caps), axis=0)

    scale = max(float(np.max(np.abs(avg_ch0))), 1e-12)
    inp = torch.FloatTensor(avg_ch1 / scale).view(1, 1, -1).to(device)
    tgt = torch.FloatTensor(avg_ch0 / scale).view(1, 1, -1).to(device)

    model = HybridCalibrationModel(taps=63).to(device)
    loss_fn = ComplexMSELoss()

    print("=== Stage 1: Relative Delay & Gain ===")
    opt1 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 1e-2},
            {"params": model.gain, "lr": 1e-2},
            {"params": model.conv.parameters(), "lr": 0.0},
        ]
    )
    for _ in range(301):
        opt1.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt1.step()

    print("=== Stage 2: Relative FIR ===")
    opt2 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 0.0},
            {"params": model.gain, "lr": 0.0},
            {"params": model.conv.parameters(), "lr": 5e-4},
        ],
        betas=(0.5, 0.9),
    )
    for _ in range(1001):
        opt2.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt2.step()

    return model, float(scale)


def calculate_metrics(
    sim: TIADCSimulator,
    model: HybridCalibrationModel,
    scale: float,
    device: str,
    post_even: nn.Module,
    post_odd: nn.Module,
    post_nl: Optional[PostNLPTV2Poly23] = None,
    *,
    test_freqs: Optional[np.ndarray] = None,
    eval_n: int = 16384,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    # 为了 alpha 扫描做公平对比：允许固定随机种子（否则每次评估噪声不同）
    if seed is not None:
        np.random.seed(int(seed))
    if test_freqs is None:
        # 限制测试频率在有效输入带宽内（避免超过滤波器截止频率导致性能崩溃）
        test_freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    test_freqs = np.asarray(test_freqs, dtype=np.float64)
    eval_n = int(eval_n)
    keys = [
        "sinad_pre",
        "sinad_post_lin",
        "sinad_post_eq",
        "enob_pre",
        "enob_post_lin",
        "enob_post_eq",
        "thd_pre",
        "thd_post_lin",
        "thd_post_eq",
        "sfdr_pre",
        "sfdr_post_lin",
        "sfdr_post_eq",
    ]
    if post_nl is not None:
        keys += ["sinad_post_nl", "enob_post_nl", "thd_post_nl", "sfdr_post_nl"]
    m = {k: [] for k in keys}

    def _interleave(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
        s0 = c0_full[0::2]
        s1 = c1_full[1::2]
        L = min(len(s0), len(s1))
        out = np.zeros(2 * L, dtype=np.float64)
        out[0::2] = s0[:L]
        out[1::2] = s1[:L]
        return out

    def _spec(sig: np.ndarray, fs: float, fin: float) -> tuple[float, float, float, float]:
        sig = np.asarray(sig, dtype=np.float64)
        n = len(sig)
        win = np.blackman(n)
        cg = float(np.mean(win))
        S = np.fft.rfft(sig * win)
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
            ss = max(0, k - span)
            ee = min(len(mag), k + span + 1)
            kk = ss + int(np.argmax(mag[ss:ee]))
            p_h += float(mag[kk] ** 2)
        thd = 10.0 * np.log10((p_h + 1e-30) / p_fund)
        return float(sinad), float(enob), float(thd), float(sfdr)

    for f in test_freqs:
        src = np.sin(2 * np.pi * f * (np.arange(eval_n) / sim.fs)) * 0.9
        # 关键修改：验证时也要使用真实ADC条件（噪声+量化）
        # 这样才能看到Post-EQ/Post-NL在真实条件下的改善效果
        y0 = sim.apply_channel_effect(src, jitter_std=100e-15, n_bits=12, **CH0)
        y1 = sim.apply_channel_effect(src, jitter_std=100e-15, n_bits=12, **CH1)
        with torch.no_grad():
            x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
            y1_cal = model(x1).cpu().numpy().flatten() * float(scale)

        margin = 500
        c0 = y0[margin:-margin]
        c1_raw = y1[margin:-margin]
        c1_cal = y1_cal[margin:-margin]
        pre = _interleave(c0, c1_raw)
        post_lin = _interleave(c0, c1_cal)
        post_eq = apply_post_eq_fir_ptv2(post_lin, post_fir_even=post_even, post_fir_odd=post_odd, device=device)
        post_nl_sig = None
        if post_nl is not None:
            post_nl_sig = apply_post_nl(post_eq, post_nl=post_nl, device=device)

        s0, e0, t0, sf0 = _spec(pre, sim.fs, f)
        s1, e1, t1, sf1 = _spec(post_lin, sim.fs, f)
        s2, e2, t2, sf2 = _spec(post_eq, sim.fs, f)
        if post_nl_sig is not None:
            s3, e3, t3, sf3 = _spec(post_nl_sig, sim.fs, f)

        m["sinad_pre"].append(s0); m["sinad_post_lin"].append(s1); m["sinad_post_eq"].append(s2)
        m["enob_pre"].append(e0); m["enob_post_lin"].append(e1); m["enob_post_eq"].append(e2)
        m["thd_pre"].append(t0); m["thd_post_lin"].append(t1); m["thd_post_eq"].append(t2)
        m["sfdr_pre"].append(sf0); m["sfdr_post_lin"].append(sf1); m["sfdr_post_eq"].append(sf2)
        if post_nl_sig is not None:
            m["sinad_post_nl"].append(s3)
            m["enob_post_nl"].append(e3)
            m["thd_post_nl"].append(t3)
            m["sfdr_post_nl"].append(sf3)

    return test_freqs, m


def plot_metrics(test_freqs: np.ndarray, m: dict) -> None:
    fghz = test_freqs / 1e9
    plt.figure(figsize=(10, 13))
    plt.subplot(4, 1, 1)
    plt.plot(fghz, m["sinad_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sinad_post_lin"], "g-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sinad_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    if "sinad_post_nl" in m:
        plt.plot(fghz, m["sinad_post_nl"], color="#2ca02c", linestyle="--", marker="o", linewidth=2, label="Post-NL (PTV2-Poly23)")
    plt.title("SINAD"); plt.ylabel("dB"); plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["enob_post_lin"], "m-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["enob_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    if "enob_post_nl" in m:
        plt.plot(fghz, m["enob_post_nl"], color="#2ca02c", linestyle="--", marker="o", linewidth=2, label="Post-NL (PTV2-Poly23)")
    plt.title("ENOB"); plt.ylabel("Bits"); plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["thd_post_lin"], "b-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["thd_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    if "thd_post_nl" in m:
        plt.plot(fghz, m["thd_post_nl"], color="#2ca02c", linestyle="--", marker="o", linewidth=2, label="Post-NL (PTV2-Poly23)")
    plt.title("THD"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sfdr_post_lin"], "c-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sfdr_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    if "sfdr_post_nl" in m:
        plt.plot(fghz, m["sfdr_post_nl"], color="#2ca02c", linestyle="--", marker="o", linewidth=2, label="Post-NL (PTV2-Poly23)")
    plt.title("SFDR"); plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    save_current_figure("metrics_vs_freq")


def _score_alpha(
    *,
    base: dict,
    cand: dict,
    max_drop_db: float,
) -> tuple[bool, float, dict]:
    """
    评分逻辑（用于“不能顾此失彼”）：
    - 约束：SINAD/SFDR/ENOB 不得比 Post‑EQ 退化超过 max_drop_db；THD 不得明显变差
    - 目标：在满足约束前提下，让 SINAD/SFDR 提升且 THD 更负（更好）
    """
    max_drop_db = float(max_drop_db)
    sinad_eq = np.asarray(base["sinad_post_eq"])
    sfdr_eq = np.asarray(base["sfdr_post_eq"])
    enob_eq = np.asarray(base["enob_post_eq"])
    thd_eq = np.asarray(base["thd_post_eq"])

    sinad_nl = np.asarray(cand["sinad_post_nl"])
    sfdr_nl = np.asarray(cand["sfdr_post_nl"])
    enob_nl = np.asarray(cand["enob_post_nl"])
    thd_nl = np.asarray(cand["thd_post_nl"])

    d_sinad = sinad_nl - sinad_eq
    d_sfdr = sfdr_nl - sfdr_eq
    d_enob = enob_nl - enob_eq
    # THD：越小（更负）越好；用 (eq - nl) 作为改善量
    d_thd = thd_eq - thd_nl

    ok = True
    if np.min(d_sinad) < -max_drop_db:
        ok = False
    if np.min(d_sfdr) < -max_drop_db:
        ok = False
    if np.min(d_enob) < -(max_drop_db / 6.02):  # 近似换算到 bit
        ok = False
    if np.min(d_thd) < -max_drop_db:
        ok = False

    summary = {
        "d_sinad_mean": float(np.mean(d_sinad)),
        "d_sfdr_mean": float(np.mean(d_sfdr)),
        "d_enob_mean": float(np.mean(d_enob)),
        "d_thd_mean": float(np.mean(d_thd)),
        "d_sinad_min": float(np.min(d_sinad)),
        "d_sfdr_min": float(np.min(d_sfdr)),
        "d_thd_min": float(np.min(d_thd)),
    }

    # score：优先 SINAD/SFDR，其次 THD（不让 THD 主导，避免“为拉 THD 伤整体”）
    score = 1.0 * summary["d_sinad_mean"] + 0.6 * summary["d_sfdr_mean"] + 0.25 * summary["d_thd_mean"]
    return ok, float(score), summary


def select_best_alpha_post_nl(
    *,
    sim: TIADCSimulator,
    model: HybridCalibrationModel,
    scale: float,
    device: str,
    post_even: nn.Module,
    post_odd: nn.Module,
    post_nl: PostNLPTV2Poly23,
    alphas: list[float],
    eval_freqs_hz: np.ndarray,
    max_drop_db: float,
    eval_n: int,
) -> tuple[float, dict]:
    """
    在少量代表频点上扫描 alpha，找到“整体提升且不退化”的最优点。
    """
    eval_freqs_hz = np.asarray(eval_freqs_hz, dtype=np.float64)
    seed0 = 20260123
    base_freqs, base = calculate_metrics(
        sim,
        model,
        scale,
        device,
        post_even,
        post_odd,
        post_nl=None,
        test_freqs=eval_freqs_hz,
        eval_n=int(eval_n),
        seed=seed0,
    )
    assert len(base_freqs) == len(eval_freqs_hz)

    best_alpha = float(alphas[0])
    best_score = -1e18
    best_summary: dict = {}
    any_ok = False

    for a in alphas:
        post_nl.set_alpha(float(a))
        _, cand = calculate_metrics(
            sim,
            model,
            scale,
            device,
            post_even,
            post_odd,
            post_nl=post_nl,
            test_freqs=eval_freqs_hz,
            eval_n=int(eval_n),
            seed=seed0,
        )
        ok, score, summary = _score_alpha(base=base, cand=cand, max_drop_db=max_drop_db)
        summary["alpha"] = float(a)
        summary["ok"] = bool(ok)
        # 详细调试输出
        print(f"[DEBUG] alpha={a:.2f} | ok={ok} | score={score:.4f} | dSINAD(mean/min)={summary['d_sinad_mean']:.3f}/{summary['d_sinad_min']:.3f} dB | dTHD(mean/min)={summary['d_thd_mean']:.3f}/{summary['d_thd_min']:.3f} dB")
        if ok:
            any_ok = True
        if ok and score > best_score:
            best_score = float(score)
            best_alpha = float(a)
            best_summary = summary

    # 如果没有任何 ok，就选 score 最大的（但会标注为 non-ok）
    if not any_ok:
        best_alpha = float(alphas[0])
        best_score = -1e18
        best_summary = {}
        for a in alphas:
            post_nl.set_alpha(float(a))
            _, cand = calculate_metrics(
                sim,
                model,
                scale,
                device,
                post_even,
                post_odd,
                post_nl=post_nl,
                test_freqs=eval_freqs_hz,
                eval_n=int(eval_n),
                seed=seed0,
            )
            ok, score, summary = _score_alpha(base=base, cand=cand, max_drop_db=max_drop_db)
            summary["alpha"] = float(a)
            summary["ok"] = bool(ok)
            if score > best_score:
                best_score = float(score)
                best_alpha = float(a)
                best_summary = summary

    post_nl.set_alpha(best_alpha)
    best_summary["score"] = float(best_score)
    return best_alpha, best_summary


if False and __name__ == "__main__":  # legacy入口：已停用（仅保留文件历史，不参与运行）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = TIADCSimulator(fs=20e9)

    print("\n=== 运行配置摘要 ===")
    print(f"PostEQ taps={POST_EQ_TAPS}, steps={POST_EQ_TRAIN_STEPS}, lr={POST_EQ_TRAIN_LR:g}, batch={POST_EQ_BATCH}, N={POST_EQ_TONE_N}, guard={POST_EQ_GUARD}")
    print(f"w_time={POST_EQ_W_TIME:g} w_freq={POST_EQ_W_FREQ:g} w_delta={POST_EQ_W_DELTA:g} w_img={POST_EQ_W_IMG:g} w_spurmax={POST_EQ_W_SPURMAX:g} w_fund={POST_EQ_W_FUND:g}")
    if POST_NL_ENABLE:
        print(
            f"PostNL enable=1 | steps={POST_NL_STEPS} lr={POST_NL_LR:g} batch={POST_NL_BATCH} "
            f"N={POST_NL_TONE_N} harm_guard={POST_NL_GUARD} | alpha={POST_NL_ALPHA:g} | "
            f"w_harm={POST_NL_W_HARM:g} w_spurmax={POST_NL_W_SPURMAX:g} w_fund={POST_NL_W_FUND:g} w_delta={POST_NL_W_DELTA:g}"
        )

    model, scale = train_relative_calibration(sim, device=device)
    post_even, post_odd = train_post_eq_ptv2_ddsp_multi_tone(
        simulator=sim,
        model_stage12=model,
        params_ch0=CH0,
        params_ch1=CH1,
        params_ref=REF,
        device=device,
        taps=int(POST_EQ_TAPS),
        band_hz=float(POST_EQ_BAND_HZ),
        hf_weight=float(POST_EQ_HF_WEIGHT),
        ridge=float(POST_EQ_RIDGE),
        diff_reg=float(POST_EQ_PTV2_DIFF),
        steps=int(POST_EQ_TRAIN_STEPS),
        lr=float(POST_EQ_TRAIN_LR),
        batch=int(POST_EQ_BATCH),
        tone_n=int(POST_EQ_TONE_N),
        fmin_hz=float(POST_EQ_FMIN_HZ),
        fmax_hz=float(POST_EQ_FMAX_HZ),
        guard=int(POST_EQ_GUARD),
        w_time=float(POST_EQ_W_TIME),
        w_freq=float(POST_EQ_W_FREQ),
        w_delta=float(POST_EQ_W_DELTA),
        w_img=float(POST_EQ_W_IMG),
        w_spurmax=float(POST_EQ_W_SPURMAX),
        w_fund=float(POST_EQ_W_FUND),
        w_smooth=float(POST_EQ_W_SMOOTH),
    )

    post_nl = None
    if POST_NL_ENABLE:
        if POST_NL_MODE == "poly23":
            print("\n=== Post‑NL: PTV2‑Poly23 (harmonic suppression) ===")
            post_nl = train_post_nl_ptv2_poly23_multi_tone(
                simulator=sim,
                model_stage12=model,
                post_even=post_even,
                post_odd=post_odd,
                params_ch0=CH0,
                params_ch1=CH1,
                params_ref=REF,
                device=device,
                steps=int(POST_NL_STEPS),
                lr=float(POST_NL_LR),
                batch=int(POST_NL_BATCH),
                tone_n=int(POST_NL_TONE_N),
                fmin_hz=float(POST_NL_FMIN_HZ),
                fmax_hz=float(POST_NL_FMAX_HZ),
                guard=int(POST_NL_GUARD),
                w_time=float(POST_NL_W_TIME),
                w_freq=float(POST_NL_W_FREQ),
                w_harm=float(POST_NL_W_HARM),
                w_spurmax=float(POST_NL_W_SPURMAX),
                w_fund=float(POST_NL_W_FUND),
                w_delta=float(POST_NL_W_DELTA),
                ridge=float(POST_NL_RIDGE),
            )
        else:
            print("\n=== Post‑NL: PTV2‑Hammerstein (线路A, with memory taps) ===")
            post_nl = train_post_nl_ptv2_hammerstein_multi_tone(
                simulator=sim,
                model_stage12=model,
                post_even=post_even,
                post_odd=post_odd,
                params_ch0=CH0,
                params_ch1=CH1,
                params_ref=REF,
                device=device,
                taps=int(POST_NL_TAPS),
                steps=int(POST_NL_STEPS),
                lr=float(POST_NL_LR),
                batch=int(POST_NL_BATCH),
                tone_n=int(POST_NL_TONE_N),
                fmin_hz=float(POST_NL_FMIN_HZ),
                fmax_hz=float(POST_NL_FMAX_HZ),
                guard=int(POST_NL_GUARD),
                w_time=float(POST_NL_W_TIME),
                w_freq=float(POST_NL_W_FREQ),
                w_harm=float(POST_NL_W_HARM),
                w_spurmax=float(POST_NL_W_SPURMAX),
                w_fund=float(POST_NL_W_FUND),
                w_delta=float(POST_NL_W_DELTA),
                ridge=float(POST_NL_RIDGE),
                w_smooth=float(POST_NL_W_SMOOTH),
            )
        # alpha 扫描：优先保证不退化，再看能否整体提升
        eval_freqs = np.array([0.5e9, 1.6e9, 3.6e9, 5.1e9, 6.1e9], dtype=np.float64)
        # 移除0.0，避免在微小改善时错误地选择关闭Post-NL
        alpha_list = [0.15, 0.25, 0.35, 0.5, 0.7, 1.0]
        best_alpha, summary = select_best_alpha_post_nl(
            sim=sim,
            model=model,
            scale=scale,
            device=device,
            post_even=post_even,
            post_odd=post_odd,
            post_nl=post_nl,
            alphas=alpha_list,
            eval_freqs_hz=eval_freqs,
            max_drop_db=float(POST_NL_MAX_DROP_DB),
            eval_n=int(POST_NL_EVAL_N),
        )
        print(
            "\n=== Post‑NL alpha 选择结果（代表频点）===\n"
            f"best_alpha={best_alpha:g} | ok={summary.get('ok')} | score={summary.get('score'):.3f}\n"
            f"dSINAD(mean/min)={summary.get('d_sinad_mean'):.3f}/{summary.get('d_sinad_min'):.3f} dB, "
            f"dSFDR(mean/min)={summary.get('d_sfdr_mean'):.3f}/{summary.get('d_sfdr_min'):.3f} dB, "
            f"dTHD(mean/min)={summary.get('d_thd_mean'):.3f}/{summary.get('d_thd_min'):.3f} dB"
        )

    # 用“选择后的 best alpha”做全扫频指标（并画图）
    test_freqs, metrics = calculate_metrics(sim, model, scale, device, post_even, post_odd, post_nl=post_nl)
    plot_metrics(test_freqs, metrics)
    # 防止文件中历史遗留的其它 main/旧逻辑继续执行
    raise SystemExit(0)
    # 防止文件中历史遗留的其它 main/旧逻辑继续执行
    raise SystemExit(0)
    # 历史遗留代码：文件后面被拼接了旧脚本片段，且有“多余缩进”的行。
    # 为了不影响本脚本主流程（Stage1/2 + Post‑EQ），这里用一个永假分支吸收这些缩进行，
    # 避免 Python 解析时报 IndentationError。
    if False:
        c0 = F.conv1d(xp, self.k0)
        c1 = F.conv1d(xp, self.k1)
        c2 = F.conv1d(xp, self.k2)
        c3 = F.conv1d(xp, self.k3)
        # return c0 + d * (c1 + d * (c2 + d * c3))
        pass


class HybridCalibrationModel(nn.Module):
    def __init__(self, taps: int = 63):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, taps // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        x = self.global_delay(x)
        return self.conv(x)


class ComplexMSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, model: HybridCalibrationModel) -> torch.Tensor:
        crop = 300
        yp = y_pred[..., crop:-crop]
        yt = y_target[..., crop:-crop]
        loss_time = torch.mean((yp - yt) ** 2)
        Yp = torch.fft.rfft(yp, dim=-1, norm="ortho")
        Yt = torch.fft.rfft(yt, dim=-1, norm="ortho")
        loss_freq = torch.mean(torch.abs(Yp - Yt) ** 2)
        w = model.conv.weight.view(-1)
        loss_reg = torch.mean((w[1:] - w[:-1]) ** 2)
        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg


# ==============================================================================
# 3) Post‑EQ：PTV2‑DDSP（全 Nyquist 口径）
# ==============================================================================
def tiadc_interleave_from_fullrate(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
    s0 = c0_full[0::2]
    s1 = c1_full[1::2]
    L = min(len(s0), len(s1))
    out = np.zeros(2 * L, dtype=np.float64)
    out[0::2] = s0[:L]
    out[1::2] = s1[:L]
    return out


def _ptv2_init_identity(conv: nn.Conv1d) -> None:
    taps = int(conv.weight.shape[-1])
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, taps // 2] = 1.0


def _apply_ptv2(x: torch.Tensor, *, post_even: nn.Conv1d, post_odd: nn.Conv1d) -> torch.Tensor:
    ye = post_even(x)
    yo = post_odd(x)
    out = ye.clone()
    out[..., 1::2] = yo[..., 1::2]
    return out


def _make_full_nyquist_weight(
    *,
    freqs_r: torch.Tensor,
    fs: float,
    band_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    hf_weight: float,
) -> torch.Tensor:
    fs = float(fs)
    nyq = fs / 2.0
    band_hz = float(band_hz)
    fmin_hz = float(fmin_hz)
    fmax_hz = float(fmax_hz)

    m_in = freqs_r <= band_hz
    img_lo = max(0.0, nyq - fmax_hz)
    img_hi = min(nyq, nyq - fmin_hz)
    m_img = (freqs_r >= img_lo) & (freqs_r <= img_hi)

    wf = torch.full_like(freqs_r, 0.15)

    if torch.any(m_in):
        u = torch.clamp(freqs_r / max(band_hz, 1.0), 0.0, 1.0)
        wf_in = (0.2 + 0.8 * u) ** float(hf_weight)
        wf = torch.where(m_in, wf_in, wf)

    if img_hi > img_lo and torch.any(m_img):
        u2 = torch.clamp((freqs_r - img_lo) / max(img_hi - img_lo, 1.0), 0.0, 1.0)
        wf_img = (0.25 + 0.75 * u2) ** float(hf_weight)
        wf = torch.where(m_img, wf_img, wf)

    wf = torch.where(freqs_r < 1e6, torch.zeros_like(wf), wf)
    return wf.view(1, 1, -1)


def train_post_eq_ptv2_ddsp_multi_tone(
    *,
    simulator: TIADCSimulator,
    model_stage12: HybridCalibrationModel,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    device: str,
    taps: int,
    band_hz: float,
    hf_weight: float,
    ridge: float,
    diff_reg: float,
    steps: int,
    lr: float,
    batch: int,
    tone_n: int,
    fmin_hz: float,
    fmax_hz: float,
    spur_guard: int,
    w_time: float,
    w_freq: float,
    w_delta: float,
    w_spur: float,
    w_fund: float,
    w_smooth: float,
) -> Tuple[nn.Conv1d, nn.Conv1d]:
    taps = int(taps)
    if taps % 2 == 0:
        taps += 1

    post_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    post_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    _ptv2_init_identity(post_even)
    _ptv2_init_identity(post_odd)

    opt = optim.Adam(list(post_even.parameters()) + list(post_odd.parameters()), lr=float(lr), betas=(0.5, 0.9))

    fs = float(simulator.fs)
    N = int(tone_n)
    t = torch.arange(N, device=device, dtype=torch.float32) / float(fs)

    k_min = max(int(np.ceil(float(fmin_hz) * N / fs)), 8)
    k_max = min(int(np.floor(float(fmax_hz) * N / fs)), N // 2 - 8)

    freqs_r = torch.fft.rfftfreq(N, d=1.0 / fs).to(device)
    wf = _make_full_nyquist_weight(
        freqs_r=freqs_r,
        fs=fs,
        band_hz=float(band_hz),
        fmin_hz=float(fmin_hz),
        fmax_hz=float(fmax_hz),
        hf_weight=float(hf_weight),
    )

    eps = 1e-12

    for step in range(int(steps)):
        opt.zero_grad()
        loss_acc = 0.0

        for _ in range(int(batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(N)

            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            # 重要：训练时加入噪声和量化，模拟真实ADC条件
            # 这样Post-EQ才能学到有意义的校正（否则在干净数据上无残差可优化）
            y0 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch0)
            y1 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch1)
            # 关键修复：REF也应该有噪声+量化！训练目标应该是realistic的
            yr = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ref)

            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1_cal = model_stage12(x1).detach().cpu().numpy().flatten() * s12

            y_in = tiadc_interleave_from_fullrate(y0, y1_cal)
            y_tgt = yr[: len(y_in)]

            s = max(float(np.max(np.abs(y_tgt))), 1e-12)
            x_in = torch.FloatTensor(y_in / s).view(1, 1, -1).to(device)
            x_tgt = torch.FloatTensor(y_tgt / s).view(1, 1, -1).to(device)

            y_hat = _apply_ptv2(x_in, post_even=post_even, post_odd=post_odd)

            # 全频时域贴合
            loss_time = torch.mean((y_hat - x_tgt) ** 2)

            # 全频加权复杂谱贴合（含镜像带）
            Yh = torch.fft.rfft(y_hat, dim=-1, norm="ortho")
            Yt = torch.fft.rfft(x_tgt, dim=-1, norm="ortho")
            loss_freq = torch.mean(torch.abs((Yh - Yt) * wf) ** 2)

            # 最小改动（全频）
            loss_delta = torch.mean((y_hat - x_in) ** 2)

            # spur（全频，除 fund 邻域）
            P = (Yh.real**2 + Yh.imag**2)
            s0 = max(0, k - int(spur_guard))
            e0 = min(P.shape[-1], k + int(spur_guard) + 1)
            fund = torch.sum(P[..., s0:e0], dim=-1) + eps

            mask = torch.ones_like(P, dtype=torch.bool)
            mask[..., :5] = False
            mask[..., s0:e0] = False
            spur = torch.sum(P[mask]) + eps
            loss_spur = spur / torch.sum(fund)

            # 基波保护：对齐目标的 fund 能量
            Pt = (Yt.real**2 + Yt.imag**2)
            fund_t = torch.sum(Pt[..., s0:e0], dim=-1) + eps
            loss_fund = torch.mean((torch.log(fund) - torch.log(fund_t)) ** 2)

            we = post_even.weight.view(-1)
            wo = post_odd.weight.view(-1)
            loss_smooth = torch.mean((we[1:] - we[:-1]) ** 2) + torch.mean((wo[1:] - wo[:-1]) ** 2)
            loss_diff = torch.mean((we - wo) ** 2)
            loss_ridge = torch.mean(we**2) + torch.mean(wo**2)

            loss_one = (
                float(w_time) * loss_time
                + float(w_freq) * loss_freq
                + float(w_delta) * loss_delta
                + float(w_spur) * loss_spur
                + float(w_fund) * loss_fund
                + float(w_smooth) * loss_smooth
                + float(diff_reg) * loss_diff
                + float(ridge) * loss_ridge
            )
            loss_acc = loss_acc + loss_one

        loss = loss_acc / float(max(int(batch), 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(post_even.parameters()) + list(post_odd.parameters()), max_norm=1.0)
        opt.step()

        if step % 100 == 0:
            print(f"PostEQ(PTV2-DDSP) step {step:4d}/{int(steps)} | loss={loss.item():.3e}")

    return post_even, post_odd


def apply_post_eq_fir_ptv2(sig: np.ndarray, *, post_fir_even: nn.Module, post_fir_odd: nn.Module, device: str) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    s = max(float(np.max(np.abs(sig))), 1e-12)
    with torch.no_grad():
        x = torch.FloatTensor(sig / s).view(1, 1, -1).to(device)
        ye = post_fir_even(x).cpu().numpy().flatten()
        yo = post_fir_odd(x).cpu().numpy().flatten()

        
    out = ye.copy()
    out[1::2] = yo[1::2]
    return out * s


# ==============================================================================
# 4) 配置（仅保留需要的）
# ==============================================================================
# 更贴近实采数据的通道参数设置
# 带宽设置明显高于测试上限，避免滤波器截止导致的性能崩溃
# **关键修改**：两个通道都有适度非线性，差异合理（2-3倍，不极端）
# Post-Linear只能校正线性失配，非线性需要Post-NL
CH0 = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 3e-3, "hd3": 2e-3}
CH1 = {"cutoff_freq": 7.8e9, "delay_samples": 0.25, "gain": 0.98, "hd2": 8e-3, "hd3": 6e-3}  # 2.7x/3x差异
# **关键修复**：REF使用与CH0相同的参数！
# Post-EQ/NL的目标是：把interleave(CH0,CH1)校正到接近CH0的水平
# 而不是校正到"完美信号"（那是不可能的）
REF = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 3e-3, "hd3": 2e-3, "snr_target": None}  # 使用CH0参数（噪声口径一致）

POST_EQ_TAPS = _env_int("POST_EQ_TAPS", 127)
POST_EQ_BAND_HZ = _env_float("POST_EQ_BAND_HZ", 6.2e9)
POST_EQ_HF_WEIGHT = _env_float("POST_EQ_HF_WEIGHT", 1.8)
POST_EQ_RIDGE = _env_float("POST_EQ_RIDGE", 2e-3)
POST_EQ_PTV2_DIFF = _env_float("POST_EQ_PTV2_DIFF", 2e-2)

POST_EQ_TRAIN_STEPS = _env_int("POST_EQ_TRAIN_STEPS", 1200)
POST_EQ_TRAIN_LR = _env_float("POST_EQ_TRAIN_LR", 6e-4)


POST_EQ_BATCH = _env_int("POST_EQ_BATCH", 6)
POST_EQ_TONE_N = _env_int("POST_EQ_TONE_N", 8192)
POST_EQ_FMIN_HZ = _env_float("POST_EQ_FMIN_HZ", 0.2e9)
POST_EQ_FMAX_HZ = _env_float("POST_EQ_FMAX_HZ", 6.2e9)
POST_EQ_SPUR_GUARD = _env_int("POST_EQ_SPUR_GUARD", 3)

POST_EQ_W_TIME = _env_float("POST_EQ_W_TIME", 10.0)
POST_EQ_W_FREQ = _env_float("POST_EQ_W_FREQ", 25.0)
POST_EQ_W_DELTA = _env_float("POST_EQ_W_DELTA", 25.0)
POST_EQ_W_SPUR = _env_float("POST_EQ_W_SPUR", 6.0)
POST_EQ_W_FUND = _env_float("POST_EQ_W_FUND", 2.0)
POST_EQ_W_SMOOTH = _env_float("POST_EQ_W_SMOOTH", 5e-4)


def train_relative_calibration(sim: TIADCSimulator, *, device: str) -> tuple[HybridCalibrationModel, float]:
    np.random.seed(42)
    torch.manual_seed(42)

    N_train = 32768
    M_average = 16
    white = np.random.randn(N_train)
    # 训练信号带宽限制在7GHz（略高于测试范围6.5GHz）
    b_src, a_src = signal.butter(6, 7.0e9 / (sim.fs / 2), btype="low")
    base_sig = signal.lfilter(b_src, a_src, white)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9

    ch0_caps = []
    ch1_caps = []
    for _ in range(M_average):
        ch0_caps.append(sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **CH0))
        ch1_caps.append(sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **CH1))
    avg_ch0 = np.mean(np.stack(ch0_caps), axis=0)
    avg_ch1 = np.mean(np.stack(ch1_caps), axis=0)

    scale = max(float(np.max(np.abs(avg_ch0))), 1e-12)
    inp = torch.FloatTensor(avg_ch1 / scale).view(1, 1, -1).to(device)
    tgt = torch.FloatTensor(avg_ch0 / scale).view(1, 1, -1).to(device)

    model = HybridCalibrationModel(taps=63).to(device)
    loss_fn = ComplexMSELoss()

    print("=== Stage 1: Relative Delay & Gain ===")
    opt1 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 1e-2},
            {"params": model.gain, "lr": 1e-2},
            {"params": model.conv.parameters(), "lr": 0.0},
        ]
    )
    for _ in range(301):
        opt1.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt1.step()

    print("=== Stage 2: Relative FIR ===")
    opt2 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 0.0},
            {"params": model.gain, "lr": 0.0},
            {"params": model.conv.parameters(), "lr": 5e-4},
        ],
        betas=(0.5, 0.9),
    )
    for _ in range(1001):
        opt2.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt2.step()

    return model, float(scale)


def calculate_metrics(sim: TIADCSimulator, model: HybridCalibrationModel, scale: float, device: str, post_even: nn.Module, post_odd: nn.Module) -> tuple[np.ndarray, dict]:
    print("\n=== 正在计算综合指标对比 (TIADC 交织输出，Nyquist=10GHz) ===")
    # 限制测试频率在有效输入带宽内（避免超过滤波器截止频率导致性能崩溃）
    test_freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    m = {
        "sinad_pre": [],
        "sinad_post_lin": [],
        "sinad_post_eq": [],
        "enob_pre": [],
        "enob_post_lin": [],
        "enob_post_eq": [],
        "thd_pre": [],
        "thd_post_lin": [],
        "thd_post_eq": [],
        "sfdr_pre": [],
        "sfdr_post_lin": [],
        "sfdr_post_eq": [],
    }

    def _interleave(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
        s0 = c0_full[0::2]
        s1 = c1_full[1::2]
        L = min(len(s0), len(s1))
        out = np.zeros(2 * L, dtype=np.float64)
        out[0::2] = s0[:L]
        out[1::2] = s1[:L]
        return out

    def _spec(sig: np.ndarray, fs: float, fin: float) -> tuple[float, float, float, float]:
        sig = np.asarray(sig, dtype=np.float64)
        n = len(sig)
        win = np.blackman(n)
        cg = float(np.mean(win))
        S = np.fft.rfft(sig * win)
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
            ss = max(0, k - span)
            ee = min(len(mag), k + span + 1)
            kk = ss + int(np.argmax(mag[ss:ee]))
            p_h += float(mag[kk] ** 2)
        thd = 10.0 * np.log10((p_h + 1e-30) / p_fund)
        return float(sinad), float(enob), float(thd), float(sfdr)

    for f in test_freqs:
        src = np.sin(2 * np.pi * f * (np.arange(16384) / sim.fs)) * 0.9
        # 验证口径必须与训练一致：使用真实 ADC 条件（噪声+量化）
        y0 = sim.apply_channel_effect(src, jitter_std=100e-15, n_bits=12, **CH0)
        y1 = sim.apply_channel_effect(src, jitter_std=100e-15, n_bits=12, **CH1)

        with torch.no_grad():
            x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
            y1_cal = model(x1).cpu().numpy().flatten() * float(scale)

        margin = 500
        c0 = y0[margin:-margin]
        c1_raw = y1[margin:-margin]
        c1_cal = y1_cal[margin:-margin]
        pre = _interleave(c0, c1_raw)
        post_lin = _interleave(c0, c1_cal)
        post_eq = apply_post_eq_fir_ptv2(post_lin, post_fir_even=post_even, post_fir_odd=post_odd, device=device)

        s0, e0, t0, sf0 = _spec(pre, sim.fs, f)
        s1, e1, t1, sf1 = _spec(post_lin, sim.fs, f)
        s2, e2, t2, sf2 = _spec(post_eq, sim.fs, f)

        m["sinad_pre"].append(s0); m["sinad_post_lin"].append(s1); m["sinad_post_eq"].append(s2)
        m["enob_pre"].append(e0); m["enob_post_lin"].append(e1); m["enob_post_eq"].append(e2)
        m["thd_pre"].append(t0); m["thd_post_lin"].append(t1); m["thd_post_eq"].append(t2)
        m["sfdr_pre"].append(sf0); m["sfdr_post_lin"].append(sf1); m["sfdr_post_eq"].append(sf2)

    return test_freqs, m


def plot_metrics(test_freqs: np.ndarray, m: dict) -> None:
    fghz = test_freqs / 1e9
    plt.figure(figsize=(10, 13))

    plt.subplot(4, 1, 1)
    plt.plot(fghz, m["sinad_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sinad_post_lin"], "g-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sinad_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("SINAD")
    plt.ylabel("dB")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["enob_post_lin"], "m-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["enob_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("ENOB")
    plt.ylabel("Bits")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["thd_post_lin"], "b-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["thd_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("THD")
    plt.ylabel("dBc")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sfdr_post_lin"], "c-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sfdr_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("SFDR")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("dBc")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_current_figure("metrics_vs_freq")


if False and __name__ == "__main__":  # legacy入口：已停用（仅保留文件历史，不参与运行）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = TIADCSimulator(fs=20e9)

    print("\n=== 运行配置摘要（全 Nyquist Post‑EQ）===")
    print(f"taps={POST_EQ_TAPS} | steps={POST_EQ_TRAIN_STEPS} | lr={POST_EQ_TRAIN_LR:g} | batch={POST_EQ_BATCH} | N={POST_EQ_TONE_N} | band={POST_EQ_BAND_HZ/1e9:.2f}GHz")
    print(f"w_time={POST_EQ_W_TIME:g} w_freq={POST_EQ_W_FREQ:g} w_delta={POST_EQ_W_DELTA:g} w_spur={POST_EQ_W_SPUR:g} w_fund={POST_EQ_W_FUND:g}")

    model, scale = train_relative_calibration(sim, device=device)

    print("\n=== Post‑EQ: PTV2‑DDSP (full Nyquist objective) ===")
    post_even, post_odd = train_post_eq_ptv2_ddsp_multi_tone(
        simulator=sim,
        model_stage12=model,
        params_ch0=CH0,
        params_ch1=CH1,
        params_ref=REF,
        device=device,
        taps=int(POST_EQ_TAPS),
        band_hz=float(POST_EQ_BAND_HZ),
        hf_weight=float(POST_EQ_HF_WEIGHT),
        ridge=float(POST_EQ_RIDGE),
        diff_reg=float(POST_EQ_PTV2_DIFF),
        steps=int(POST_EQ_TRAIN_STEPS),
        lr=float(POST_EQ_TRAIN_LR),
        batch=int(POST_EQ_BATCH),
        tone_n=int(POST_EQ_TONE_N),
        fmin_hz=float(POST_EQ_FMIN_HZ),
        fmax_hz=float(POST_EQ_FMAX_HZ),
        spur_guard=int(POST_EQ_SPUR_GUARD),
        w_time=float(POST_EQ_W_TIME),
        w_freq=float(POST_EQ_W_FREQ),
        w_delta=float(POST_EQ_W_DELTA),
        w_spur=float(POST_EQ_W_SPUR),
        w_fund=float(POST_EQ_W_FUND),
        w_smooth=float(POST_EQ_W_SMOOTH),
    )

    test_freqs, metrics = calculate_metrics(sim, model, scale, device, post_even, post_odd)
    plot_metrics(test_freqs, metrics)
    # 防止历史遗留代码（文件后半段）被继续执行
    raise SystemExit(0)


# ==============================================================================
# 1) 仿真器（物理通道）
# ==============================================================================
class TIADCSimulator:
    def __init__(self, fs: float = 20e9):
        self.fs = float(fs)

    def fractional_delay(self, sig: np.ndarray, delay: float) -> np.ndarray:
        """4-tap 三次 Farrow/Lagrange 分数延时（避免 FFT 循环延时绕回）。"""
        d = float(delay) % 1.0
        if abs(d) < 1e-12:
            return sig
        k0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        k1 = np.array([-1 / 3, -1 / 2, 1.0, -1 / 6], dtype=np.float64)
        k2 = np.array([1 / 2, -1.0, 1 / 2, 0.0], dtype=np.float64)
        k3 = np.array([-1 / 6, 1 / 2, -1 / 2, 1 / 6], dtype=np.float64)

        xpad = np.pad(sig.astype(np.float64), (1, 2), mode="edge")
        c0 = np.convolve(xpad, k0[::-1], mode="valid")
        c1 = np.convolve(xpad, k1[::-1], mode="valid")
        c2 = np.convolve(xpad, k2[::-1], mode="valid")
        c3 = np.convolve(xpad, k3[::-1], mode="valid")
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out.astype(sig.dtype, copy=False)

    def apply_channel_effect(
        self,
        sig: np.ndarray,
        *,
        cutoff_freq: float,
        delay_samples: float,
        gain: float = 1.0,
        jitter_std: float = 100e-15,
        n_bits: Optional[int] = 12,
        hd2: float = 0.0,
        hd3: float = 0.0,
        snr_target: Optional[float] = None,
    ) -> np.ndarray:
        nyquist = self.fs / 2

        # 1) 非线性
        sig_nl = sig + hd2 * (sig**2) + hd3 * (sig**3)

        # 2) 带宽
        b, a = signal.butter(5, float(cutoff_freq) / nyquist, btype="low")
        sig_bw = signal.lfilter(b, a, sig_nl)

        # 3) 增益
        sig_gain = sig_bw * float(gain)

        # 4) 延时
        sig_delayed = self.fractional_delay(sig_gain, float(delay_samples))

        # 5) 抖动
        if jitter_std and jitter_std > 0:
            slope = np.gradient(sig_delayed) * self.fs
            dt_noise = np.random.normal(0, float(jitter_std), len(sig_delayed))
            sig_j = sig_delayed + slope * dt_noise
        else:
            sig_j = sig_delayed

        # 6) 热噪声（按目标 SNR 或默认底噪）
        if snr_target is not None:
            sig_power = float(np.mean(sig_j**2))
            noise_power = sig_power / (10 ** (float(snr_target) / 10))
            thermal = np.random.normal(0, np.sqrt(noise_power), len(sig_j))
        else:
            thermal = np.random.normal(0, 1e-4, len(sig_j))
        sig_noisy = sig_j + thermal

        # 7) 量化
        if n_bits is not None:
            v_range = 2.0
            levels = 2 ** int(n_bits)
            step = v_range / levels
            sig_clip = np.clip(sig_noisy, -1.0, 1.0)
            sig_out = np.round(sig_clip / step) * step
        else:
            sig_out = sig_noisy
        return sig_out


# ==============================================================================
# 2) Stage1/2：相对线性校准（Ch1 -> Ch0）
# ==============================================================================
class FarrowDelay(nn.Module):
    """torch 版 4-tap Farrow 分数延时（只支持小数部分；整数部分不在此做）。"""

    def __init__(self):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor(0.0))

        # (out_len=4) Farrow coefficients，和 numpy 版一致
        k0 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        k1 = torch.tensor([-1 / 3, -1 / 2, 1.0, -1 / 6], dtype=torch.float32)
        k2 = torch.tensor([1 / 2, -1.0, 1 / 2, 0.0], dtype=torch.float32)
        k3 = torch.tensor([-1 / 6, 1 / 2, -1 / 2, 1 / 6], dtype=torch.float32)

        self.register_buffer("k0", k0.view(1, 1, -1))
        self.register_buffer("k1", k1.view(1, 1, -1))
        self.register_buffer("k2", k2.view(1, 1, -1))
        self.register_buffer("k3", k3.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,T)
        d = torch.remainder(self.delay, 1.0)
        if torch.all(torch.abs(d) < 1e-9):
            return x
        # replicate pad (1,2)
        xp = F.pad(x, (1, 2), mode="replicate")
        c0 = F.conv1d(xp, self.k0)
        c1 = F.conv1d(xp, self.k1)
        c2 = F.conv1d(xp, self.k2)
        c3 = F.conv1d(xp, self.k3)
        return c0 + d * (c1 + d * (c2 + d * c3))


class HybridCalibrationModel(nn.Module):
    """只保留 Stage1/2 需要的线性路径：gain + delay + FIR。"""

    def __init__(self, taps: int = 63):
        super().__init__()
        if taps % 2 == 0:
            taps += 1
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, taps // 2] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        x = self.global_delay(x)
        return self.conv(x)


class ComplexMSELoss(nn.Module):
    """Stage1/2 的稳定损失：时域 + 频域 + FIR 平滑正则。"""

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, model: HybridCalibrationModel) -> torch.Tensor:
        crop = 300
        yp = y_pred[..., crop:-crop]
        yt = y_target[..., crop:-crop]

        # 时域
        loss_time = torch.mean((yp - yt) ** 2)

        # 频域（复杂谱）
        Yp = torch.fft.rfft(yp, dim=-1, norm="ortho")
        Yt = torch.fft.rfft(yt, dim=-1, norm="ortho")
        loss_freq = torch.mean(torch.abs(Yp - Yt) ** 2)

        # FIR 平滑
        w = model.conv.weight.view(-1)
        loss_reg = torch.mean((w[1:] - w[:-1]) ** 2)

        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg


# ==============================================================================
# 3) Post‑EQ：PTV2‑DDSP（even/odd 两套 FIR，用参考驱动训练）
# ==============================================================================
def tiadc_interleave_from_fullrate(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
    """20GSps 交织：偶点取 c0[0::2]，奇点取 c1[1::2]。"""
    s0 = c0_full[0::2]
    s1 = c1_full[1::2]
    L = min(len(s0), len(s1))
    out = np.zeros(2 * L, dtype=np.float64)
    out[0::2] = s0[:L]
    out[1::2] = s1[:L]
    return out


def _ptv2_init_identity(conv: nn.Conv1d) -> None:
    taps = int(conv.weight.shape[-1])
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, taps // 2] = 1.0


def _make_full_nyquist_weight(
    *,
    freqs_r: torch.Tensor,
    fs: float,
    band_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    hf_weight: float,
) -> torch.Tensor:
    """
    为“全 Nyquist 提升”构造频域权重：
    - 带内：0..band_hz
    - 镜像带：fs/2 - fmax .. fs/2 - fmin（对应输入频段的 image）
    - 其余频段：仍保留小权重，避免训练“看不见”导致指标劣化
    """
    fs = float(fs)
    nyq = fs / 2.0
    band_hz = float(band_hz)
    fmin_hz = float(fmin_hz)
    fmax_hz = float(fmax_hz)

    # 主带
    m_in = freqs_r <= band_hz

    # 镜像带（与训练 tone 频段对应）
    img_lo = max(0.0, nyq - fmax_hz)
    img_hi = min(nyq, nyq - fmin_hz)
    m_img = (freqs_r >= img_lo) & (freqs_r <= img_hi)

    # 带内高频权重 ramp（0.2~1.0）^hf_weight
    wf = torch.full_like(freqs_r, 0.15)  # 其它频段也给一点权重，避免“放飞”
    if torch.any(m_in):
        u = torch.clamp(freqs_r / max(band_hz, 1.0), 0.0, 1.0)
        wf_in = (0.2 + 0.8 * u) ** float(hf_weight)
        wf = torch.where(m_in, wf_in, wf)

    # 镜像带也给同等量级权重（避免 image/spur 被训练忽略）
    if img_hi > img_lo and torch.any(m_img):
        # 镜像带内部也做一个 ramp（靠近 nyq 处略高）
        u2 = torch.clamp((freqs_r - img_lo) / max(img_hi - img_lo, 1.0), 0.0, 1.0)
        wf_img = (0.25 + 0.75 * u2) ** float(hf_weight)
        wf = torch.where(m_img, wf_img, wf)

    # DC 附近不给权重（避免把 DC 当成优化目标）
    wf = torch.where(freqs_r < 1e6, torch.zeros_like(wf), wf)
    return wf.view(1, 1, -1)


def _apply_ptv2(x: torch.Tensor, *, post_even: nn.Conv1d, post_odd: nn.Conv1d) -> torch.Tensor:
    ye = post_even(x)
    yo = post_odd(x)
    out = ye.clone()
    out[..., 1::2] = yo[..., 1::2]
    return out


def train_post_eq_ptv2_ddsp_multi_tone(
    *,
    simulator: TIADCSimulator,
    model_stage12: HybridCalibrationModel,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    device: str,
    taps: int,
    band_hz: float,
    hf_weight: float,
    ridge: float,
    diff_reg: float,
    steps: int,
    batch: int,
    tone_n: int,
    fmin_hz: float,
    fmax_hz: float,
    spur_guard: int,
    w_time: float,
    w_freq: float,
    w_delta: float,
    w_spur: float,
    w_fund: float,
    w_smooth: float,
) -> Tuple[nn.Conv1d, nn.Conv1d]:
    """
    关键改动点（对应你要的“继续改”）：
    - 训练激励从单一 base_sig 改为 “多单音 batch”（和评估扫频口径一致）
    - 增加 spur/基波保护项，避免 DDSP 为贴波形引入新 spur 导致 SFDR/SINAD 变差
    """
    taps = int(taps)
    if taps % 2 == 0:
        taps += 1

    post_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    post_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    _ptv2_init_identity(post_even)
    _ptv2_init_identity(post_odd)

    opt = optim.Adam(list(post_even.parameters()) + list(post_odd.parameters()), lr=6e-4, betas=(0.5, 0.9))

    fs = float(simulator.fs)
    N = int(tone_n)
    t = torch.arange(N, device=device, dtype=torch.float32) / float(fs)

    # 选择相干采样的 k，使得 fin = k*fs/N
    k_min = int(np.ceil(float(fmin_hz) * N / fs))
    k_max = int(np.floor(float(fmax_hz) * N / fs))
    k_min = max(k_min, 8)
    k_max = min(k_max, N // 2 - 8)

    freqs_r = torch.fft.rfftfreq(N, d=1.0 / fs).to(device)
    wf = _make_full_nyquist_weight(
        freqs_r=freqs_r,
        fs=fs,
        band_hz=float(band_hz),
        fmin_hz=float(fmin_hz),
        fmax_hz=float(fmax_hz),
        hf_weight=float(hf_weight),
    )

    eps = 1e-12

    for step in range(int(steps)):
        opt.zero_grad()

        loss_acc = 0.0
        for _ in range(int(batch)):
            k = np.random.randint(k_min, k_max + 1)
            fin = float(k) * fs / float(N)

            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            # 重要：训练目标与输入必须同等噪声/量化口径（避免“去噪/反量化”导致反作用）
            y0 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch0)
            y1 = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ch1)
            yr = simulator.apply_channel_effect(src_np, jitter_std=100e-15, n_bits=12, **params_ref)

            # Stage1/2 校正 Ch1（注意归一化尺度用 y0，避免数值漂）
            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1_cal = model_stage12(x1).detach().cpu().numpy().flatten() * s12

            y_in = tiadc_interleave_from_fullrate(y0, y1_cal)
            y_tgt = yr[: len(y_in)]

            s = max(float(np.max(np.abs(y_tgt))), 1e-12)
            x_in = torch.FloatTensor(y_in / s).view(1, 1, -1).to(device)
            x_tgt = torch.FloatTensor(y_tgt / s).view(1, 1, -1).to(device)

            y_hat = _apply_ptv2(x_in, post_even=post_even, post_odd=post_odd)

            # === 全 Nyquist 口径：直接在全频优化 ===
            # 1) 时域贴合（全频）
            loss_time = torch.mean((y_hat - x_tgt) ** 2)

            # 2) 频域贴合（复杂谱，权重覆盖：带内 + 镜像带 + 其它频段小权重）
            Yh = torch.fft.rfft(y_hat, dim=-1, norm="ortho")
            Yt = torch.fft.rfft(x_tgt, dim=-1, norm="ortho")
            loss_freq = torch.mean(torch.abs((Yh - Yt) * wf) ** 2)

            # 3) 最小改动（全频）：避免 Post‑EQ 把系统“改形”引入新 spur
            loss_delta = torch.mean((y_hat - x_in) ** 2)

            # 4) spur 约束（类似“最大杂散/噪声能量”压制，保护 SFDR/SINAD）
            #    因为是相干单音，不需要窗；这里直接用谱能量比做 differentiable proxy
            Y = torch.fft.rfft(y_hat, dim=-1, norm="ortho")
            P = (Y.real**2 + Y.imag**2)
            k0 = int(k)
            s0 = max(0, k0 - int(spur_guard))
            e0 = min(P.shape[-1], k0 + int(spur_guard) + 1)
            fund = torch.sum(P[..., s0:e0], dim=-1) + eps

            mask = torch.ones_like(P, dtype=torch.bool)
            mask[..., :5] = False  # DC 附近
            mask[..., s0:e0] = False
            spur = torch.sum(P[mask]) + eps
            loss_spur = spur / torch.sum(fund)

            # 5) 基波保护（避免压基波或相位乱跑）
            Tspec = torch.fft.rfft(x_tgt, dim=-1, norm="ortho")
            Pt = (Tspec.real**2 + Tspec.imag**2)
            fund_t = torch.sum(Pt[..., s0:e0], dim=-1) + eps
            loss_fund = torch.mean((torch.log(fund) - torch.log(fund_t)) ** 2)

            # 6) FIR 平滑 + even/odd 相似性
            we = post_even.weight.view(-1)
            wo = post_odd.weight.view(-1)
            loss_smooth = torch.mean((we[1:] - we[:-1]) ** 2) + torch.mean((wo[1:] - wo[:-1]) ** 2)
            loss_diff = torch.mean((we - wo) ** 2)
            loss_ridge = torch.mean(we**2) + torch.mean(wo**2)

            loss_one = (
                float(w_time) * loss_time
                + float(w_freq) * loss_freq
                + float(w_delta) * loss_delta
                + float(w_spur) * loss_spur
                + float(w_fund) * loss_fund
                + float(w_smooth) * loss_smooth
                + float(diff_reg) * loss_diff
                + float(ridge) * loss_ridge
            )
            loss_acc = loss_acc + loss_one

        loss = loss_acc / float(max(int(batch), 1))
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"PostEQ(PTV2-DDSP) step {step:4d}/{int(steps)} | loss={loss.item():.3e}")

    return post_even, post_odd


def apply_post_eq_fir_ptv2(sig: np.ndarray, *, post_fir_even: nn.Module, post_fir_odd: nn.Module, device: str) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    s = max(float(np.max(np.abs(sig))), 1e-12)
    with torch.no_grad():
        x = torch.FloatTensor(sig / s).view(1, 1, -1).to(device)
        ye = post_fir_even(x).cpu().numpy().flatten()
        yo = post_fir_odd(x).cpu().numpy().flatten()
    out = ye.copy()
    out[1::2] = yo[1::2]
    return out * s


# ==============================================================================
# 4) 训练流程：Stage1/2 + Post‑EQ(PTV2‑DDSP)
# ==============================================================================
# 物理通道（待校准）- 更贴近实采数据的参数设置
# 带宽明显高于测试上限，避免滤波器衰减影响
# **关键修改**：两个通道都有适度非线性，差异合理（2-3倍）
# Post-Linear校正线性失配，Post-NL校正非线性失配
CH0 = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 3e-3, "hd3": 2e-3}
CH1 = {"cutoff_freq": 7.8e9, "delay_samples": 0.25, "gain": 0.98, "hd2": 8e-3, "hd3": 6e-3}  # 2.7x/3x差异

# 参考仪器（target）
REF_SNR_DB = 140.0

# Post‑EQ 配置（默认使用 DDSP 训练版本）
POST_EQ_MODE = "ptv2_ddsp"
POST_EQ_TAPS = 127
POST_EQ_BAND_HZ = 6.2e9
POST_EQ_HF_WEIGHT = 1.8
POST_EQ_RIDGE = 2e-3
POST_EQ_PTV2_DIFF = 2e-2

POST_EQ_TRAIN_STEPS = 1200
POST_EQ_BATCH = 6
POST_EQ_TONE_N = 8192
POST_EQ_FMIN_HZ = 0.2e9
POST_EQ_FMAX_HZ = 6.2e9
POST_EQ_SPUR_GUARD = 3

POST_EQ_W_TIME = 25.0
POST_EQ_W_FREQ = 25.0
POST_EQ_W_DELTA = 35.0
POST_EQ_W_SPUR = 2.0
POST_EQ_W_FUND = 2.0
POST_EQ_W_SMOOTH = 5e-4

# 环境变量覆盖
POST_EQ_MODE = _env_str("POST_EQ_MODE", POST_EQ_MODE)
POST_EQ_TAPS = _env_int("POST_EQ_TAPS", POST_EQ_TAPS)
POST_EQ_BAND_HZ = _env_float("POST_EQ_BAND_HZ", POST_EQ_BAND_HZ)
POST_EQ_HF_WEIGHT = _env_float("POST_EQ_HF_WEIGHT", POST_EQ_HF_WEIGHT)
POST_EQ_RIDGE = _env_float("POST_EQ_RIDGE", POST_EQ_RIDGE)
POST_EQ_PTV2_DIFF = _env_float("POST_EQ_PTV2_DIFF", POST_EQ_PTV2_DIFF)

POST_EQ_TRAIN_STEPS = _env_int("POST_EQ_TRAIN_STEPS", POST_EQ_TRAIN_STEPS)
POST_EQ_BATCH = _env_int("POST_EQ_BATCH", POST_EQ_BATCH)
POST_EQ_TONE_N = _env_int("POST_EQ_TONE_N", POST_EQ_TONE_N)
POST_EQ_FMIN_HZ = _env_float("POST_EQ_FMIN_HZ", POST_EQ_FMIN_HZ)
POST_EQ_FMAX_HZ = _env_float("POST_EQ_FMAX_HZ", POST_EQ_FMAX_HZ)
POST_EQ_SPUR_GUARD = _env_int("POST_EQ_SPUR_GUARD", POST_EQ_SPUR_GUARD)

POST_EQ_W_TIME = _env_float("POST_EQ_W_TIME", POST_EQ_W_TIME)
POST_EQ_W_FREQ = _env_float("POST_EQ_W_FREQ", POST_EQ_W_FREQ)
POST_EQ_W_DELTA = _env_float("POST_EQ_W_DELTA", POST_EQ_W_DELTA)
POST_EQ_W_SPUR = _env_float("POST_EQ_W_SPUR", POST_EQ_W_SPUR)
POST_EQ_W_FUND = _env_float("POST_EQ_W_FUND", POST_EQ_W_FUND)
POST_EQ_W_SMOOTH = _env_float("POST_EQ_W_SMOOTH", POST_EQ_W_SMOOTH)


def train_relative_calibration(sim: TIADCSimulator, *, device: str) -> tuple[HybridCalibrationModel, float, dict, dict, dict]:
    """Stage1/2 训练 + Post‑EQ(PTV2‑DDSP) 训练，并把 post_fir_even/odd 挂到 model 上。"""
    np.random.seed(42)
    torch.manual_seed(42)

    # 训练数据：带限噪声（只是给 Stage1/2 用，Post‑EQ 训练会用多单音）
    N_train = 32768
    M_average = 16
    white = np.random.randn(N_train)
    # 训练信号带宽限制在7GHz（略高于测试范围6.5GHz）
    b_src, a_src = signal.butter(6, 7.0e9 / (sim.fs / 2), btype="low")
    base_sig = signal.lfilter(b_src, a_src, white)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9

    ch0_caps = []
    ch1_caps = []
    for _ in range(M_average):
        ch0_caps.append(sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **CH0))
        ch1_caps.append(sim.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **CH1))
    avg_ch0 = np.mean(np.stack(ch0_caps), axis=0)
    avg_ch1 = np.mean(np.stack(ch1_caps), axis=0)

    scale = max(float(np.max(np.abs(avg_ch0))), 1e-12)
    inp = torch.FloatTensor(avg_ch1 / scale).view(1, 1, -1).to(device)
    tgt = torch.FloatTensor(avg_ch0 / scale).view(1, 1, -1).to(device)

    model = HybridCalibrationModel(taps=63).to(device)
    loss_fn = ComplexMSELoss()

    print("=== Stage 1: Relative Delay & Gain ===")
    opt1 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 1e-2},
            {"params": model.gain, "lr": 1e-2},
            {"params": model.conv.parameters(), "lr": 0.0},
        ]
    )
    for _ in range(301):
        opt1.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt1.step()

    print("=== Stage 2: Relative FIR ===")
    opt2 = optim.Adam(
        [
            {"params": model.global_delay.parameters(), "lr": 0.0},
            {"params": model.gain, "lr": 0.0},
            {"params": model.conv.parameters(), "lr": 5e-4},
        ],
        betas=(0.5, 0.9),
    )
    for _ in range(1001):
        opt2.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt2.step()

    # Post‑EQ：参考驱动 PTV2‑DDSP（只保留这一条主流程）
    params_ref = {
        "cutoff_freq": float(CH0["cutoff_freq"]),
        "delay_samples": 0.0,
        "gain": 1.0,
        "hd2": 0.0,
        "hd3": 0.0,
        # 训练目标与输入必须同等噪声口径；不让模型“去噪/反量化”
        "snr_target": None,
    }

    if str(POST_EQ_MODE).strip().lower() != "ptv2_ddsp":
        raise ValueError("精简版脚本只保留 POST_EQ_MODE='ptv2_ddsp'")

    print("\n=== Post‑EQ: PTV2‑DDSP (multi-tone + spur/fund protection) ===")
    post_even, post_odd = train_post_eq_ptv2_ddsp_multi_tone(
        simulator=sim,
        model_stage12=model,
        params_ch0=CH0,
        params_ch1=CH1,
        params_ref=params_ref,
        device=device,
        taps=int(POST_EQ_TAPS),
        band_hz=float(POST_EQ_BAND_HZ),
        hf_weight=float(POST_EQ_HF_WEIGHT),
        ridge=float(POST_EQ_RIDGE),
        diff_reg=float(POST_EQ_PTV2_DIFF),
        steps=int(POST_EQ_TRAIN_STEPS),
        batch=int(POST_EQ_BATCH),
        tone_n=int(POST_EQ_TONE_N),
        fmin_hz=float(POST_EQ_FMIN_HZ),
        fmax_hz=float(POST_EQ_FMAX_HZ),
        spur_guard=int(POST_EQ_SPUR_GUARD),
        w_time=float(POST_EQ_W_TIME),
        w_freq=float(POST_EQ_W_FREQ),
        w_delta=float(POST_EQ_W_DELTA),
        w_spur=float(POST_EQ_W_SPUR),
        w_fund=float(POST_EQ_W_FUND),
        w_smooth=float(POST_EQ_W_SMOOTH),
    )
    model.post_fir_even = post_even
    model.post_fir_odd = post_odd
    model.reference_params = params_ref

    return model, float(scale), CH0, CH1, params_ref


# ==============================================================================
# 5) 指标计算与绘图（只保留对比口径：Pre / Post-Linear / Post-EQ）
# ==============================================================================
def calculate_metrics_detailed(sim: TIADCSimulator, model: HybridCalibrationModel, scale: float, device: str) -> tuple[np.ndarray, dict]:
    print("\n=== 正在计算综合指标对比 (TIADC 交织输出) ===")

    # 限制测试频率在有效输入带宽内（避免超过滤波器截止频率导致性能崩溃）
    # 对于6GHz带宽限制，测试到6.5GHz已经足够展示性能
    test_freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    metrics = {
        "sinad_pre": [],
        "sinad_post_lin": [],
        "sinad_post_eq": [],
        "enob_pre": [],
        "enob_post_lin": [],
        "enob_post_eq": [],
        "thd_pre": [],
        "thd_post_lin": [],
        "thd_post_eq": [],
        "sfdr_pre": [],
        "sfdr_post_lin": [],
        "sfdr_post_eq": [],
    }

    def _interleave(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
        s0 = c0_full[0::2]
        s1 = c1_full[1::2]
        L = min(len(s0), len(s1))
        out = np.zeros(2 * L, dtype=np.float64)
        out[0::2] = s0[:L]
        out[1::2] = s1[:L]
        return out

    def _spec_metrics(sig: np.ndarray, fs: float, fin: float) -> tuple[float, float, float, float]:
        sig = np.asarray(sig, dtype=np.float64)
        n = len(sig)
        win = np.blackman(n)
        cg = float(np.mean(win))
        S = np.fft.rfft(sig * win)
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

        # THD：2~5 次谐波折叠
        p_h = 0.0
        for h in range(2, 6):
            hf = (fin * h) % fs
            if hf > fs / 2:
                hf = fs - hf
            if hf < 1e6 or hf > fs / 2 - 1e6:
                continue
            k = int(np.argmin(np.abs(freqs - hf)))
            ss = max(0, k - span)
            ee = min(len(mag), k + span + 1)
            kk = ss + int(np.argmax(mag[ss:ee]))
            p_h += float(mag[kk] ** 2)
        thd = 10.0 * np.log10((p_h + 1e-30) / p_fund)
        return float(sinad), float(enob), float(thd), float(sfdr)

    for f in test_freqs:
        src = np.sin(2 * np.pi * f * (np.arange(16384) / sim.fs)) * 0.9
        # 验证口径必须与训练一致：使用真实 ADC 条件（噪声+量化）
        y0 = sim.apply_channel_effect(src, jitter_std=100e-15, n_bits=12, **CH0)
        y1 = sim.apply_channel_effect(src, jitter_std=100e-15, n_bits=12, **CH1)

        with torch.no_grad():
            x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
            y1_cal = model(x1).cpu().numpy().flatten() * float(scale)

        margin = 500
        c0 = y0[margin:-margin]
        c1_raw = y1[margin:-margin]
        c1_cal = y1_cal[margin:-margin]

        pre = _interleave(c0, c1_raw)
        post_lin = _interleave(c0, c1_cal)
        post_eq = apply_post_eq_fir_ptv2(post_lin, post_fir_even=model.post_fir_even, post_fir_odd=model.post_fir_odd, device=device)

        s0, e0, t0, sf0 = _spec_metrics(pre, sim.fs, f)
        s1, e1, t1, sf1 = _spec_metrics(post_lin, sim.fs, f)
        s2, e2, t2, sf2 = _spec_metrics(post_eq, sim.fs, f)

        metrics["sinad_pre"].append(s0)
        metrics["sinad_post_lin"].append(s1)
        metrics["sinad_post_eq"].append(s2)
        metrics["enob_pre"].append(e0)
        metrics["enob_post_lin"].append(e1)
        metrics["enob_post_eq"].append(e2)
        metrics["thd_pre"].append(t0)
        metrics["thd_post_lin"].append(t1)
        metrics["thd_post_eq"].append(t2)
        metrics["sfdr_pre"].append(sf0)
        metrics["sfdr_post_lin"].append(sf1)
        metrics["sfdr_post_eq"].append(sf2)

    return test_freqs, metrics


def plot_metrics(test_freqs: np.ndarray, m: dict) -> None:
    fghz = test_freqs / 1e9
    plt.figure(figsize=(10, 13))

    plt.subplot(4, 1, 1)
    plt.plot(fghz, m["sinad_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sinad_post_lin"], "g-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sinad_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("SINAD Improvement")
    plt.ylabel("dB")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["enob_post_lin"], "m-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["enob_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("ENOB Improvement")
    plt.ylabel("Bits")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["thd_post_lin"], "b-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["thd_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("THD Comparison")
    plt.ylabel("dBc")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], "r--o", alpha=0.5, label="Pre-Calib")
    plt.plot(fghz, m["sfdr_post_lin"], "c-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sfdr_post_eq"], color="#ff7f0e", marker="o", linewidth=2, label="Post-EQ (PTV2-DDSP)")
    plt.title("SFDR Improvement")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("dBc")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_current_figure("metrics_vs_freq")


# ==============================================================================
# Main
# ==============================================================================
if False and __name__ == "__main__":  # legacy入口：已停用（仅保留文件历史，不参与运行）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== 运行配置摘要（精简版：Stage1/2 + Post‑EQ‑DDSP）===")
    print(f"POST_EQ_MODE={POST_EQ_MODE} | taps={POST_EQ_TAPS} | band={POST_EQ_BAND_HZ/1e9:.2f}GHz | steps={POST_EQ_TRAIN_STEPS} | batch={POST_EQ_BATCH} | N={POST_EQ_TONE_N}")

    sim = TIADCSimulator(fs=20e9)
    model, scale, _, _, _ = train_relative_calibration(sim, device=device)
    test_freqs, metrics = calculate_metrics_detailed(sim, model, scale, device)
    plot_metrics(test_freqs, metrics)
    # 重要：文件历史原因曾被误拼接旧代码；这里强制结束并截断文件内容。
    # 文件历史原因：曾经把旧版大脚本内容拼接到了本文件后面。
    # 这里强制结束，避免执行到任何遗留逻辑；同时本文件应在此处结束。
    raise SystemExit(0)


def design_post_eq_fir_wiener(
    *,
    y: np.ndarray,
    x_target: np.ndarray,
    fs_hz: float,
    taps: int,
    band_hz: float,
    hf_weight: float,
    ridge: float,
) -> np.ndarray:
    """
    设计交织后 Post-EQ FIR（维纳反卷积）。

    目标：找 g，使得 g * y ≈ x_target（带内）。

    参数：
    - y: 交织后的 TIADC 输出（20GSps）
    - x_target: “理想恒等”目标（建议先做同带宽低通，避免带外反卷积）
    - band_hz: 仅在该带宽内施加恒等目标（带外权重为 0）
    - hf_weight: 带内频率权重指数，>1 更偏向高频幅相贴合
    - ridge: 反卷积正则项（越大越保守，越不容易噪声放大）

    返回：
    - g: 时域 FIR（长度 taps，中心对齐；离线“非因果”意义下更适合恒等相位对齐）
    """
    y = np.asarray(y, dtype=np.float64)
    x_target = np.asarray(x_target, dtype=np.float64)
    n = min(len(y), len(x_target))
    y = y[:n]
    x_target = x_target[:n]

    taps = int(taps)
    if taps % 2 == 0:
        taps += 1

    win = np.hanning(n).astype(np.float64)
    Y = np.fft.rfft(y * win, norm="ortho")
    X = np.fft.rfft(x_target * win, norm="ortho")
    Pyy = (Y.real**2 + Y.imag**2).astype(np.float64)

    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))
    band = freqs <= float(band_hz)
    wf = np.zeros_like(freqs, dtype=np.float64)
    if np.any(band):
        u = freqs[band] / max(float(band_hz), 1.0)
        wf[band] = (0.2 + 0.8 * u) ** float(hf_weight)

    # Wiener: G = X*conj(Y) / (|Y|^2 + ridge)，带权重 wf（带外=0）
    G = (X * np.conj(Y) * wf) / (Pyy + float(ridge))
    g_full = np.fft.irfft(G, n=n, norm="ortho")

    # 让最大峰对齐到中心，截断 taps；并用窗函数缓解截断振铃
    k = int(np.argmax(np.abs(g_full)))
    g_roll = np.roll(g_full, -k)
    g = g_roll[:taps].copy()
    g = np.roll(g, taps // 2)
    g = g * np.hanning(taps)
    return g.astype(np.float64, copy=False)


def design_post_eq_fir_ptv2_ridge(
    *,
    y: np.ndarray,
    x_target: np.ndarray,
    fs_hz: float,
    band_hz: float,
    taps: int,
    ridge: float,
    diff_reg: float = 0.0,
    crop: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    设计交织后 Post-EQ 的 2 相位 PTV(周期=2) FIR：even/odd 两套系数。

    核心思想：
    - TIADC 交织后的残差 spur 很多呈现“周期=2 的时变”特征；
    - 用单一 LTI FIR 很难同时兼顾所有 spur；
    - 这里直接在全速率时域上做岭回归最小二乘：
        out[n] = sum_k g_{parity(n)}[k] * y[n+k]

    参数：
    - y:      交织后的 TIADC 输出（20GSps）
    - x_target: 目标（推荐用 reference instrument 输出）
    - taps:   FIR taps（奇数更便于中心对齐）
    - ridge:  岭回归强度（越大越保守）
    - crop:   去掉两端边界，避免 padding/边缘影响

    返回：
    - g_even, g_odd: 两套时域 FIR 系数（长度 taps）
    """
    y = np.asarray(y, dtype=np.float64)
    x_target = np.asarray(x_target, dtype=np.float64)
    n = min(len(y), len(x_target))
    y = y[:n]
    x_target = x_target[:n]

    taps = int(taps)
    if taps % 2 == 0:
        taps += 1
    pad = taps // 2
    crop = int(max(crop, pad + 2))

    # 只在带内拟合（用同带宽低通做“带内目标投影”），避免 PTV2 在带外乱补导致发散
    if float(band_hz) > 0:
        wn = float(band_hz) / (float(fs_hz) / 2.0)
        wn = float(np.clip(wn, 1e-6, 0.999999))
        b, a = signal.butter(6, wn, btype="low")
        y_fit = signal.lfilter(b, a, y)
        x_fit = signal.lfilter(b, a, x_target)
    else:
        y_fit = y
        x_fit = x_target

    # 滑窗特征矩阵：Phi[t] = y[t-pad : t+pad]
    yp = np.pad(y_fit, (pad, pad), mode="edge")
    Phi = np.lib.stride_tricks.sliding_window_view(yp, taps)  # (n, taps)

    idx = np.arange(crop, n - crop, dtype=np.int64)
    idx_even = idx[(idx % 2) == 0]
    idx_odd = idx[(idx % 2) == 1]

    def _fallback_identity() -> np.ndarray:
        g0 = np.zeros((taps,), dtype=np.float64)
        g0[taps // 2] = 1.0
        return g0

    if (len(idx_even) < taps + 4) or (len(idx_odd) < taps + 4):
        g_even = _fallback_identity()
        g_odd = _fallback_identity()
        return g_even.astype(np.float64, copy=False), g_odd.astype(np.float64, copy=False)

    Xe = Phi[idx_even, :]
    Xo = Phi[idx_odd, :]
    de = x_fit[idx_even]
    do = x_fit[idx_odd]

    Ae = Xe.T @ Xe
    Ao = Xo.T @ Xo
    be = Xe.T @ de
    bo = Xo.T @ do

    r = float(ridge)
    dreg = float(max(diff_reg, 0.0))
    I = np.eye(taps, dtype=np.float64)

    if dreg <= 0.0:
        g_even = np.linalg.solve(Ae + r * I, be)
        g_odd = np.linalg.solve(Ao + r * I, bo)
    else:
        # 联合求解：增加 ||g_even - g_odd||^2 约束，抑制“带外发散”
        # [Ae+(r+d)I, -dI] [ge] = [be]
        # [-dI, Ao+(r+d)I] [go]   [bo]
        Z = -dreg * I
        M = np.block([
            [Ae + (r + dreg) * I, Z],
            [Z, Ao + (r + dreg) * I],
        ])
        bb = np.concatenate([be, bo], axis=0)
        sol = np.linalg.solve(M, bb)
        g_even = sol[:taps]
        g_odd = sol[taps:]

    return g_even.astype(np.float64, copy=False), g_odd.astype(np.float64, copy=False)


def attach_post_eq_fir_ptv2(model: nn.Module, *, g_even: np.ndarray, g_odd: np.ndarray, device: str) -> None:
    """
    把 PTV2 FIR 挂载到模型：
    - model.post_fir_even / model.post_fir_odd: nn.Conv1d
    """
    g_even = np.asarray(g_even, dtype=np.float64)
    g_odd = np.asarray(g_odd, dtype=np.float64)
    taps = int(len(g_even))
    if len(g_odd) != taps:
        raise ValueError("g_even/g_odd taps mismatch")

    model.post_fir_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    model.post_fir_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    with torch.no_grad():
        we = torch.from_numpy(g_even[::-1].astype(np.float32)).view(1, 1, taps).to(device)
        wo = torch.from_numpy(g_odd[::-1].astype(np.float32)).view(1, 1, taps).to(device)
        model.post_fir_even.weight.copy_(we)
        model.post_fir_odd.weight.copy_(wo)


def apply_post_eq_fir_ptv2(sig: np.ndarray, *, post_fir_even: nn.Module, post_fir_odd: nn.Module, device: str) -> np.ndarray:
    """
    应用 PTV2 Post-EQ：
    - 先分别算两套卷积输出 ye/yo
    - 再按奇偶采样点选择拼回全速率输出
    """
    sig = np.asarray(sig, dtype=np.float64)
    s = max(float(np.max(np.abs(sig))), 1e-12)
    with torch.no_grad():
        x = torch.FloatTensor(sig / s).view(1, 1, -1).to(device)
        ye = post_fir_even(x).cpu().numpy().flatten()
        yo = post_fir_odd(x).cpu().numpy().flatten()
    out = ye.copy()
    out[1::2] = yo[1::2]
    return out * s


def attach_post_eq_fir(model: nn.Module, *, g: np.ndarray, device: str) -> None:
    """
    把时域 FIR g 挂载到 model.post_fir（nn.Conv1d），用于验证/推理。

    注意：
    - torch 的 Conv1d 本质是相关运算；这里通过翻转 g 实现“卷积等价”。
    """
    g = np.asarray(g, dtype=np.float64)
    taps = int(len(g))
    model.post_fir = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    with torch.no_grad():
        w = torch.from_numpy(g[::-1].astype(np.float32)).view(1, 1, taps).to(device)
        model.post_fir.weight.copy_(w)


def apply_post_eq_fir(sig: np.ndarray, *, post_fir: nn.Module, device: str) -> np.ndarray:
    """
    应用交织后 Post-EQ FIR。
    - 使用幅度归一化避免数值问题；输出恢复原尺度。
    """
    sig = np.asarray(sig, dtype=np.float64)
    s = max(float(np.max(np.abs(sig))), 1e-12)
    with torch.no_grad():
        x = torch.FloatTensor(sig / s).view(1, 1, -1).to(device)
        y = post_fir(x).cpu().numpy().flatten() * s
    return y


def _ptv2_init_identity(conv: nn.Conv1d) -> None:
    """把 1D 卷积初始化为“直通”FIR（中心 tap=1，其余为 0）。"""
    taps = int(conv.weight.shape[-1])
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, taps // 2] = 1.0


def _lowpass_torch(x: torch.Tensor, *, fs: float, band_hz: float) -> torch.Tensor:
    """
    在 torch 中用 rFFT 做“理想低通”投影（带外清零），用于 Post‑EQ 的带内损失。
    x: (1,1,T)
    """
    if band_hz <= 0:
        return x
    T = int(x.shape[-1])
    X = torch.fft.rfft(x, dim=-1, norm="ortho")
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(fs)).to(x.device)
    mask = (freqs <= float(band_hz)).view(1, 1, -1)
    X = X * mask
    return torch.fft.irfft(X, n=T, dim=-1, norm="ortho")


def train_post_eq_ptv2_ddsp(
    *,
    simulator,
    model_stage12: nn.Module,
    base_sig: np.ndarray,
    scale: float,
    params_ch0: dict,
    params_ch1: dict,
    params_ref: dict,
    device: str,
    taps: int,
    band_hz: float,
    hf_weight: float,
    ridge: float,
    diff_reg: float,
    steps: int,
    lr: float,
    w_time: float,
    w_freq: float,
    w_delta: float,
    w_smooth: float,
) -> tuple[nn.Conv1d, nn.Conv1d]:
    """
    Post‑EQ（PTV2‑DDSP）：训练 even/odd 两套 FIR 去贴 reference target。

    训练口径：
    - 只在带内计算损失（用理想低通投影）
    - 频域用“复杂谱”MSE（保幅相），并给高频更大权重（hf_weight）
    - 加最小改动约束（delta）+ 平滑正则 + even/odd 相似性约束（diff_reg）
    """
    taps = int(taps)
    if taps % 2 == 0:
        taps += 1

    post_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    post_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps // 2, bias=False).to(device)
    _ptv2_init_identity(post_even)
    _ptv2_init_identity(post_odd)

    opt = optim.Adam(list(post_even.parameters()) + list(post_odd.parameters()), lr=float(lr), betas=(0.5, 0.9))

    # 频域权重（只在带内，且高频权重更高）
    T = int(len(base_sig))
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(simulator.fs)).to(device)
    band = freqs <= float(band_hz)
    wf = torch.zeros_like(freqs)
    if torch.any(band):
        u = freqs[band] / max(float(band_hz), 1.0)
        wf[band] = (0.2 + 0.8 * u) ** float(hf_weight)
    wf = wf.view(1, 1, -1)

    # 干净样本（无 jitter/无量化）
    y0 = simulator.apply_channel_effect(base_sig, jitter_std=0, n_bits=None, **params_ch0)
    y1 = simulator.apply_channel_effect(base_sig, jitter_std=0, n_bits=None, **params_ch1)
    yref = simulator.apply_channel_effect(base_sig, jitter_std=0, n_bits=None, **params_ref)

    with torch.no_grad():
        x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
        y1_cal = model_stage12(x1).detach().cpu().numpy().flatten() * float(scale)
    y_in = tiadc_interleave_from_fullrate(y0, y1_cal)
    y_tgt = yref[: len(y_in)]

    s = max(float(np.max(np.abs(y_tgt))), 1e-12)
    x_in = torch.FloatTensor(y_in / s).view(1, 1, -1).to(device)
    x_tgt = torch.FloatTensor(y_tgt / s).view(1, 1, -1).to(device)

    def _apply_ptv2(x: torch.Tensor) -> torch.Tensor:
        ye = post_even(x)
        yo = post_odd(x)
        out = ye.clone()
        out[..., 1::2] = yo[..., 1::2]
        return out

    for step in range(int(steps)):
        opt.zero_grad()
        y_hat = _apply_ptv2(x_in)

        yb = _lowpass_torch(y_hat, fs=float(simulator.fs), band_hz=float(band_hz))
        tb = _lowpass_torch(x_tgt, fs=float(simulator.fs), band_hz=float(band_hz))
        ib = _lowpass_torch(x_in, fs=float(simulator.fs), band_hz=float(band_hz))

        loss_time = torch.mean((yb - tb) ** 2)

        Yh = torch.fft.rfft(yb, dim=-1, norm="ortho")
        Yt = torch.fft.rfft(tb, dim=-1, norm="ortho")
        loss_freq = torch.mean(torch.abs((Yh - Yt) * wf) ** 2)

        loss_delta = torch.mean((yb - ib) ** 2)

        we = post_even.weight.view(-1)
        wo = post_odd.weight.view(-1)
        loss_smooth = torch.mean((we[1:] - we[:-1]) ** 2) + torch.mean((wo[1:] - wo[:-1]) ** 2)
        loss_diff = torch.mean((we - wo) ** 2)
        loss_ridge = torch.mean(we**2) + torch.mean(wo**2)

        loss = (
            float(w_time) * loss_time
            + float(w_freq) * loss_freq
            + float(w_delta) * loss_delta
            + float(w_smooth) * loss_smooth
            + float(diff_reg) * loss_diff
            + float(ridge) * loss_ridge
        )
        loss.backward()
        opt.step()

        if step % 150 == 0:
            with torch.no_grad():
                print(
                    f"PostEQ(PTV2-DDSP) step {step:4d}/{int(steps)} | "
                    f"L={loss.item():.3e} | Lt={loss_time.item():.3e} | Lf={loss_freq.item():.3e} | "
                    f"Ld={loss_delta.item():.3e} | diff={loss_diff.item():.3e}"
                )

    return post_even, post_odd


def build_and_attach_post_eq(
    *,
    simulator,
    model: nn.Module,
    base_sig: np.ndarray,
    scale: float,
    params_ch0: dict,
    params_ch1: dict,
    device: str,
    taps: int,
    band_hz: float,
    hf_weight: float,
    ridge: float,
    use_reference_target: bool,
    reference_snr_db: float,
    mode: str,
) -> None:
    """
    Post-EQ 一站式封装：
    - 构造训练样本（使用无抖动/无量化的通道输出，避免把随机噪声当成可逆）
    - 得到交织输出 y_tiadc（Ch0 + Cal(Ch1)）
    - 构造带宽受限的理想目标 x_target
    - 设计 FIR 并挂载到 model.post_fir
    """
    if use_reference_target:
        print("\n=== Post-EQ: Interleaved output -> Reference Instrument (target) ===")
    else:
        print("\n=== Post-EQ: Interleaved output -> Unit-Impulse (Identity) ===")

    # 1) Post-QE（物理信息静态逆，多项式）训练口径修正：
    # - 必须与 Stage3/验证一致：使用“真实 ADC 条件”（噪声+量化），并用相干单音集合训练
    # - 目标必须是“同口径 reference”（同噪声/量化），避免学到“去噪/反量化”的坏解
    params_ref = {
        "cutoff_freq": float(params_ch0["cutoff_freq"]),
        "delay_samples": 0.0,
        "gain": 1.0,
        "hd2": 0.0,
        "hd3": 0.0,
        "snr_target": None,
    }
    # 让指标绘图里也能看到 reference baseline（与 Stage3 复用）
    model.reference_params = params_ref

    # 2) 用“物理信息静态逆模型层”替代 Post‑EQ（命名为 Post‑QE）
    #
    # 说明：
    # - 原 Post‑EQ 是“交织后线性 FIR 反卷积/均衡”，属于黑盒/信号处理求解。
    # - 这里改成显式的多项式逆模型层，只学习少量具有物理意义的系数（w1/w2/w3...）。
    # - 该层默认 PTV（even/odd 两套系数），更符合 TIADC 子通道差异来源。

    # 清理旧字段，避免口径混淆
    model.post_fir = None
    model.post_fir_even = None
    model.post_fir_odd = None
    model.post_fir_meta = None

    # 训练配置（默认更保守；可用环境变量覆盖）
    qe_order = _env_int("POST_QE_ORDER", 3)
    qe_steps = _env_int("POST_QE_STEPS", 450)
    qe_lr = _env_float("POST_QE_LR", 5e-4)
    qe_batch = _env_int("POST_QE_BATCH", 6)
    qe_tone_n = _env_int("POST_QE_TONE_N", 8192)
    qe_fmin_hz = _env_float("POST_QE_FMIN_HZ", 0.2e9)
    qe_fmax_hz = _env_float("POST_QE_FMAX_HZ", 6.2e9)
    qe_guard = _env_int("POST_QE_GUARD", 3)
    qe_w_time = _env_float("POST_QE_W_TIME", 10.0)
    qe_w_harm = _env_float("POST_QE_W_HARM", 450.0)
    qe_w_fund = _env_float("POST_QE_W_FUND", 8.0)
    qe_w_delta = _env_float("POST_QE_W_DELTA", 25.0)
    qe_ridge = _env_float("POST_QE_RIDGE", 2e-4)

    post_qe = PhysicalNonlinearityLayer(order=int(qe_order), ptv=True).to(device)
    opt = optim.Adam(post_qe.parameters(), lr=float(qe_lr), betas=(0.5, 0.9))

    fs = float(simulator.fs)
    N = int(qe_tone_n)
    t = torch.arange(N, device=device, dtype=torch.float32) / float(fs)
    k_min = max(int(np.ceil(float(qe_fmin_hz) * N / fs)), 8)
    k_max = min(int(np.floor(float(qe_fmax_hz) * N / fs)), N // 2 - 8)

    print(f">> Post-QE(Physics-Poly) training: order={qe_order} steps={qe_steps} lr={qe_lr:g} batch={qe_batch} N={qe_tone_n} guard={qe_guard}")
    for step in range(int(qe_steps)):
        loss_acc = 0.0
        for _ in range(int(qe_batch)):
            k = int(np.random.randint(k_min, k_max + 1))
            fin = float(k) * fs / float(N)
            amp = float(np.random.uniform(0.4, 0.95))
            src = torch.sin(2.0 * np.pi * fin * t) * amp
            src_np = src.detach().cpu().numpy().astype(np.float64)

            # 真实口径：噪声+量化（与验证一致）
            y0 = simulator.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0)
            y1 = simulator.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1)
            yr = simulator.apply_channel_effect(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ref)

            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                y1_lin = model(x1).cpu().numpy().flatten() * scale

            tiadc_in = tiadc_interleave_from_fullrate(y0, y1_lin)
            target = yr[: len(tiadc_in)] if bool(use_reference_target) else tiadc_interleave_from_fullrate(y0, y0)[: len(tiadc_in)]

            s0 = max(float(np.max(np.abs(tiadc_in))), float(np.max(np.abs(target))), 1e-12)
            x_in = torch.FloatTensor((tiadc_in / s0)).view(1, 1, -1).to(device)
            y_tgt = torch.FloatTensor((target / s0)).view(1, 1, -1).to(device)
            x_in = torch.clamp(x_in, -1.0, 1.0)
            y_tgt = torch.clamp(y_tgt, -1.0, 1.0)

            y_hat = post_qe(x_in)

            # 以“残差谐波”为主目标，避免为了压谐波去扭曲基波/全带
            residual = y_hat - y_tgt
            loss_harm = residual_harmonic_loss_torch(residual, y_tgt, int(k), max_h=5, guard=int(qe_guard))

            # 基波保护
            win = torch.hann_window(N, device=device, dtype=x_in.dtype).view(1, 1, -1)
            Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
            Yr = torch.fft.rfft(y_tgt * win, dim=-1, norm="ortho")
            fund_mag_hat = torch.abs(Yh[..., int(k)]) + 1e-12
            fund_mag_ref = torch.abs(Yr[..., int(k)]) + 1e-12
            loss_fund = torch.mean((torch.log10(fund_mag_hat) - torch.log10(fund_mag_ref)) ** 2)

            # 最小改动 + 小权重时域贴合
            loss_delta = torch.mean((y_hat - x_in) ** 2)
            loss_time = torch.mean((y_hat - y_tgt) ** 2)

            # 正则（raw 参数 L2，越保守越不易出坏解）
            reg = 0.0
            for p in post_qe.parameters():
                reg = reg + torch.mean(p ** 2)

            loss = (
                float(qe_w_time) * loss_time
                + float(qe_w_harm) * loss_harm
                + float(qe_w_fund) * loss_fund
                + float(qe_w_delta) * loss_delta
                + float(qe_ridge) * reg
            )
            loss_acc = loss_acc + loss

        loss_acc = loss_acc / float(max(int(qe_batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_qe.parameters(), max_norm=1.0)
        opt.step()

        if step % 150 == 0:
            print(f"Post-QE(Physics-Poly) step {step:4d}/{int(qe_steps)} | loss={loss_acc.item():.3e}")

    # 挂载到模型上，供验证/绘图使用
    model.post_qe = post_qe
    model.post_qe_meta = {
        "type": "physics_poly_inverse",
        "order": int(qe_order),
        "steps": int(qe_steps),
        "lr": float(qe_lr),
        "batch": int(qe_batch),
        "tone_n": int(qe_tone_n),
        "ridge": float(qe_ridge),
        "use_reference_target": bool(use_reference_target),
    }
    print(">> Post-QE(Physics-Poly) ready.")

# Windows 控制台常见编码问题（不影响算法，只为避免乱码）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 锁定随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 运行配置（v0119_v2）
# ==============================================================================
# 说明：
# - Stage1/2：相对线性校准（Delay/Gain + FIR），把 Ch1 贴到 Ch0
# - Stage3：非线性误差“修整”（对 TIADC 交织输出做 Volterra/记忆多项式后校正）
#   这一阶段若仍用 Ch0 当 target，则 THD 的上限会被 Ch0 自身 hd2/hd3 锁死；
#   因此默认使用“参考仪器”仿真数据作为 target（更贴近现实用高性能仪器做 foreground calibration）
ENABLE_STAGE3_NONLINEAR = True  # 需要对比 Post-NL 时建议开启；如只看线性校准可关掉
STAGE3_USE_REFERENCE_TARGET = True   # True：用“参考仪器”作为上限目标；False：会被 Ch0 非线性锁死上限
STAGE3_REFERENCE_SNR_DB = 120.0

# Stage3 非线性模型选择（用于寻找“最好方案”做 A/B）
# - ptv_poly:      PTV 记忆无关多项式（每通道 even/odd 独立 a2/a3），最轻量、最易上 FPGA
# - ptv_volterra:  PTV Volterra/Hammerstein（x^2/x^3 + 每通道独立 FIR taps），能力更强
#
# [新增] 更稳的求解方式：在 *_ls 模式下，用岭回归一次性解出最优参数（避免 Adam 训练把指标拉坏）
# - ptv_poly_ls / ptv_volterra_ls: 推荐优先试（通常更稳、更不容易引入新 spur）
STAGE3_SCHEME = "pinn_mem_poly"  # 物理信息版：PTV 记忆多项式（可微分 + 少量可解释参数）
STAGE3_SWEEP = False            # True 时：自动对比多种 scheme，选最优挂载
STAGE3_SWEEP_SCHEMES = ("pinn_mem_poly", "odd_poly_ls", "odd_volterra_ls", "ptv_poly_ls", "ptv_volterra_ls", "ptv_poly", "ptv_volterra")
STAGE3_SWEEP_STEPS = 350        # sweep 时每个 scheme 的训练步数（快速找方向）

# [新增] 自动搜索：在多个“目标定义 + 模型结构 + taps + 岭回归强度”上自动挑最优（推荐开启后跑一次）
STAGE3_AUTOSEARCH = False
STAGE3_AUTOSEARCH_MAX_TRIALS = 36          # 最大尝试次数（控制总耗时）
STAGE3_AUTOSEARCH_EVAL_FREQS_GHZ = (0.5, 1.6, 2.1, 3.6, 5.1, 6.1)  # 快速评估的代表频点（覆盖中频与高频）
STAGE3_AUTOSEARCH_FINE_FREQS_GHZ = (0.5, 1.0, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1, 5.6, 6.1)
STAGE3_AUTOSEARCH_EVAL_N_COARSE = 8192
STAGE3_AUTOSEARCH_EVAL_N_FINE = 16384
STAGE3_AUTOSEARCH_FINE_TOPK = 4            # 粗筛后保留多少个候选进入精搜
STAGE3_AUTOSEARCH_THD_WORSEN_MAX_DB = 0.3  # 硬约束：任意评估频点 THD 变差不得超过该阈值（dB）
STAGE3_AUTOSEARCH_TAPS_LIST = (5, 7, 9, 11, 15, 21)  # 只对 *volterra 生效（odd taps 更稳）
STAGE3_AUTOSEARCH_RIDGE_LIST = (5e-1, 3e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3)  # 岭回归强度（越大越保守）
STAGE3_AUTOSEARCH_LS_ROUNDS = (1, 2, 3)    # *_ls 的“轮数”
STAGE3_AUTOSEARCH_HARM_WDELTA_LIST = (5.0, 10.0, 20.0, 40.0, 80.0)  # 仅对 *_harm_ls 生效

# ------------------------------------------------------------------------------
# [新方案] 交织后“单位冲激/恒等”校准（替代 Stage3 非线性残差修整）
# - 先用 Stage1/2 做相对对齐（Ch1->Ch0）
# - 再在全速率交织输出上训练/求解一个 LTI FIR，让整体响应尽量逼近“恒等系统”
#   （等价于冲激响应逼近单位冲激，且相位/群延时对齐）
# ------------------------------------------------------------------------------
ENABLE_POST_EQ = True
POST_EQ_TAPS = 127
POST_EQ_RIDGE = 2e-3            # 维纳反卷积正则（越大越保守）
POST_EQ_BAND_HZ = 6.2e9         # 只在该带宽内做“恒等”目标，带外不强行反卷积
POST_EQ_HF_WEIGHT = 1.8         # 高频权重：>1 更偏向高频幅相贴合
# [新增] Post-EQ 的结构：
# - "lti"      : 单一 LTI FIR（稳，但对“周期=2 spur”能力有限）
# - "ptv2"     : 2 相位 PTV（even/odd 两套 FIR），闭式岭回归一次性求解
# - "ptv2_ddsp": 2 相位 PTV（even/odd 两套 FIR），用 DDSP/Adam 训练去贴 reference target（更灵活，但需强约束）
POST_EQ_MODE = "ptv2_ddsp"
# [新增] Post-EQ 的训练目标：是否使用“参考仪器输出”作为 target
# - False: target = 低通后的 base_sig（恒等/单位冲激口径）
# - True : target = reference instrument capture（更像 foreground calibration）
#
# 注意：如果 reference target 的噪声/量化口径比输入更“干净”，会诱导 Post‑EQ 学“去噪/反量化”，
# 在真实噪声条件下往往表现为“没收益甚至反作用”。因此默认关闭 reference target。
POST_EQ_USE_REFERENCE_TARGET = False
POST_EQ_REFERENCE_SNR_DB = 120.0  # 仅用于记录/对照，不作为训练目标的噪声口径
# [新增] PTV2 约束：even/odd 两套 FIR 不要差太多（抑制带外“发散”）
POST_EQ_PTV2_DIFF = 2e-2
# [新增] PTV2-DDSP 训练配置（只对 POST_EQ_MODE="ptv2_ddsp" 生效）
POST_EQ_TRAIN_STEPS = 900
POST_EQ_TRAIN_LR = 6e-4
POST_EQ_W_TIME = 60.0        # 带内时域 MSE
POST_EQ_W_FREQ = 40.0        # 带内频域（复杂谱）MSE（带高频权重）
POST_EQ_W_DELTA = 25.0       # 最小改动约束：限制 y_hat 相对 y_in 的能量变化，避免“乱补”
POST_EQ_W_SMOOTH = 5e-4      # FIR 平滑正则：邻差平方

STAGE3_TONE_N = 16384          # FFT 长度（建议 2^n，便于相干采样）
STAGE3_BATCH = 9               # 每步 tone 个数（将按低/中/高分层采样）
STAGE3_STEPS = 1200            # 训练步数（THD 优先，但避免过拟合）
STAGE3_LR = 4e-4               # 学习率（回到更稳的区间）
STAGE3_TAPS = 21               # Volterra 卷积 taps（越大越能拟合记忆效应，但更难训）
STAGE3_W_TIME = 40.0           # 时域约束（保证主信号不被拉坏）
STAGE3_W_HARM = 600.0          # 谐波惩罚（对残差谐波下手，力度适中）
STAGE3_W_FUND = 6.0            # 基波幅度约束（避免压基波）
STAGE3_W_REG = 2e-4            # 权重正则
STAGE3_W_DELTA = 8.0           # 最小改动约束：限制 y_hat-x 的能量，避免引入新失真
STAGE3_HARM_GUARD = 3          # 谐波 bin 周围保护带（抗窗泄漏）

# [新增] Stage3 最小二乘/岭回归求解配置（用于 *_ls）
STAGE3_LS_RIDGE = 2e-2         # 岭回归强度：越大越保守，越不容易引入新 spur
STAGE3_LS_CROP = 300           # 去掉两端边界（避免卷积 padding/边界影响）
STAGE3_LS_TONE_BATCH = 6       # 每轮用多少个 tone 累积法方程（太大易慢）

# [新增] 更“对症”的频域（谐波 bin）最小二乘：直接压谐波（推荐）
STAGE3_HARM_LS_MAX_H = 5
STAGE3_HARM_LS_GUARD = 2        # 每个谐波 bin 左右扩展，抗窗泄漏
STAGE3_HARM_LS_INCLUDE_FUND = False  # 一般关闭，避免动基波幅度
STAGE3_HARM_LS_W_DELTA = 20.0   # “最小改动”强度：越大越保守，越不易引入新 spur

# 训练频点偏置：THD 优先时，更多采样中频（如 2.1GHz 附近）
STAGE3_BIAS_F0_GHZ = 2.1
STAGE3_BIAS_BW_GHZ = 0.8
STAGE3_BIAS_PROB = 0.55        # 有多少概率从“关注频段”采样

# ------------------------------------------------------------------------------
# 校准模式（absolute 路径已移除，仅保留 relative）
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 环境变量覆盖（不用改代码就能跑 A/B）
# PowerShell 示例：
#   $env:ENABLE_STAGE3_NONLINEAR="1"
#   $env:STAGE3_USE_REFERENCE_TARGET="1"
#   $env:STAGE3_SCHEME="ptv_volterra_ls"
#   $env:STAGE3_SWEEP="1"
# ------------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)

ENABLE_STAGE3_NONLINEAR = _env_bool("ENABLE_STAGE3_NONLINEAR", ENABLE_STAGE3_NONLINEAR)
STAGE3_USE_REFERENCE_TARGET = _env_bool("STAGE3_USE_REFERENCE_TARGET", STAGE3_USE_REFERENCE_TARGET)
STAGE3_REFERENCE_SNR_DB = _env_float("STAGE3_REFERENCE_SNR_DB", STAGE3_REFERENCE_SNR_DB)
STAGE3_SCHEME = _env_str("STAGE3_SCHEME", STAGE3_SCHEME)
STAGE3_SWEEP = _env_bool("STAGE3_SWEEP", STAGE3_SWEEP)
STAGE3_SWEEP_STEPS = _env_int("STAGE3_SWEEP_STEPS", STAGE3_SWEEP_STEPS)
STAGE3_STEPS = _env_int("STAGE3_STEPS", STAGE3_STEPS)
STAGE3_LS_RIDGE = _env_float("STAGE3_LS_RIDGE", STAGE3_LS_RIDGE)
STAGE3_LS_TONE_BATCH = _env_int("STAGE3_LS_TONE_BATCH", STAGE3_LS_TONE_BATCH)
STAGE3_HARM_LS_MAX_H = _env_int("STAGE3_HARM_LS_MAX_H", STAGE3_HARM_LS_MAX_H)
STAGE3_HARM_LS_GUARD = _env_int("STAGE3_HARM_LS_GUARD", STAGE3_HARM_LS_GUARD)
STAGE3_HARM_LS_INCLUDE_FUND = _env_bool("STAGE3_HARM_LS_INCLUDE_FUND", STAGE3_HARM_LS_INCLUDE_FUND)
STAGE3_HARM_LS_W_DELTA = _env_float("STAGE3_HARM_LS_W_DELTA", STAGE3_HARM_LS_W_DELTA)
STAGE3_AUTOSEARCH = _env_bool("STAGE3_AUTOSEARCH", STAGE3_AUTOSEARCH)
STAGE3_AUTOSEARCH_MAX_TRIALS = _env_int("STAGE3_AUTOSEARCH_MAX_TRIALS", STAGE3_AUTOSEARCH_MAX_TRIALS)
STAGE3_AUTOSEARCH_FINE_TOPK = _env_int("STAGE3_AUTOSEARCH_FINE_TOPK", STAGE3_AUTOSEARCH_FINE_TOPK)
STAGE3_AUTOSEARCH_THD_WORSEN_MAX_DB = _env_float("STAGE3_AUTOSEARCH_THD_WORSEN_MAX_DB", STAGE3_AUTOSEARCH_THD_WORSEN_MAX_DB)
ENABLE_POST_EQ = _env_bool("ENABLE_POST_EQ", ENABLE_POST_EQ)
POST_EQ_MODE = _env_str("POST_EQ_MODE", POST_EQ_MODE)
POST_EQ_TAPS = _env_int("POST_EQ_TAPS", POST_EQ_TAPS)
POST_EQ_RIDGE = _env_float("POST_EQ_RIDGE", POST_EQ_RIDGE)
POST_EQ_BAND_HZ = _env_float("POST_EQ_BAND_HZ", POST_EQ_BAND_HZ)
POST_EQ_HF_WEIGHT = _env_float("POST_EQ_HF_WEIGHT", POST_EQ_HF_WEIGHT)
POST_EQ_USE_REFERENCE_TARGET = _env_bool("POST_EQ_USE_REFERENCE_TARGET", POST_EQ_USE_REFERENCE_TARGET)
POST_EQ_REFERENCE_SNR_DB = _env_float("POST_EQ_REFERENCE_SNR_DB", POST_EQ_REFERENCE_SNR_DB)
POST_EQ_PTV2_DIFF = _env_float("POST_EQ_PTV2_DIFF", POST_EQ_PTV2_DIFF)
POST_EQ_TRAIN_STEPS = _env_int("POST_EQ_TRAIN_STEPS", POST_EQ_TRAIN_STEPS)
POST_EQ_TRAIN_LR = _env_float("POST_EQ_TRAIN_LR", POST_EQ_TRAIN_LR)
POST_EQ_W_TIME = _env_float("POST_EQ_W_TIME", POST_EQ_W_TIME)
POST_EQ_W_FREQ = _env_float("POST_EQ_W_FREQ", POST_EQ_W_FREQ)
POST_EQ_W_DELTA = _env_float("POST_EQ_W_DELTA", POST_EQ_W_DELTA)
POST_EQ_W_SMOOTH = _env_float("POST_EQ_W_SMOOTH", POST_EQ_W_SMOOTH)

# ------------------------------------------------------------------------------
# 方向B增强：带记忆的非线性校正（Parallel Hammerstein）
# ------------------------------------------------------------------------------
ENABLE_HAMMERSTEIN_NL = False # True
HAMMERSTEIN_TAPS = 21
HAMMERSTEIN_STAGE3_STEPS = 1200
HAMMERSTEIN_STAGE3_LR = 6e-4
HAMMERSTEIN_STAGE3_BATCH = 6
HAMMERSTEIN_W_TIME = 0.0        # 关闭时域项（避免 NL 被线性误差驱动）
HAMMERSTEIN_W_FREQ = 0.0        # 关闭整体频域 MSE
HAMMERSTEIN_W_HARM = 900.0      # THD 主导
HAMMERSTEIN_W_FUND = 35.0       # 基波保护/幅度对齐
HAMMERSTEIN_W_DELTA = 60.0      # 最小改动：限制 y_hat 相对线性输出的变化
HAMMERSTEIN_W_REG = 2e-4        # 非线性分支正则
HAMMERSTEIN_HARM_GUARD = 3

# Stage2 线性精修：高频加权（改善 4~6GHz 的幅相一致性，避免 NL 去“硬扛”线性误差）
FIR_HF_REFINE_STEPS = 0
FIR_HF_REFINE_LR = 0.0

# Hammerstein 训练频点：轻微偏向高频，但保持全带覆盖
HAMMERSTEIN_HIGH_PROB = 0.30

# （absolute 路径已移除）交织后线性微调相关配置已移除
# ==============================================================================
# 0. FPGA 辅助工具
# ==============================================================================
class Quantizer(nn.Module):
    """
    模拟 FPGA 定点数量化效应
    forward 过程: Float -> Fixed(Int) -> Float (用于梯度回传和误差评估)
    """
    def __init__(self, bit_width=16, dynamic_range=2.0):
        super().__init__()
        self.bit_width = bit_width
        self.dynamic_range = dynamic_range
        self.scale = (2**(bit_width - 1) - 1) / (dynamic_range / 2)

    def forward(self, x):
        # 模拟量化：缩放 -> 取整 -> 截断 -> 还原
        x_int = torch.round(x * self.scale)
        # 饱和截断 (Saturate)
        max_val = 2**(self.bit_width - 1) - 1
        min_val = -2**(self.bit_width - 1)
        x_clamped = torch.clamp(x_int, min_val, max_val)
        return x_clamped / self.scale

class DDSPVolterraNetwork(nn.Module):
    """
    [DDSP 核心模块] 物理模型驱动的 Volterra 级数网络
    
    理论架构:
    y[n] = x[n] + Sum_p( Conv( Basis_p(x), W_p[n] ) )
    
    关键特性:
    1. Basis Generator: 使用标准 Volterra 基底 (x^2, x^3)
       [REVERTED] 放弃 Chebyshev，回归物理本源，消除 DC Offset 问题。
    2. PTV Filtering (Demux): 权重 W[n] 随时间/通道周期性变化，实现通道独立的非线性校正。
    3. [Pure Nonlinear] 移除了线性分支 (conv1)，强制只学习非线性残差，避免与 Stage 1/2 线性校准冲突。
    """
    def __init__(self, taps=21):
        super().__init__()
        self.taps = taps
        
        # 定义两组独立的 Volterra 核，分别对应 Ch0 (Even) 和 Ch1 (Odd)
        # 这对应于理论中的 "Channel Demux" 或 "PTV Weights"
        
        # --- Ch0 Kernels (Even Samples) ---
        self.conv2_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        self.conv3_even = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        
        # --- Ch1 Kernels (Odd Samples) ---
        self.conv2_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        self.conv3_odd = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)

        # 初始化
        with torch.no_grad():
            # Non-linear terms: Near zero
            for m in [self.conv2_even, self.conv3_even, self.conv2_odd, self.conv3_odd]:
                m.weight.data.normal_(0, 1e-6)

    def forward(self, x):
        # x: (Batch, 1, Time)
        
        # [Safety] Hard Clamping to prevent numerical explosion
        x = torch.clamp(x, -1.0, 1.0)
        
        B, C, T = x.shape
        
        # 1. Basis Generator (基函数生成算子)
        # [SIMPLIFIED] 标准 Volterra 基底 (Skip x^1 to avoid linear interference)
        # basis_t1 = x 
        basis_t2 = x ** 2
        basis_t3 = x ** 3
        
        # 2. PTV Filtering (周期时变滤波 / 加权求和算子)
        
        # 路径 A: Ch0 参数 (Even)
        out_even_params = self.conv2_even(basis_t2) + self.conv3_even(basis_t3)
        
        # 路径 B: Ch1 参数 (Odd)
        out_odd_params  = self.conv2_odd(basis_t2)  + self.conv3_odd(basis_t3)
        
        # 3. Channel Mux / Switch (通道选择)
        mask_even = torch.zeros((1, 1, T), device=x.device)
        mask_even[:, :, 0::2] = 1.0
        
        mask_odd = 1.0 - mask_even
        
        # 组合误差预测
        error_pred = out_even_params * mask_even + out_odd_params * mask_odd
        
        # 4. Residual Connection (残差连接)
        # y = x + error_pred 
        return x + error_pred


class PTVMemorylessPoly23(nn.Module):
    """
    最轻量的 PTV 非线性校正（记忆无关，2/3 次）：
      y[n] = x[n] + (a2_even*x^2 + a3_even*x^3)*mask_even + (a2_odd*x^2 + a3_odd*x^3)*mask_odd

    用途：
    - 作为“最小复杂度基线”，用于和 ptv_volterra 做对比，快速判断是否真的需要 taps/记忆
    """
    def __init__(self):
        super().__init__()
        self.a2_even = nn.Parameter(torch.tensor(0.0))
        self.a3_even = nn.Parameter(torch.tensor(0.0))
        self.a2_odd = nn.Parameter(torch.tensor(0.0))
        self.a3_odd = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,T)
        x = torch.clamp(x, -1.0, 1.0)
        B, C, T = x.shape
        x2 = x ** 2
        x3 = x ** 3

        even = self.a2_even * x2 + self.a3_even * x3
        odd = self.a2_odd * x2 + self.a3_odd * x3

        mask_even = torch.zeros((1, 1, T), device=x.device, dtype=x.dtype)
        mask_even[:, :, 0::2] = 1.0
        mask_odd = 1.0 - mask_even

        return x + even * mask_even + odd * mask_odd


class OddOnlyVolterraNetwork(nn.Module):
    """
    [新方案] 只修 odd(Ch1) 采样相位的 Volterra(2/3 次 + 记忆 taps)。

    设计动机：
    - 子 ADC 的非线性主要发生在各自采样相位上；交织后再“全局修”会把 even/odd 耦合在一起，容易欠定并引入新 spur。
    - 只对 odd 相位修正，能显著降低自由度与耦合，稳定性通常更好。

    形式：
      y[n] = x[n] + mask_odd * (Conv(x^2, w2) + Conv(x^3, w3))
    """

    def __init__(self, taps: int = 21):
        super().__init__()
        self.taps = int(taps)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=self.taps, padding=self.taps // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=self.taps, padding=self.taps // 2, bias=False)
        with torch.no_grad():
            for m in [self.conv2, self.conv3]:
                m.weight.data.normal_(0, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        B, C, T = x.shape
        x2 = x ** 2
        x3 = x ** 3
        corr = self.conv2(x2) + self.conv3(x3)
        mask_even = torch.zeros((1, 1, T), device=x.device, dtype=x.dtype)
        mask_even[:, :, 0::2] = 1.0
        mask_odd = 1.0 - mask_even
        return x + corr * mask_odd


class OddOnlyMemorylessPoly23(nn.Module):
    """
    [新方案] 只修 odd(Ch1) 采样相位的记忆无关 2/3 次多项式：
      y[n] = x[n] + (a2*x^2 + a3*x^3) * mask_odd
    """

    def __init__(self):
        super().__init__()
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        B, C, T = x.shape
        x2 = x ** 2
        x3 = x ** 3
        corr = self.a2 * x2 + self.a3 * x3
        mask_even = torch.zeros((1, 1, T), device=x.device, dtype=x.dtype)
        mask_even[:, :, 0::2] = 1.0
        mask_odd = 1.0 - mask_even
        return x + corr * mask_odd


class PhysicalNonlinearityLayer(nn.Module):
    """
    物理信息静态逆模型层（memoryless polynomial inverse）：
      y = w1*x + w2*x^2 + w3*x^3 (+ ...)

    - 默认使用 PTV（even/odd 两套系数），更贴近 TIADC 子通道非线性差异
    - 初始化为“近似线性”：w1=1，其它=0
    """

    def __init__(self, *, order: int = 3, ptv: bool = True):
        super().__init__()
        self.order = int(order)
        if self.order < 1:
            raise ValueError("order must be >= 1")
        self.ptv = bool(ptv)
        # 设计目标：
        # - 不破坏 Stage1/2 的线性解：线性项固定为 1，只学习非线性残差（p>=2）
        # - 系数必须限幅（tanh），避免出现“把信号打爆”的坏解
        # - 使用 residual/alpha 小步修正：y = x + alpha * f_nl(x)
        self.alpha = float(os.getenv("POST_QE_ALPHA", "0.25"))

        # p=2..order 的系数上限（经验值：与 hd2/hd3 同量级，默认 1e-2 以内）
        # 对应项：x^2, x^3, x^4...
        max_p2 = float(os.getenv("POST_QE_MAX_P2", "8e-3"))
        max_p3 = float(os.getenv("POST_QE_MAX_P3", "6e-3"))
        max_hi = float(os.getenv("POST_QE_MAX_PHI", "2e-3"))
        scales = []
        for p in range(2, self.order + 1):
            if p == 2:
                scales.append(max_p2)
            elif p == 3:
                scales.append(max_p3)
            else:
                scales.append(max_hi)
        self.register_buffer("_scales", torch.tensor(scales, dtype=torch.float32))  # (order-1,)

        if self.ptv:
            self.raw_even = nn.Parameter(torch.zeros(self.order - 1, dtype=torch.float32))
            self.raw_odd = nn.Parameter(torch.zeros(self.order - 1, dtype=torch.float32))
        else:
            self.raw = nn.Parameter(torch.zeros(self.order - 1, dtype=torch.float32))

    def _nl_residual(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # f_nl(x) = sum_{p=2..order} a_p * x^p
        a = self._scales.to(device=x.device, dtype=x.dtype) * torch.tanh(raw.to(device=x.device, dtype=x.dtype))
        y = torch.zeros_like(x)
        for i, p in enumerate(range(2, self.order + 1)):
            y = y + a[i] * (x ** p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        alpha = float(self.alpha)
        if not self.ptv:
            return x + alpha * self._nl_residual(x, self.raw)

        xe = x[..., 0::2]
        xo = x[..., 1::2]
        re = self._nl_residual(xe, self.raw_even)
        ro = self._nl_residual(xo, self.raw_odd)
        y = x.clone()
        y[..., 0::2] = xe + alpha * re
        y[..., 1::2] = xo + alpha * ro
        return y


class DifferentiableMemoryPolynomial(nn.Module):
    """
    可微分记忆多项式（Memory Polynomial, MP）：
      y[n] = sum_{k=0..K-1} sum_{p=1..P} c[k,p] * x[n-k]^p

    - 默认使用 PTV（even/odd 两套 c[k,p]），等价于“每个子 ADC 的 MP 逆模型”
    - 初始化为恒等：c[0,1]=1，其它=0
    """

    def __init__(self, *, memory_depth: int = 3, nonlinear_order: int = 3, ptv: bool = True):
        super().__init__()
        self.K = int(memory_depth)
        self.P = int(nonlinear_order)
        if self.K < 1 or self.P < 1:
            raise ValueError("memory_depth and nonlinear_order must be >= 1")
        self.ptv = bool(ptv)
        # 关键：避免破坏线性校准（Stage1/2）
        # - 线性项固定：仅 x[n] 的系数为 1，其余线性 taps = 0
        # - 只学习 p>=2 的非线性项，并做限幅 + residual/alpha
        self.alpha = float(os.getenv("POST_NL_ALPHA", "0.25"))

        if self.P < 2:
            raise ValueError("nonlinear_order must be >= 2 for post-nonlinear stage")

        # p=2..P 的系数上限（与 hd2/hd3 同量级），高阶更小
        max_p2 = float(os.getenv("POST_NL_MAX_P2", "8e-3"))
        max_p3 = float(os.getenv("POST_NL_MAX_P3", "6e-3"))
        max_hi = float(os.getenv("POST_NL_MAX_PHI", "2e-3"))
        scales = []
        for p in range(2, self.P + 1):
            if p == 2:
                scales.append(max_p2)
            elif p == 3:
                scales.append(max_p3)
            else:
                scales.append(max_hi)
        self.register_buffer("_scales", torch.tensor(scales, dtype=torch.float32))  # (P-1,)

        # raw coeffs: (K, P-1) 对应 p=2..P
        if self.ptv:
            self.raw_even = nn.Parameter(torch.zeros(self.K, self.P - 1, dtype=torch.float32))
            self.raw_odd = nn.Parameter(torch.zeros(self.K, self.P - 1, dtype=torch.float32))
        else:
            self.raw = nn.Parameter(torch.zeros(self.K, self.P - 1, dtype=torch.float32))

    @staticmethod
    def _delay(x: torch.Tensor, k: int) -> torch.Tensor:
        # 因果延迟：x[n-k]，左侧补 0
        if k <= 0:
            return x
        xpad = F.pad(x, (k, 0), mode="constant", value=0.0)
        return xpad[..., :-k]

    def _mp_residual(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # f_nl(x) = sum_{k=0..K-1} sum_{p=2..P} c[k,p] * x[n-k]^p
        x = torch.clamp(x, -1.0, 1.0)
        scales = self._scales.to(device=x.device, dtype=x.dtype)  # (P-1,)
        raw_t = raw.to(device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)
        for k in range(self.K):
            xd = self._delay(x, k)
            a = scales * torch.tanh(raw_t[k, :])
            for i, p in enumerate(range(2, self.P + 1)):
                y = y + a[i] * (xd ** p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        alpha = float(self.alpha)
        if not self.ptv:
            # 线性项恒等：y = x + alpha * f_nl(x)
            return x + alpha * self._mp_residual(x, self.raw)

        xe = x[..., 0::2]
        xo = x[..., 1::2]
        y = x.clone()
        y[..., 0::2] = xe + alpha * self._mp_residual(xe, self.raw_even)
        y[..., 1::2] = xo + alpha * self._mp_residual(xo, self.raw_odd)
        return y


def _fold_freq_to_nyquist(f_hz: float, fs_hz: float) -> float:
    """把频率折叠到 [0, fs/2]（用于谐波混叠处理）。"""
    f = f_hz % fs_hz
    if f > fs_hz / 2:
        f = fs_hz - f
    return f


def _harmonic_bins_from_fund_bin(fund_bin: int, n: int, fs_hz: float, max_h: int = 5):
    """
    在相干采样（fund_bin 为整数 bin）前提下，计算 2~max_h 次谐波在 rfft 上的折叠 bin 索引。
    rfft bin 范围: [0, n/2]
    """
    bins = []
    for h in range(2, max_h + 1):
        # “原始” bin
        k = (fund_bin * h) % n
        # 折叠到 [0, n/2]：利用频谱共轭对称
        if k > n // 2:
            k = n - k
        # 排除 DC/Nyquist
        if k <= 1 or k >= n // 2 - 1:
            continue
        bins.append(int(k))
    return bins


def harmonic_ratio_loss_torch(x: torch.Tensor, fund_bin: int, *, max_h: int = 5, guard: int = 2, eps: float = 1e-12) -> torch.Tensor:
    """
    THD 友好的损失：谐波能量 / 基波能量（线性域）。
    x: [1,1,N]，默认相干采样，fund_bin 为整数 bin。
    """
    n = x.shape[-1]
    # Hann 窗（更稳，避免训练时因微小频偏产生泄漏）
    win = torch.hann_window(n, device=x.device, dtype=x.dtype).view(1, 1, -1)
    X = torch.fft.rfft(x * win, dim=-1, norm="ortho")
    P = (X.real**2 + X.imag**2)  # 功率谱

    # 基波能量：fund_bin 附近 guard 窗口求和
    s0 = max(0, fund_bin - guard)
    e0 = min(P.shape[-1], fund_bin + guard + 1)
    p_fund = torch.sum(P[..., s0:e0], dim=-1) + eps

    # 谐波能量：2~max_h 次谐波折叠后，同样用 guard 窗口求和
    harm_bins = _harmonic_bins_from_fund_bin(fund_bin, n, 0.0, max_h=max_h)
    p_harm = 0.0
    for k in harm_bins:
        s = max(0, k - guard)
        e = min(P.shape[-1], k + guard + 1)
        p_harm = p_harm + torch.sum(P[..., s:e], dim=-1)
    p_harm = p_harm + eps

    # 返回 batch 维均值
    return torch.mean(p_harm / p_fund)


def harmonic_power_loss_torch(x: torch.Tensor, fund_bin: int, *, max_h: int = 5, guard: int = 3, eps: float = 1e-12) -> torch.Tensor:
    """
    直接惩罚 2~max_h 次谐波“绝对功率”（线性域），并用基波功率做归一化，避免量纲问题。
    比 harmonic_ratio_loss 更“狠”，更偏向 THD 下降。
    """
    n = x.shape[-1]
    win = torch.hann_window(n, device=x.device, dtype=x.dtype).view(1, 1, -1)
    X = torch.fft.rfft(x * win, dim=-1, norm="ortho")
    P = (X.real**2 + X.imag**2)

    s0 = max(0, fund_bin - guard)
    e0 = min(P.shape[-1], fund_bin + guard + 1)
    p_fund = torch.sum(P[..., s0:e0], dim=-1) + eps

    harm_bins = _harmonic_bins_from_fund_bin(fund_bin, n, 0.0, max_h=max_h)
    p_harm = 0.0
    for k in harm_bins:
        s = max(0, k - guard)
        e = min(P.shape[-1], k + guard + 1)
        p_harm = p_harm + torch.sum(P[..., s:e], dim=-1)
    p_harm = p_harm + eps

    return torch.mean(p_harm / p_fund)


def residual_harmonic_loss_torch(residual: torch.Tensor, target: torch.Tensor, fund_bin: int, *, max_h: int = 5, guard: int = 3, eps: float = 1e-12) -> torch.Tensor:
    """
    更稳的 THD 优化方式：
    - 只惩罚“残差”(y_hat - target) 在 2~max_h 次谐波折叠 bin 附近的能量
    - 用 target 的基波能量做归一化
    这样可以避免为了压谐波而扭曲基波/其它频段，从而拖垮 SINAD/SFDR。
    """
    n = residual.shape[-1]
    win = torch.hann_window(n, device=residual.device, dtype=residual.dtype).view(1, 1, -1)

    R = torch.fft.rfft(residual * win, dim=-1, norm="ortho")
    Pr = (R.real**2 + R.imag**2)

    T = torch.fft.rfft(target * win, dim=-1, norm="ortho")
    Pt = (T.real**2 + T.imag**2)

    s0 = max(0, fund_bin - guard)
    e0 = min(Pt.shape[-1], fund_bin + guard + 1)
    p_fund = torch.sum(Pt[..., s0:e0], dim=-1) + eps

    harm_bins = _harmonic_bins_from_fund_bin(fund_bin, n, 0.0, max_h=max_h)
    p_harm_res = 0.0
    for k in harm_bins:
        s = max(0, k - guard)
        e = min(Pr.shape[-1], k + guard + 1)
        p_harm_res = p_harm_res + torch.sum(Pr[..., s:e], dim=-1)
    p_harm_res = p_harm_res + eps

    return torch.mean(p_harm_res / p_fund)

# ==============================================================================
# 1. 物理仿真引擎
# ==============================================================================
class TIADCSimulator:
    def __init__(self, fs=20e9):
        self.fs = fs

    def fractional_delay(self, sig, delay):
        """
        更接近现实的分数延时：使用 4-tap 三次 Farrow/Lagrange（避免 FFT 循环延时的绕回伪影）。
        delay: 主要用于 <1 sample 的分数延时（本脚本失配通常在 0~0.5 sample）
        """
        d = float(delay) % 1.0
        if abs(d) < 1e-12:
            return sig
        # 4-tap Farrow coefficients (same as FarrowDelay module)
        k0 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        k1 = np.array([-1/3, -1/2, 1.0, -1/6], dtype=np.float64)
        k2 = np.array([1/2, -1.0, 1/2, 0.0], dtype=np.float64)
        k3 = np.array([-1/6, 1/2, -1/2, 1/6], dtype=np.float64)

        # replicate pad: (1,2) to match torch implementation
        xpad = np.pad(sig.astype(np.float64), (1, 2), mode='edge')
        c0 = np.convolve(xpad, k0[::-1], mode='valid')
        c1 = np.convolve(xpad, k1[::-1], mode='valid')
        c2 = np.convolve(xpad, k2[::-1], mode='valid')
        c3 = np.convolve(xpad, k3[::-1], mode='valid')
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out.astype(sig.dtype, copy=False)

    def generate_tone_data(self, freq, N=8192):
        t = np.arange(N) / self.fs
        src = np.sin(2 * np.pi * freq * t)
        return src * 0.9

    def apply_channel_effect(self, sig, cutoff_freq, delay_samples, gain=1.0, 
                             jitter_std=100e-15, n_bits=12, 
                             hd2=0.0, hd3=0.0, snr_target=None): 
        nyquist = self.fs / 2
        
        # 1. 非线性失真
        sig_nonlinear = sig + hd2 * (sig**2) + hd3 * (sig**3)
        
        # 2. 带宽限制 (模拟模拟前端滤波器)
        b, a = signal.butter(5, cutoff_freq / nyquist, btype='low')
        sig_bw = signal.lfilter(b, a, sig_nonlinear)
        
        # 3. 增益失配
        sig_gain = sig_bw * gain
        
        # 4. 时延失配
        sig_delayed = self.fractional_delay(sig_gain, delay_samples)
        
        # 5. 时钟抖动 (Jitter)
        # Jitter 产生的噪声功率与信号频率和抖动值成正比
        if jitter_std > 0:
            slope = np.gradient(sig_delayed) * self.fs # 信号斜率
            dt_noise = np.random.normal(0, jitter_std, len(sig_delayed))
            sig_jittered = sig_delayed + slope * dt_noise
        else:
            sig_jittered = sig_delayed
            
        # 6. 热噪声 (Thermal Noise)
        # 如果指定了 n_bits (量化)，通常热噪声底应该低于量化噪声或与其相当
        # 这里改为更科学的 SNR 设置，或者基于位宽的本底噪声
        if snr_target is not None:
             # 根据目标 SNR 添加高斯白噪声
            sig_power = np.mean(sig_jittered**2)
            noise_power = sig_power / (10**(snr_target/10))
            thermal_noise = np.random.normal(0, np.sqrt(noise_power), len(sig_jittered))
        else:
            # 默认热噪声 (模拟约 14bit 系统的热噪声底)
            thermal_noise = np.random.normal(0, 1e-4, len(sig_jittered))
            
        sig_noisy = sig_jittered + thermal_noise
        
        # 7. 量化 (ADC Digitization)
        if n_bits is not None:
            v_range = 2.0 
            levels = 2**n_bits
            step = v_range / levels
            # 模拟 ADC 的输入截断 (Clipping)
            sig_clipped = np.clip(sig_noisy, -1.0, 1.0)
            # 模拟量化
            sig_out = np.round(sig_clipped / step) * step
        else:
            sig_out = sig_noisy
            
        return sig_out

# ==============================================================================
# 2. 混合校准模型
# ==============================================================================
class FarrowDelay(nn.Module):
    def __init__(self):
        super().__init__()
        self.delay = nn.Parameter(torch.tensor(0.0))
        # FPGA 部署提示: 这些系数在 FPGA 中需要预先计算并量化
        self.register_buffer('kernel_0', torch.tensor([[[0, 1, 0, 0]]], dtype=torch.float32))
        self.register_buffer('kernel_1', torch.tensor([[[-1/3, -1/2, 1, -1/6]]], dtype=torch.float32))
        self.register_buffer('kernel_2', torch.tensor([[[1/2, -1, 1/2, 0]]], dtype=torch.float32))
        self.register_buffer('kernel_3', torch.tensor([[[-1/6, 1/2, -1/2, 1/6]]], dtype=torch.float32))

    def forward(self, x):
        d = self.delay
        padded_x = F.pad(x, (1, 2), mode='replicate') 
        c0 = F.conv1d(padded_x, self.kernel_0)
        c1 = F.conv1d(padded_x, self.kernel_1)
        c2 = F.conv1d(padded_x, self.kernel_2)
        c3 = F.conv1d(padded_x, self.kernel_3)
        # Horner's method 结构，适合 FPGA DSP Slice 级联
        out = c0 + d * (c1 + d * (c2 + d * c3))
        return out


class PolynomialCorrection(nn.Module):
    """
    记忆无关的 2/3 次多项式非线性校正（用于方向B：每通道先做非线性修整，再交织）。
    y = x + a2*x^2 + a3*x^3
    """
    def __init__(self):
        super().__init__()
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x + self.a2 * (x ** 2) + self.a3 * (x ** 3)


class HammersteinNonlinearity(nn.Module):
    """
    Parallel Hammerstein 非线性（带记忆）：
      y_nl = FIR2(x^2) + FIR3(x^3)
    用小 taps 学习非线性记忆效应，通常比纯多项式更能改善中高频 THD。
    """
    def __init__(self, taps=21):
        super().__init__()
        self.taps = taps
        self.conv2 = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        with torch.no_grad():
            # 初始化为 0（不改变系统）
            self.conv2.weight.zero_()
            self.conv3.weight.zero_()

    def forward(self, x):
        x = torch.clamp(x, -1.0, 1.0)
        b2 = x ** 2
        b3 = x ** 3
        return self.conv2(b2) + self.conv3(b3)

class NonlinearPostCorrector(nn.Module):
    """
    线性校准之后的“后级非线性修整器”（与线性校准解耦）：
      y = x + a2*x^2 + a3*x^3 + FIR2(x^2) + FIR3(x^3)

    设计目标：
    - 默认是恒等映射（初始不改变线性输出）
    - 训练时只更新本模块参数，保证线性校准（gain/delay/FIR）不被非线性影响
    """
    def __init__(self, taps: int = 21, *, enable_hammerstein: bool = True):
        super().__init__()
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))
        self.hammerstein = HammersteinNonlinearity(taps=taps) if enable_hammerstein else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        poly = self.a2 * (x ** 2) + self.a3 * (x ** 3)
        if self.hammerstein is None:
            return x + poly
        return x + poly + self.hammerstein(x)

class HybridCalibrationModel(nn.Module): 
    # [FPGA Optimization] 默认 Taps 从 511 降至 63，以适应 FPGA 资源限制
    def __init__(self, taps=63, fpga_simulation=False):
        super().__init__()
        # 非线性校正（方向B建议启用；相对校准可冻结或关闭）
        self.poly = PolynomialCorrection()
        self.hammerstein = HammersteinNonlinearity(taps=HAMMERSTEIN_TAPS) if ENABLE_HAMMERSTEIN_NL else None
        # 默认关闭：避免线性校准被非线性分支耦合影响
        self.enable_nonlinear = False
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.global_delay = FarrowDelay()
        
        self.conv = nn.Conv1d(1, 1, kernel_size=taps, padding=taps//2, bias=False)
        with torch.no_grad():
            self.conv.weight.data.fill_(0.0)
            self.conv.weight.data[0, 0, taps//2] = 1.0
            
        self.fpga_simulation = fpga_simulation
        self.data_quant = Quantizer(bit_width=16, dynamic_range=4.0)
        self.weight_quant = Quantizer(bit_width=18, dynamic_range=4.0)

    def forward(self, x):
        # 线性校准默认只做 gain/delay/FIR；非线性需显式开启
        x_in = self.poly(x) if self.enable_nonlinear else x
        x_gain = x_in * self.gain
        x_delay = self.global_delay(x_gain)

        # 线性路径
        if self.fpga_simulation:
            w_lin = self.weight_quant(self.conv.weight)
            x_lin = self.data_quant(x_delay)
            out_lin = F.conv1d(x_lin, w_lin, padding=self.conv.padding, stride=self.conv.stride)
        else:
            out_lin = self.conv(x_delay)

        # 非线性路径（Parallel Hammerstein）：仅在 enable_nonlinear=True 时参与 forward
        if (not self.enable_nonlinear) or (self.hammerstein is None):
            return out_lin

        if self.fpga_simulation:
            # 对 NL 分支也做量化仿真（保持口径一致）
            w2 = self.weight_quant(self.hammerstein.conv2.weight)
            w3 = self.weight_quant(self.hammerstein.conv3.weight)
            x_nl = self.data_quant(x_delay)
            b2 = x_nl ** 2
            b3 = x_nl ** 3
            out_nl = (
                F.conv1d(b2, w2, padding=self.hammerstein.conv2.padding, stride=self.hammerstein.conv2.stride)
                + F.conv1d(b3, w3, padding=self.hammerstein.conv3.padding, stride=self.hammerstein.conv3.stride)
            )
        else:
            out_nl = self.hammerstein(x_delay)

        return out_lin + out_nl

    def forward_linear(self, x):
        """
        仅线性路径（gain+delay+FIR），用于保证线性校准口径固定（不含任何非线性项）。
        """
        x_gain = x * self.gain
        x_delay = self.global_delay(x_gain)

        if self.fpga_simulation:
            w_lin = self.weight_quant(self.conv.weight)
            x_lin = self.data_quant(x_delay)
            out_lin = F.conv1d(x_lin, w_lin, padding=self.conv.padding, stride=self.conv.stride)
        else:
            out_lin = self.conv(x_delay)
        return out_lin

# ==============================================================================
# 3. 损失函数
# ==============================================================================
class ComplexMSELoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

    def forward(self, y_pred, y_target, model):
        crop = 300 
        y_p = y_pred[..., crop:-crop]
        y_t = y_target[..., crop:-crop]
        
        # [Stage 1 & 2] 标准 MSE
        fft_p = torch.fft.rfft(y_p, dim=-1, norm='ortho')
        fft_t = torch.fft.rfft(y_t, dim=-1, norm='ortho')
        loss_freq = torch.mean(torch.abs(fft_p - fft_t)**2)
        loss_time = torch.mean((y_p - y_t)**2)
        w = model.conv.weight
        loss_reg = torch.mean((w[:,:,1:] - w[:,:,:-1])**2)
        return 100.0 * loss_time + 100.0 * loss_freq + 0.001 * loss_reg

# ==============================================================================
# 4. 训练流程 (相对校准)
# ==============================================================================
def train_relative_calibration(simulator, device='cpu'):
    N_train = 32768
    M_average = 16 
    
    # 模拟真实的物理通道非线性（更贴近实采数据的参数设置）
    # 关键改进：
    # 1. 带宽应明显高于测试频率上限
    # 2. **增强非线性失配**：让Post-NL有发挥空间
    #    - CH0: hd2=1e-3, hd3=0.5e-3（较好的通道）
    #    - CH1: hd2=5e-3, hd3=3e-3（较差的通道，5倍差异）
    #    实采ADC的非线性失配通常在3-10倍范围
    params_ch0 = {'cutoff_freq': 8.0e9, 'delay_samples': 0.0, 'gain': 1.0, 'hd2': 1e-3, 'hd3': 0.5e-3}
    params_ch1 = {'cutoff_freq': 7.8e9, 'delay_samples': 0.25, 'gain': 0.98, 'hd2': 5e-3, 'hd3': 3e-3}
    
    print(f"开始相对校准训练: Mapping Ch1 -> Ch0")

    np.random.seed(42)
    white_noise = np.random.randn(N_train)
    # 训练信号带宽限制在7GHz（略高于测试范围，确保覆盖）
    b_src, a_src = signal.butter(6, 7.0e9/(simulator.fs/2), btype='low')
    base_sig = signal.lfilter(b_src, a_src, white_noise)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9
    base_sig = base_sig + 0.5e-3 * base_sig**2 

    ch0_captures = []
    ch1_captures = []
    for _ in range(M_average):
        sig0 = simulator.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch0)
        ch0_captures.append(sig0)
        sig1 = simulator.apply_channel_effect(base_sig, jitter_std=100e-15, n_bits=12, **params_ch1)
        ch1_captures.append(sig1)
    
    avg_ch0 = np.mean(np.stack(ch0_captures), axis=0) # Target
    avg_ch1 = np.mean(np.stack(ch1_captures), axis=0) # Input
    
    scale = np.max(np.abs(avg_ch0))
    inp_t = torch.FloatTensor(avg_ch1/scale).view(1, 1, -1).to(device)
    tgt_t = torch.FloatTensor(avg_ch0/scale).view(1, 1, -1).to(device)
    
    model = HybridCalibrationModel(taps=63, fpga_simulation=False).to(device)
    loss_fn = ComplexMSELoss(device=device)
    
    print("=== Stage 1: Relative Delay & Gain (粗调) ===")
    opt_s1 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 1e-2},
        {'params': model.gain, 'lr': 1e-2},
        {'params': model.conv.parameters(), 'lr': 0.0},
    ])
    for ep in range(301):
        opt_s1.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s1.step()

    print("=== Stage 2: Relative FIR (精调 - 频率响应匹配) ===")
    opt_s2 = optim.Adam([
        {'params': model.global_delay.parameters(), 'lr': 0.0},
        {'params': model.gain, 'lr': 0.0},
        {'params': model.conv.parameters(), 'lr': 5e-4},
    ], betas=(0.5, 0.9))
    for ep in range(1001):
        opt_s2.zero_grad()
        loss = loss_fn(model(inp_t), tgt_t, model)
        loss.backward()
        opt_s2.step()

    # --------------------------------------------------------------------------
    # [替代 Stage3] 交织后 Post-EQ（封装版）：把整体 TIADC 输出往“恒等/单位冲激”拉
    # --------------------------------------------------------------------------
    if ENABLE_POST_EQ:
        build_and_attach_post_eq(
            simulator=simulator,
            model=model,
            base_sig=base_sig,
            scale=float(scale),
            params_ch0=params_ch0,
            params_ch1=params_ch1,
            device=device,
            taps=int(POST_EQ_TAPS),
            band_hz=float(POST_EQ_BAND_HZ),
            hf_weight=float(POST_EQ_HF_WEIGHT),
            ridge=float(POST_EQ_RIDGE),
            use_reference_target=bool(POST_EQ_USE_REFERENCE_TARGET),
            reference_snr_db=float(POST_EQ_REFERENCE_SNR_DB),
            mode=str(POST_EQ_MODE),
        )
    else:
        model.post_fir = None
        model.post_fir_meta = None

    # --------------------------------------------------------------------------
    # Stage3 自动搜索：固定 Stage1/2，只在 Stage3 上做有限预算搜索，自动选最优
    # --------------------------------------------------------------------------
    def _calc_spectrum_metrics_fast(sig: np.ndarray, fs: float, input_freq: float):
        # 与 calculate_metrics_detailed 的口径保持一致（Blackman + 谐波折叠）
        sig = np.asarray(sig, dtype=np.float64)
        n = len(sig)
        win = np.blackman(n)
        cg = float(np.mean(win))
        fft_spec = np.fft.rfft(sig * win)
        fft_mag = np.abs(fft_spec) / (n / 2 * cg + 1e-20)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        idx_fund = int(np.argmin(np.abs(freqs - input_freq)))
        span = 5
        s = max(0, idx_fund - span)
        e = min(len(fft_mag), idx_fund + span + 1)
        idx_peak = s + int(np.argmax(fft_mag[s:e]))
        p_fund = float(fft_mag[idx_peak] ** 2) + 1e-30

        mask = np.ones_like(fft_mag, dtype=bool)
        mask[:5] = False
        mask[max(0, idx_peak - span): min(len(mask), idx_peak + span + 1)] = False
        p_noise_dist = float(np.sum(fft_mag[mask] ** 2)) + 1e-30
        sinad = 10.0 * np.log10(p_fund / p_noise_dist)
        enob = (sinad - 1.76) / 6.02
        p_spur_max = float(np.max(fft_mag[mask] ** 2)) if np.any(mask) else 1e-30
        sfdr = 10.0 * np.log10(p_fund / (p_spur_max + 1e-30))

        # THD (2~5 次谐波，折叠到 Nyquist 内)
        p_harm = 0.0
        for h in range(2, 6):
            hf = (input_freq * h) % fs
            if hf > fs / 2:
                hf = fs - hf
            if hf < 1e6 or hf > fs / 2 - 1e6:
                continue
            k = int(np.argmin(np.abs(freqs - hf)))
            ss = max(0, k - span)
            ee = min(len(fft_mag), k + span + 1)
            kk = ss + int(np.argmax(fft_mag[ss:ee]))
            p_harm += float(fft_mag[kk] ** 2)
        thd = 10.0 * np.log10((p_harm + 1e-30) / p_fund)
        return float(sinad), float(enob), float(thd), float(sfdr)

    def _interleave_for_metric(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
        # 与 calculate_metrics_detailed 的 interleave 保持一致（先抽取再交织）
        s0 = c0_full[0::2]
        s1 = c1_full[1::2]
        L = min(len(s0), len(s1))
        out = np.zeros(2 * L, dtype=np.float64)
        out[0::2] = s0[:L]
        out[1::2] = s1[:L]
        return out

    def _apply_stage3_post(sig_post_qe: np.ndarray) -> np.ndarray:
        if not (hasattr(model, "post_linearizer") and model.post_linearizer is not None):
            return sig_post_qe
        s = max(float(np.max(np.abs(sig_post_qe))), 1e-12)
        with torch.no_grad():
            x = torch.FloatTensor(sig_post_qe / s).view(1, 1, -1).to(device)
            y = model.post_linearizer(x).cpu().numpy().flatten() * s
        return y

    def _quick_eval(in_freqs_hz, *, tone_n: int):
        # 返回 dict：线性与非线性后的“逐频点”与汇总（均值/最差下降）
        sin_lin, sf_lin, th_lin = [], [], []
        sin_nl, sf_nl, th_nl = [], [], []

        for f in in_freqs_hz:
            src = simulator.generate_tone_data(f, N=int(tone_n))
            # 评估口径必须与训练一致：使用真实 ADC 条件（噪声+量化）
            y0 = simulator.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0)
            y1 = simulator.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                y1c = model(x1).cpu().numpy().flatten() * scale

            sig_post_lin = _interleave_for_metric(y0, y1c)

            # 新链路：Post-QE（如果存在）再接 Stage3(Post-NL)
            sig_post_qe = sig_post_lin
            if hasattr(model, "post_qe") and (model.post_qe is not None):
                s_qe = max(float(np.max(np.abs(sig_post_lin))), 1e-12)
                with torch.no_grad():
                    xq = torch.FloatTensor(sig_post_lin / s_qe).view(1, 1, -1).to(device)
                    yq = model.post_qe(xq).cpu().numpy().flatten() * s_qe
                sig_post_qe = yq

            sig_post_nl = _apply_stage3_post(sig_post_qe)

            s1, _, t1, sf1 = _calc_spectrum_metrics_fast(sig_post_qe, simulator.fs, f)
            s2, _, t2, sf2 = _calc_spectrum_metrics_fast(sig_post_nl, simulator.fs, f)
            sin_lin.append(s1); sf_lin.append(sf1); th_lin.append(t1)
            sin_nl.append(s2); sf_nl.append(sf2); th_nl.append(t2)

        sin_lin = np.asarray(sin_lin, dtype=np.float64)
        sf_lin = np.asarray(sf_lin, dtype=np.float64)
        th_lin = np.asarray(th_lin, dtype=np.float64)
        sin_nl = np.asarray(sin_nl, dtype=np.float64)
        sf_nl = np.asarray(sf_nl, dtype=np.float64)
        th_nl = np.asarray(th_nl, dtype=np.float64)

        return {
            "sin_lin_arr": sin_lin,
            "sfdr_lin_arr": sf_lin,
            "thd_lin_arr": th_lin,
            "sin_nl_arr": sin_nl,
            "sfdr_nl_arr": sf_nl,
            "thd_nl_arr": th_nl,
            "sin_lin": float(np.mean(sin_lin)),
            "sfdr_lin": float(np.mean(sf_lin)),
            "thd_lin": float(np.mean(th_lin)),
            "sin_nl": float(np.mean(sin_nl)),
            "sfdr_nl": float(np.mean(sf_nl)),
            "thd_nl": float(np.mean(th_nl)),
            "sin_drop_max": float(np.max(sin_lin - sin_nl)),
            "sfdr_drop_max": float(np.max(sf_lin - sf_nl)),
        }

    def _autosearch_stage3():
        # 备份全局配置，试完恢复
        global STAGE3_TAPS, STAGE3_LS_RIDGE, STAGE3_USE_REFERENCE_TARGET, STAGE3_SCHEME, STAGE3_STEPS, STAGE3_HARM_LS_W_DELTA, STAGE3_LS_TONE_BATCH
        bak = (STAGE3_TAPS, STAGE3_LS_RIDGE, STAGE3_USE_REFERENCE_TARGET, STAGE3_SCHEME, STAGE3_STEPS, STAGE3_HARM_LS_W_DELTA, STAGE3_LS_TONE_BATCH)

        eval_freqs = [float(x) * 1e9 for x in STAGE3_AUTOSEARCH_EVAL_FREQS_GHZ]
        fine_freqs = [float(x) * 1e9 for x in STAGE3_AUTOSEARCH_FINE_FREQS_GHZ]

        # baseline：不挂 Stage3
        model.post_linearizer = None
        model.post_linearizer_scale = None
        model.reference_params = None
        base_m = _quick_eval(eval_freqs, tone_n=int(STAGE3_AUTOSEARCH_EVAL_N_COARSE))
        print(f"[AutoSearch] baseline (no stage3) | SINAD={base_m['sin_lin']:.2f} | SFDR={base_m['sfdr_lin']:.2f} | THD={base_m['thd_lin']:.2f}")

        trials = []

        # 构造候选：odd_* 默认用 match_ch0(target=False)，ptv_* 两种 target 都试
        schemes = ["odd_poly_ls", "odd_volterra_ls", "ptv_poly_ls", "ptv_volterra_ls", "ptv_volterra_harm_ls", "ptv_poly_harm_ls"]
        for scheme in schemes:
            is_volterra = ("volterra" in scheme)
            taps_list = list(STAGE3_AUTOSEARCH_TAPS_LIST) if is_volterra else [STAGE3_TAPS]
            ridge_list = list(STAGE3_AUTOSEARCH_RIDGE_LIST)
            rounds_list = list(STAGE3_AUTOSEARCH_LS_ROUNDS) if scheme.endswith("_ls") or scheme.endswith("_harm_ls") else [int(STAGE3_STEPS)]
            wdelta_list = list(STAGE3_AUTOSEARCH_HARM_WDELTA_LIST) if scheme.endswith("_harm_ls") else [float(STAGE3_HARM_LS_W_DELTA)]
            if scheme.startswith("odd_"):
                target_list = [False]  # 只对齐到 Ch0（避免为追 reference 而扭曲 odd）
            else:
                target_list = [True, False]

            for use_ref in target_list:
                for taps in taps_list:
                    for ridge in ridge_list:
                        for rounds in rounds_list:
                            for wdelta in wdelta_list:
                                trials.append((scheme, use_ref, int(taps), float(ridge), int(rounds), float(wdelta)))

        # 控制预算：先按“保守优先”排序（ridge 大、rounds 小优先），更容易不翻车
        trials.sort(key=lambda x: (-x[3], x[4], x[2], -x[5]))
        trials = trials[: int(STAGE3_AUTOSEARCH_MAX_TRIALS)]

        # ----------------------------
        # Phase A: 粗筛（快速、保守）
        # ----------------------------
        coarse_results = []
        best = None
        best_dd = None
        best_ref = None

        for idx, (scheme, use_ref, taps, ridge, rounds, wdelta) in enumerate(trials, 1):
            STAGE3_SCHEME = scheme
            STAGE3_USE_REFERENCE_TARGET = bool(use_ref)
            STAGE3_TAPS = int(taps)
            STAGE3_LS_RIDGE = float(ridge)
            STAGE3_STEPS = int(rounds)
            STAGE3_HARM_LS_W_DELTA = float(wdelta)
            # 粗筛时 tone_batch 小一点更快
            STAGE3_LS_TONE_BATCH = min(int(STAGE3_LS_TONE_BATCH), 6)

            print(f"\n[AutoSearch] coarse {idx}/{len(trials)} try scheme={scheme} use_ref={use_ref} taps={taps} ridge={ridge:g} rounds={rounds} wdelta={wdelta:g}")
            try:
                dd, params_ref = _train_stage3_one_scheme(scheme=scheme, steps=int(rounds))
            except Exception as e:
                print(f"[AutoSearch] skip (train error): {e}")
                continue

            model.post_linearizer = dd
            model.post_linearizer_scale = None
            model.reference_params = params_ref if use_ref else None

            m = _quick_eval(eval_freqs, tone_n=int(STAGE3_AUTOSEARCH_EVAL_N_COARSE))

            # 评分：硬约束使用“最差下降”（防止某个频点明显变差），再看均值收益
            sin_drop_max = float(m["sin_drop_max"])
            sf_drop_max = float(m["sfdr_drop_max"])
            thd_worsen_max = float(np.max(m["thd_nl_arr"] - m["thd_lin_arr"]))
            ok = (sin_drop_max <= 0.2) and (sf_drop_max <= 0.5) and (thd_worsen_max <= float(STAGE3_AUTOSEARCH_THD_WORSEN_MAX_DB))

            sin_gain = float(m["sin_nl"] - base_m["sin_lin"])
            sf_gain = float(m["sfdr_nl"] - base_m["sfdr_lin"])
            th_gain = float(((-m["thd_nl"]) - (-base_m["thd_lin"])))  # THD 更负更好

            score = sf_gain + 0.2 * th_gain + 0.2 * sin_gain - 6.0 * max(0.0, sin_drop_max) - 2.0 * max(0.0, sf_drop_max)
            print(
                f"[AutoSearch] eval | SINAD_nl={m['sin_nl']:.2f} (mean gain {sin_gain:+.2f}, max drop {sin_drop_max:+.2f}) | "
                f"SFDR_nl={m['sfdr_nl']:.2f} (mean gain {sf_gain:+.2f}, max drop {sf_drop_max:+.2f}) | "
                f"THD_nl={m['thd_nl']:.2f} (max worsen {thd_worsen_max:+.2f}) | score={score:.3f} | ok={ok}"
            )

            if not ok:
                continue

            cand = (score, m, scheme, use_ref, taps, ridge, rounds, wdelta)
            coarse_results.append(cand)
            if (best is None) or (cand[0] > best[0]):
                best = cand
                best_dd = dd
                best_ref = params_ref if use_ref else None

        # ----------------------------
        # Phase B: 精搜（围绕 TopK 做更严格评估）
        # ----------------------------
        coarse_results.sort(key=lambda x: x[0], reverse=True)
        topk = coarse_results[: int(max(1, STAGE3_AUTOSEARCH_FINE_TOPK))]
        if len(topk) > 0:
            print(f"\n[AutoSearch] fine stage: refine top {len(topk)} candidates")

        for rank, (score0, m0, scheme0, use_ref0, taps0, ridge0, rounds0, wdelta0) in enumerate(topk, 1):
            # 细化：ridge 在周围做 0.5x/1x/2x，rounds 固定 1（避免过拟合）但 tone_batch 加大更稳
            ridge_cands = sorted(set([ridge0 * 0.5, ridge0, ridge0 * 2.0]))
            ridge_cands = [float(r) for r in ridge_cands if 1e-4 <= r <= 1.0]
            rounds_cands = [1]
            wdelta_cands = [wdelta0]
            if str(scheme0).endswith("_harm_ls"):
                wdelta_cands = list(STAGE3_AUTOSEARCH_HARM_WDELTA_LIST)

            for ridge in ridge_cands:
                for rounds in rounds_cands:
                    for wdelta in wdelta_cands:
                        STAGE3_SCHEME = scheme0
                        STAGE3_USE_REFERENCE_TARGET = bool(use_ref0)
                        STAGE3_TAPS = int(taps0)
                        STAGE3_LS_RIDGE = float(ridge)
                        STAGE3_STEPS = int(rounds)
                        STAGE3_HARM_LS_W_DELTA = float(wdelta)
                        # 精搜：tone_batch 加大一些，估计更稳
                        STAGE3_LS_TONE_BATCH = max(int(STAGE3_LS_TONE_BATCH), 10)

                        print(f"\n[AutoSearch] fine {rank}/{len(topk)} try scheme={scheme0} use_ref={use_ref0} taps={taps0} ridge={ridge:g} rounds={rounds} wdelta={wdelta:g}")
                        try:
                            dd, params_ref = _train_stage3_one_scheme(scheme=scheme0, steps=int(rounds))
                        except Exception as e:
                            print(f"[AutoSearch] fine skip (train error): {e}")
                            continue

                        model.post_linearizer = dd
                        model.post_linearizer_scale = None
                        model.reference_params = params_ref if use_ref0 else None

                        m = _quick_eval(fine_freqs, tone_n=int(STAGE3_AUTOSEARCH_EVAL_N_FINE))
                        sin_drop_max = float(m["sin_drop_max"])
                        sf_drop_max = float(m["sfdr_drop_max"])
                        thd_worsen_max = float(np.max(m["thd_nl_arr"] - m["thd_lin_arr"]))

                        # 精搜硬约束更严格
                        ok = (sin_drop_max <= 0.15) and (sf_drop_max <= 0.35) and (thd_worsen_max <= float(STAGE3_AUTOSEARCH_THD_WORSEN_MAX_DB))

                        sin_gain = float(m["sin_nl"] - base_m["sin_lin"])
                        sf_gain = float(m["sfdr_nl"] - base_m["sfdr_lin"])
                        th_gain = float(((-m["thd_nl"]) - (-base_m["thd_lin"])))
                        score = sf_gain + 0.25 * th_gain + 0.25 * sin_gain - 8.0 * max(0.0, sin_drop_max) - 3.0 * max(0.0, sf_drop_max)

                        print(
                            f"[AutoSearch] fine eval | SINAD_nl={m['sin_nl']:.2f} (mean gain {sin_gain:+.2f}, max drop {sin_drop_max:+.2f}) | "
                            f"SFDR_nl={m['sfdr_nl']:.2f} (mean gain {sf_gain:+.2f}, max drop {sf_drop_max:+.2f}) | "
                            f"THD_nl={m['thd_nl']:.2f} (max worsen {thd_worsen_max:+.2f}) | score={score:.3f} | ok={ok}"
                        )

                        if not ok:
                            continue

                        cand = (score, m, scheme0, use_ref0, taps0, ridge, rounds, wdelta)
                        if (best is None) or (cand[0] > best[0]):
                            best = cand
                            best_dd = dd
                            best_ref = params_ref if use_ref0 else None

        # 恢复全局配置（但把最佳模型挂回去）
        STAGE3_TAPS, STAGE3_LS_RIDGE, STAGE3_USE_REFERENCE_TARGET, STAGE3_SCHEME, STAGE3_STEPS, STAGE3_HARM_LS_W_DELTA, STAGE3_LS_TONE_BATCH = bak

        if best is None:
            print("[AutoSearch] 未找到比 baseline 更稳的 Stage3（自动关闭 Stage3）。")
            model.post_linearizer = None
            model.post_linearizer_scale = None
            model.reference_params = None
            return None

        _, m, scheme, use_ref, taps, ridge, rounds, wdelta = best
        print(
            f"[AutoSearch] BEST: scheme={scheme} use_ref={use_ref} taps={taps} ridge={ridge:g} rounds={rounds} | "
            f"SINAD={m['sin_nl']:.2f} SFDR={m['sfdr_nl']:.2f} THD={m['thd_nl']:.2f} | "
            f"max_drop: SINAD {m['sin_drop_max']:+.2f}, SFDR {m['sfdr_drop_max']:+.2f}"
        )

        model.post_linearizer = best_dd
        model.post_linearizer_scale = None
        model.reference_params = best_ref
        best_cfg = {
            "scheme": scheme,
            "use_ref": bool(use_ref),
            "taps": int(taps),
            "ridge": float(ridge),
            "rounds": int(rounds),
            "harm_wdelta": float(wdelta),
            "eval_freqs_ghz": list(STAGE3_AUTOSEARCH_EVAL_FREQS_GHZ),
            "fine_freqs_ghz": list(STAGE3_AUTOSEARCH_FINE_FREQS_GHZ),
        }
        try:
            out_dir = _get_plot_dir()
            out_path = out_dir / "best_stage3_config.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(best_cfg, f, ensure_ascii=False, indent=2)
            print(f"[AutoSearch] saved config: {out_path}")
        except Exception as e:
            print(f"[AutoSearch] save config failed: {e}")
        return best_cfg

    # === Stage 3: Nonlinear Error Trimming (Volterra / Memory Polynomial) ===
    #
    # 目的：对“交织后的 TIADC 输出”做非线性残差修整（主要改善 THD，同时兼顾 SFDR/SINAD）。
    # 为什么要放在 Stage3（而不是 Stage4）：
    # - Stage1/2 已完成线性对齐；非线性修整应基于“线性已对齐的输出”继续优化。
    # - 若把非线性修整放到更后面但 target 仍是 Ch0，会被 Ch0 自身非线性锁死上限。
    #
    # 优化点（针对 v0119_v2 原 Stage4 效果差）：
    # - 不用 chirp 做主训练目标：chirp 的谐波会在频域铺开，THD 优化梯度不集中；
    #   改为“相干采样的单音集合”，直接对 2~5 次谐波做惩罚，梯度更干净、收敛更快。
    # - target 默认用“参考仪器通道”（仿真）：同带宽、极低非线性/高 SNR，模拟现实用更准仪器采集真值做 foreground calibration。
    def _train_stage3_one_scheme(*, scheme: str, steps: int):
        # scheme 解析：支持 *_ls
        if scheme.endswith("_harm_ls"):
            solver = "harm_ls"
            base = scheme[: -len("_harm_ls")]
        elif scheme.endswith("_ls"):
            solver = "ls"
            base = scheme[: -len("_ls")]
        else:
            solver = "adam"
            base = scheme

        print("\n=== Stage 3: Nonlinear Error Trimming (Volterra / Memory Polynomial) ===")
        print(">> 输入: TIADC 交织输出 (Ch0 + Cal(Ch1))")
        print(f">> Scheme: {scheme} (base={base}, solver={solver}) | steps={steps}")
        if STAGE3_USE_REFERENCE_TARGET:
            print(">> Target: Reference Instrument (same BW, near-ideal linearity)")
        else:
            print(">> Target: Ideal TIADC (Ch0 + Ch0) [上限受 Ch0 非线性锁死]")

        # 参考仪器参数（同带宽，hd2/hd3=0）
        params_ref = {
            'cutoff_freq': params_ch0['cutoff_freq'],
            'delay_samples': 0.0,
            'gain': 1.0,
            'hd2': 0.0,
            'hd3': 0.0,
            'snr_target': STAGE3_REFERENCE_SNR_DB,
        }

        # 交织（全速率：偶取 c0，奇取 c1）
        def interleave_fullrate(c0, c1):
            L = min(len(c0), len(c1))
            out = np.zeros(L, dtype=np.float64)
            out[0::2] = c0[0:L:2]
            out[1::2] = c1[1:L:2]
            return out

        # 构造“相干采样”bin 列表：f = k * fs / N
        n = STAGE3_TONE_N
        fs = simulator.fs
        fmin = 0.2e9
        fmax = 6.2e9
        k_min = int(np.ceil(fmin * n / fs))
        k_max = int(np.floor(fmax * n / fs))
        k_min = max(k_min, 5)
        k_max = min(k_max, n // 2 - 5)
        cand_bins = np.arange(k_min, k_max, dtype=int)

        def _bins_in_band(f_lo, f_hi):
            k_lo = int(np.ceil(f_lo * n / fs))
            k_hi = int(np.floor(f_hi * n / fs))
            k_lo = max(k_lo, k_min)
            k_hi = min(k_hi, k_max)
            b = cand_bins[(cand_bins >= k_lo) & (cand_bins <= k_hi)]
            return b if len(b) > 0 else cand_bins

        low_bins = _bins_in_band(0.2e9, 1.0e9)
        f0 = STAGE3_BIAS_F0_GHZ * 1e9
        bw = STAGE3_BIAS_BW_GHZ * 1e9
        mid_bins = _bins_in_band(max(0.2e9, f0 - bw), min(fmax, f0 + bw))
        high_bins = _bins_in_band(3.5e9, 6.2e9)

        if base == "pinn_mem_poly":
            # 物理信息版 Post-NL：PTV 记忆多项式（可微分 + 反向传播仅优化少量物理参数）
            ddsp_model = DifferentiableMemoryPolynomial(memory_depth=int(STAGE3_TAPS), nonlinear_order=3, ptv=True).to(device)
        elif base == "ptv_volterra":
            ddsp_model = DDSPVolterraNetwork(taps=STAGE3_TAPS).to(device)
        elif base == "ptv_poly":
            ddsp_model = PTVMemorylessPoly23().to(device)
        elif base == "odd_volterra":
            ddsp_model = OddOnlyVolterraNetwork(taps=STAGE3_TAPS).to(device)
        elif base == "odd_poly":
            ddsp_model = OddOnlyMemorylessPoly23().to(device)
        else:
            raise ValueError(f"Unknown scheme={scheme!r} (base={base!r})")

        # ----------------------------
        # PINN/MP 训练分支（Adam）
        # - 输入链路：Stage1/2 (线性) -> Post-QE(静态物理逆，多项式) -> Post-NL(记忆多项式)
        # ----------------------------
        if base == "pinn_mem_poly":
            if solver != "adam":
                raise ValueError("pinn_mem_poly 仅支持 Adam 训练（不支持 *_ls / *_harm_ls）")

            opt3 = optim.Adam(ddsp_model.parameters(), lr=float(STAGE3_LR), betas=(0.5, 0.9))

            # 训练：用相干采样单音，直接压谐波 + 基波保护 + 最小改动
            for step in range(int(steps)):
                loss_acc = 0.0
                for _ in range(int(STAGE3_BATCH)):
                    # 频点采样（带偏置）
                    if np.random.rand() < float(STAGE3_BIAS_PROB):
                        k = int(np.random.choice(mid_bins))
                    else:
                        r = np.random.rand()
                        if r < 0.25:
                            k = int(np.random.choice(low_bins))
                        elif r < 0.70:
                            k = int(np.random.choice(high_bins))
                        else:
                            k = int(np.random.choice(cand_bins))

                    T = int(n)
                    fin = float(k) * float(fs) / float(T)
                    t = np.arange(T) / float(fs)
                    amp = 0.7 + 0.2 * np.random.rand()
                    src = np.sin(2 * np.pi * fin * t) * amp

                    # 训练口径：真实 ADC 条件（噪声+量化）
                    y0 = simulator.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0)
                    y1 = simulator.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1)

                    with torch.no_grad():
                        x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                        y1_lin = model(x1).cpu().numpy().flatten() * scale

                    tiadc_in = interleave_fullrate(y0, y1_lin)

                    if STAGE3_USE_REFERENCE_TARGET:
                        y_ref = simulator.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ref)
                        target = y_ref
                    else:
                        target = interleave_fullrate(y0, y0)

                    s0 = max(float(np.max(np.abs(tiadc_in))), float(np.max(np.abs(target))), 1e-12)
                    x = torch.FloatTensor((tiadc_in / s0)).view(1, 1, -1).to(device)
                    y = torch.FloatTensor((target / s0)).view(1, 1, -1).to(device)
                    x = torch.clamp(x, -1.0, 1.0)
                    y = torch.clamp(y, -1.0, 1.0)

                    # 先过 Post-QE（如果已训练/挂载）
                    if hasattr(model, "post_qe") and (model.post_qe is not None):
                        with torch.no_grad():
                            x = model.post_qe(x)

                    y_hat = ddsp_model(x)

                    # loss: 残差谐波（更稳） + 基波保护 + 最小改动 + 小幅时域贴合
                    residual = y_hat - y
                    loss_harm = residual_harmonic_loss_torch(residual, y, int(k), max_h=5, guard=int(STAGE3_HARM_GUARD))

                    # 基波保护（对数幅度差）
                    win = torch.hann_window(T, device=device, dtype=x.dtype).view(1, 1, -1)
                    Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
                    Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
                    fund_mag_hat = torch.abs(Yh[..., int(k)]) + 1e-12
                    fund_mag_ref = torch.abs(Yr[..., int(k)]) + 1e-12
                    loss_fund = torch.mean((torch.log10(fund_mag_hat) - torch.log10(fund_mag_ref)) ** 2)

                    loss_delta = torch.mean((y_hat - x) ** 2)
                    loss_time = torch.mean((y_hat - y) ** 2)

                    # 正则：系数 L2（更保守）
                    reg = 0.0
                    # 新 MP 实现：参数名为 raw_even/raw_odd（已做限幅），这里直接对所有参数做 L2 即可
                    for p in ddsp_model.parameters():
                        reg = reg + torch.mean(p ** 2)

                    loss = (
                        float(STAGE3_W_TIME) * loss_time
                        + float(STAGE3_W_HARM) * loss_harm
                        + float(STAGE3_W_FUND) * loss_fund
                        + float(STAGE3_W_DELTA) * loss_delta
                        + float(STAGE3_W_REG) * reg
                    )
                    loss_acc = loss_acc + loss

                loss_acc = loss_acc / float(max(int(STAGE3_BATCH), 1))
                opt3.zero_grad()
                loss_acc.backward()
                torch.nn.utils.clip_grad_norm_(ddsp_model.parameters(), max_norm=1.0)
                opt3.step()

                if step % 150 == 0:
                    with torch.no_grad():
                        w_mean = float(torch.mean(torch.stack([p.abs().mean() for p in ddsp_model.parameters()])).detach().cpu())
                    print(f"Stage3(Post-NL, PINN-MP) step {step:4d}/{int(steps)} | loss={loss_acc.item():.3e} | w_mean~{w_mean:.2e}")

            return ddsp_model, params_ref

        def _solve_ridge(A: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
            P = A.shape[0]
            return np.linalg.solve(A + ridge * np.eye(P, dtype=A.dtype), b)

        def _sliding_windows_1d(x: np.ndarray, taps: int) -> np.ndarray:
            # x: (T,)
            pad = taps // 2
            xp = np.pad(x, (pad, pad), mode="edge")
            return np.lib.stride_tricks.sliding_window_view(xp, taps)  # (T, taps)

        def _fit_ls_once(ks: np.ndarray):
            # 累积法方程：A = Phi^T Phi, b = Phi^T r
            T = STAGE3_TONE_N
            taps = STAGE3_TAPS
            crop = int(STAGE3_LS_CROP)

            if base == "ptv_volterra":
                P = 4 * taps
            elif base == "ptv_poly":
                P = 4
            elif base == "odd_volterra":
                P = 2 * taps
            elif base == "odd_poly":
                P = 2
            else:
                raise ValueError(f"Unknown base={base!r}")

            A = np.zeros((P, P), dtype=np.float64)
            bb = np.zeros((P,), dtype=np.float64)

            even = (np.arange(T) % 2 == 0)
            odd = ~even

            for k in ks:
                t = np.arange(T) / fs
                fin = k * fs / T
                amp = 0.7 + 0.2 * np.random.rand()
                src = np.sin(2 * np.pi * fin * t) * amp

                y0 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch0)
                y1 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch1)

                with torch.no_grad():
                    x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                    y1_lin = model(x1).cpu().numpy().flatten() * scale

                tiadc_in = interleave_fullrate(y0, y1_lin)

                if STAGE3_USE_REFERENCE_TARGET:
                    y_ref = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ref)
                    target = y_ref
                else:
                    target = interleave_fullrate(y0, y0)

                s = max(np.max(np.abs(tiadc_in)), np.max(np.abs(target)), 1e-12)
                x = np.clip((tiadc_in / s).astype(np.float64, copy=False), -1.0, 1.0)
                y = np.clip((target / s).astype(np.float64, copy=False), -1.0, 1.0)

                # 目标是拟合“需要加到 x 上的误差项”
                r = (y - x)

                if crop and crop * 2 < T:
                    sl = slice(crop, T - crop)
                    x = x[sl]; y = y[sl]; r = r[sl]
                    even_loc = even[sl]
                    odd_loc = odd[sl]
                else:
                    even_loc = even
                    odd_loc = odd

                x2 = x ** 2
                x3 = x ** 3

                if base == "ptv_poly":
                    Phi = np.zeros((len(x), 4), dtype=np.float64)
                    Phi[even_loc, 0] = x2[even_loc]
                    Phi[even_loc, 1] = x3[even_loc]
                    Phi[odd_loc, 2] = x2[odd_loc]
                    Phi[odd_loc, 3] = x3[odd_loc]
                elif base == "ptv_volterra":
                    x2w = _sliding_windows_1d(x2, taps)
                    x3w = _sliding_windows_1d(x3, taps)
                    Phi = np.zeros((len(x), 4 * taps), dtype=np.float64)
                    Phi[even_loc, 0:taps] = x2w[even_loc]
                    Phi[even_loc, taps:2 * taps] = x3w[even_loc]
                    Phi[odd_loc, 2 * taps:3 * taps] = x2w[odd_loc]
                    Phi[odd_loc, 3 * taps:4 * taps] = x3w[odd_loc]
                elif base == "odd_poly":
                    # 只拟合 odd 相位，降低耦合（更稳）
                    idx = odd_loc
                    Phi = np.zeros((len(x), 2), dtype=np.float64)
                    Phi[idx, 0] = x2[idx]
                    Phi[idx, 1] = x3[idx]
                elif base == "odd_volterra":
                    idx = odd_loc
                    x2w = _sliding_windows_1d(x2, taps)
                    x3w = _sliding_windows_1d(x3, taps)
                    Phi = np.zeros((len(x), 2 * taps), dtype=np.float64)
                    Phi[idx, 0:taps] = x2w[idx]
                    Phi[idx, taps:2 * taps] = x3w[idx]
                else:
                    raise ValueError(f"Unknown base={base!r}")

                A[:] = A + Phi.T @ Phi
                bb[:] = bb + Phi.T @ r

            w = _solve_ridge(A, bb, float(STAGE3_LS_RIDGE))

            # 写回模型参数
            if base == "ptv_poly":
                with torch.no_grad():
                    ddsp_model.a2_even.copy_(torch.tensor(w[0], device=device, dtype=torch.float32))
                    ddsp_model.a3_even.copy_(torch.tensor(w[1], device=device, dtype=torch.float32))
                    ddsp_model.a2_odd.copy_(torch.tensor(w[2], device=device, dtype=torch.float32))
                    ddsp_model.a3_odd.copy_(torch.tensor(w[3], device=device, dtype=torch.float32))
            elif base == "ptv_volterra":
                taps = STAGE3_TAPS
                w = w.astype(np.float32, copy=False)
                with torch.no_grad():
                    ddsp_model.conv2_even.weight.copy_(torch.from_numpy(w[0:taps]).view(1, 1, taps).to(device))
                    ddsp_model.conv3_even.weight.copy_(torch.from_numpy(w[taps:2 * taps]).view(1, 1, taps).to(device))
                    ddsp_model.conv2_odd.weight.copy_(torch.from_numpy(w[2 * taps:3 * taps]).view(1, 1, taps).to(device))
                    ddsp_model.conv3_odd.weight.copy_(torch.from_numpy(w[3 * taps:4 * taps]).view(1, 1, taps).to(device))
            elif base == "odd_poly":
                with torch.no_grad():
                    ddsp_model.a2.copy_(torch.tensor(w[0], device=device, dtype=torch.float32))
                    ddsp_model.a3.copy_(torch.tensor(w[1], device=device, dtype=torch.float32))
            elif base == "odd_volterra":
                taps = STAGE3_TAPS
                w = w.astype(np.float32, copy=False)
                with torch.no_grad():
                    ddsp_model.conv2.weight.copy_(torch.from_numpy(w[0:taps]).view(1, 1, taps).to(device))
                    ddsp_model.conv3.weight.copy_(torch.from_numpy(w[taps:2 * taps]).view(1, 1, taps).to(device))
            else:
                raise ValueError(f"Unknown base={base!r}")

        def _fit_harm_ls_once(ks: np.ndarray):
            """
            频域（谐波 bin）最小二乘（更稳版）：
            - 目标不是“去拟合 residual”，而是直接最小化校正后输出 y_hat = x + Phi*w 在谐波 bin 上的能量：
                minimize || X_h + F_h * w ||^2
            - 同时加入“最小改动”惩罚，限制 ||Phi*w||^2，避免引入新 spur / 拉坏 SINAD
            w 为实数，按 Re/Im 展开做法方程累积。
            """
            T = STAGE3_TONE_N
            taps = STAGE3_TAPS
            # 频域谐波最小二乘依赖“相干采样/整数 bin”的前提；对时域做 crop 会破坏相干性，
            # 并导致谐波 bin 索引失配。因此在 harm_ls 中强制不 crop。

            if base == "ptv_volterra":
                P = 4 * taps
            else:
                P = 4

            A = np.zeros((P, P), dtype=np.float64)
            bb = np.zeros((P,), dtype=np.float64)

            even = (np.arange(T) % 2 == 0)
            odd = ~even

            win = np.hanning(T).astype(np.float64)

            def _select_bins(fund_bin: int):
                bins = []
                if STAGE3_HARM_LS_INCLUDE_FUND:
                    bins.append(int(fund_bin))
                hb = _harmonic_bins_from_fund_bin(int(fund_bin), int(T), 0.0, max_h=int(STAGE3_HARM_LS_MAX_H))
                for k0 in hb:
                    for kk in range(int(k0) - int(STAGE3_HARM_LS_GUARD), int(k0) + int(STAGE3_HARM_LS_GUARD) + 1):
                        if 2 <= kk <= (T // 2 - 2):
                            bins.append(int(kk))
                # 去重并排序
                return sorted(set(bins))

            for k in ks:
                t = np.arange(T) / fs
                fin = k * fs / T
                amp = 0.7 + 0.2 * np.random.rand()
                src = np.sin(2 * np.pi * fin * t) * amp

                y0 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch0)
                y1 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch1)

                with torch.no_grad():
                    x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                    y1_lin = model(x1).cpu().numpy().flatten() * scale

                tiadc_in = interleave_fullrate(y0, y1_lin)

                if STAGE3_USE_REFERENCE_TARGET:
                    y_ref = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ref)
                    target = y_ref
                else:
                    target = interleave_fullrate(y0, y0)

                s = max(np.max(np.abs(tiadc_in)), np.max(np.abs(target)), 1e-12)
                x = np.clip((tiadc_in / s).astype(np.float64, copy=False), -1.0, 1.0)
                y = np.clip((target / s).astype(np.float64, copy=False), -1.0, 1.0)

                x_loc = x
                even_loc = even
                odd_loc = odd
                win_loc = win

                x2 = x_loc ** 2
                x3 = x_loc ** 3

                if base == "ptv_poly":
                    Phi = np.zeros((len(x_loc), 4), dtype=np.float64)
                    Phi[even_loc, 0] = x2[even_loc]
                    Phi[even_loc, 1] = x3[even_loc]
                    Phi[odd_loc, 2] = x2[odd_loc]
                    Phi[odd_loc, 3] = x3[odd_loc]
                else:
                    x2w = _sliding_windows_1d(x2, taps)
                    x3w = _sliding_windows_1d(x3, taps)
                    Phi = np.zeros((len(x_loc), 4 * taps), dtype=np.float64)
                    Phi[even_loc, 0:taps] = x2w[even_loc]
                    Phi[even_loc, taps:2 * taps] = x3w[even_loc]
                    Phi[odd_loc, 2 * taps:3 * taps] = x2w[odd_loc]
                    Phi[odd_loc, 3 * taps:4 * taps] = x3w[odd_loc]

                # 只选谐波 bin（相干采样前提下，fund_bin = k）
                sel = _select_bins(int(k))

                # 计算每个特征列在 selected bins 上的复频域值（线性）
                # F: (Fbins, P), X: (Fbins,)
                F = np.fft.rfft(Phi * win_loc[:, None], axis=0, norm="ortho")
                X = np.fft.rfft(x_loc * win_loc, norm="ortho")
                # 安全：避免越界（尤其是未来改动 n/crop 时）
                sel = [kk for kk in sel if 0 <= kk < F.shape[0]]
                if len(sel) == 0:
                    continue
                Fsel = F[sel, :]  # complex
                Xsel = X[sel]     # complex

                # w 为实数：A += Re^T Re + Im^T Im, b += Re^T Re(R) + Im^T Im(R)
                ReF = Fsel.real
                ImF = Fsel.imag
                ReX = Xsel.real
                ImX = Xsel.imag

                # 频域谐波最小化：min ||X + Fw||^2 => A += F^H F, b += -F^H X
                A[:] = A + (ReF.T @ ReF) + (ImF.T @ ImF)
                bb[:] = bb + (-(ReF.T @ ReX) - (ImF.T @ ImX))

                # “最小改动”惩罚：限制 ||Phi*w||^2
                if STAGE3_HARM_LS_W_DELTA and STAGE3_HARM_LS_W_DELTA > 0:
                    A[:] = A + float(STAGE3_HARM_LS_W_DELTA) * (Phi.T @ Phi)

            w = _solve_ridge(A, bb, float(STAGE3_LS_RIDGE))

            # 写回模型参数
            if base == "ptv_poly":
                with torch.no_grad():
                    ddsp_model.a2_even.copy_(torch.tensor(w[0], device=device, dtype=torch.float32))
                    ddsp_model.a3_even.copy_(torch.tensor(w[1], device=device, dtype=torch.float32))
                    ddsp_model.a2_odd.copy_(torch.tensor(w[2], device=device, dtype=torch.float32))
                    ddsp_model.a3_odd.copy_(torch.tensor(w[3], device=device, dtype=torch.float32))
            else:
                w = w.astype(np.float32, copy=False)
                with torch.no_grad():
                    ddsp_model.conv2_even.weight.copy_(torch.from_numpy(w[0:taps]).view(1, 1, taps).to(device))
                    ddsp_model.conv3_even.weight.copy_(torch.from_numpy(w[taps:2 * taps]).view(1, 1, taps).to(device))
                    ddsp_model.conv2_odd.weight.copy_(torch.from_numpy(w[2 * taps:3 * taps]).view(1, 1, taps).to(device))
                    ddsp_model.conv3_odd.weight.copy_(torch.from_numpy(w[3 * taps:4 * taps]).view(1, 1, taps).to(device))

        if solver == "ls":
            # steps 解释为“轮数”，每轮重新采样 tone 并解一次岭回归（通常 1~3 就够稳）
            rounds = max(1, int(steps))
            for rr in range(rounds):
                ks = np.random.choice(cand_bins, size=int(STAGE3_LS_TONE_BATCH), replace=True).astype(int)
                _fit_ls_once(ks)
                if rr == 0 or rr == rounds - 1:
                    if base == "ptv_volterra":
                        w_mean = ddsp_model.conv2_even.weight.abs().mean().item()
                    else:
                        w_mean = float(torch.mean(torch.stack([p.abs().mean() for p in ddsp_model.parameters()])).item())
                    print(f"Stage 3 LS round {rr+1}/{rounds}: W_mean={w_mean:.2e}")
            return ddsp_model, params_ref

        if solver == "harm_ls":
            if base in ("odd_poly", "odd_volterra"):
                raise ValueError("odd_* 暂不支持 _harm_ls（建议用 *_ls：更稳且更符合 odd-only 设计）")
            rounds = max(1, int(steps))
            for rr in range(rounds):
                ks = np.random.choice(cand_bins, size=int(STAGE3_LS_TONE_BATCH), replace=True).astype(int)
                _fit_harm_ls_once(ks)
                if rr == 0 or rr == rounds - 1:
                    if base == "ptv_volterra":
                        w_mean = ddsp_model.conv2_even.weight.abs().mean().item()
                    else:
                        w_mean = float(torch.mean(torch.stack([p.abs().mean() for p in ddsp_model.parameters()])).item())
                    print(f"Stage 3 HarmLS round {rr+1}/{rounds}: W_mean={w_mean:.2e}")
            return ddsp_model, params_ref

        # 退化为 Adam 训练（旧逻辑）
        opt_nl = optim.Adam(ddsp_model.parameters(), lr=STAGE3_LR)

        for step in range(steps):
            n_low = 2
            n_high = 2
            n_mid = max(STAGE3_BATCH - n_low - n_high, 1)
            ks = np.concatenate([
                np.random.choice(low_bins, size=n_low, replace=True),
                np.random.choice(high_bins, size=n_high, replace=True),
                np.random.choice(mid_bins, size=n_mid, replace=True),
            ]).astype(int)

            loss_acc = 0.0
            for k in ks:
                t = np.arange(n) / fs
                fin = k * fs / n
                amp = 0.7 + 0.2 * np.random.rand()
                src = np.sin(2 * np.pi * fin * t) * amp

                y0 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch0)
                y1 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch1)

                with torch.no_grad():
                    x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                    y1_lin = model(x1).cpu().numpy().flatten() * scale

                tiadc_in = interleave_fullrate(y0, y1_lin)

                if STAGE3_USE_REFERENCE_TARGET:
                    y_ref = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ref)
                    target = y_ref
                else:
                    target = interleave_fullrate(y0, y0)

                s = max(np.max(np.abs(tiadc_in)), np.max(np.abs(target)), 1e-12)
                x = torch.FloatTensor(tiadc_in / s).view(1, 1, -1).to(device)
                y = torch.FloatTensor(target / s).view(1, 1, -1).to(device)

                y_hat = ddsp_model(x)

                loss_time = torch.mean((y_hat - y) ** 2)
                res = y_hat - y
                loss_harm = residual_harmonic_loss_torch(
                    res, y, int(k), max_h=5, guard=STAGE3_HARM_GUARD
                )

                win = torch.hann_window(n, device=device).view(1, 1, -1)
                Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
                Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
                fund_mag_hat = torch.abs(Yh[..., int(k)]) + 1e-12
                fund_mag_ref = torch.abs(Yr[..., int(k)]) + 1e-12
                loss_fund = torch.mean((torch.log10(fund_mag_hat) - torch.log10(fund_mag_ref)) ** 2)

                reg = 0.0
                if isinstance(ddsp_model, DDSPVolterraNetwork):
                    for m in [ddsp_model.conv2_even, ddsp_model.conv3_even, ddsp_model.conv2_odd, ddsp_model.conv3_odd]:
                        reg = reg + torch.mean(m.weight ** 2)
                else:
                    for p in ddsp_model.parameters():
                        reg = reg + torch.mean(p ** 2)

                loss_delta = torch.mean((y_hat - x) ** 2)

                loss = (
                    STAGE3_W_TIME * loss_time
                    + STAGE3_W_HARM * loss_harm
                    + STAGE3_W_FUND * loss_fund
                    + STAGE3_W_DELTA * loss_delta
                    + STAGE3_W_REG * reg
                )
                loss_acc = loss_acc + loss

            loss_acc = loss_acc / float(STAGE3_BATCH)
            opt_nl.zero_grad()
            loss_acc.backward()
            torch.nn.utils.clip_grad_norm_(ddsp_model.parameters(), max_norm=1.0)
            opt_nl.step()

            if step % 150 == 0:
                if isinstance(ddsp_model, DDSPVolterraNetwork):
                    w_mean = ddsp_model.conv2_even.weight.abs().mean().item()
                else:
                    w_mean = float(torch.mean(torch.stack([p.abs().mean() for p in ddsp_model.parameters()])).item())
                print(f"Stage 3 Step {step}: Loss={loss_acc.item():.6f}, W_mean={w_mean:.2e}")

        return ddsp_model, params_ref

    ddsp_model = None
    # 运行链路收敛：Stage1/2 + Post-QE(物理静态逆) + Stage3(Post-NL 记忆多项式)
    # 旧逻辑：启用 Post-EQ 时关闭 Stage3。现在 Post-EQ 已被替换为 Post-QE，且 Stage3 是“新方案”必选项，因此不再互斥。
    if ENABLE_STAGE3_NONLINEAR:
        if STAGE3_AUTOSEARCH:
            print("\n=== Stage 3: AutoSearch (fixed Stage1/2, search best Stage3 config) ===")
            best_cfg = _autosearch_stage3()
            ddsp_model = getattr(model, "post_linearizer", None)
            if best_cfg is not None:
                print(f"[AutoSearch] chosen config: {best_cfg}")
            else:
                print("[AutoSearch] chosen config: None (Stage3 disabled)")
        elif STAGE3_SWEEP:
            # 只训练一次线性，然后对比不同 scheme，选一个“全带更稳”的
            best = None
            best_score = None
            best_scheme = None
            best_ref = None

            for scheme in STAGE3_SWEEP_SCHEMES:
                dd, params_ref = _train_stage3_one_scheme(scheme=scheme, steps=int(STAGE3_SWEEP_STEPS))
                # 临时挂载用于评估
                model.post_linearizer = dd
                model.post_linearizer_scale = None
                model.reference_params = params_ref if STAGE3_USE_REFERENCE_TARGET else None

                tf, mm = calculate_metrics_detailed(simulator, model, scale, device, params_ch0, params_ch1)
                tf = np.asarray(tf)
                inband = tf <= 6.2e9

                sin_lin = float(np.mean(np.asarray(mm['sinad_post'])[inband]))
                sin_nl = float(np.mean(np.asarray(mm['sinad_post_nl'])[inband]))
                thd_nl = float(np.mean(np.asarray(mm['thd_post_nl'])[inband]))  # dBc（越小越好，越负越好）
                sfdr_nl = float(np.mean(np.asarray(mm['sfdr_post_nl'])[inband]))

                # 简单评分：优先不伤 SINAD，其次压 THD，同时看 SFDR
                sin_penalty = max(0.0, sin_lin - sin_nl)  # 若非线性让 SINAD 下降则惩罚
                score = (-thd_nl) + 0.15 * sfdr_nl - 2.0 * sin_penalty
                print(f"[Stage3 Sweep] scheme={scheme} | inband mean SINAD lin={sin_lin:.2f} nl={sin_nl:.2f} | THD_nl={thd_nl:.2f} | SFDR_nl={sfdr_nl:.2f} | score={score:.3f}")

                if (best_score is None) or (score > best_score):
                    best_score = score
                    best = dd
                    best_scheme = scheme
                    best_ref = params_ref

            ddsp_model = best
            model.post_linearizer = ddsp_model
            model.post_linearizer_scale = None
            model.reference_params = best_ref if STAGE3_USE_REFERENCE_TARGET else None
            print(f"[Stage3 Sweep] BEST scheme={best_scheme} | score={best_score:.3f}")
        else:
            ddsp_model, params_ref = _train_stage3_one_scheme(scheme=STAGE3_SCHEME, steps=int(STAGE3_STEPS))
            model.post_linearizer = ddsp_model
            model.post_linearizer_scale = None
            model.reference_params = params_ref if STAGE3_USE_REFERENCE_TARGET else None
    else:
        # 明确关闭旧 Stage3
        model.post_linearizer = None
        model.post_linearizer_scale = None

    print(f"训练完成。返回 5 个对象。")
    return model, ddsp_model, scale, params_ch0, params_ch1


def train_absolute_calibration(simulator, device='cpu'):
    """
    方向B：分别训练两套校准器
      - Cal0: Ch0 -> Ref
      - Cal1: Ch1 -> Ref
    然后验证时做 interleave(Cal0(Ch0), Cal1(Ch1))
    """
    raise RuntimeError("absolute 路径已移除：请使用相对校准（train_relative_calibration）")
    N_train = 32768
    M_average = 16

    # 模拟真实的物理通道（待校准对象）- 更贴近实采数据的参数设置
    # 带宽明显高于测试上限，避免滤波器截止影响
    # **非线性失配加强**：让Post-NL有明显改善空间
    params_ch0 = {'cutoff_freq': 8.0e9, 'delay_samples': 0.0, 'gain': 1.0, 'hd2': 1e-3, 'hd3': 0.5e-3}
    params_ch1 = {'cutoff_freq': 7.8e9, 'delay_samples': 0.25, 'gain': 0.98, 'hd2': 5e-3, 'hd3': 3e-3}

    # 更接近真实仪器的参考通道（同带宽，低非线性、低抖动、高分辨率，但不是“完美无噪声”）
    params_ref = {
        'cutoff_freq': params_ch0['cutoff_freq'],
        'delay_samples': 0.0,
        'gain': 1.0,
        'hd2': 0.0,
        'hd3': 0.0,
        # 训练目标与输入必须同等噪声口径；不让模型“去噪/反量化”
        'snr_target': None,
    }

    print("开始绝对校准训练: Mapping (Ch0, Ch1) -> Reference Instrument")

    np.random.seed(42)
    white_noise = np.random.randn(N_train)
    # 训练信号带宽限制在7GHz（略高于测试范围6.5GHz）
    b_src, a_src = signal.butter(6, 7.0e9 / (simulator.fs / 2), btype='low')
    base_sig = signal.lfilter(b_src, a_src, white_noise)
    base_sig = base_sig / np.max(np.abs(base_sig)) * 0.9
    base_sig = base_sig + 0.5e-3 * base_sig**2

    # 多次采集平均，模拟现实“多段采样平均/叠加”降低随机噪声
    ch0_caps, ch1_caps, ref_caps = [], [], []
    for _ in range(M_average):
        y0 = simulator.apply_channel_effect(base_sig, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0)
        y1 = simulator.apply_channel_effect(base_sig, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1)
        yr = simulator.apply_channel_effect(base_sig, jitter_std=REF_JITTER_STD, n_bits=REF_NBITS, **params_ref)
        ch0_caps.append(y0)
        ch1_caps.append(y1)
        ref_caps.append(yr)

    avg_ch0 = np.mean(np.stack(ch0_caps), axis=0)
    avg_ch1 = np.mean(np.stack(ch1_caps), axis=0)
    avg_ref = np.mean(np.stack(ref_caps), axis=0)

    # 以参考为尺度（更贴近“对齐到仪器真值”的做法）
    scale = np.max(np.abs(avg_ref)) + 1e-12
    tgt_t = torch.FloatTensor(avg_ref / scale).view(1, 1, -1).to(device)

    inp0_t = torch.FloatTensor(avg_ch0 / scale).view(1, 1, -1).to(device)
    inp1_t = torch.FloatTensor(avg_ch1 / scale).view(1, 1, -1).to(device)

    model0 = HybridCalibrationModel(taps=63, fpga_simulation=False).to(device)
    model1 = HybridCalibrationModel(taps=63, fpga_simulation=False).to(device)
    loss_fn = ComplexMSELoss(device=device)

    def _train_one(name, model, inp):
        print(f"\n=== {name}: Stage 1 (Delay & Gain) ===")
        opt1 = optim.Adam([
            {'params': model.global_delay.parameters(), 'lr': 1e-2},
            {'params': model.gain, 'lr': 1e-2},
            {'params': model.conv.parameters(), 'lr': 0.0},
            {'params': model.poly.parameters(), 'lr': 0.0},
        ])
        for _ in range(301):
            opt1.zero_grad()
            loss = loss_fn(model(inp), tgt_t, model)
            loss.backward()
            opt1.step()

        print(f"=== {name}: Stage 2 (FIR) ===")
        opt2 = optim.Adam([
            {'params': model.global_delay.parameters(), 'lr': 0.0},
            {'params': model.gain, 'lr': 0.0},
            {'params': model.conv.parameters(), 'lr': 5e-4},
            {'params': model.poly.parameters(), 'lr': 0.0},
        ], betas=(0.5, 0.9))
        for _ in range(1001):
            opt2.zero_grad()
            loss = loss_fn(model(inp), tgt_t, model)
            loss.backward()
            opt2.step()

        # --- Stage 2b: 高频线性精修（可选）---
        if FIR_HF_REFINE_STEPS and FIR_HF_REFINE_STEPS > 0:
            # 目的：把高频线性幅相尽量贴近 Ref，避免 Stage3 非线性分支去补线性误差导致 THD 退化
            print(f"=== {name}: Stage 2b (FIR HF-Weighted Refinement) ===")
            opt2b = optim.Adam([
                {'params': model.conv.parameters(), 'lr': FIR_HF_REFINE_LR},
            ])

            n_eff = 8192
            pad = 2048
            n_total = n_eff + 2 * pad
            fs = simulator.fs
            t = np.arange(n_total) / fs
            chirp = signal.chirp(t, f0=0.3e9, t1=t[-1], f1=6.2e9) * 0.9

            # 固定一条训练样本（稳定、快速）；实际工程可改为多样本/多幅度
            if "Cal0" in name:
                y_raw = simulator.apply_channel_effect(chirp, jitter_std=0, n_bits=None, **params_ch0)[pad:pad + n_eff]
            else:
                y_raw = simulator.apply_channel_effect(chirp, jitter_std=0, n_bits=None, **params_ch1)[pad:pad + n_eff]
            y_ref = simulator.apply_channel_effect(chirp, jitter_std=0, n_bits=None, **params_ref)[pad:pad + n_eff]

            s = max(np.max(np.abs(y_ref)), 1e-12)
            x_fix = torch.FloatTensor(y_raw / s).view(1, 1, -1).to(device)
            y_fix = torch.FloatTensor(y_ref / s).view(1, 1, -1).to(device)

            # 频域加权：越靠近高频权重越大
            w = torch.linspace(0.2, 1.0, steps=(n_eff // 2 + 1), device=device) ** 2

            for _ in range(FIR_HF_REFINE_STEPS):
                opt2b.zero_grad()
                y_hat = model(x_fix)
                # 轻时域约束 + 高频加权频域约束
                loss_t = torch.mean((y_hat - y_fix) ** 2)
                Yh = torch.fft.rfft(y_hat, dim=-1, norm="ortho")
                Yr = torch.fft.rfft(y_fix, dim=-1, norm="ortho")
                diff = (Yh - Yr)
                loss_f = torch.mean((diff.real**2 + diff.imag**2) * w)
                loss = 10.0 * loss_t + 200.0 * loss_f
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.conv.parameters(), max_norm=1.0)
                opt2b.step()

        # === Stage 3: 后级非线性修整（与线性校准解耦） ===
        # 目标：验证非线性方案，但绝不允许“反向影响” Stage1/2 的线性校准解。
        if ENABLE_HAMMERSTEIN_NL:
            print(f"=== {name}: Stage 3 (Post Nonlinear Trimming - Decoupled) ===")

            post_nl = NonlinearPostCorrector(
                taps=HAMMERSTEIN_TAPS,
                enable_hammerstein=True,
            ).to(device)

            opt3 = optim.Adam(post_nl.parameters(), lr=HAMMERSTEIN_STAGE3_LR)

            n_eff = 8192
            pad = 2048
            n_total = n_eff + 2 * pad
            fs = simulator.fs

            fmin = 0.3e9
            fmax = 6.2e9
            k_min = max(int(np.ceil(fmin * n_eff / fs)), 5)
            k_max = min(int(np.floor(fmax * n_eff / fs)), n_eff // 2 - 5)
            cand_bins = np.arange(k_min, k_max, dtype=int)

            def _bins(f_lo, f_hi):
                kk0 = max(int(np.ceil(f_lo * n_eff / fs)), k_min)
                kk1 = min(int(np.floor(f_hi * n_eff / fs)), k_max)
                bb = cand_bins[(cand_bins >= kk0) & (cand_bins <= kk1)]
                return bb if len(bb) > 0 else cand_bins

            low_bins = _bins(0.3e9, 1.5e9)
            mid_bins = _bins(1.5e9, 3.5e9)
            high_bins = _bins(3.5e9, 6.2e9)

            for step in range(HAMMERSTEIN_STAGE3_STEPS):
                ks = []
                ks.append(int(np.random.choice(low_bins)))
                ks.append(int(np.random.choice(mid_bins)))
                ks.append(int(np.random.choice(high_bins)))
                while len(ks) < HAMMERSTEIN_STAGE3_BATCH:
                    if np.random.rand() < HAMMERSTEIN_HIGH_PROB:
                        ks.append(int(np.random.choice(high_bins)))
                    else:
                        ks.append(int(np.random.choice(cand_bins)))
                ks = np.array(ks, dtype=int)

                loss_acc = 0.0
                for k in ks:
                    t = np.arange(n_total) / fs
                    fin = k * fs / n_eff
                    amp = 0.75 + 0.15 * np.random.rand()
                    src = np.sin(2 * np.pi * fin * t) * amp

                    if "Cal0" in name:
                        y_raw = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch0)
                    else:
                        y_raw = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch1)

                    y_ref = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ref)

                    y_raw = y_raw[pad:pad + n_eff]
                    y_ref = y_ref[pad:pad + n_eff]

                    s = max(np.max(np.abs(y_ref)), 1e-12)
                    x_raw = torch.FloatTensor(y_raw / s).view(1, 1, -1).to(device)
                    y = torch.FloatTensor(y_ref / s).view(1, 1, -1).to(device)

                    # 固定线性校准输出（不回传梯度）
                    with torch.no_grad():
                        y_lin = model.forward_linear(x_raw)

                    y_hat = post_nl(y_lin)

                    loss_time = torch.mean((y_hat - y) ** 2)
                    loss_harm = harmonic_power_loss_torch(y_hat, int(k), max_h=5, guard=HAMMERSTEIN_HARM_GUARD)

                    win = torch.hann_window(n_eff, device=device).view(1, 1, -1)
                    Yh = torch.fft.rfft(y_hat * win, dim=-1, norm="ortho")
                    Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
                    fund_mag_hat = torch.abs(Yh[..., int(k)]) + 1e-12
                    fund_mag_ref = torch.abs(Yr[..., int(k)]) + 1e-12
                    loss_fund = torch.mean((torch.log10(fund_mag_hat) - torch.log10(fund_mag_ref)) ** 2)

                    # 最小改动：限制相对线性输出的变化
                    loss_delta = torch.mean((y_hat - y_lin) ** 2)

                    reg = torch.mean(post_nl.a2 ** 2) + torch.mean(post_nl.a3 ** 2)
                    if post_nl.hammerstein is not None:
                        reg = reg + torch.mean(post_nl.hammerstein.conv2.weight ** 2) + torch.mean(post_nl.hammerstein.conv3.weight ** 2)

                    loss = (
                        HAMMERSTEIN_W_TIME * loss_time
                        + HAMMERSTEIN_W_HARM * loss_harm
                        + HAMMERSTEIN_W_FUND * loss_fund
                        + HAMMERSTEIN_W_DELTA * loss_delta
                        + HAMMERSTEIN_W_REG * reg
                    )
                    loss_acc = loss_acc + loss

                loss_acc = loss_acc / float(HAMMERSTEIN_STAGE3_BATCH)
                opt3.zero_grad()
                loss_acc.backward()
                torch.nn.utils.clip_grad_norm_(post_nl.parameters(), max_norm=1.0)
                opt3.step()

                if step % 150 == 0:
                    w_mean = 0.0
                    if post_nl.hammerstein is not None:
                        w_mean = post_nl.hammerstein.conv2.weight.abs().mean().item()
                    print(f"{name} PostNL step {step}: loss={loss_acc.item():.6e}, nl_w={w_mean:.2e}")

            # 挂载到模型上：后级非线性修整器（保持线性校准解不变）
            model.post_nonlinear = post_nl
        else:
            model.post_nonlinear = None

    _train_one("Cal0 (Ch0->Ref)", model0, inp0_t)
    _train_one("Cal1 (Ch1->Ref)", model1, inp1_t)

    # 保存参考参数，便于验证阶段输出基线
    model0.reference_params = params_ref
    model1.reference_params = params_ref

    # 可选：交织后线性微调（主要改善残余镜像 spur / 线性失配）
    post_fir = None
    if ENABLE_POST_INTERLEAVE_FIR:
        print("\n=== Post-Interleave Linear FIR (Residual Image Spur Cleanup) ===")
        post_fir = nn.Conv1d(1, 1, kernel_size=POST_INTERLEAVE_TAPS, padding=POST_INTERLEAVE_TAPS//2, bias=False).to(device)
        with torch.no_grad():
            post_fir.weight.zero_()
            post_fir.weight[0, 0, POST_INTERLEAVE_TAPS//2] = 1.0
        opt_p = optim.Adam(post_fir.parameters(), lr=POST_INTERLEAVE_LR)

        def interleave_fullrate(c0, c1):
            L = min(len(c0), len(c1))
            out = np.zeros(L, dtype=np.float64)
            out[0::2] = c0[0:L:2]
            out[1::2] = c1[1:L:2]
            return out

        n = 16384
        fs = simulator.fs
        fmin = 0.3e9
        fmax = 6.2e9
        k_min = max(int(np.ceil(fmin * n / fs)), 5)
        k_max = min(int(np.floor(fmax * n / fs)), n // 2 - 5)
        cand_bins = np.arange(k_min, k_max, dtype=int)

        for step in range(POST_INTERLEAVE_STEPS):
            k = int(np.random.choice(cand_bins))
            t = np.arange(n) / fs
            fin = k * fs / n
            amp = 0.75 + 0.15 * np.random.rand()
            src = np.sin(2 * np.pi * fin * t) * amp

            y0 = simulator.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch0)
            y1 = simulator.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **params_ch1)
            yref = simulator.apply_channel_effect(src, jitter_std=REF_JITTER_STD, n_bits=REF_NBITS, **params_ref)

            with torch.no_grad():
                x0 = torch.FloatTensor(y0 / scale).view(1, 1, -1).to(device)
                x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                y0c = model0(x0).cpu().numpy().flatten() * scale
                y1c = model1(x1).cpu().numpy().flatten() * scale

            tiadc = interleave_fullrate(y0c, y1c)
            s = max(np.max(np.abs(yref)), 1e-12)
            x = torch.FloatTensor(tiadc / s).view(1, 1, -1).to(device)
            y = torch.FloatTensor(yref / s).view(1, 1, -1).to(device)

            y_hat = post_fir(x)
            loss = torch.mean((y_hat - y) ** 2)
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()
            if step % 100 == 0:
                print(f"PostFIR step {step}: loss={loss.item():.6e}")

    return model0, model1, post_fir, scale, params_ch0, params_ch1, params_ref

# ==============================================================================
# 5. 验证工具 (新增 THD 和 频谱分析)
# ==============================================================================
def calculate_metrics_detailed(sim, model, scale, device, p_ch0, p_ch1):
    """
    [Corrected] 计算 SINAD, ENOB, THD, SFDR 的对比数据 (Pre vs Post)
    基于 TIADC 交织后的全速率数据进行计算，正确反映失配带来的影响。
    """
    print("\n=== 正在计算综合指标对比 (基于 TIADC 交织输出) ===")
    
    # 开启 FPGA 仿真模式进行验证，查看定点化影响
    model.fpga_simulation = True 
    print(">> 注意: 验证阶段已开启 FPGA 定点化模拟 (Fixed-Point Simulation)")
    
    # 限制测试频率在有效输入带宽内（避免超过滤波器截止频率导致性能崩溃）
    # 对于6GHz带宽限制，测试到6.5GHz已经足够展示性能，更贴近实采数据
    test_freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    
    # 存储结果的字典
    metrics = {
        'sinad_pre': [], 'sinad_post': [],
        'enob_pre': [],  'enob_post': [],
        'thd_pre': [],   'thd_post': [],
        'sfdr_pre': [],  'sfdr_post': []
    }
    # [新增] 若启用了交织后 Post-EQ（post_fir），单独给出 Post-EQ 曲线（避免被 Stage3 覆盖看不到）
    metrics.update({
        'sinad_post_eq': [], 'enob_post_eq': [], 'thd_post_eq': [], 'sfdr_post_eq': []
    })

    # 若启用了非线性后级，额外给出一套“线性+非线性”的指标，保证线性口径不被覆盖
    metrics.update({
        'sinad_post_nl': [], 'enob_post_nl': [], 'thd_post_nl': [], 'sfdr_post_nl': []
    })

    # 若 Stage3 使用了参考仪器 target，则这里同时给出参考基线（用于对照上限）
    p_ref = getattr(model, 'reference_params', None)
    if p_ref is not None:
        metrics.update({'sinad_ref': [], 'enob_ref': [], 'thd_ref': [], 'sfdr_ref': []})
    
    # 辅助：交织函数 (修正版：包含抽取)
    def interleave(c0_full, c1_full):
        # 模拟 20GSps TIADC：
        # ADC0 取偶数点 (0, 2, 4...)
        # ADC1 取奇数点 (1, 3, 5...)
        
        s0 = c0_full[0::2] # 降采样到 10GSps
        s1 = c1_full[1::2] # 降采样到 10GSps
        
        L = min(len(s0), len(s1))
        out = np.zeros(2*L)
        out[0::2] = s0[:L]
        out[1::2] = s1[:L]
        return out

    # 辅助：基于频谱计算指标
    def calc_spectrum_metrics(sig, fs, input_freq):
        # 加窗 (Blackman)
        win = np.blackman(len(sig))
        # 归一化因子 (Coherent Gain)
        cg = np.mean(win)
        
        # FFT
        fft_spec = np.fft.rfft(sig * win)
        fft_mag = np.abs(fft_spec) / (len(sig)/2 * cg) # 归一化幅度
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0/fs)
        
        # 1. 寻找基波峰值
        idx_fund = np.argmin(np.abs(fft_freqs - input_freq))
        span = 5
        s = max(0, idx_fund - span)
        e = min(len(fft_mag), idx_fund + span)
        idx_peak = s + np.argmax(fft_mag[s:e])
        
        p_fund = fft_mag[idx_peak]**2
        
        # 2. 计算噪声+失真 (NAD)
        # 将基波及其附近的能量剔除
        mask = np.ones_like(fft_mag, dtype=bool)
        mask[:5] = False # DC
        mask[idx_peak-span : idx_peak+span+1] = False # Fund
        
        # 剩余所有频点能量和
        p_noise_dist = np.sum(fft_mag[mask]**2) + 1e-20
        
        # SINAD
        sinad = 10 * np.log10(p_fund / p_noise_dist)
        
        # ENOB
        enob = (sinad - 1.76) / 6.02
        
        # SFDR (最大杂散)
        p_spur_max = np.max(fft_mag[mask]**2) if np.any(mask) else 1e-20
        sfdr = 10 * np.log10(p_fund / p_spur_max)
        
        # THD (前5次谐波，含混叠折叠到 Nyquist 内)
        #
        # 旧实现仅统计 Nyquist 内的“原始谐波频率”，当输入频率较高（如 > fs/4）时，
        # 2次/3次谐波会落到 fs/2 以外并发生混叠回带内；若不折叠，会出现 p_harm_sum=0，
        # 从而得到接近 -200 dBc 的“假 THD”。
        p_harm_sum = 0.0
        for h in range(2, 6):
            hf_raw = input_freq * h
            # 折叠到 [0, fs/2]
            hf = hf_raw % fs
            if hf > fs / 2:
                hf = fs - hf
            # 排除 DC/极靠近 Nyquist 的点（容易受窗函数/边界影响）
            if hf < 1e6 or hf > fs / 2 - 1e6:
                continue
            idx_h = np.argmin(np.abs(fft_freqs - hf))
            s_h = max(0, idx_h - span)
            e_h = min(len(fft_mag), idx_h + span + 1)
            if e_h > s_h:
                # 用邻域最大值近似该谐波功率（与 fund 的取法一致）
                p_harm_sum += np.max(fft_mag[s_h:e_h]) ** 2

        thd = 10 * np.log10((p_harm_sum + 1e-20) / (p_fund + 1e-20))
        
        return sinad, enob, thd, sfdr
    
    for f in test_freqs:
        # 生成稍长的数据
        src = sim.generate_tone_data(f, N=8192*2)
        
        # Ch0 (Ref)
        # 验证口径必须与训练一致：使用真实 ADC 条件（噪声+量化）
        y_ch0 = sim.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p_ch0)
        # Ch1 (Bad)
        y_ch1_raw = sim.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p_ch1)
        
        # Ch1 (Cal)
        with torch.no_grad():
            inp_t = torch.FloatTensor(y_ch1_raw/scale).view(1,1,-1).to(device)
            y_ch1_cal_tensor = model(inp_t)
            y_ch1_cal = y_ch1_cal_tensor.cpu().numpy().flatten() * scale
            
        # 截断稳定区
        margin = 500
        c0 = y_ch0[margin:-margin]
        c1_raw = y_ch1_raw[margin:-margin]
        c1_cal = y_ch1_cal[margin:-margin]
        
        # 构建 TIADC 全速率信号
        sig_pre = interleave(c0, c1_raw)
        sig_post_lin = interleave(c0, c1_cal)

        # 先算 Post-QE（替代原 Post-EQ）：物理信息静态逆模型层
        # 兼容：如果仍挂载了旧的 post_fir/post_fir_even/odd，也保留旧分支
        if hasattr(model, "post_qe") and (model.post_qe is not None):
            s_qe = max(np.max(np.abs(sig_post_lin)), 1e-12)
            with torch.no_grad():
                xq = torch.FloatTensor(sig_post_lin / s_qe).view(1, 1, -1).to(device)
                yq = model.post_qe(xq).cpu().numpy().flatten() * s_qe
            sig_post_eq = yq
        elif hasattr(model, "post_fir_even") and hasattr(model, "post_fir_odd") and (model.post_fir_even is not None) and (model.post_fir_odd is not None):
            sig_post_eq = apply_post_eq_fir_ptv2(sig_post_lin, post_fir_even=model.post_fir_even, post_fir_odd=model.post_fir_odd, device=device)
        elif hasattr(model, "post_fir") and model.post_fir is not None:
            sig_post_eq = apply_post_eq_fir(sig_post_lin, post_fir=model.post_fir, device=device)
        else:
            sig_post_eq = sig_post_lin

        # 再算 Post-NL（交织后非线性修整）：与新链路保持一致，作用于 Post-QE 的输出
        if hasattr(model, 'post_linearizer') and model.post_linearizer is not None:
            scale_pl = getattr(model, 'post_linearizer_scale', None)
            if scale_pl is None:
                scale_pl = max(np.max(np.abs(sig_post_eq)), 1e-12)
            sig_post_t = torch.FloatTensor(sig_post_eq / scale_pl).view(1, 1, -1).to(device)
            with torch.no_grad():
                sig_final_t = model.post_linearizer(sig_post_t)
            sig_post_nl = sig_final_t.cpu().numpy().flatten() * scale_pl
        else:
            sig_post_nl = sig_post_eq

        # 参考基线（可选）
        if p_ref is not None:
            # 参考基线也保持与 ADC 同口径（避免“完美参考”诱导误判）
            y_ref = sim.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p_ref)
            ref_seg = y_ref[margin:-margin]
            s_ref, e_ref, t_ref, sf_ref = calc_spectrum_metrics(ref_seg, sim.fs, f)
            metrics['sinad_ref'].append(s_ref)
            metrics['enob_ref'].append(e_ref)
            metrics['thd_ref'].append(t_ref)
            metrics['sfdr_ref'].append(sf_ref)
        
        # 计算指标
        s_pre, e_pre, t_pre, sf_pre = calc_spectrum_metrics(sig_pre, sim.fs, f)
        s_post, e_post, t_post, sf_post = calc_spectrum_metrics(sig_post_lin, sim.fs, f)
        s_eq, e_eq, t_eq, sf_eq = calc_spectrum_metrics(sig_post_eq, sim.fs, f)
        s_nl, e_nl, t_nl, sf_nl = calc_spectrum_metrics(sig_post_nl, sim.fs, f)
        
        metrics['sinad_pre'].append(s_pre)
        metrics['sinad_post'].append(s_post)
        metrics['enob_pre'].append(e_pre)
        metrics['enob_post'].append(e_post)
        metrics['thd_pre'].append(t_pre)
        metrics['thd_post'].append(t_post)
        metrics['sfdr_pre'].append(sf_pre)
        metrics['sfdr_post'].append(sf_post)

        metrics['sinad_post_eq'].append(s_eq)
        metrics['enob_post_eq'].append(e_eq)
        metrics['thd_post_eq'].append(t_eq)
        metrics['sfdr_post_eq'].append(sf_eq)

        metrics['sinad_post_nl'].append(s_nl)
        metrics['enob_post_nl'].append(e_nl)
        metrics['thd_post_nl'].append(t_nl)
        metrics['sfdr_post_nl'].append(sf_nl)

    # 验证完成后关闭 FPGA 仿真模式 (或者保持开启取决于后续用途)
    model.fpga_simulation = False
    return test_freqs, metrics


def calculate_metrics_absolute(sim, model0, model1, post_fir, scale, device, p_ch0, p_ch1, p_ref):
    """
    方向B验证：
      Pre : interleave(Ch0_raw, Ch1_raw)
      Post: interleave(Cal0(Ch0_raw), Cal1(Ch1_raw))
      Ref : reference instrument capture（单通道 20GSps）
    """
    raise RuntimeError("absolute 路径已移除：请使用 calculate_metrics_detailed（相对校准口径）")
    print("\n=== 正在计算综合指标对比 (绝对校准: Ch0/Ch1 -> Ref) ===")

    model0.fpga_simulation = True
    model1.fpga_simulation = True
    print(">> 注意: 验证阶段已开启 FPGA 定点化模拟 (Fixed-Point Simulation)")

    # 限制测试频率在有效输入带宽内（避免超过滤波器截止频率导致性能崩溃）
    # 对于6GHz带宽限制，测试到6.5GHz已经足够展示性能，更贴近实采数据
    test_freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    metrics = {
        'sinad_pre': [], 'sinad_post': [], 'sinad_ref': [],
        'enob_pre': [], 'enob_post': [], 'enob_ref': [],
        'thd_pre': [], 'thd_post': [], 'thd_ref': [],
        'sfdr_pre': [], 'sfdr_post': [], 'sfdr_ref': []
    }
    metrics.update({
        'sinad_post_nl': [], 'enob_post_nl': [], 'thd_post_nl': [], 'sfdr_post_nl': []
    })

    def interleave(c0_full, c1_full):
        s0 = c0_full[0::2]
        s1 = c1_full[1::2]
        L = min(len(s0), len(s1))
        out = np.zeros(2 * L)
        out[0::2] = s0[:L]
        out[1::2] = s1[:L]
        return out

    def calc_spectrum_metrics(sig, fs, input_freq):
        win = np.blackman(len(sig))
        cg = np.mean(win)
        fft_spec = np.fft.rfft(sig * win)
        fft_mag = np.abs(fft_spec) / (len(sig) / 2 * cg)
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)

        idx_fund = np.argmin(np.abs(fft_freqs - input_freq))
        span = 5
        s = max(0, idx_fund - span)
        e = min(len(fft_mag), idx_fund + span)
        idx_peak = s + np.argmax(fft_mag[s:e])
        p_fund = fft_mag[idx_peak] ** 2

        mask = np.ones_like(fft_mag, dtype=bool)
        mask[:5] = False
        mask[idx_peak - span: idx_peak + span + 1] = False
        p_noise_dist = np.sum(fft_mag[mask] ** 2) + 1e-20

        sinad = 10 * np.log10(p_fund / p_noise_dist)
        enob = (sinad - 1.76) / 6.02
        p_spur_max = np.max(fft_mag[mask] ** 2) if np.any(mask) else 1e-20
        sfdr = 10 * np.log10(p_fund / p_spur_max)

        # THD with alias folding
        p_harm_sum = 0.0
        for h in range(2, 6):
            hf_raw = input_freq * h
            hf = hf_raw % fs
            if hf > fs / 2:
                hf = fs - hf
            if hf < 1e6 or hf > fs / 2 - 1e6:
                continue
            idx_h = np.argmin(np.abs(fft_freqs - hf))
            s_h = max(0, idx_h - span)
            e_h = min(len(fft_mag), idx_h + span + 1)
            if e_h > s_h:
                p_harm_sum += np.max(fft_mag[s_h:e_h]) ** 2
        thd = 10 * np.log10((p_harm_sum + 1e-20) / (p_fund + 1e-20))

        return sinad, enob, thd, sfdr

    for f in test_freqs:
        src = sim.generate_tone_data(f, N=8192 * 2)

        y0 = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch0)
        y1 = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch1)
        yref = sim.apply_channel_effect(src, jitter_std=REF_JITTER_STD, n_bits=REF_NBITS, **p_ref)

        with torch.no_grad():
            x0 = torch.FloatTensor(y0 / scale).view(1, 1, -1).to(device)
            x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
            y0c = model0(x0).cpu().numpy().flatten() * scale
            y1c = model1(x1).cpu().numpy().flatten() * scale

        # 后级非线性（每通道独立），不影响线性输出 y0c/y1c 的口径
        y0c_nl, y1c_nl = None, None
        if hasattr(model0, 'post_nonlinear') and model0.post_nonlinear is not None:
            s0 = max(np.max(np.abs(y0c)), 1e-12)
            with torch.no_grad():
                y0c_nl = model0.post_nonlinear(torch.FloatTensor(y0c / s0).view(1, 1, -1).to(device)).cpu().numpy().flatten() * s0
        if hasattr(model1, 'post_nonlinear') and model1.post_nonlinear is not None:
            s1 = max(np.max(np.abs(y1c)), 1e-12)
            with torch.no_grad():
                y1c_nl = model1.post_nonlinear(torch.FloatTensor(y1c / s1).view(1, 1, -1).to(device)).cpu().numpy().flatten() * s1

        margin = 500
        pre = interleave(y0[margin:-margin], y1[margin:-margin])
        post = interleave(y0c[margin:-margin], y1c[margin:-margin])
        post_nl = None
        if (y0c_nl is not None) and (y1c_nl is not None):
            post_nl = interleave(y0c_nl[margin:-margin], y1c_nl[margin:-margin])
        if post_fir is not None:
            # 交织后线性微调
            s = max(np.max(np.abs(post)), 1e-12)
            with torch.no_grad():
                x = torch.FloatTensor(post / s).view(1, 1, -1).to(device)
                y = post_fir(x).cpu().numpy().flatten() * s
            post = y
            if post_nl is not None:
                s = max(np.max(np.abs(post_nl)), 1e-12)
                with torch.no_grad():
                    x = torch.FloatTensor(post_nl / s).view(1, 1, -1).to(device)
                    y = post_fir(x).cpu().numpy().flatten() * s
                post_nl = y
        ref = yref[margin:-margin]

        s_pre, e_pre, t_pre, sf_pre = calc_spectrum_metrics(pre, sim.fs, f)
        s_post, e_post, t_post, sf_post = calc_spectrum_metrics(post, sim.fs, f)
        if post_nl is not None:
            s_nl, e_nl, t_nl, sf_nl = calc_spectrum_metrics(post_nl, sim.fs, f)
        else:
            s_nl, e_nl, t_nl, sf_nl = s_post, e_post, t_post, sf_post
        s_ref, e_ref, t_ref, sf_ref = calc_spectrum_metrics(ref, sim.fs, f)

        metrics['sinad_pre'].append(s_pre); metrics['sinad_post'].append(s_post); metrics['sinad_ref'].append(s_ref)
        metrics['enob_pre'].append(e_pre);   metrics['enob_post'].append(e_post);   metrics['enob_ref'].append(e_ref)
        metrics['thd_pre'].append(t_pre);    metrics['thd_post'].append(t_post);    metrics['thd_ref'].append(t_ref)
        metrics['sfdr_pre'].append(sf_pre);  metrics['sfdr_post'].append(sf_post);  metrics['sfdr_ref'].append(sf_ref)
        metrics['sinad_post_nl'].append(s_nl)
        metrics['enob_post_nl'].append(e_nl)
        metrics['thd_post_nl'].append(t_nl)
        metrics['sfdr_post_nl'].append(sf_nl)

    model0.fpga_simulation = False
    model1.fpga_simulation = False
    return test_freqs, metrics

def analyze_and_plot_spectrum(sim, model, scale, device, p_ch0, p_ch1, plot_freq=2.1e9):
    """
    [新增] 频谱对比分析
    画出 Reference, Uncalibrated, Calibrated 三者的频谱对比
    这是论文中用于展示"镜像杂散消除"最直观的图
    """
    print(f"\n=== 正在绘制 {plot_freq/1e9}GHz 处的频谱对比图 ===")
    src = sim.generate_tone_data(plot_freq, N=16384) # 稍微长一点以获得高频率分辨率
    
    # 生成数据 (模拟 TIADC 交织输出)
    # Ref: Ideal TIADC (Ch0 + Ch0) -> 代表校准的终极目标
    # Bad: Uncal TIADC (Ch0 + Ch1_Bad) -> 存在镜像杂散
    # Cal: Calib TIADC (Ch0 + Ch1_Cal) -> 杂散应消失
    
    # 频谱分析也保持与指标口径一致：真实 ADC 条件
    ch0_real = sim.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p_ch0)
    ch1_bad  = sim.apply_channel_effect(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p_ch1)
    
    with torch.no_grad():
        inp_t = torch.FloatTensor(ch1_bad/scale).view(1,1,-1).to(device)
        ch1_cal = model(inp_t).cpu().numpy().flatten() * scale
        
    def interleave(c0, c1):
        L = min(len(c0), len(c1))
        L -= L % 2
        out = np.zeros(L)
        out[0::2] = c0[:L:2]
        out[1::2] = c1[1:L:2]
        return out
        
    tiadc_ref = interleave(ch0_real, ch0_real)
    tiadc_bad = interleave(ch0_real, ch1_bad)
    tiadc_cal = interleave(ch0_real, ch1_cal)

    # 如果存在后级 DDSP，再对校准结果做一次后处理（保持与指标计算一致的归一化尺度）
    if hasattr(model, 'post_linearizer') and model.post_linearizer is not None:
        scale_pl = getattr(model, 'post_linearizer_scale', scale)
        with torch.no_grad():
            x = torch.FloatTensor(tiadc_cal / scale_pl).view(1, 1, -1).to(device)
            y = model.post_linearizer(x).cpu().numpy().flatten() * scale_pl
        tiadc_cal = y
    
    # 计算 PSD
    def get_psd(sig):
        win = np.blackman(len(sig))
        freqs, psd = signal.periodogram(sig, sim.fs, window='blackman', scaling='spectrum')
        psd_db = 10 * np.log10(psd + 1e-15)
        return freqs, psd_db

    f, psd_ref = get_psd(tiadc_ref)
    _, psd_bad = get_psd(tiadc_bad)
    _, psd_cal = get_psd(tiadc_cal)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(f/1e9, psd_bad, 'r', alpha=0.5, label='Uncalibrated (Mismatch Spur)')
    plt.plot(f/1e9, psd_cal, 'b', alpha=0.8, label='Calibrated (Proposed)')
    # plt.plot(f/1e9, psd_ref, 'k--', alpha=0.3, label='Reference (Ch0 Baseline)') # 可选
    
    # 标注镜像频率
    image_freq = sim.fs/2 - plot_freq
    plt.axvline(image_freq/1e9, color='g', linestyle='--', alpha=0.5, label='Image Freq Location')
    
    plt.title(f"TIADC Spectrum Analysis @ {plot_freq/1e9} GHz Input")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power Spectrum (dB)")
    plt.ylim(-120, 0)
    plt.legend()
    plt.grid(True)
    save_current_figure(f"spectrum_psd_{plot_freq/1e9:.3f}GHz".replace(".", "p"))

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n=== 运行配置摘要 ===")
    print(f"ENABLE_STAGE3_NONLINEAR={ENABLE_STAGE3_NONLINEAR} | STAGE3_SCHEME={STAGE3_SCHEME} | STAGE3_USE_REFERENCE_TARGET={STAGE3_USE_REFERENCE_TARGET}")
    print(f"ENABLE_POST_EQ={ENABLE_POST_EQ} | (Post-QE physics layer) | POST_EQ_MODE={POST_EQ_MODE} | POST_EQ_USE_REFERENCE_TARGET={POST_EQ_USE_REFERENCE_TARGET} | POST_EQ_BAND_HZ={POST_EQ_BAND_HZ/1e9:.2f}GHz | POST_EQ_TAPS={POST_EQ_TAPS} | POST_EQ_RIDGE={POST_EQ_RIDGE:g}")
    if not ENABLE_STAGE3_NONLINEAR:
        print(">> Stage3 已关闭：Post-NL 将与 Post-Linear 完全相同（这是预期行为）")
    sim = TIADCSimulator(fs=20e9)
    # 相对校准
    results = train_relative_calibration(sim, device)
    model = results[0]
    scale = results[2]
    p_ch0 = results[3]
    p_ch1 = results[4]
    test_freqs, m = calculate_metrics_detailed(sim, model, scale, device, p_ch0, p_ch1)

    # 打印若干代表频点，方便快速判断“校正是否变好”（避免只看图误判）
    def _pick(freq_ghz_list):
        idxs = []
        for g in freq_ghz_list:
            f0 = g * 1e9
            idxs.append(int(np.argmin(np.abs(test_freqs - f0))))
        return sorted(set(idxs))

    print("\n=== 关键频点指标（Pre -> Post）===")
    # 调整关键频点范围以匹配新的测试范围（0.1-6.5GHz）
    for i in _pick([0.1, 1.5, 3.0, 4.5, 5.5, 6.0, 6.5]):
        fghz = test_freqs[i] / 1e9
        ref_str = ""
        if 'thd_ref' in m:
            ref_str = (
                f" | Ref: SINAD {m['sinad_ref'][i]:>6.2f} dB"
                f", THD {m['thd_ref'][i]:>7.2f} dBc"
                f", SFDR {m['sfdr_ref'][i]:>6.2f} dBc"
            )
        eq_str = ""
        if (
            (hasattr(model, "post_qe") and model.post_qe is not None)
            or (hasattr(model, "post_fir") and model.post_fir is not None)
            or (
                hasattr(model, "post_fir_even")
                and hasattr(model, "post_fir_odd")
                and (model.post_fir_even is not None)
                and (model.post_fir_odd is not None)
            )
        ):
            eq_str = (
                f" | Post-QE: SINAD {m['sinad_post_eq'][i]:>6.2f} dB"
                f", THD {m['thd_post_eq'][i]:>7.2f} dBc"
                f", SFDR {m['sfdr_post_eq'][i]:>6.2f} dBc"
            )
        nl_str = ""
        if 'thd_post_nl' in m:
            nl_str = (
                f" | Post-NL: SINAD {m['sinad_post_nl'][i]:>6.2f} dB"
                f", THD {m['thd_post_nl'][i]:>7.2f} dBc"
                f", SFDR {m['sfdr_post_nl'][i]:>6.2f} dBc"
            )
        print(
            f"f={fghz:>4.1f}GHz | "
            f"SINAD {m['sinad_pre'][i]:>6.2f}->{m['sinad_post'][i]:>6.2f} dB | "
            f"ENOB {m['enob_pre'][i]:>5.2f}->{m['enob_post'][i]:>5.2f} bits | "
            f"THD {m['thd_pre'][i]:>7.2f}->{m['thd_post'][i]:>7.2f} dBc | "
            f"SFDR {m['sfdr_pre'][i]:>6.2f}->{m['sfdr_post'][i]:>6.2f} dBc"
            f"{ref_str}{eq_str}{nl_str}"
        )
    
    # 绘图
    plt.figure(figsize=(10, 13))
    
    # 1. SINAD 对比
    plt.subplot(4,1,1)
    plt.plot(test_freqs/1e9, m['sinad_pre'], 'r--o', alpha=0.5, label='Pre-Calib')
    plt.plot(test_freqs/1e9, m['sinad_post'], 'g-o', linewidth=2, label='Post-Linear')
    if (
        (hasattr(model, "post_qe") and model.post_qe is not None)
        or (hasattr(model, "post_fir") and model.post_fir is not None)
        or (
            hasattr(model, "post_fir_even")
            and hasattr(model, "post_fir_odd")
            and (model.post_fir_even is not None)
            and (model.post_fir_odd is not None)
        )
    ):
        plt.plot(test_freqs/1e9, m['sinad_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-QE')
    if 'sinad_post_nl' in m:
        plt.plot(test_freqs/1e9, m['sinad_post_nl'], 'b--o', alpha=0.85, label='Post-NL')
    plt.title("SINAD Improvement")
    plt.ylabel("dB"); plt.legend(); plt.grid(True)
    
    # 2. ENOB 对比
    plt.subplot(4,1,2)
    plt.plot(test_freqs/1e9, m['enob_pre'], 'r--o', alpha=0.5, label='Pre-Calib')
    plt.plot(test_freqs/1e9, m['enob_post'], 'm-o', linewidth=2, label='Post-Linear')
    if (
        (hasattr(model, "post_qe") and model.post_qe is not None)
        or (hasattr(model, "post_fir") and model.post_fir is not None)
        or (
            hasattr(model, "post_fir_even")
            and hasattr(model, "post_fir_odd")
            and (model.post_fir_even is not None)
            and (model.post_fir_odd is not None)
        )
    ):
        plt.plot(test_freqs/1e9, m['enob_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-QE')
    if 'enob_post_nl' in m:
        plt.plot(test_freqs/1e9, m['enob_post_nl'], 'b--o', alpha=0.85, label='Post-NL')
    plt.title("ENOB Improvement")
    plt.ylabel("Bits"); plt.legend(); plt.grid(True)
    
    # 3. THD 对比
    plt.subplot(4,1,3)
    plt.plot(test_freqs/1e9, m['thd_pre'], 'r--o', alpha=0.5, label='Pre-Calib (Ch1 Raw)')
    plt.plot(test_freqs/1e9, m['thd_post'], 'b-o', linewidth=2, label='Post-Linear')
    if (
        (hasattr(model, "post_qe") and model.post_qe is not None)
        or (hasattr(model, "post_fir") and model.post_fir is not None)
        or (
            hasattr(model, "post_fir_even")
            and hasattr(model, "post_fir_odd")
            and (model.post_fir_even is not None)
            and (model.post_fir_odd is not None)
        )
    ):
        plt.plot(test_freqs/1e9, m['thd_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-QE')
    if 'thd_post_nl' in m:
        plt.plot(test_freqs/1e9, m['thd_post_nl'], 'k--o', alpha=0.85, label='Post-NL')
    # 可选：画出 Ch0 的 THD 作为基准线 (Target)
    # plt.axhline(-60, color='k', linestyle='--', alpha=0.3, label='Reference (Ch0)')
    plt.title("THD Comparison")
    plt.ylabel("dBc"); plt.legend(); plt.grid(True)

    # 4. SFDR 对比
    plt.subplot(4,1,4)
    plt.plot(test_freqs/1e9, m['sfdr_pre'], 'r--o', alpha=0.5, label='Pre-Calib (Ch1 Raw)')
    plt.plot(test_freqs/1e9, m['sfdr_post'], 'c-o', linewidth=2, label='Post-Linear')
    if (
        (hasattr(model, "post_qe") and model.post_qe is not None)
        or (hasattr(model, "post_fir") and model.post_fir is not None)
        or (
            hasattr(model, "post_fir_even")
            and hasattr(model, "post_fir_odd")
            and (model.post_fir_even is not None)
            and (model.post_fir_odd is not None)
        )
    ):
        plt.plot(test_freqs/1e9, m['sfdr_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-QE')
    if 'sfdr_post_nl' in m:
        plt.plot(test_freqs/1e9, m['sfdr_post_nl'], 'b--o', alpha=0.85, label='Post-NL')
    plt.title("SFDR Improvement")

    plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    save_current_figure("metrics_vs_freq")