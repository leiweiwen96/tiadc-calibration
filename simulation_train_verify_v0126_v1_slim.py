"""
v0126_v1_slim
仅保留主流程：Stage1/2（线性） + Post‑QE（物理静态逆） + Post‑NL（记忆多项式）

注意：此文件刻意保持“干净”，不包含旧脚本的历史遗留段落。
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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
ADC_NBITS = int(os.getenv("ADC_NBITS", "12"))


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


class TIADCSimulator:
    def __init__(self, fs: float = 20e9):
        self.fs = float(fs)

    def tone(self, fin: float, n: int, amp: float = 0.9) -> np.ndarray:
        t = np.arange(int(n), dtype=np.float64) / self.fs
        return (np.sin(2 * np.pi * float(fin) * t) * float(amp)).astype(np.float64)

    def _frac_delay(self, sig: np.ndarray, delay: float) -> np.ndarray:
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
        return (c0 + d * (c1 + d * (c2 + d * c3))).astype(sig.dtype, copy=False)

    def apply(
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

        sig = sig + float(hd2) * (sig**2) + float(hd3) * (sig**3)
        b, a = signal.butter(5, float(cutoff_freq) / nyq, btype="low")
        sig = signal.lfilter(b, a, sig)
        sig = self._frac_delay(sig * float(gain), float(delay_samples))

        if jitter_std and float(jitter_std) > 0:
            slope = np.gradient(sig) * self.fs
            dt = np.random.normal(0.0, float(jitter_std), len(sig))
            sig = sig + slope * dt

        if snr_target is None:
            sig = sig + np.random.normal(0.0, 1e-4, len(sig))
        else:
            p = float(np.mean(sig**2))
            n_p = p / (10 ** (float(snr_target) / 10.0))
            sig = sig + np.random.normal(0.0, np.sqrt(n_p), len(sig))

        if n_bits is None:
            return sig.astype(np.float64, copy=False)
        levels = 2 ** int(n_bits)
        step = 2.0 / levels
        sig = np.clip(sig, -1.0, 1.0)
        return (np.round(sig / step) * step).astype(np.float64, copy=False)


def interleave(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
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


class Stage12Model(nn.Module):
    def __init__(self, taps: int = 63):
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


class Stage12Loss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor, model: Stage12Model) -> torch.Tensor:
        crop = 300
        yp = y_pred[..., crop:-crop]
        yt = y_target[..., crop:-crop]
        lt = torch.mean((yp - yt) ** 2)
        Yp = torch.fft.rfft(yp, dim=-1, norm="ortho")
        Yt = torch.fft.rfft(yt, dim=-1, norm="ortho")
        lf = torch.mean(torch.abs(Yp - Yt) ** 2)
        w = model.fir.weight.view(-1)
        lr = torch.mean((w[1:] - w[:-1]) ** 2)
        return 100.0 * lt + 100.0 * lf + 0.001 * lr


class PhysicalNonlinearityLayer(nn.Module):
    def __init__(self, order: int = 3):
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
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.raw = nn.Parameter(torch.zeros(self.order - 1))

    def _res(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        a = self.scales.to(x.device, x.dtype) * torch.tanh(raw.to(x.device, x.dtype))
        y = torch.zeros_like(x)
        for i, p in enumerate(range(2, self.order + 1)):
            y = y + a[i] * (x**p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        return x + float(self.alpha) * self._res(x, self.raw)


class DifferentiableMemoryPolynomial(nn.Module):
    def __init__(self, memory_depth: int = 21, nonlinear_order: int = 3):
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
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.raw = nn.Parameter(torch.zeros(self.K, self.P - 1))

    @staticmethod
    def _delay(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return x
        xp = F.pad(x, (k, 0), mode="constant", value=0.0)
        return xp[..., :-k]

    def _res(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        scales = self.scales.to(x.device, x.dtype)
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
        return x + float(self.alpha) * self._res(x, self.raw)


def _harm_bins_folded(k: int, n: int, max_h: int = 5) -> list[int]:
    bins: list[int] = []
    for h in range(2, max_h + 1):
        kk = (h * int(k)) % int(n)
        if kk > n // 2:
            kk = n - kk
        if 1 <= kk <= (n // 2 - 1):
            bins.append(int(kk))
    return sorted(set(bins))


def residual_harmonic_loss(residual: torch.Tensor, target: torch.Tensor, fund_bin: int, guard: int = 3) -> torch.Tensor:
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


def train_stage12(sim: TIADCSimulator, *, device: str, p0: dict, p1: dict) -> Tuple[Stage12Model, float]:
    np.random.seed(42)
    torch.manual_seed(42)
    n_train = 32768
    m_avg = 16
    white = np.random.randn(n_train)
    b, a = signal.butter(6, 7.0e9 / (sim.fs / 2), btype="low")
    base = signal.lfilter(b, a, white)
    base = base / np.max(np.abs(base)) * 0.9
    caps0, caps1 = [], []
    for _ in range(m_avg):
        caps0.append(sim.apply(base, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0))
        caps1.append(sim.apply(base, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1))
    avg0 = np.mean(np.stack(caps0), axis=0)
    avg1 = np.mean(np.stack(caps1), axis=0)
    scale = max(float(np.max(np.abs(avg0))), 1e-12)
    inp = torch.FloatTensor(avg1 / scale).view(1, 1, -1).to(device)
    tgt = torch.FloatTensor(avg0 / scale).view(1, 1, -1).to(device)

    model = Stage12Model(taps=63).to(device)
    loss_fn = Stage12Loss()

    print("=== Stage1: Delay & Gain ===")
    opt1 = optim.Adam(
        [
            {"params": model.delay.parameters(), "lr": 1e-2},
            {"params": model.gain, "lr": 1e-2},
            {"params": model.fir.parameters(), "lr": 0.0},
        ]
    )
    for _ in range(301):
        opt1.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt1.step()

    print("=== Stage2: FIR ===")
    opt2 = optim.Adam(
        [
            {"params": model.delay.parameters(), "lr": 0.0},
            {"params": model.gain, "lr": 0.0},
            {"params": model.fir.parameters(), "lr": 5e-4},
        ],
        betas=(0.5, 0.9),
    )
    for _ in range(1001):
        opt2.zero_grad()
        loss = loss_fn(model(inp), tgt, model)
        loss.backward()
        opt2.step()
    return model, float(scale)


def train_post_qe(sim: TIADCSimulator, *, stage12: Stage12Model, device: str, p0: dict, p1: dict, pref: dict) -> PhysicalNonlinearityLayer:
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

    post_qe = PhysicalNonlinearityLayer(order).to(device)
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
            y0 = sim.apply(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0)
            y1 = sim.apply(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1)
            yr = sim.apply(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **pref)
            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1c = stage12(x1).detach().cpu().numpy().flatten() * s12
            xin = interleave(y0, y1c)
            yt = yr[: len(xin)]
            s0 = max(float(np.max(np.abs(xin))), float(np.max(np.abs(yt))), 1e-12)
            x = torch.FloatTensor(xin / s0).view(1, 1, -1).to(device)
            y = torch.FloatTensor(yt / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.clamp(y, -1.0, 1.0)
            yhat = post_qe(x)
            lh = residual_harmonic_loss(yhat - y, y, int(k), guard)
            win = torch.hann_window(n, device=device, dtype=x.dtype).view(1, 1, -1)
            Yh = torch.fft.rfft(yhat * win, dim=-1, norm="ortho")
            Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
            lf = torch.mean((torch.log10(torch.abs(Yh[..., int(k)]) + 1e-12) - torch.log10(torch.abs(Yr[..., int(k)]) + 1e-12)) ** 2)
            ld = torch.mean((yhat - x) ** 2)
            lt = torch.mean((yhat - y) ** 2)
            reg = sum(torch.mean(p**2) for p in post_qe.parameters())
            loss_acc = loss_acc + (w_time * lt + w_harm * lh + w_fund * lf + w_delta * ld + ridge * reg)
        loss_acc = loss_acc / float(max(int(batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_qe.parameters(), max_norm=1.0)
        opt.step()
        if step % 150 == 0:
            print(f"Post-QE step {step:4d}/{steps} | loss={loss_acc.item():.3e}")
    return post_qe


def train_post_nl(sim: TIADCSimulator, *, stage12: Stage12Model, post_qe: Optional[PhysicalNonlinearityLayer], device: str, p0: dict, p1: dict, pref: dict) -> DifferentiableMemoryPolynomial:
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

    post_nl = DifferentiableMemoryPolynomial(taps, order).to(device)
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
            y0 = sim.apply(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0)
            y1 = sim.apply(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1)
            yr = sim.apply(src_np, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **pref)
            s12 = max(float(np.max(np.abs(y0))), 1e-12)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / s12).view(1, 1, -1).to(device)
                y1c = stage12(x1).detach().cpu().numpy().flatten() * s12
            xin = interleave(y0, y1c)
            yt = yr[: len(xin)]
            s0 = max(float(np.max(np.abs(xin))), float(np.max(np.abs(yt))), 1e-12)
            x = torch.FloatTensor(xin / s0).view(1, 1, -1).to(device)
            y = torch.FloatTensor(yt / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            y = torch.clamp(y, -1.0, 1.0)
            if post_qe is not None:
                with torch.no_grad():
                    x = post_qe(x)
            yhat = post_nl(x)
            lh = residual_harmonic_loss(yhat - y, y, int(k), guard)
            win = torch.hann_window(n, device=device, dtype=x.dtype).view(1, 1, -1)
            Yh = torch.fft.rfft(yhat * win, dim=-1, norm="ortho")
            Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
            lf = torch.mean((torch.log10(torch.abs(Yh[..., int(k)]) + 1e-12) - torch.log10(torch.abs(Yr[..., int(k)]) + 1e-12)) ** 2)
            ld = torch.mean((yhat - x) ** 2)
            lt = torch.mean((yhat - y) ** 2)
            reg = sum(torch.mean(p**2) for p in post_nl.parameters())
            loss_acc = loss_acc + (w_time * lt + w_harm * lh + w_fund * lf + w_delta * ld + w_reg * reg)
        loss_acc = loss_acc / float(max(int(batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_nl.parameters(), max_norm=1.0)
        opt.step()
        if step % 150 == 0:
            w_mean = float(torch.mean(torch.stack([p.abs().mean() for p in post_nl.parameters()])).detach().cpu())
            print(f"Post-NL step {step:4d}/{steps} | loss={loss_acc.item():.3e} | w_mean~{w_mean:.2e}")
    return post_nl


def calc_metrics(sig: np.ndarray, fs: float, fin: float) -> tuple[float, float, float, float]:
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


def _rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    r = float(np.sqrt(np.mean(x * x)) + 1e-30)
    return 20.0 * np.log10(r + 1e-30)


def _top_spurs_db(sig: np.ndarray, *, fs: float, fin: float, k_top: int = 8, guard: int = 5) -> list[tuple[float, float]]:
    """Return list[(spur_freq_hz, spur_level_dBc)] excluding DC and fundamental ±guard bins."""
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    win = np.blackman(n)
    cg = float(np.mean(win))
    S = np.fft.rfft(sig * win)
    mag = np.abs(S) / (n / 2 * cg + 1e-20)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - fin)))
    span = int(guard)
    s0 = max(0, idx - span)
    e0 = min(len(mag), idx + span + 1)
    idxp = s0 + int(np.argmax(mag[s0:e0]))
    a_fund = float(mag[idxp]) + 1e-30
    mask = np.ones_like(mag, dtype=bool)
    mask[:5] = False
    mask[max(0, idxp - span) : min(len(mask), idxp + span + 1)] = False
    # greedy pick top spurs while masking neighbors
    out: list[tuple[float, float]] = []
    m2 = mag.copy()
    for _ in range(int(k_top)):
        if not np.any(mask):
            break
        kk = int(np.argmax(m2 * mask))
        lvl = 20.0 * np.log10(float(m2[kk]) / a_fund + 1e-30)
        out.append((float(freqs[kk]), float(lvl)))
        mask[max(0, kk - span) : min(len(mask), kk + span + 1)] = False
    return out


def effect_report(
    sim: TIADCSimulator,
    *,
    stage12: Stage12Model,
    post_qe: Optional[PhysicalNonlinearityLayer],
    post_nl: Optional[DifferentiableMemoryPolynomial],
    scale: float,
    device: str,
    p0: dict,
    p1: dict,
) -> None:
    """A/B 验证：看参数幅度、输出改变量、以及最大 spur 是否变化。"""
    if post_qe is None and post_nl is None:
        print("\n[effect] Post-QE/Post-NL 都未启用，跳过。")
        return

    try:
        if post_qe is not None:
            r = post_qe.raw.detach().cpu().numpy()
            print(f"\n[effect] Post-QE | alpha={float(post_qe.alpha):g} | raw_abs_mean={float(np.mean(np.abs(r))):.2e} | raw_abs_max={float(np.max(np.abs(r))):.2e}")
        if post_nl is not None:
            r = post_nl.raw.detach().cpu().numpy()
            print(f"[effect] Post-NL | alpha={float(post_nl.alpha):g} | raw_abs_mean={float(np.mean(np.abs(r))):.2e} | raw_abs_max={float(np.max(np.abs(r))):.2e}")
    except Exception:
        pass

    fs = float(sim.fs)
    n = _env_int("EFFECT_N", 16384)
    amps = [float(x) for x in os.getenv("EFFECT_AMPS", "0.3,0.6,0.9").split(",")]
    freqs_ghz = [float(x) for x in os.getenv("EFFECT_FREQS_GHZ", "0.5,2.5,5.5").split(",")]

    print("\n[effect] A/B 消融（同一输入，同一 Stage1/2）")
    for fghz in freqs_ghz:
        fin = fghz * 1e9
        for amp in amps:
            src = sim.tone(fin, n, amp)
            y0 = sim.apply(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0)
            y1 = sim.apply(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
                y1c = stage12(x1).cpu().numpy().flatten() * float(scale)
            margin = 500
            c0 = y0[margin:-margin]
            c1c = y1c[margin:-margin]
            post = interleave(c0, c1c)

            qe = post
            if post_qe is not None:
                s = max(float(np.max(np.abs(post))), 1e-12)
                with torch.no_grad():
                    x = torch.FloatTensor(post / s).view(1, 1, -1).to(device)
                    qe = post_qe(x).cpu().numpy().flatten() * s
            nl = qe
            if post_nl is not None:
                s = max(float(np.max(np.abs(qe))), 1e-12)
                with torch.no_grad():
                    x = torch.FloatTensor(qe / s).view(1, 1, -1).to(device)
                    nl = post_nl(x).cpu().numpy().flatten() * s

            # 关键：看“到底改没改”，以及“改的是谐波还是最大 spur”
            d_qe = _rms_db(qe - post)
            d_nl = _rms_db(nl - qe)
            s_post, _, t_post, sf_post = calc_metrics(post, fs, fin)
            s_qe, _, t_qe, sf_qe = calc_metrics(qe, fs, fin)
            s_nl, _, t_nl, sf_nl = calc_metrics(nl, fs, fin)
            spur_post = _top_spurs_db(post, fs=fs, fin=fin, k_top=1)[0][1]
            spur_qe = _top_spurs_db(qe, fs=fs, fin=fin, k_top=1)[0][1]
            spur_nl = _top_spurs_db(nl, fs=fs, fin=fin, k_top=1)[0][1]

            print(
                f"f={fghz:>3.1f}GHz amp={amp:>3.1f} | "
                f"ΔQE_rms={d_qe:>6.1f}dBFS ΔNL_rms={d_nl:>6.1f}dBFS | "
                f"SINAD {s_post:>5.2f}->{s_qe:>5.2f}->{s_nl:>5.2f} | "
                f"THD {t_post:>6.2f}->{t_qe:>6.2f}->{t_nl:>6.2f} | "
                f"SFDR {sf_post:>5.2f}->{sf_qe:>5.2f}->{sf_nl:>5.2f} | "
                f"maxSpur {spur_post:>6.1f}->{spur_qe:>6.1f}->{spur_nl:>6.1f} dBc"
            )


def evaluate(sim: TIADCSimulator, *, stage12: Stage12Model, post_qe: Optional[PhysicalNonlinearityLayer], post_nl: Optional[DifferentiableMemoryPolynomial], scale: float, device: str, p0: dict, p1: dict) -> tuple[np.ndarray, dict]:
    freqs = np.arange(0.1e9, 6.6e9, 0.2e9)
    m = {k: [] for k in ["sinad_pre", "sinad_post", "sinad_qe", "sinad_nl",
                         "enob_pre", "enob_post", "enob_qe", "enob_nl",
                         "thd_pre", "thd_post", "thd_qe", "thd_nl",
                         "sfdr_pre", "sfdr_post", "sfdr_qe", "sfdr_nl"]}
    for f in freqs:
        src = sim.tone(float(f), 16384, 0.9)
        y0 = sim.apply(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p0)
        y1 = sim.apply(src, jitter_std=ADC_JITTER_STD, n_bits=ADC_NBITS, **p1)
        with torch.no_grad():
            x1 = torch.FloatTensor(y1 / float(scale)).view(1, 1, -1).to(device)
            y1c = stage12(x1).cpu().numpy().flatten() * float(scale)
        margin = 500
        c0 = y0[margin:-margin]
        c1r = y1[margin:-margin]
        c1c = y1c[margin:-margin]
        pre = interleave(c0, c1r)
        post = interleave(c0, c1c)
        qe = post
        if post_qe is not None:
            s = max(float(np.max(np.abs(post))), 1e-12)
            with torch.no_grad():
                x = torch.FloatTensor(post / s).view(1, 1, -1).to(device)
                qe = post_qe(x).cpu().numpy().flatten() * s
        nl = qe
        if post_nl is not None:
            s = max(float(np.max(np.abs(qe))), 1e-12)
            with torch.no_grad():
                x = torch.FloatTensor(qe / s).view(1, 1, -1).to(device)
                nl = post_nl(x).cpu().numpy().flatten() * s
        s0, e0, t0, sf0 = calc_metrics(pre, sim.fs, float(f))
        s1, e1, t1, sf1 = calc_metrics(post, sim.fs, float(f))
        s2, e2, t2, sf2 = calc_metrics(qe, sim.fs, float(f))
        s3, e3, t3, sf3 = calc_metrics(nl, sim.fs, float(f))
        m["sinad_pre"].append(s0); m["sinad_post"].append(s1); m["sinad_qe"].append(s2); m["sinad_nl"].append(s3)
        m["enob_pre"].append(e0); m["enob_post"].append(e1); m["enob_qe"].append(e2); m["enob_nl"].append(e3)
        m["thd_pre"].append(t0); m["thd_post"].append(t1); m["thd_qe"].append(t2); m["thd_nl"].append(t3)
        m["sfdr_pre"].append(sf0); m["sfdr_post"].append(sf1); m["sfdr_qe"].append(sf2); m["sfdr_nl"].append(sf3)
    return freqs, m


def plot_metrics(freqs: np.ndarray, m: dict) -> None:
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
    savefig("metrics_vs_freq")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== v0126_v1_slim ===")
    print(f"device={device} | ENABLE_POST_QE={_env_bool('ENABLE_POST_QE', True)} | ENABLE_STAGE3_NONLINEAR={_env_bool('ENABLE_STAGE3_NONLINEAR', True)}")

    sim = TIADCSimulator(fs=20e9)
    p0 = {"cutoff_freq": 8.0e9, "delay_samples": 0.0, "gain": 1.0, "hd2": 1e-3, "hd3": 0.5e-3}
    p1 = {"cutoff_freq": 7.8e9, "delay_samples": 0.25, "gain": 0.98, "hd2": 5e-3, "hd3": 3e-3}
    pref = {"cutoff_freq": p0["cutoff_freq"], "delay_samples": 0.0, "gain": 1.0, "hd2": 0.0, "hd3": 0.0, "snr_target": None}

    stage12, scale = train_stage12(sim, device=device, p0=p0, p1=p1)

    post_qe = train_post_qe(sim, stage12=stage12, device=device, p0=p0, p1=p1, pref=pref) if _env_bool("ENABLE_POST_QE", True) else None
    post_nl = train_post_nl(sim, stage12=stage12, post_qe=post_qe, device=device, p0=p0, p1=p1, pref=pref) if _env_bool("ENABLE_STAGE3_NONLINEAR", True) else None

    freqs, m = evaluate(sim, stage12=stage12, post_qe=post_qe, post_nl=post_nl, scale=scale, device=device, p0=p0, p1=p1)

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
    if _env_bool("EFFECT_REPORT", False):
        effect_report(
            sim,
            stage12=stage12,
            post_qe=post_qe,
            post_nl=post_nl,
            scale=scale,
            device=device,
            p0=p0,
            p1=p1,
        )


if __name__ == "__main__":
    main()
