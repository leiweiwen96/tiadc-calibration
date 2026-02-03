"""
v0126_v2

- **仿真模式**：Stage1/2（线性） + Post‑QE（静态物理逆） + Post‑NL（记忆多项式）
- **实测 CSV 模式**：读取示波器导出的 `Uni-t000.csv`（两路交织后的全速数据），
  按 1,3,5... / 2,4,6... 拆成两组（等价于 0::2 / 1::2），并用“交织镜像 spur 最小化”
  的自监督方式训练 Stage1/2（可选再训练 Post‑QE/Post‑NL 做 THD/谐波抑制）。

开关：
- `USE_SCOPE_CSV=1` 启用实测 CSV 模式
- `CSV_PATH=Uni-t000.csv` 指定 CSV 路径（可写 `Unit-t000.csv`，会自动兜底）
"""

import os
import sys
import glob
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
        # 更“看得见”的默认值：允许更强校正，但用 res_clip 防止失控
        self.alpha = float(os.getenv("POST_QE_ALPHA", "0.8"))
        self.res_clip = float(os.getenv("POST_QE_RES_CLIP", "0.2"))
        max_p2 = float(os.getenv("POST_QE_MAX_P2", "5e-2"))
        max_p3 = float(os.getenv("POST_QE_MAX_P3", "3e-2"))
        max_hi = float(os.getenv("POST_QE_MAX_PHI", "1e-2"))
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
        res = self._res(x, self.raw)
        res = torch.clamp(res, -float(self.res_clip), float(self.res_clip))
        return x + float(self.alpha) * res


class DifferentiableMemoryPolynomial(nn.Module):
    def __init__(self, memory_depth: int = 21, nonlinear_order: int = 3):
        super().__init__()
        self.K = int(memory_depth)
        self.P = int(nonlinear_order)
        if self.K < 1 or self.P < 2:
            raise ValueError("memory_depth>=1, nonlinear_order>=2 required")
        self.alpha = float(os.getenv("POST_NL_ALPHA", "0.8"))
        self.res_clip = float(os.getenv("POST_NL_RES_CLIP", "0.2"))
        max_p2 = float(os.getenv("POST_NL_MAX_P2", "5e-2"))
        max_p3 = float(os.getenv("POST_NL_MAX_P3", "3e-2"))
        max_hi = float(os.getenv("POST_NL_MAX_PHI", "1e-2"))
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
        res = self._res(x, self.raw)
        res = torch.clamp(res, -float(self.res_clip), float(self.res_clip))
        return x + float(self.alpha) * res


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
    # 更偏向“看得见的提升”：加大谐波权重，弱化“尽量不动输入”的约束
    w_time = _env_float("POST_QE_W_TIME", 15.0)
    w_harm = _env_float("POST_QE_W_HARM", 1800.0)
    w_fund = _env_float("POST_QE_W_FUND", 2.0)
    w_delta = _env_float("POST_QE_W_DELTA", 1.0)
    ridge = _env_float("POST_QE_RIDGE", 5e-5)

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
    w_time = _env_float("STAGE3_W_TIME", 60.0)
    w_harm = _env_float("STAGE3_W_HARM", 2400.0)
    w_fund = _env_float("STAGE3_W_FUND", 1.0)
    w_delta = _env_float("STAGE3_W_DELTA", 0.6)
    w_reg = _env_float("STAGE3_W_REG", 5e-5)

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
        # 若某个高次谐波折叠后落到基波附近，会导致 THD 被“基波能量”污染（甚至变成 0 dBc）。
        # 这种情况在 fin 接近 Nyquist、且只用离散序列计算 THD 时很常见。
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


def load_scope_csv(path: str) -> tuple[float, np.ndarray, np.ndarray]:
    """
    读取示波器 CSV（前面有若干行 key:value 头），返回 (fs_hz, t_s, y_volt).
    该仓库里的 `Uni-t000.csv` 是 `X-axis(μs), Y-axis(mV)`。
    """
    p = Path(path)
    if not p.exists():
        # 用户有时会写成 Unit-*.csv
        alt = p.with_name(p.name.replace("Unit-", "Uni-"))
        if alt.exists():
            p = alt
        else:
            alt2 = p.with_name(p.name.replace("Uni-", "Unit-"))
            if alt2.exists():
                p = alt2
            else:
                raise FileNotFoundError(str(p))

    fs_hz: Optional[float] = None
    data_start: Optional[int] = None
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s.lower().startswith("sample rate:"):
                # e.g. "Sample Rate:80.00GSa/s"
                v = s.split(":", 1)[1].strip()
                v = v.replace("Sa/s", "").strip()
                mul = 1.0
                if v.endswith("G"):
                    mul = 1e9
                    v = v[:-1]
                elif v.endswith("M"):
                    mul = 1e6
                    v = v[:-1]
                elif v.endswith("k") or v.endswith("K"):
                    mul = 1e3
                    v = v[:-1]
                fs_hz = float(v) * mul
            if "x-axis" in s.lower() and "y-axis" in s.lower():
                data_start = i + 1
                break
    if fs_hz is None:
        raise ValueError(f"无法从 CSV 头解析 Sample Rate: {p}")
    if data_start is None:
        raise ValueError(f"无法定位数据起始行（X-axis...）: {p}")

    arr = np.loadtxt(str(p), delimiter=",", skiprows=int(data_start), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"CSV 数据列数异常: shape={arr.shape} file={p}")
    t_us = arr[:, 0]
    y_mv = arr[:, 1]
    t_s = t_us * 1e-6
    y_v = y_mv * 1e-3
    return float(fs_hz), t_s.astype(np.float64, copy=False), y_v.astype(np.float64, copy=False)


def _scope_effective_fs(fs_header_hz: float) -> float:
    """
    实测数据的“有效采样率”有时应按 TIADC 结构来取，而不是示波器导出头里的采样率。
    - `SCOPE_INTERLEAVED_FS_GSA`：直接指定交织后全速采样率（GSa/s）
    - `SCOPE_SUBADC_FS_GSA`：指定单片/子 ADC 采样率（GSa/s），两路交织则 fs = 2 * subadc
    """
    v = os.getenv("SCOPE_INTERLEAVED_FS_GSA")
    if v is not None:
        return float(v) * 1e9
    v = os.getenv("SCOPE_SUBADC_FS_GSA")
    if v is not None:
        return 2.0 * float(v) * 1e9
    return float(fs_header_hz)


def collect_scope_csv_paths() -> list[Path]:
    """
    支持批量扫频：
    - `CSV_GLOB`：例如 `Uni-t*.csv`
    - 否则退化为单文件 `CSV_PATH`
    """
    g = os.getenv("CSV_GLOB")
    if g:
        paths = [Path(p) for p in glob.glob(g)]
        paths = [p for p in paths if p.exists() and p.is_file()]
        return sorted(paths)
    p = Path(os.getenv("CSV_PATH", "Uni-t000.csv"))
    if not p.exists():
        # 兼容 Unit-/Uni- 拼写
        alt = p.with_name(p.name.replace("Unit-", "Uni-"))
        if alt.exists():
            p = alt
        else:
            alt2 = p.with_name(p.name.replace("Uni-", "Unit-"))
            if alt2.exists():
                p = alt2
    return [p]


def _maybe_resample_scope_stream(
    y_v: np.ndarray, t_s: np.ndarray, *, fs_header: float, fs_effective: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    当示波器 CSV 以更高采样率导出（例如 80GSa/s）但你希望按 TIADC 有效采样率
    （例如两路交织 20GSa/s）来做数字域分析时，需要把波形重采样到 fs_effective。

    默认仅在 (fs_header / fs_effective) 接近整数倍时启用，并可用环境变量关闭：
    - `SCOPE_RESAMPLE=0`：不重采样（直接用 CSV 原始采样率）
    """
    if not _env_bool("SCOPE_RESAMPLE", True):
        return y_v, t_s
    r = float(fs_header) / max(float(fs_effective), 1e-12)
    m = int(round(r))
    if m < 2 or abs(r - float(m)) > 1e-3:
        return y_v, t_s

    method = os.getenv("SCOPE_DOWNSAMPLE_METHOD", "pick_best").strip().lower()
    y_v = np.asarray(y_v, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)

    if method in ("resample", "resample_poly", "poly"):
        # polyphase 重采样（带抗混叠滤波，适合“连续时间波形”场景）
        y_rs = signal.resample_poly(y_v, up=1, down=m).astype(np.float64, copy=False)
        t0 = float(t_s[0]) if len(t_s) else 0.0
        t_rs = (t0 + np.arange(len(y_rs), dtype=np.float64) / float(fs_effective)).astype(np.float64, copy=False)
        print(f"[scope] resample {fs_header/1e9:.2f}G -> {fs_effective/1e9:.2f}G (x{m}) | N {len(y_v)} -> {len(y_rs)}")
        return y_rs, t_rs

    # 默认：按相位挑选（适合示波器过采样观测“离散采样保持输出”的场景）
    # 在 0..m-1 的相位中选一个，使得基波峰值最大（近似“采样点对齐”）
    best_o = 0
    best_peak = -1.0
    for o in range(m):
        yy = y_v[o::m]
        if len(yy) < 1024:
            continue
        yy0 = yy - float(np.mean(yy))
        n = len(yy0)
        win = np.blackman(n)
        S = np.fft.rfft(yy0 * win)
        mag = np.abs(S)
        mag[:5] = 0.0
        peak = float(np.max(mag))
        if peak > best_peak:
            best_peak = peak
            best_o = o
    y_ds = y_v[best_o::m].astype(np.float64, copy=False)
    t0 = float(t_s[0]) if len(t_s) else 0.0
    t_ds = (t0 + np.arange(len(y_ds), dtype=np.float64) / float(fs_effective)).astype(np.float64, copy=False)
    print(f"[scope] downsample pick_best: phase={best_o}/{m} | N {len(y_v)} -> {len(y_ds)}")
    return y_ds, t_ds

def split_two_way_interleaved(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    两路交织：1,3,5... 为一组；2,4,6... 为一组（等价于 0::2 / 1::2）。
    返回 (grp_odd_1based, grp_even_1based) -> (y[0::2], y[1::2])
    """
    y = np.asarray(y, dtype=np.float64)
    return y[0::2].copy(), y[1::2].copy()


def _estimate_fund_bin(sig: np.ndarray, fs: float) -> tuple[int, float]:
    sig = np.asarray(sig, dtype=np.float64)
    n = len(sig)
    win = np.blackman(n)
    S = np.fft.rfft((sig - np.mean(sig)) * win)
    mag = np.abs(S)
    mag[:5] = 0.0
    k = int(np.argmax(mag))
    fin = float(k) * float(fs) / float(n)
    return k, fin


def plot_scope_compare(
    *,
    fs: float,
    fin: float,
    pre: np.ndarray,
    post_lin: np.ndarray,
    qe: Optional[np.ndarray] = None,
    nl: Optional[np.ndarray] = None,
    nfft: int = 16384,
) -> None:
    """实测单次捕获：保存时域片段 + 频谱对比（dBc）。"""
    pre = np.asarray(pre, dtype=np.float64)
    post_lin = np.asarray(post_lin, dtype=np.float64)
    qe = None if qe is None else np.asarray(qe, dtype=np.float64)
    nl = None if nl is None else np.asarray(nl, dtype=np.float64)

    n = int(min(len(pre), len(post_lin), len(qe) if qe is not None else len(pre), len(nl) if nl is not None else len(pre)))
    n = max(2048, min(n, int(nfft)))
    pre = pre[:n]
    post_lin = post_lin[:n]
    if qe is not None:
        qe = qe[:n]
    if nl is not None:
        nl = nl[:n]

    def _spec_db(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        win = np.blackman(len(sig))
        cg = float(np.mean(win))
        S = np.fft.rfft((sig - float(np.mean(sig))) * win)
        mag = np.abs(S) / (len(sig) / 2 * cg + 1e-20)
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / float(fs))
        idx = int(np.argmin(np.abs(freqs - float(fin))))
        span = 5
        s0 = max(0, idx - span)
        e0 = min(len(mag), idx + span + 1)
        idxp = s0 + int(np.argmax(mag[s0:e0]))
        a_fund = float(mag[idxp]) + 1e-30
        db = 20.0 * np.log10(mag / a_fund + 1e-30)
        return freqs, db

    f, db_pre = _spec_db(pre)
    _, db_post = _spec_db(post_lin)
    db_qe = None if qe is None else _spec_db(qe)[1]
    db_nl = None if nl is None else _spec_db(nl)[1]

    # 时域片段（便于看是否“改坏波形”）
    L = min(2000, n)
    t_ns = np.arange(L, dtype=np.float64) / float(fs) * 1e9
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_ns, pre[:L], label="Pre")
    plt.plot(t_ns, post_lin[:L], label="Post-Linear", alpha=0.9)
    if qe is not None:
        plt.plot(t_ns, qe[:L], label="Post-QE", alpha=0.85)
    if nl is not None:
        plt.plot(t_ns, nl[:L], label="Post-NL", alpha=0.85)
    plt.grid(True)
    plt.xlabel("Time (ns)")
    plt.ylabel("Norm. Amp")
    plt.title(f"Scope capture (time) | fin≈{fin/1e9:.4f}GHz")
    plt.legend()

    # 频谱（dBc）
    plt.subplot(2, 1, 2)
    plt.plot(f / 1e9, db_pre, label="Pre", alpha=0.6)
    plt.plot(f / 1e9, db_post, label="Post-Linear", linewidth=2)
    if db_qe is not None:
        plt.plot(f / 1e9, db_qe, label="Post-QE", linewidth=2)
    if db_nl is not None:
        plt.plot(f / 1e9, db_nl, label="Post-NL", linewidth=2)
    plt.ylim(-120, 5)
    plt.xlim(0, float(fs) / 2 / 1e9)
    plt.grid(True)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("dBc (ref to fund)")
    plt.title("Spectrum (dBc)")
    plt.legend()
    plt.tight_layout()
    savefig("scope_time_and_spectrum")


def _interleave_torch(c0_full: torch.Tensor, c1_full: torch.Tensor) -> torch.Tensor:
    s0 = c0_full[..., 0::2]
    s1 = c1_full[..., 1::2]
    L = int(min(s0.shape[-1], s1.shape[-1]))
    out = torch.zeros(c0_full.shape[:-1] + (2 * L,), device=c0_full.device, dtype=c0_full.dtype)
    out[..., 0::2] = s0[..., :L]
    out[..., 1::2] = s1[..., :L]
    return out


def _image_spur_loss(sig: torch.Tensor, *, fund_bin: int, guard: int = 3) -> torch.Tensor:
    """
    2-way TIADC 典型镜像 spur 在 fs/2 ± fin 处。对离散 bin，主镜像近似在 k_img = |N/2 - k0|.
    这里返回 (P_img / P_fund)，越小越好。
    """
    n = int(sig.shape[-1])
    win = torch.hann_window(n, device=sig.device, dtype=sig.dtype).view(1, 1, -1)
    S = torch.fft.rfft((sig - torch.mean(sig, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    P = (S.real**2 + S.imag**2)
    k0 = int(fund_bin)
    k_img = abs(n // 2 - k0)
    # 保护：避免落到 DC 或太靠近基波
    k_img = max(1, min(k_img, P.shape[-1] - 2))
    eps = 1e-12
    s0 = max(0, k0 - int(guard))
    e0 = min(P.shape[-1], k0 + int(guard) + 1)
    p_fund = torch.sum(P[..., s0:e0], dim=-1) + eps
    si = max(0, k_img - int(guard))
    ei = min(P.shape[-1], k_img + int(guard) + 1)
    p_img = torch.sum(P[..., si:ei], dim=-1) + eps
    return torch.mean(p_img / p_fund)


def train_stage12_from_scope(
    y_full: np.ndarray,
    *,
    fs: float,
    device: str,
    taps: int = 63,
) -> Stage12Model:
    """
    实测数据自监督训练 Stage1/2：
    - 将 y_full 拆成两组（0::2 与 1::2），填回 full-rate 数组的偶/奇位置
    - 只校正“第二组”（奇位置），并最小化交织镜像 spur
    """
    y_full = np.asarray(y_full, dtype=np.float64)
    # 轻微去直流并归一化，避免定标问题
    y_full = y_full - float(np.mean(y_full))
    s = max(float(np.max(np.abs(y_full))), 1e-12)
    y_full = y_full / s

    g0, g1 = split_two_way_interleaved(y_full)
    c0 = np.zeros_like(y_full)
    c1 = np.zeros_like(y_full)
    c0[0::2] = g0
    c1[1::2] = g1

    k0, fin = _estimate_fund_bin(y_full, fs)
    print(f"[scope] fs={fs/1e9:.2f}GSa/s | N={len(y_full)} | est fin≈{fin/1e9:.4f}GHz (k0={k0})")

    x0 = torch.tensor(c0, dtype=torch.float32, device=device).view(1, 1, -1)
    x1 = torch.tensor(c1, dtype=torch.float32, device=device).view(1, 1, -1)
    y_ref = torch.tensor(y_full, dtype=torch.float32, device=device).view(1, 1, -1)

    model = Stage12Model(taps=int(taps)).to(device)
    steps = _env_int("SCOPE_STAGE12_STEPS", 2200)
    lr = _env_float("SCOPE_STAGE12_LR", 8e-4)
    w_img = _env_float("SCOPE_W_IMG", 3000.0)
    w_time = _env_float("SCOPE_W_TIME", 2.0)
    w_reg = _env_float("SCOPE_W_REG", 1e-4)
    guard = _env_int("SCOPE_IMG_GUARD", 3)

    opt = optim.Adam(model.parameters(), lr=float(lr), betas=(0.5, 0.9))
    print(f">> [scope] Stage1/2: taps={taps} steps={steps} lr={lr:g} w_img={w_img:g}")
    for step in range(int(steps)):
        opt.zero_grad()
        y1c = model(x1)
        yhat = _interleave_torch(x0, y1c)
        lim = min(yhat.shape[-1], y_ref.shape[-1])
        yhat = yhat[..., :lim]
        yr = y_ref[..., :lim]

        l_img = _image_spur_loss(yhat, fund_bin=int(k0), guard=int(guard))
        l_time = torch.mean((yhat - yr) ** 2)
        reg = torch.mean(model.fir.weight.view(-1) ** 2)
        loss = float(w_img) * l_img + float(w_time) * l_time + float(w_reg) * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if step % 200 == 0:
            print(f"[scope] step {step:4d}/{steps} | loss={loss.item():.3e} | img={l_img.item():.3e}")
    return model


def _harmonic_ratio_loss(sig: torch.Tensor, *, fund_bin: int, guard: int = 3, max_h: int = 5) -> torch.Tensor:
    n = int(sig.shape[-1])
    win = torch.hann_window(n, device=sig.device, dtype=sig.dtype).view(1, 1, -1)
    S = torch.fft.rfft((sig - torch.mean(sig, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    P = (S.real**2 + S.imag**2)
    eps = 1e-12
    k0 = int(fund_bin)
    s0 = max(0, k0 - int(guard))
    e0 = min(P.shape[-1], k0 + int(guard) + 1)
    p_fund = torch.sum(P[..., s0:e0], dim=-1) + eps
    p_h = 0.0
    for kh in _harm_bins_folded(int(k0), int(n), max_h=int(max_h)):
        ss = max(0, kh - int(guard))
        ee = min(P.shape[-1], kh + int(guard) + 1)
        p_h = p_h + torch.sum(P[..., ss:ee], dim=-1)
    p_h = p_h + eps
    return torch.mean(p_h / p_fund)


def train_post_on_scope(
    x_full: np.ndarray,
    *,
    fs: float,
    device: str,
    stage: str = "qe",
    taps: int = 21,
    order: int = 3,
) -> nn.Module:
    """
    实测自监督训练 Post‑QE/Post‑NL：最小化输出自身的谐波比（THD proxy），并保持基波/时域变化受控。
    """
    x_full = np.asarray(x_full, dtype=np.float64)
    x_full = x_full - float(np.mean(x_full))
    s = max(float(np.max(np.abs(x_full))), 1e-12)
    x_full = np.clip(x_full / s, -1.0, 1.0)
    k0, fin = _estimate_fund_bin(x_full, fs)
    print(f">> [scope] Post-{stage.upper()} | est fin≈{fin/1e9:.4f}GHz (k0={k0})")

    x = torch.tensor(x_full, dtype=torch.float32, device=device).view(1, 1, -1)
    if stage.lower() == "qe":
        mod: nn.Module = PhysicalNonlinearityLayer(_env_int("POST_QE_ORDER", order)).to(device)
        steps = _env_int("SCOPE_POST_QE_STEPS", 1200)
        lr = _env_float("SCOPE_POST_QE_LR", 6e-4)
        w_h = _env_float("SCOPE_POST_QE_W_HARM", 1200.0)
        w_f = _env_float("SCOPE_POST_QE_W_FUND", 3.0)
        w_d = _env_float("SCOPE_POST_QE_W_DELTA", 0.8)
        w_r = _env_float("SCOPE_POST_QE_W_REG", 1e-4)
    else:
        mod = DifferentiableMemoryPolynomial(_env_int("STAGE3_TAPS", taps), _env_int("STAGE3_ORDER", order)).to(device)
        steps = _env_int("SCOPE_POST_NL_STEPS", 1800)
        lr = _env_float("SCOPE_POST_NL_LR", 4e-4)
        w_h = _env_float("SCOPE_POST_NL_W_HARM", 1600.0)
        w_f = _env_float("SCOPE_POST_NL_W_FUND", 3.0)
        w_d = _env_float("SCOPE_POST_NL_W_DELTA", 0.8)
        w_r = _env_float("SCOPE_POST_NL_W_REG", 1e-4)

    opt = optim.Adam(mod.parameters(), lr=float(lr), betas=(0.5, 0.9))
    guard = _env_int("SCOPE_POST_GUARD", 3)
    for step in range(int(steps)):
        opt.zero_grad()
        y = mod(x)
        l_h = _harmonic_ratio_loss(y, fund_bin=int(k0), guard=int(guard))
        # 基波幅度保持：用 log 幅度对齐
        win = torch.hann_window(x.shape[-1], device=device, dtype=x.dtype).view(1, 1, -1)
        X = torch.fft.rfft(x * win, dim=-1, norm="ortho")
        Y = torch.fft.rfft(y * win, dim=-1, norm="ortho")
        l_f = torch.mean((torch.log10(torch.abs(Y[..., int(k0)]) + 1e-12) - torch.log10(torch.abs(X[..., int(k0)]) + 1e-12)) ** 2)
        l_d = torch.mean((y - x) ** 2)
        reg = sum(torch.mean(p**2) for p in mod.parameters())
        loss = float(w_h) * l_h + float(w_f) * l_f + float(w_d) * l_d + float(w_r) * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), max_norm=1.0)
        opt.step()
        if step % 300 == 0:
            print(f"[scope] Post-{stage.upper()} step {step:4d}/{steps} | loss={loss.item():.3e} | harm={l_h.item():.3e}")
    return mod


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
    use_scope = _env_bool("USE_SCOPE_CSV", False)
    print("\n=== v0126_v2 ===")
    print(f"device={device} | USE_SCOPE_CSV={use_scope} | ENABLE_POST_QE={_env_bool('ENABLE_POST_QE', True)} | ENABLE_STAGE3_NONLINEAR={_env_bool('ENABLE_STAGE3_NONLINEAR', True)}")

    if use_scope:
        out_dir = _plot_dir()
        paths = collect_scope_csv_paths()
        if len(paths) == 1:
            print("[scope] 单文件输入：所以 metrics_vs_freq 只能是单点。要扫频请提供多份 CSV 并用 CSV_GLOB。")
        else:
            print(f"[scope] 批量扫频：{len(paths)} files via CSV_GLOB")

        # 先用第一份文件训练 Stage1/2，再应用到所有文件做评估（更接近工程“单套系数跨频点使用”）
        stage12: Optional[Stage12Model] = None
        fs_used: Optional[float] = None

        freqs_hz: list[float] = []
        sinad_pre: list[float] = []
        sinad_post: list[float] = []
        thd_pre: list[float] = []
        thd_post: list[float] = []
        enob_pre: list[float] = []
        enob_post: list[float] = []
        sfdr_pre: list[float] = []
        sfdr_post: list[float] = []

        for i, p in enumerate(paths):
            fs_header, t_s, y_v = load_scope_csv(str(p))
            fs = _scope_effective_fs(fs_header)
            if fs_used is None:
                fs_used = fs
            if abs(fs - fs_header) / max(fs_header, 1e-12) > 1e-6:
                y_v, t_s = _maybe_resample_scope_stream(y_v, t_s, fs_header=fs_header, fs_effective=fs)

            y = y_v - float(np.mean(y_v))
            y = y / max(float(np.max(np.abs(y))), 1e-12)
            g1, g2 = split_two_way_interleaved(y)

            if i == 0:
                # 保存第一份解析，方便核对
                np.savetxt(out_dir / "scope_group1_1based_odd.csv", g1, delimiter=",")
                np.savetxt(out_dir / "scope_group2_1based_even.csv", g2, delimiter=",")
                stage12 = train_stage12_from_scope(y, fs=fs, device=device, taps=_env_int("SCOPE_STAGE12_TAPS", 63))

            assert stage12 is not None
            c0 = np.zeros_like(y)
            c1 = np.zeros_like(y)
            c0[0::2] = g1
            c1[1::2] = g2
            with torch.no_grad():
                x0 = torch.tensor(c0, dtype=torch.float32, device=device).view(1, 1, -1)
                x1 = torch.tensor(c1, dtype=torch.float32, device=device).view(1, 1, -1)
                y1c = stage12(x1)
                post_lin = _interleave_torch(x0, y1c).cpu().numpy().flatten()

            _, fin = _estimate_fund_bin(y, fs)
            pre_m = calc_metrics(y, fs, fin)
            post_m = calc_metrics(post_lin, fs, fin)

            freqs_hz.append(float(fin))
            sinad_pre.append(pre_m[0]); sinad_post.append(post_m[0])
            enob_pre.append(pre_m[1]); enob_post.append(post_m[1])
            thd_pre.append(pre_m[2]); thd_post.append(post_m[2])
            sfdr_pre.append(pre_m[3]); sfdr_post.append(post_m[3])

            print(f"[scope] {p.name} | fin≈{fin/1e9:.4f}GHz | SINAD {pre_m[0]:.2f}->{post_m[0]:.2f} | THD {pre_m[2]:.2f}->{post_m[2]:.2f} | SFDR {pre_m[3]:.2f}->{post_m[3]:.2f}")

        # 组装成与旧图一致的 m 字典（Post-QE/Post-NL 先用 Post-Linear 占位；需要扫频 Post-NL 再单独加）
        order = np.argsort(np.array(freqs_hz))
        freqs = np.array(freqs_hz, dtype=np.float64)[order]
        m = {
            "sinad_pre": list(np.array(sinad_pre)[order]),
            "sinad_post": list(np.array(sinad_post)[order]),
            "sinad_qe": list(np.array(sinad_post)[order]),
            "sinad_nl": list(np.array(sinad_post)[order]),
            "enob_pre": list(np.array(enob_pre)[order]),
            "enob_post": list(np.array(enob_post)[order]),
            "enob_qe": list(np.array(enob_post)[order]),
            "enob_nl": list(np.array(enob_post)[order]),
            "thd_pre": list(np.array(thd_pre)[order]),
            "thd_post": list(np.array(thd_post)[order]),
            "thd_qe": list(np.array(thd_post)[order]),
            "thd_nl": list(np.array(thd_post)[order]),
            "sfdr_pre": list(np.array(sfdr_pre)[order]),
            "sfdr_post": list(np.array(sfdr_post)[order]),
            "sfdr_qe": list(np.array(sfdr_post)[order]),
            "sfdr_nl": list(np.array(sfdr_post)[order]),
        }

        if _env_bool("PLOT_SCOPE_METRICS", True):
            plot_metrics(freqs, m)

        # 保存一份表格便于你后处理
        summary = np.stack(
            [
                freqs,
                np.array(sinad_pre)[order],
                np.array(sinad_post)[order],
                np.array(enob_pre)[order],
                np.array(enob_post)[order],
                np.array(thd_pre)[order],
                np.array(thd_post)[order],
                np.array(sfdr_pre)[order],
                np.array(sfdr_post)[order],
            ],
            axis=1,
        )
        np.savetxt(
            out_dir / "scope_metrics_summary.csv",
            summary,
            delimiter=",",
            header="fin_hz,sinad_pre,sinad_post,enob_pre,enob_post,thd_pre,thd_post,sfdr_pre,sfdr_post",
            comments="",
        )
        print(f"[scope] saved summary: {out_dir / 'scope_metrics_summary.csv'}")
        return

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
