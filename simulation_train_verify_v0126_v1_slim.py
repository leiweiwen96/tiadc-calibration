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


def _coherent_bin(fs: float, n: int, fin_hz: float, *, k_min: int = 8, k_guard: int = 8) -> tuple[int, float]:
    """
    相干采样（coherent sampling）工具：
    给定目标频率 fin_hz，把它吸附到 FFT 的整数 bin：
        k = round(fin_hz * N / fs), fin_coh = k * fs / N

    重要性：
    - 不做相干采样时，谱泄漏会把 spur 能量“摊平”，导致 SINAD/SFDR 看起来没提升甚至变差；
    - 训练里我们经常是按 bin 设计 loss（镜像 spur、谐波/IMD bin），评估也必须 bin 对齐才公平。
    """
    fs = float(fs)
    n = int(n)
    k = int(round(float(fin_hz) * n / fs))
    k = max(int(k_min), min(k, n // 2 - int(k_guard)))
    fin = float(k) * fs / float(n)
    return k, fin


def _sample_k_biased(k_min: int, k_max: int, *, high_bias: float) -> int:
    """
    频点采样偏置：high_bias<1 会偏向高频（u**high_bias 更靠近 1）。
    用来解决“训练频点太平均 => 高频学不到 => 4GHz 后没效果”的常见问题。
    """
    u = float(np.random.rand())
    u = u ** float(max(1e-3, high_bias))
    return int(k_min + round((k_max - k_min) * u))


def _stage12_prepare_input(y1: np.ndarray) -> np.ndarray:
    """
    Stage1/2 输入口径开关：
    - `STAGE12_SPARSE_INPUT=1`（默认）：只提供奇相样本（偶相置 0），更贴近真实 TIADC
    - `STAGE12_SPARSE_INPUT=0`：提供全速 y1（仿真里会更强，但现实中不可得）
    """
    sparse = _env_bool("STAGE12_SPARSE_INPUT", True)
    y1 = np.asarray(y1, dtype=np.float64)
    if not sparse:
        return y1
    c1 = np.zeros_like(y1)
    c1[1::2] = y1[1::2]
    return c1


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

    def _analog_fullrate(
        self,
        sig: np.ndarray,
        *,
        cutoff_freq: float,
        gain: float = 1.0,
        offset: float = 0.0,
        hd2: float = 0.0,
        hd3: float = 0.0,
    ) -> np.ndarray:
        """
        更贴近现实的“模拟前端 + 静态非线性”模型（全速网格上近似连续时间）：
        - 非线性：x + hd2*x^2 + hd3*x^3
        - 模拟带宽：5阶 Butter 低通
        - 增益/偏置：gain/offset

        注意：这里只建模“模拟链路”，不做采样抖动/量化/采样噪声。
        这些应当发生在“采样时刻”（只对真实存在的采样点发生）。
        """
        sig = np.asarray(sig, dtype=np.float64)
        nyq = self.fs / 2.0
        y = sig + float(hd2) * (sig**2) + float(hd3) * (sig**3)
        b, a = signal.butter(5, float(cutoff_freq) / nyq, btype="low")
        y = signal.lfilter(b, a, y)
        y = y * float(gain) + float(offset)
        return y.astype(np.float64, copy=False)

    def _sample_and_quantize(
        self,
        y_fullrate: np.ndarray,
        *,
        sample_offset: int,
        stride: int,
        delay_samples: float,
        jitter_std: float,
        snr_target: Optional[float],
        n_bits: Optional[int],
    ) -> np.ndarray:
        """
        在“真实采样点”上发生的过程：
        - 采样时刻偏差：用 fractional delay 近似（在全速网格上做亚采样点对齐）
        - 采样抖动：在采样点用 slope*dt 近似（dt~N(0, jitter_std)）
        - 采样噪声：按 snr_target 或默认噪声
        - 量化：n_bits
        """
        y = np.asarray(y_fullrate, dtype=np.float64)
        # “采样时刻偏差/固定偏斜”：对连续时间波形做亚采样延时后再抽取
        y = self._frac_delay(y, float(delay_samples))

        idx = np.arange(int(sample_offset), len(y), int(stride))
        s = y[idx].astype(np.float64, copy=False)

        if jitter_std and float(jitter_std) > 0:
            slope = np.gradient(y) * self.fs
            dt = np.random.normal(0.0, float(jitter_std), len(idx))
            s = s + slope[idx] * dt

        # 噪声：这里近似“采样噪声/热噪声”，发生在采样点
        if snr_target is None:
            s = s + np.random.normal(0.0, 1e-4, len(s))
        else:
            p = float(np.mean(s**2))
            n_p = p / (10 ** (float(snr_target) / 10.0))
            s = s + np.random.normal(0.0, np.sqrt(n_p), len(s))

        if n_bits is None:
            return s.astype(np.float64, copy=False)
        levels = 2 ** int(n_bits)
        step = 2.0 / levels
        s = np.clip(s, -1.0, 1.0)
        return (np.round(s / step) * step).astype(np.float64, copy=False)

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

    def capture_two_way(
        self,
        sig: np.ndarray,
        *,
        p0: dict,
        p1: dict,
        n_bits: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        更贴近现实两片 ADC 的“2-way TIADC 采集”：
        - 通道0只在偶相采样点产生数据（0,2,4,...）
        - 通道1只在奇相采样点产生数据（1,3,5,...）
        - 抖动/量化/噪声只对“真实采样点”生效

        返回：
        - c0_full：长度 N 的 full-rate 数组，仅偶位置有值
        - c1_full：长度 N 的 full-rate 数组，仅奇位置有值
        - y_full：交织后的 full-rate 输出（等价于 c0_full + c1_full）
        """
        x = np.asarray(sig, dtype=np.float64)

        # 解析参数（兼容旧字段）
        def _get(p: dict, name: str, default):
            return p[name] if name in p else default

        # 模拟前端
        a0 = self._analog_fullrate(
            x,
            cutoff_freq=float(_get(p0, "cutoff_freq", _get(p0, "cutoff_hz", 8.0e9))),
            gain=float(_get(p0, "gain", 1.0)),
            offset=float(_get(p0, "offset", 0.0)),
            hd2=float(_get(p0, "hd2", 0.0)),
            hd3=float(_get(p0, "hd3", 0.0)),
        )
        a1 = self._analog_fullrate(
            x,
            cutoff_freq=float(_get(p1, "cutoff_freq", _get(p1, "cutoff_hz", 8.0e9))),
            gain=float(_get(p1, "gain", 1.0)),
            offset=float(_get(p1, "offset", 0.0)),
            hd2=float(_get(p1, "hd2", 0.0)),
            hd3=float(_get(p1, "hd3", 0.0)),
        )

        # 采样 + 量化（只在真实采样点）
        s0 = self._sample_and_quantize(
            a0,
            sample_offset=0,
            stride=2,
            delay_samples=float(_get(p0, "delay_samples", 0.0)),
            jitter_std=float(_get(p0, "jitter_std", ADC_JITTER_STD)),
            snr_target=_get(p0, "snr_target", None),
            n_bits=n_bits,
        )
        s1 = self._sample_and_quantize(
            a1,
            sample_offset=1,
            stride=2,
            delay_samples=float(_get(p1, "delay_samples", 0.0)),
            jitter_std=float(_get(p1, "jitter_std", ADC_JITTER_STD)),
            snr_target=_get(p1, "snr_target", None),
            n_bits=n_bits,
        )

        c0 = np.zeros_like(x, dtype=np.float64)
        c1 = np.zeros_like(x, dtype=np.float64)
        c0[0::2] = s0[: len(c0[0::2])]
        c1[1::2] = s1[: len(c1[1::2])]
        y = c0 + c1
        return c0, c1, y


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


class FractionalDelayFIR(nn.Module):
    """
    可微分的窗函数 sinc 分数延时（比 3 阶 Farrow/Lagrange 在接近 Nyquist 时更稳）。

    现实动机：
    - 在多相/半速域里，4.5GHz 对应 fs_sub=10GHz 的 0.45*fs_sub，已经非常接近 Nyquist(5GHz)；
    - 3 阶 Farrow 在该区域幅相误差明显，会直接限制“镜像 spur”的可压制上限；
    - 用窗函数 sinc 生成的分数延时 FIR 能显著改善高频相位对齐能力。
    """

    def __init__(self, taps: int = 21, *, max_delay: float = 0.5):
        super().__init__()
        taps = int(taps)
        if taps < 5:
            taps = 5
        if taps % 2 == 0:
            taps += 1
        self.taps = int(taps)
        self.max_delay = float(max_delay)
        self.delay = nn.Parameter(torch.tensor(0.0))
        n = torch.arange(self.taps, dtype=torch.float32) - float(self.taps // 2)
        self.register_buffer("n", n.view(1, 1, -1))
        self.register_buffer("win", torch.hann_window(self.taps, periodic=False, dtype=torch.float32).view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将 delay 限制在 [-max_delay, max_delay]，避免训练不稳定
        d = torch.tanh(self.delay) * float(self.max_delay)
        n = self.n.to(device=x.device, dtype=x.dtype)
        w = self.win.to(device=x.device, dtype=x.dtype)
        h = torch.sinc(n - d) * w
        h = h / (torch.sum(h, dim=-1, keepdim=True) + 1e-12)
        return F.conv1d(x, h, padding=self.taps // 2)


class Stage12Model(nn.Module):
    def __init__(self, taps: int = 63):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.gain = nn.Parameter(torch.tensor(1.0))
        delay_impl = os.getenv("STAGE12_DELAY_IMPL", "fir").strip().lower()  # fir | farrow
        if delay_impl == "farrow":
            self.delay = FarrowDelay()
        else:
            d_taps = _env_int("STAGE12_DELAY_TAPS", 21)
            d_max = _env_float("STAGE12_DELAY_MAX", 0.5)
            self.delay = FractionalDelayFIR(d_taps, max_delay=float(d_max))
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


class PolyphaseStage12Model(nn.Module):
    """
    2-way TIADC 的“正确结构”的 Stage1/2（线性）：
    - 在半速子序列域建模：g0[n]=y[2n], g1[n]=y[2n+1]
    - 分别对 g0/g1 做 gain + frac-delay + FIR（全都在 10G 域进行）
    - 再交织回全速输出

    为什么要这样做？
    - 真实系统每片 ADC 只产出半速子序列，等化器应在子序列域设计；
    - 之前“零插 full-rate + Conv1d”会引入上采样镜像/调制效应，导致全频段难以稳定均衡。
    """

    def __init__(self, taps: int = 257, *, train_both: bool = False):
        super().__init__()
        taps = int(taps)
        if taps % 2 == 0:
            taps += 1
        self.train_both = bool(train_both)
        self.even = Stage12Model(taps=taps)  # 偶相子序列 g0（默认作为参考不动）
        self.odd = Stage12Model(taps=taps)  # 奇相子序列 g1（默认只训练这一支）
        # 2x2 交叉耦合 FIR（半速域 MIMO 等化）：
        # 现实动机：仅“逐路等化”在 fs/4 附近可能无法同时压下 (4.5,5.5) 这种成对镜像 spur；
        # 允许少量 cross-coupling（g0/g1 互相补偿）更接近实际的多相校正器结构。
        self.use_xcouple = _env_bool("STAGE12_USE_XCOUPLE", True)
        if self.use_xcouple:
            mtaps = _env_int("STAGE12_XCOUPLE_TAPS", 49)
            mtaps = int(mtaps)
            if mtaps % 2 == 0:
                mtaps += 1
            self.mix_fir = nn.Conv1d(2, 2, kernel_size=mtaps, padding=mtaps // 2, bias=False)
            with torch.no_grad():
                self.mix_fir.weight.zero_()
                self.mix_fir.weight[0, 0, mtaps // 2] = 1.0
                self.mix_fir.weight[1, 1, mtaps // 2] = 1.0
        # image-canceller（全速域的 π-shift 分支）：
        # y = y + (-1)^n * FIR( (-1)^n * y )
        # 等价于对 fs/2 搬移项做可学习抑制，比“只靠两路分别等化”更容易压下 4~6GHz 的镜像 spur。
        self.use_img_cancel = _env_bool("STAGE12_USE_IMG_CANCEL", True)
        if self.use_img_cancel:
            itaps = _env_int("STAGE12_IMG_CANCEL_TAPS", 193)
            itaps = int(itaps)
            if itaps % 2 == 0:
                itaps += 1
            self.img_fir = nn.Conv1d(1, 1, kernel_size=itaps, padding=itaps // 2, bias=False)
            with torch.no_grad():
                self.img_fir.weight.zero_()

        # 工程默认：固定一片 ADC 为参考（更符合实测自校准口径，且避免欠定导致整体漂移）
        if not self.train_both:
            with torch.no_grad():
                # even 初始化为严格 identity
                self.even.gain.fill_(1.0)
                self.even.delay.delay.fill_(0.0)
                self.even.fir.weight.zero_()
                self.even.fir.weight[0, 0, taps // 2] = 1.0
            for p in self.even.parameters():
                p.requires_grad_(False)

    def forward(self, g0: torch.Tensor, g1: torch.Tensor) -> torch.Tensor:
        # g0/g1: [B,1,N/2]
        if self.train_both:
            g0c, g1c = self.even(g0), self.odd(g1)
        else:
            g0c, g1c = g0, self.odd(g1)
        if self.use_xcouple:
            x2 = torch.cat([g0c, g1c], dim=1)
            y2 = self.mix_fir(x2)
            g0c, g1c = y2[:, 0:1, :], y2[:, 1:2, :]
        y = _interleave_halfrate_torch(g0c, g1c)
        if self.use_img_cancel:
            n2 = int(y.shape[-1])
            # sign: [1,1,N] = (+1,-1,+1,-1,...)
            idx = torch.arange(n2, device=y.device)
            sign = (1.0 - 2.0 * (idx % 2).to(y.dtype)).view(1, 1, -1)
            z = y * sign
            zf = self.img_fir(z)
            y = y + zf * sign
        return y


def _interleave_halfrate_torch(g0: torch.Tensor, g1: torch.Tensor) -> torch.Tensor:
    """g0/g1 为半速序列，交织成全速序列。"""
    L = int(min(g0.shape[-1], g1.shape[-1]))
    out = torch.zeros(g0.shape[:-1] + (2 * L,), device=g0.device, dtype=g0.dtype)
    out[..., 0::2] = g0[..., :L]
    out[..., 1::2] = g1[..., :L]
    return out


def _design_windowed_sinc_lpf(*, num_taps: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """
    设计一个简单的窗函数 sinc 低通 FIR，用于“只滤校正量 corr，不滤原信号”：
    - 现实原因：非线性校正如果允许产生接近 Nyquist 的高频分量，容易在高频单音测试时
      把折叠谐波/边缘带外成分“抬起来”，造成你看到的高频 THD 变差；
    - 工程做法：对校正量做带宽限制（band-limited correction），只在关注带宽内起作用。
    """
    num_taps = int(num_taps)
    if num_taps < 3:
        raise ValueError("num_taps must be >=3")
    if num_taps % 2 == 0:
        num_taps += 1
    fs = float(fs)
    cutoff_hz = float(cutoff_hz)
    cutoff_hz = max(0.0, min(cutoff_hz, 0.499 * fs))
    if cutoff_hz <= 0.0:
        # disable
        h = np.zeros(num_taps, dtype=np.float32)
        h[num_taps // 2] = 1.0
        return h
    fc = cutoff_hz / fs  # cycles/sample, in (0,0.5)
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    h = 2.0 * fc * np.sinc(2.0 * fc * n)  # np.sinc(x)=sin(pi x)/(pi x)
    w = np.hamming(num_taps).astype(np.float64)
    h = h * w
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        h = np.zeros(num_taps, dtype=np.float32)
        h[num_taps // 2] = 1.0
        return h
    h = (h / s).astype(np.float32, copy=False)
    return h


def _apply_fir_same(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """对 [B,1,N] 做 same padding FIR（h: [1,1,T]）。"""
    t = int(h.shape[-1])
    if t <= 1:
        return x
    return F.conv1d(x, h, padding=t // 2)


class PhysicalNonlinearityLayer(nn.Module):
    def __init__(self, order: int = 3, *, fs: float = 20e9, corr_lpf_hz: float = 0.0, corr_lpf_taps: int = 33):
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
        # TIADC：even/odd 两相往往具有不同的静态非线性，必须分相建模
        self.raw_even = nn.Parameter(torch.zeros(self.order - 1))
        self.raw_odd = nn.Parameter(torch.zeros(self.order - 1))
        # 对“校正量”做带宽限制（默认关闭；设置 POST_QE_CORR_LPF_HZ>0 启用）
        self.corr_lpf_hz = float(corr_lpf_hz)
        self.corr_lpf_taps = int(corr_lpf_taps)
        if self.corr_lpf_hz > 0.0:
            h = _design_windowed_sinc_lpf(num_taps=int(self.corr_lpf_taps), cutoff_hz=float(self.corr_lpf_hz), fs=float(fs))
            self.register_buffer("corr_fir", torch.tensor(h, dtype=torch.float32).view(1, 1, -1))
        else:
            self.register_buffer("corr_fir", torch.zeros(1, 1, 1, dtype=torch.float32))

    def _res(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        a = self.scales.to(x.device, x.dtype) * torch.tanh(raw.to(x.device, x.dtype))
        y = torch.zeros_like(x)
        for i, p in enumerate(range(2, self.order + 1)):
            y = y + a[i] * (x**p)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        # 分相校正（even/odd）
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        ye = xe + float(self.alpha) * torch.clamp(self._res(xe, self.raw_even), -float(self.res_clip), float(self.res_clip))
        yo = xo + float(self.alpha) * torch.clamp(self._res(xo, self.raw_odd), -float(self.res_clip), float(self.res_clip))
        y = x.clone()
        y[..., 0::2] = ye
        y[..., 1::2] = yo
        # band-limited correction：只滤校正量 (y-x)，避免高频被“校正量”抬起来
        if float(self.corr_lpf_hz) > 0.0 and int(self.corr_fir.shape[-1]) > 1:
            corr = y - x
            hf = self.corr_fir.to(device=x.device, dtype=x.dtype)
            y = x + _apply_fir_same(corr, hf)
        return y


class DifferentiableMemoryPolynomial(nn.Module):
    def __init__(self, memory_depth: int = 21, nonlinear_order: int = 3, *, fs: float = 20e9, corr_lpf_hz: float = 0.0, corr_lpf_taps: int = 33):
        super().__init__()
        self.K = int(memory_depth)
        self.P = int(nonlinear_order)
        if self.K < 1 or self.P < 2:
            raise ValueError("memory_depth>=1, nonlinear_order>=2 required")
        # 经验上记忆多项式更容易“过度补偿”并在高频产生副作用，因此默认更保守一些：
        # - 通过较小的 alpha/res_clip 控制校正量幅度
        # - 仍保留环境变量可手动调大
        self.alpha = float(os.getenv("POST_NL_ALPHA", "0.6"))
        self.res_clip = float(os.getenv("POST_NL_RES_CLIP", "0.15"))
        max_p2 = float(os.getenv("POST_NL_MAX_P2", "4e-2"))
        max_p3 = float(os.getenv("POST_NL_MAX_P3", "2.5e-2"))
        max_hi = float(os.getenv("POST_NL_MAX_PHI", "8e-3"))
        scales = []
        for p in range(2, self.P + 1):
            scales.append(max_p2 if p == 2 else max_p3 if p == 3 else max_hi)
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        # TIADC：even/odd 两相是两个子 ADC（采样间隔为 2Ts），记忆多项式也应分相
        self.raw_even = nn.Parameter(torch.zeros(self.K, self.P - 1))
        self.raw_odd = nn.Parameter(torch.zeros(self.K, self.P - 1))
        # 对“校正量”做带宽限制（默认关闭；设置 POST_NL_CORR_LPF_HZ>0 启用）
        self.corr_lpf_hz = float(corr_lpf_hz)
        self.corr_lpf_taps = int(corr_lpf_taps)
        if self.corr_lpf_hz > 0.0:
            h = _design_windowed_sinc_lpf(num_taps=int(self.corr_lpf_taps), cutoff_hz=float(self.corr_lpf_hz), fs=float(fs))
            self.register_buffer("corr_fir", torch.tensor(h, dtype=torch.float32).view(1, 1, -1))
        else:
            self.register_buffer("corr_fir", torch.zeros(1, 1, 1, dtype=torch.float32))

    @staticmethod
    def _delay(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return x
        xp = F.pad(x, (k, 0), mode="constant", value=0.0)
        return xp[..., :-k]

    def _res_seq(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
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
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        re = torch.clamp(self._res_seq(xe, self.raw_even), -float(self.res_clip), float(self.res_clip))
        ro = torch.clamp(self._res_seq(xo, self.raw_odd), -float(self.res_clip), float(self.res_clip))
        ye = xe + float(self.alpha) * re
        yo = xo + float(self.alpha) * ro
        y = x.clone()
        y[..., 0::2] = ye
        y[..., 1::2] = yo
        if float(self.corr_lpf_hz) > 0.0 and int(self.corr_fir.shape[-1]) > 1:
            corr = y - x
            hf = self.corr_fir.to(device=x.device, dtype=x.dtype)
            y = x + _apply_fir_same(corr, hf)
        return y


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
    2-way TIADC 的典型交织镜像 spur 在 fs/2 ± fin 附近。
    离散 bin 上，主镜像近似为 k_img = |N/2 - k0|. 返回 P_img / P_fund（越小越好）。
    """
    n = int(sig.shape[-1])
    win = torch.hann_window(n, device=sig.device, dtype=sig.dtype).view(1, 1, -1)
    S = torch.fft.rfft((sig - torch.mean(sig, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    P = (S.real**2 + S.imag**2)
    k0 = int(fund_bin)
    k_img = abs(n // 2 - k0)
    k_img = max(1, min(k_img, P.shape[-1] - 2))
    eps = 1e-12
    s0 = max(0, k0 - int(guard))
    e0 = min(P.shape[-1], k0 + int(guard) + 1)
    p_fund = torch.sum(P[..., s0:e0], dim=-1) + eps
    si = max(0, k_img - int(guard))
    ei = min(P.shape[-1], k_img + int(guard) + 1)
    p_img = torch.sum(P[..., si:ei], dim=-1) + eps
    return torch.mean(p_img / p_fund)


def _fund_hold_loss(sig: torch.Tensor, ref: torch.Tensor, *, fund_bin: int, guard: int = 3) -> torch.Tensor:
    """
    基波保持：只压 spur 很容易把“基波也压掉/拉爆”出现投机解。
    用 log 幅度对齐更稳定（与量级无关）。
    """
    n = int(sig.shape[-1])
    win = torch.hann_window(n, device=sig.device, dtype=sig.dtype).view(1, 1, -1)
    S = torch.fft.rfft((sig - torch.mean(sig, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    R = torch.fft.rfft((ref - torch.mean(ref, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    P = (S.real**2 + S.imag**2)
    Q = (R.real**2 + R.imag**2)
    s0 = max(0, int(fund_bin) - int(guard))
    e0 = min(P.shape[-1], int(fund_bin) + int(guard) + 1)
    ps = torch.sum(P[..., s0:e0], dim=-1) + 1e-12
    pr = torch.sum(Q[..., s0:e0], dim=-1) + 1e-12
    return torch.mean((torch.log10(ps) - torch.log10(pr)) ** 2)


def _funds_hold_loss(sig: torch.Tensor, ref: torch.Tensor, *, fund_bins: list[int], guard: int = 3) -> torch.Tensor:
    """
    multisine 场景下的“多基波保持”：
    - 单音时，用一个 fund_bin 保持即可；
    - 多音时，如果只保持其中一个代表 bin，其它基波就可能被“牺牲掉”来换更小的镜像 spur 比（投机解）。

    做法：对每个基波 bin（各自 ±guard）计算 log 能量对齐误差，然后取平均。
    """
    if not fund_bins:
        return torch.zeros((), device=sig.device, dtype=sig.dtype)
    n = int(sig.shape[-1])
    win = torch.hann_window(n, device=sig.device, dtype=sig.dtype).view(1, 1, -1)
    S = torch.fft.rfft((sig - torch.mean(sig, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    R = torch.fft.rfft((ref - torch.mean(ref, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    P = (S.real**2 + S.imag**2)
    Q = (R.real**2 + R.imag**2)
    losses = 0.0
    cnt = 0
    for k in sorted(set(int(x) for x in fund_bins)):
        s0 = max(0, k - int(guard))
        e0 = min(P.shape[-1], k + int(guard) + 1)
        ps = torch.sum(P[..., s0:e0], dim=-1) + 1e-12
        pr = torch.sum(Q[..., s0:e0], dim=-1) + 1e-12
        losses = losses + (torch.log10(ps) - torch.log10(pr)) ** 2
        cnt += 1
    return torch.mean(losses / float(max(cnt, 1)))


def _fir_gain_band_loss(w: torch.Tensor, *, n_fft: int = 1024, min_gain: float = 0.7, max_gain: float = 1.5) -> torch.Tensor:
    """
    FIR 频响增益约束（工程上很关键）：
    线性等化若允许任意增益，很容易在某些频段“过度补偿”，把噪声/抖动/非线性一并放大，
    从而出现你看到的“某段频率突然崩盘”。

    这里简单约束 |H(e^{jω})| 在 [min_gain, max_gain] 内（超出部分惩罚）。
    """
    w = w.view(-1)
    H = torch.fft.rfft(w, n=int(n_fft))
    mag = torch.abs(H)
    lo = torch.relu(float(min_gain) - mag)
    hi = torch.relu(mag - float(max_gain))
    return torch.mean(lo * lo + hi * hi)


def _spec_hold_excluding_bins_loss(
    yhat: torch.Tensor,
    yin: torch.Tensor,
    *,
    fund_bin: int,
    guard: int = 3,
    extra_exclude: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    频谱保持（带内“别乱动”约束）：
    - 只优化镜像 spur 很容易通过“抬噪声底/改坏其它频点”获得更小的 spur 比；
    - 这里约束 yhat 的整体谱形状尽量贴近输入 yin（即保持噪声底/带内频响），
      但**排除**基波附近与镜像 spur 附近（这些正是我们希望改变的位置）。
    """
    n = int(yhat.shape[-1])
    win = torch.hann_window(n, device=yhat.device, dtype=yhat.dtype).view(1, 1, -1)
    Yh = torch.fft.rfft((yhat - torch.mean(yhat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    Yi = torch.fft.rfft((yin - torch.mean(yin, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    Ph = (Yh.real**2 + Yh.imag**2) + 1e-12
    Pi = (Yi.real**2 + Yi.imag**2) + 1e-12
    lh = torch.log10(Ph)
    li = torch.log10(Pi)

    mask = torch.ones_like(lh, dtype=torch.bool)
    # 排除 DC 附近
    mask[..., :5] = False
    # 排除基波附近
    k0 = int(fund_bin)
    s0 = max(0, k0 - int(guard))
    e0 = min(lh.shape[-1], k0 + int(guard) + 1)
    mask[..., s0:e0] = False
    # 排除镜像 spur 附近（2-way 主镜像）
    k_img = abs(n // 2 - k0)
    k_img = max(1, min(k_img, lh.shape[-1] - 2))
    si = max(0, k_img - int(guard))
    ei = min(lh.shape[-1], k_img + int(guard) + 1)
    mask[..., si:ei] = False
    # 额外排除（可选）
    if extra_exclude:
        for kk in extra_exclude:
            kk = int(kk)
            ss = max(0, kk - int(guard))
            ee = min(lh.shape[-1], kk + int(guard) + 1)
            mask[..., ss:ee] = False

    diff = (lh - li) * mask
    denom = torch.sum(mask.to(diff.dtype)) + 1e-12
    return torch.sum(diff * diff) / denom


def _band_energy_sum(P: torch.Tensor, bins: list[int], guard: int) -> torch.Tensor:
    """把若干 bin（各自 ±guard）能量累加。P: [B,1,N/2+1]"""
    if not bins:
        return torch.zeros(P.shape[:-1], device=P.device, dtype=P.dtype)
    out = 0.0
    for k in bins:
        s = max(0, int(k) - int(guard))
        e = min(P.shape[-1], int(k) + int(guard) + 1)
        out = out + torch.sum(P[..., s:e], dim=-1)
    return out


def _two_way_image_bins(k_funds: list[int], n: int) -> list[int]:
    """2-way 主镜像：k_img = |N/2 - k|（折叠到 rfft bin）。"""
    out: list[int] = []
    half = int(n) // 2
    for k in k_funds:
        ki = abs(half - int(k))
        ki = max(1, min(ki, half - 1))
        out.append(int(ki))
    return sorted(set(out))


def _sample_k_set(
    k_min: int,
    k_max: int,
    *,
    n_tones: int,
    min_sep: int = 24,
    n: Optional[int] = None,
    avoid_image_collision: bool = True,
) -> list[int]:
    """
    在 [k_min,k_max] 采样一组互不太接近的 bin，用于 multisine。

    关键工程细节（2-way TIADC 特有）：
    - 镜像 bin 近似 k_img = |N/2 - k|；
    - 若采样集合里同时包含 k 与其镜像 k_img（例如围绕 N/4 成对的频点），
      那么“基波”和“镜像”在频域上会互相重叠，镜像损失会变得不可辨识/自相矛盾，
      训练很容易出现“保住一个、牺牲另一个”的现象（你看到的某些频点掉下去）。
    - 这里默认避免挑选这种会互相撞车的 tone 组合（不是特殊频点补丁，而是避免欠定目标）。
    """
    n_tones = int(n_tones)
    if n_tones <= 1:
        return [int(np.random.randint(int(k_min), int(k_max) + 1))]
    pool = np.arange(int(k_min), int(k_max) + 1, dtype=np.int32)
    np.random.shuffle(pool)
    out: list[int] = []
    half = (int(n) // 2) if (n is not None and int(n) > 0) else None
    for k in pool:
        kk = int(k)
        if not all(abs(kk - x) >= int(min_sep) for x in out):
            continue
        if avoid_image_collision and half is not None:
            ki = abs(int(half) - kk)
            # 限定到有效 rfft bin 范围（这里只用于“是否撞车”的判定）
            ki = max(1, min(int(ki), int(half) - 1))
            if ki in out:
                continue
        out.append(kk)
        if len(out) >= n_tones:
            break
    if len(out) == 0:
        out = [int(k_min)]
    return sorted(out)


def _multisine_np(*, n: int, bins: list[int], amp: float = 0.85) -> np.ndarray:
    """生成相干多音（bin 对齐），相位随机，整体归一化到 amp。"""
    n = int(n)
    t = np.arange(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for k in bins:
        ph = float(np.random.uniform(0.0, 2.0 * np.pi))
        a = float(np.random.uniform(0.3, 1.0))
        y = y + a * np.sin(2.0 * np.pi * float(k) * t / float(n) + ph)
    m = float(np.max(np.abs(y))) if len(y) else 1.0
    if m < 1e-12:
        return y
    return (y / m * float(amp)).astype(np.float64, copy=False)


def _harm_bins_folded(k: int, n: int, max_h: int = 5) -> list[int]:
    bins: list[int] = []
    for h in range(2, max_h + 1):
        kk = (h * int(k)) % int(n)
        if kk > n // 2:
            kk = n - kk
        if 1 <= kk <= (n // 2 - 1):
            bins.append(int(kk))
    return sorted(set(bins))


def _fold_to_rfft_bin(k: int, n: int) -> int:
    """把任意“全 FFT bin”折叠到 rfft 的 [0, N/2]。"""
    kk = int(k) % int(n)
    if kk > n // 2:
        kk = n - kk
    return int(kk)


def _imd3_bins_folded(k1: int, k2: int, n: int) -> list[int]:
    """
    两音 IMD3（折叠后）：2f1-f2 与 2f2-f1（以及其 mod N 的折叠）。
    这是工程上评估非线性最常用的“可校正目标”（比用理想 ref 拟合波形更贴近实际）。
    """
    a = _fold_to_rfft_bin(2 * int(k1) - int(k2), int(n))
    b = _fold_to_rfft_bin(2 * int(k2) - int(k1), int(n))
    out = []
    for kk in (a, b):
        if 1 <= kk <= (n // 2 - 1):
            out.append(int(kk))
    return sorted(set(out))


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


def train_stage12(sim: TIADCSimulator, *, device: str, p0: dict, p1: dict, pref: Optional[dict] = None) -> Tuple[nn.Module, float]:
    """
    Stage1/2 线性校准（2-way TIADC）：
    - 旧版做法：把 y1 当成“全速连续波形”来校正，再取奇样本参与交织（会用到真实系统不存在的“偶样本信息”）
    - 改进做法（默认启用）：只把 y1 的奇相样本喂给模型（偶位置填 0），并用“镜像 spur 最小化”自监督训练。
      这与 2-way 频域模型中 ±π 搬移项（fs/2±fin 镜像）更一致，也更接近实测数据形态。
    """
    np.random.seed(42)
    torch.manual_seed(42)

    #use_image_mode = _env_bool("STAGE12_IMAGE_TRAIN", True)
    use_image_mode = _env_bool("STAGE12_IMAGE_TRAIN", True)
    use_polyphase = _env_bool("STAGE12_POLYPHASE", True)
    # 仿真里默认启用“参考监督”，把问题从欠定的自监督变成可辨识系统辨识：
    # ref 由 pref/pref 的理想两路采样产生（只在各自相位采样点有值，口径一致）
    # 符合实际：默认关闭参考监督（实测通常拿不到理想 ref）
    use_ref = _env_bool("STAGE12_SUPERVISED_REF", False) and (pref is not None)
    # 现实工程默认更稳：固定偶相(ADC0)为参考，只训练奇相(ADC1)的主路等化；
    # 镜像 spur 的主要抑制依赖 widely-linear 分支（img_cancel / xcouple），它们天然使用两路信息。
    train_both = _env_bool("STAGE12_TRAIN_BOTH", False)
    # 工程建议：高频想要有效，FIR taps 往往需要 >=129（默认提高）
    taps = _env_int("STAGE12_TAPS", 257)
    n = _env_int("STAGE12_TONE_N", 16384)
    # continuation：先训练“镜像抑制分支”（B(ω)），再训练主路 A(ω)
    steps0 = _env_int("STAGE12_S0_STEPS", 900)
    steps1 = _env_int("STAGE12_S1_STEPS", 800)
    steps2 = _env_int("STAGE12_S2_STEPS", 1400)
    batch = _env_int("STAGE12_BATCH", 4)
    lr_img = _env_float("STAGE12_LR_IMG", 9e-4)
    lr_dg = _env_float("STAGE12_LR_DG", 8e-4)
    lr_fir = _env_float("STAGE12_LR_FIR", 6e-4)
    # 评估里默认包含 0.1GHz，因此训练下限也应覆盖到该范围，否则该段属于“训练域外”，掉点是正常现象
    fmin = _env_float("STAGE12_FMIN_HZ", 0.1e9)
    # 默认覆盖到与评估一致的上限（6.6GHz）
    fmax = _env_float("STAGE12_FMAX_HZ", 6.6e9)
    guard = _env_int("STAGE12_IMG_GUARD", 3)
    # 权重：监督参考时以 ref 对齐为主，镜像 spur 仅作为正则
    # 高端镜像往往是 SINAD/SFDR 的主瓶颈，因此在自监督模式下默认给更高权重
    w_img = _env_float("STAGE12_W_IMG", 600.0 if use_ref else 5500.0)
    w_ref_time = _env_float("STAGE12_W_REF_TIME", 35.0)
    w_ref_spec = _env_float("STAGE12_W_REF_SPEC", 6.0)
    # 约束项：不要让 yhat 偏离输入太多；基波保持用于防投机
    w_delta = _env_float("STAGE12_W_DELTA", 2.0)
    w_fund = _env_float("STAGE12_W_FUND", 3.0)
    w_reg = _env_float("STAGE12_W_REG", 2e-5)
    w_fir_band = _env_float("STAGE12_W_FIR_BAND", 8e-3)
    fir_min_gain = _env_float("STAGE12_FIR_MIN_GAIN", 0.80)
    fir_max_gain = _env_float("STAGE12_FIR_MAX_GAIN", 1.25)
    # 全频段均衡关键：频谱保持（排除基波/镜像附近）防止“抬噪声底换 spur”
    w_spec_hold = _env_float("STAGE12_W_SPEC_HOLD", 6.0)
    # “全频段均衡”建议：用均匀覆盖采样（sampler=cycle），high_bias 设为 1.0
    sampler = os.getenv("STAGE12_SAMPLER", "cycle").strip().lower()
    high_bias = _env_float("STAGE12_HIGH_BIAS", 1.0)

    # 仍然返回一个 scale，保持 evaluate() 的接口不变
    scale = 1.0

    # 默认使用“正确结构”的多相/子序列域 Stage1/2
    model: nn.Module
    if use_polyphase:
        model = PolyphaseStage12Model(taps=taps, train_both=train_both).to(device)
    else:
        model = Stage12Model(taps=taps).to(device)

    if not use_image_mode:
        # 兼容：保留旧的 supervised 训练（但改成“稀疏输入”，避免偷用偶样本）
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

        c1 = _stage12_prepare_input(avg1)
        inp = torch.FloatTensor(c1 / scale).view(1, 1, -1).to(device)
        tgt = torch.FloatTensor(avg0 / scale).view(1, 1, -1).to(device)

        loss_fn = Stage12Loss()
        print("=== Stage1/2 (legacy supervised, sparse input) ===")
        opt = optim.Adam(model.parameters(), lr=8e-4, betas=(0.5, 0.9))
        for _ in range(1600):
            opt.zero_grad()
            # legacy path 仅适用于旧的 full-rate 模型
            if not isinstance(model, Stage12Model):
                raise RuntimeError("legacy supervised path requires STAGE12_POLYPHASE=0")
            loss = loss_fn(model(inp), tgt, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        return model, float(scale)

    # 自监督镜像 spur 训练（tone 随机扫频）
    fs = float(sim.fs)
    t = torch.arange(n, device=device, dtype=torch.float32) / fs
    k_min = max(int(np.ceil(fmin * n / fs)), 8)
    k_max = min(int(np.floor(fmax * n / fs)), n // 2 - 8)

    # 频点采样器：cycle（全覆盖）/ random（随机，可选高频偏置）
    k_pool = np.arange(int(k_min), int(k_max) + 1, dtype=np.int32)
    if len(k_pool) <= 0:
        raise ValueError("Stage12 k_pool empty: check STAGE12_FMIN_HZ/FMAX_HZ/N/fs")
    _k_idx = 0
    np.random.shuffle(k_pool)

    def _next_k() -> int:
        nonlocal _k_idx, k_pool
        if sampler == "random":
            return _sample_k_biased(int(k_min), int(k_max), high_bias=high_bias)
        # cycle：保证频段均匀覆盖；每轮洗牌一次避免周期性
        k = int(k_pool[_k_idx])
        _k_idx += 1
        if _k_idx >= len(k_pool):
            _k_idx = 0
            np.random.shuffle(k_pool)
        return k

    def _next_k_set(*, n_tones: int, min_sep: int) -> list[int]:
        """
        multisine 的“覆盖式采样”：
        之前用纯随机从 [k_min,k_max] 选 k_funds，会导致某些频点（例如 5.5GHz）在整个训练里出现次数极少，
        从而出现“整个高频段某些点效果很差/甚至变差”的现象。

        这里用 cycle 的 _next_k() 来构造 tone 集合，保证长期统计上对全频段更均匀覆盖（不是特殊频点补丁）。
        """
        out: list[int] = []
        tries = 0
        max_tries = max(64, int(n_tones) * 32)
        half = int(n) // 2
        while len(out) < int(n_tones) and tries < max_tries:
            tries += 1
            kk = int(_next_k())  # cycle coverage
            if any(abs(kk - x) < int(min_sep) for x in out):
                continue
            # 避免与镜像撞车（可辨识性）
            ki = abs(half - kk)
            ki = max(1, min(int(ki), half - 1))
            if ki in out:
                continue
            out.append(int(kk))
        if not out:
            out = [int(_next_k())]
        return sorted(out)

    def _make_batch_loss(
        *,
        kmin_override: Optional[int] = None,
        kmax_override: Optional[int] = None,
        img_scale: float = 1.0,
        stratify_mid_k: Optional[int] = None,
        spec_scale: float = 1.0,
    ) -> torch.Tensor:
        loss_acc = 0.0
        kmin_use = int(k_min if kmin_override is None else kmin_override)
        kmax_use = int(k_max if kmax_override is None else kmax_override)

        def _loss_for_src(src_np: np.ndarray, *, k_funds_local: list[int], img_scale_local: float, spec_scale_local: float) -> torch.Tensor:
            """把“给定激励 + tone bins”跑一遍前向并计算 loss（核心目标不变）。"""
            # 更贴近现实：直接模拟两片 ADC 的采样（只在各自相位产生样本）
            c0, c1, pre = sim.capture_two_way(src_np, p0=p0, p1=p1, n_bits=ADC_NBITS)
            if use_ref and pref is not None:
                _, _, ref = sim.capture_two_way(src_np, p0=pref, p1=pref, n_bits=ADC_NBITS)
            else:
                ref = None
            s = max(float(np.max(np.abs(pre))), 1e-12)

            pre_t = torch.tensor(pre / s, dtype=torch.float32, device=device).view(1, 1, -1)
            ref_t = None
            if ref is not None:
                ref_t = torch.tensor(ref / s, dtype=torch.float32, device=device).view(1, 1, -1)
            # 子序列域输入：g0/g1（半速）
            g0 = torch.tensor((c0[0::2] / s), dtype=torch.float32, device=device).view(1, 1, -1)
            g1 = torch.tensor((c1[1::2] / s), dtype=torch.float32, device=device).view(1, 1, -1)

            if isinstance(model, PolyphaseStage12Model):
                yhat = model(g0, g1)
            else:
                # fallback：旧 full-rate 模型（不推荐）
                x0 = torch.tensor(c0 / s, dtype=torch.float32, device=device).view(1, 1, -1)
                x1 = torch.tensor(c1 / s, dtype=torch.float32, device=device).view(1, 1, -1)
                y1c = model(x1)
                yhat = _interleave_torch(x0, y1c)

            # 主目标：压制交织镜像 spur（fs/2±f）
            # 采用“逐 tone 归一化，再取均值”的 l_img（每个 tone 权重更均衡）。
            k_imgs_local = _two_way_image_bins(k_funds_local, n)
            k_imgs_local = sorted(set(k_imgs_local) - set(int(x) for x in k_funds_local))
            win = torch.hann_window(n, device=device, dtype=pre_t.dtype).view(1, 1, -1)
            Y = torch.fft.rfft((yhat - torch.mean(yhat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
            P = (Y.real**2 + Y.imag**2)
            # 逐 tone 的 fund/img 比值
            p_funds = []
            p_imgs = []
            funds_set = set(int(x) for x in k_funds_local)
            for kk in k_funds_local:
                kk = int(kk)
                ki = abs((n // 2) - kk)
                ki = max(1, min(int(ki), (n // 2 - 1)))
                # 若镜像恰好落在某个基波上（不可辨识），跳过该 tone 的镜像项
                if ki in funds_set:
                    continue
                p_funds.append(_band_energy_sum(P, [kk], guard=int(guard)) + 1e-12)
                p_imgs.append(_band_energy_sum(P, [ki], guard=int(guard)) + 1e-12)
            if len(p_funds) == 0:
                # fallback：退化为总能量比
                p_fund = _band_energy_sum(P, k_funds_local, guard=int(guard)) + 1e-12
                p_img = _band_energy_sum(P, k_imgs_local, guard=int(guard)) + 1e-12
                r_out_vec = [p_img / p_fund]
            else:
                r_out_vec = [pi / pf for (pi, pf) in zip(p_imgs, p_funds)]
            # 关键“工程约束”：校正不应把镜像 spur 做得更差（hinge），同时保留一点“绝对压镜像”的梯度
            use_hinge = _env_bool("STAGE12_IMG_HINGE", True)
            if use_hinge:
                beta = _env_float("STAGE12_IMG_HINGE_BETA", 0.3)
                Yi = torch.fft.rfft((pre_t - torch.mean(pre_t, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
                Pi = (Yi.real**2 + Yi.imag**2)
                r_in_vec = []
                if len(p_funds) == 0:
                    p_fund_in = _band_energy_sum(Pi, k_funds_local, guard=int(guard)) + 1e-12
                    p_img_in = _band_energy_sum(Pi, k_imgs_local, guard=int(guard)) + 1e-12
                    r_in_vec = [p_img_in / p_fund_in]
                else:
                    for kk in k_funds_local:
                        kk = int(kk)
                        ki = abs((n // 2) - kk)
                        ki = max(1, min(int(ki), (n // 2 - 1)))
                        if ki in funds_set:
                            continue
                        pf = _band_energy_sum(Pi, [kk], guard=int(guard)) + 1e-12
                        pi = _band_energy_sum(Pi, [ki], guard=int(guard)) + 1e-12
                        r_in_vec.append(pi / pf)
                terms = [torch.relu(ro - ri) + float(beta) * ro for (ro, ri) in zip(r_out_vec, r_in_vec)]
                l_img = torch.mean(torch.stack([t.squeeze() for t in terms]))
            else:
                l_img = torch.mean(torch.stack([ro.squeeze() for ro in r_out_vec]))

            # 监督参考：ref 对齐为主；自监督：基波保持+小改动为主
            if use_ref and ref_t is not None:
                crop = 300
                l_ref_time = torch.mean((yhat[..., crop:-crop] - ref_t[..., crop:-crop]) ** 2)
                Yh = torch.fft.rfft(yhat, dim=-1, norm="ortho")
                Yr = torch.fft.rfft(ref_t, dim=-1, norm="ortho")
                Ph = (Yh.real**2 + Yh.imag**2) + 1e-12
                Pr = (Yr.real**2 + Yr.imag**2) + 1e-12
                l_ref_spec = torch.mean((torch.log10(Ph) - torch.log10(Pr)) ** 2)
                l_delta = torch.mean((yhat - pre_t) ** 2)
                loss_main = float(w_ref_time) * l_ref_time + float(w_ref_spec) * l_ref_spec + float(w_img) * l_img + 0.2 * float(w_delta) * l_delta
            else:
                k_rep = int(k_funds_local[len(k_funds_local) // 2]) if len(k_funds_local) else int(max(k_min, 8))
                l_fund = _funds_hold_loss(yhat, pre_t, fund_bins=k_funds_local, guard=int(guard)) if len(k_funds_local) else _fund_hold_loss(yhat, pre_t, fund_bin=int(k_rep), guard=int(guard))
                l_delta = torch.mean((yhat - pre_t) ** 2)
                l_spec_hold = _spec_hold_excluding_bins_loss(
                    yhat,
                    pre_t,
                    fund_bin=int(k_rep),
                    guard=int(guard),
                    extra_exclude=(list(k_funds_local) + list(k_imgs_local)),
                )
                loss_main = float(w_img) * float(img_scale_local) * l_img + float(w_fund) * l_fund + float(w_delta) * l_delta + float(w_spec_hold) * float(spec_scale_local) * l_spec_hold

            # FIR 正则：平滑 + 频响增益约束（防止噪声放大导致某段频率崩盘）
            if isinstance(model, PolyphaseStage12Model):
                w1 = model.odd.fir.weight.view(-1)
                l_reg = torch.mean((w1[1:] - w1[:-1]) ** 2)
                l_band = _fir_gain_band_loss(w1, n_fft=1024, min_gain=fir_min_gain, max_gain=fir_max_gain)
                if getattr(model, "train_both", False):
                    w0 = model.even.fir.weight.view(-1)
                    l_reg = l_reg + torch.mean((w0[1:] - w0[:-1]) ** 2)
                    l_band = l_band + _fir_gain_band_loss(w0, n_fft=1024, min_gain=fir_min_gain, max_gain=fir_max_gain)
                # 交叉耦合 FIR：默认应很小（接近单位阵），否则容易抬噪声/改坏带内谱形
                if getattr(model, "use_xcouple", False):
                    wx = model.mix_fir.weight
                    w_x_l2 = _env_float("STAGE12_W_XCOUPLE_L2", 5e-4)
                    l_reg = l_reg + float(w_x_l2) * torch.mean(wx * wx)
                # image-canceller FIR：幅度应当“小且平滑”，避免引入额外噪声/幅频畸变
                if getattr(model, "use_img_cancel", False):
                    wi = model.img_fir.weight.view(-1)
                    w_ic_l2 = _env_float("STAGE12_W_IMG_CANCEL_L2", 2e-4)
                    w_ic_band = _env_float("STAGE12_W_IMG_CANCEL_BAND", 2e-3)
                    l_reg = l_reg + float(w_ic_l2) * torch.mean(wi * wi)
                    l_band = l_band + float(w_ic_band) * _fir_gain_band_loss(wi, n_fft=1024, min_gain=0.0, max_gain=0.35)
            else:
                w = model.fir.weight.view(-1)
                l_reg = torch.mean((w[1:] - w[:-1]) ** 2)
                l_band = _fir_gain_band_loss(w, n_fft=1024, min_gain=fir_min_gain, max_gain=fir_max_gain)

            # even-anchor（若启用两路训练）
            if isinstance(model, PolyphaseStage12Model) and getattr(model, "train_both", False):
                w_anchor = _env_float("STAGE12_W_EVEN_ANCHOR", 5e-1)
                taps0 = int(model.even.fir.weight.shape[-1])
                delta = torch.zeros_like(model.even.fir.weight.view(-1))
                delta[taps0 // 2] = 1.0
                l_anchor = (model.even.gain - 1.0) ** 2 + torch.mean(model.even.delay.delay**2) + torch.mean((model.even.fir.weight.view(-1) - delta) ** 2)
                loss_main = loss_main + float(w_anchor) * l_anchor

            return loss_main + float(w_reg) * l_reg + float(w_fir_band) * l_band

        for _ in range(int(batch)):
            use_multi = _env_bool("STAGE12_USE_MULTISINE", True)
            n_tones = _env_int("STAGE12_N_TONES", 6)
            min_sep = _env_int("STAGE12_TONE_MIN_SEP", 24)

            # 关键：fs/4 两侧“成对覆盖”训练（每次都对低侧/高侧各跑一遍），避免 4.5/5.5 互相拉扯
            if use_multi and _env_bool("STAGE12_STRATIFY_FS4", True):
                mid = int(n) // 4 if stratify_mid_k is None else int(stratify_mid_k)
                lo_max = min(int(kmax_use), mid - 8)
                hi_min = max(int(kmin_use), mid + 8)
                n_hi = max(1, int(n_tones) // 2)
                n_lo = max(1, int(n_tones) - n_hi)
                # low-side multisine
                if int(kmin_use) <= lo_max:
                    k_lo = _sample_k_set(int(kmin_use), int(lo_max), n_tones=int(n_lo), min_sep=min_sep, n=int(n), avoid_image_collision=True)
                else:
                    k_lo = _sample_k_set(int(kmin_use), int(kmax_use), n_tones=int(n_lo), min_sep=min_sep, n=int(n), avoid_image_collision=True)
                src_lo = _multisine_np(n=n, bins=k_lo, amp=float(np.random.uniform(0.55, 0.92)))
                # high-side multisine
                if hi_min <= int(kmax_use):
                    k_hi = _sample_k_set(int(hi_min), int(kmax_use), n_tones=int(n_hi), min_sep=min_sep, n=int(n), avoid_image_collision=True)
                else:
                    k_hi = _sample_k_set(int(kmin_use), int(kmax_use), n_tones=int(n_hi), min_sep=min_sep, n=int(n), avoid_image_collision=True)
                src_hi = _multisine_np(n=n, bins=k_hi, amp=float(np.random.uniform(0.55, 0.92)))

                loss_acc = loss_acc + 0.5 * (
                    _loss_for_src(src_lo, k_funds_local=k_lo, img_scale_local=float(img_scale), spec_scale_local=float(spec_scale))
                    + _loss_for_src(src_hi, k_funds_local=k_hi, img_scale_local=float(img_scale), spec_scale_local=float(spec_scale))
                )

                # 额外的“镜像成对单音”约束（可辨识且更强）：随机选一个高侧 tone，并同时训练其镜像对应的低侧 tone。
                # 这样会直接对（k, N/2-k）这一对频点建立约束，让 4GHz 之后的成对频点不再“顾此失彼”。
                if _env_bool("STAGE12_PAIR_TONE", True):
                    w_pair = _env_float("STAGE12_PAIR_TONE_W", 0.7)
                    # high-side pick (avoid Nyquist/DC guard)
                    half = int(n) // 2
                    mid = int(n) // 4
                    k_hi_min = max(int(kmin_use), mid + 16)
                    # 只在 fs/4 附近窗口内做“镜像成对单音”，直接针对 4~6.xGHz 的难点
                    margin_hz = _env_float("STAGE12_PAIR_FS4_MARGIN_HZ", 1.9e9)
                    margin_bins = max(64, int(np.ceil(float(margin_hz) * float(n) / float(fs))))
                    k_hi_max = min(int(kmax_use), mid + margin_bins)
                    if k_hi_min < k_hi_max:
                        # 用 cycle 覆盖（而不是纯随机）：保证诸如 5.5GHz 这类关键高频点会在训练中持续被“点名”约束
                        tries = 0
                        k_hi0 = int(k_hi_min)
                        while tries < 64:
                            tries += 1
                            kk = int(_next_k())
                            if kk < k_hi_min or kk > k_hi_max:
                                continue
                            k_hi0 = kk
                            break
                        k_lo0 = abs(half - k_hi0)
                        k_lo0 = max(int(kmin_use), min(int(kmax_use), int(k_lo0)))
                        amp0 = float(np.random.uniform(0.55, 0.90))
                        t_np = np.arange(int(n), dtype=np.float64)
                        src_hi0 = (np.sin(2.0 * np.pi * float(k_hi0) * t_np / float(n)) * amp0).astype(np.float64, copy=False)
                        src_lo0 = (np.sin(2.0 * np.pi * float(k_lo0) * t_np / float(n)) * amp0).astype(np.float64, copy=False)
                        loss_acc = loss_acc + float(w_pair) * 0.5 * (
                            _loss_for_src(src_hi0, k_funds_local=[int(k_hi0)], img_scale_local=float(img_scale), spec_scale_local=float(spec_scale))
                            + _loss_for_src(src_lo0, k_funds_local=[int(k_lo0)], img_scale_local=float(img_scale), spec_scale_local=float(spec_scale))
                        )
                continue

            # fallback：原来的单次 multisine / 单音路径
            if use_multi:
                k_funds = _sample_k_set(int(kmin_use), int(kmax_use), n_tones=int(n_tones), min_sep=min_sep, n=int(n), avoid_image_collision=True)
                src_np = _multisine_np(n=n, bins=k_funds, amp=float(np.random.uniform(0.55, 0.92)))
                loss_acc = loss_acc + _loss_for_src(src_np, k_funds_local=k_funds, img_scale_local=float(img_scale), spec_scale_local=float(spec_scale))
            else:
                k = _next_k()
                src = (torch.sin(2.0 * np.pi * (float(k) * fs / float(n)) * t) * float(np.random.uniform(0.4, 0.95))).detach().cpu().numpy().astype(np.float64)
                loss_acc = loss_acc + _loss_for_src(src, k_funds_local=[int(k)], img_scale_local=float(img_scale), spec_scale_local=float(spec_scale))

        return loss_acc / float(max(int(batch), 1))

    print(
        f"=== Stage1/2 (polyphase={use_polyphase} | ref={use_ref} | train_both={train_both}) taps={taps} N={n} "
        f"f=[{fmin/1e9:.2f},{fmax/1e9:.2f}]GHz | sampler={sampler} | HIGH_BIAS={high_bias:g} ==="
    )
    # Phase-0：先只训练“镜像抑制分支”（widely-linear 的 B(ω)）
    img_params: list[nn.Parameter] = []
    if isinstance(model, PolyphaseStage12Model):
        if getattr(model, "use_img_cancel", False):
            img_params += list(model.img_fir.parameters())
        if getattr(model, "use_xcouple", False):
            img_params += list(model.mix_fir.parameters())
    if len(img_params) > 0 and int(steps0) > 0:
        print("=== Stage0: Image-canceller / X-couple ===")
        s0_img_scale = _env_float("STAGE12_S0_IMG_SCALE", 2.6)
        s0_spec_scale = _env_float("STAGE12_S0_SPEC_SCALE", 0.25)
        opt0 = optim.Adam(img_params, lr=float(lr_img), betas=(0.5, 0.9))
        for step in range(int(steps0)):
            opt0.zero_grad()
            loss = _make_batch_loss(img_scale=float(s0_img_scale), spec_scale=float(s0_spec_scale), stratify_mid_k=(int(n) // 4))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(img_params, max_norm=1.0)
            opt0.step()
            if step % 200 == 0:
                print(f"Stage12-S0 step {step:4d}/{steps0} | loss={loss.item():.3e}")
        # 关键：Stage0 解的是 B(ω)。后续训练 A(ω) 时最好不要再动 B(ω)，否则容易“把镜像又带回来”。
        if _env_bool("STAGE12_FREEZE_IMG_AFTER_S0", True) and isinstance(model, PolyphaseStage12Model):
            if getattr(model, "use_img_cancel", False):
                for p in model.img_fir.parameters():
                    p.requires_grad_(False)
            if getattr(model, "use_xcouple", False):
                for p in model.mix_fir.parameters():
                    p.requires_grad_(False)
    print("=== Stage1: Delay & Gain ===")
    # Phase-1：只调 gain/delay（不动 FIR），让对齐先稳定
    if isinstance(model, PolyphaseStage12Model):
        dg_params = []
        fir_params = []
        if getattr(model, "train_both", False):
            dg_params += list(model.even.delay.parameters()) + [model.even.gain]
            fir_params += list(model.even.fir.parameters())
        dg_params += list(model.odd.delay.parameters()) + [model.odd.gain]
        fir_params += list(model.odd.fir.parameters())
    else:
        dg_params = list(model.delay.parameters()) + [model.gain]
        fir_params = list(model.fir.parameters())

    opt1 = optim.Adam(
        [
            {"params": dg_params, "lr": float(lr_dg)},
            {"params": fir_params, "lr": 0.0},
        ],
        betas=(0.5, 0.9),
    )
    for step in range(int(steps1)):
        opt1.zero_grad()
        loss = _make_batch_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt1.step()
        if step % 200 == 0:
            print(f"Stage12-S1 step {step:4d}/{steps1} | loss={loss.item():.3e}")

    print("=== Stage2: FIR ===")
    # Phase-2：冻结 gain/delay，只训练 FIR（均衡带宽/幅相差）
    opt2 = optim.Adam(
        [
            {"params": dg_params, "lr": 0.0},
            {"params": fir_params, "lr": float(lr_fir)},
        ],
        betas=(0.5, 0.9),
    )
    for step in range(int(steps2)):
        opt2.zero_grad()
        loss = _make_batch_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt2.step()
        if step % 200 == 0:
            print(f"Stage12-S2 step {step:4d}/{steps2} | loss={loss.item():.3e}")
    # High-band fine-tune：>4GHz 段主瓶颈通常是 2-way 镜像 spur（fs/2-f）在 -20dBc 量级，
    # 仅靠全带宽平均训练往往压不下去。这里加一个高频段微调阶段：
    # - 只在高频段采样（默认 4.0~6.6GHz）
    # - 适度提高镜像项权重（img_scale）
    # - 仍保留 spec-hold/fund-hold/delta 等约束，避免牺牲低频与抬噪声底
    hb_steps = _env_int("STAGE12_HIGHBAND_STEPS", 0)
    if hb_steps > 0:
        hb_fmin = _env_float("STAGE12_HIGHBAND_FMIN_HZ", 4.0e9)
        hb_fmax = _env_float("STAGE12_HIGHBAND_FMAX_HZ", float(fmax))
        hb_kmin = max(int(np.ceil(hb_fmin * n / fs)), k_min)
        hb_kmax = min(int(np.floor(hb_fmax * n / fs)), k_max)
        hb_img_scale = _env_float("STAGE12_HIGHBAND_IMG_SCALE", 2.0)
        mid_k = int(n) // 4  # fs/4 对应的 bin（2-way 的“镜像对称轴”）
        print(f"=== Stage2b: High-band fine-tune f=[{hb_fmin/1e9:.2f},{hb_fmax/1e9:.2f}]GHz steps={hb_steps} img_scale={hb_img_scale:g} ===")
        for step in range(int(hb_steps)):
            opt2.zero_grad()
            loss = _make_batch_loss(kmin_override=hb_kmin, kmax_override=hb_kmax, img_scale=float(hb_img_scale), stratify_mid_k=mid_k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt2.step()
            if step % 200 == 0:
                print(f"Stage12-S2b step {step:4d}/{hb_steps} | loss={loss.item():.3e}")
    return model, float(scale)


def train_post_qe(sim: TIADCSimulator, *, stage12: nn.Module, device: str, p0: dict, p1: dict, pref: dict) -> PhysicalNonlinearityLayer:
    order = _env_int("POST_QE_ORDER", 3)
    steps = _env_int("POST_QE_STEPS", 450)
    lr = _env_float("POST_QE_LR", 5e-4)
    batch = _env_int("POST_QE_BATCH", 6)
    n = _env_int("POST_QE_TONE_N", 8192)
    # 默认覆盖到评估频段，避免“训练域外频点掉点”
    fmin = _env_float("POST_QE_FMIN_HZ", 0.1e9)
    fmax = _env_float("POST_QE_FMAX_HZ", 6.6e9)
    guard = _env_int("POST_QE_GUARD", 3)
    # Post-QE 的“现实可行训练目标”：
    # - 实测场景通常拿不到理想 ref 波形；且抖动/量化噪声也不可能被静态非线性层拟合掉
    # - 因此默认采用 self-supervised：以“HD/IMD 能量比”作为主目标，并加上基波保持 + 小改动约束
    mode = os.getenv("POST_QE_MODE", "self").strip().lower()  # self | ref
    use_twotone = _env_bool("POST_QE_USE_TWOTONE", True)
    p_twotone = _env_float("POST_QE_P_TWOTONE", 0.65)
    min_sep = _env_int("POST_QE_TONE_MIN_SEP", 48)
    w_hd = _env_float("POST_QE_W_HD", 2200.0)
    w_imd = _env_float("POST_QE_W_IMD", 2200.0)
    w_fund = _env_float("POST_QE_W_FUND", 6.0)
    w_delta = _env_float("POST_QE_W_DELTA", 1.0)
    # ref 模式下保留旧目标（用于“仿真 sanity check”），但默认不推荐当作实际标定手段
    w_time = _env_float("POST_QE_W_TIME", 15.0)
    ridge = _env_float("POST_QE_RIDGE", 5e-5)

    corr_lpf_hz = _env_float("POST_QE_CORR_LPF_HZ", 7.0e9)
    corr_lpf_taps = _env_int("POST_QE_CORR_LPF_TAPS", 33)
    post_qe = PhysicalNonlinearityLayer(order, fs=float(sim.fs), corr_lpf_hz=float(corr_lpf_hz), corr_lpf_taps=int(corr_lpf_taps)).to(device)


    opt = optim.Adam(post_qe.parameters(), lr=float(lr), betas=(0.5, 0.9))
    fs = float(sim.fs)
    t = torch.arange(n, device=device, dtype=torch.float32) / fs
    k_min = max(int(np.ceil(fmin * n / fs)), 8)
    k_max = min(int(np.floor(fmax * n / fs)), n // 2 - 8)

    print(f">> Post-QE: mode={mode} twotone={use_twotone} order={order} steps={steps} lr={lr:g} batch={batch} N={n}")
    for step in range(int(steps)):
        loss_acc = 0.0
        for _ in range(int(batch)):
            # 训练激励：
            # - ref 模式：沿用单音（与旧版本一致）

            
            # - self 模式：默认用 two-tone（更能约束 IMD3，避免“只对单音有效”）
            do_twotone = (mode != "ref") and bool(use_twotone) and (float(np.random.rand()) < float(p_twotone))
            if mode == "ref" or (not do_twotone):
                k1 = int(np.random.randint(k_min, k_max + 1))
                k2 = None
                bins = [int(k1)]
            else:
                # 选两音时也要避免与 2-way 镜像关系撞车（可辨识性）
                bins = _sample_k_set(int(k_min), int(k_max), n_tones=2, min_sep=int(min_sep), n=int(n), avoid_image_collision=True)
                k1 = int(bins[0])
                k2 = int(bins[1]) if len(bins) > 1 else int(min(k1 + int(min_sep), k_max))
                bins = [int(k1), int(k2)]
            amp = float(np.random.uniform(0.45, 0.92))
            src_np = _multisine_np(n=int(n), bins=bins, amp=amp)
            c0, c1, y_in = sim.capture_two_way(src_np, p0=p0, p1=p1, n_bits=ADC_NBITS)
            # ref：两路“理想一致”的采样（同样只在相位点采样）
            yr = None
            if mode == "ref":
                _, _, yr = sim.capture_two_way(src_np, p0=pref, p1=pref, n_bits=ADC_NBITS)
            s12 = max(float(np.max(np.abs(y_in))), 1e-12)
            with torch.no_grad():
                # Stage1/2：多相结构下先校正两路，再交织
                if isinstance(stage12, PolyphaseStage12Model):
                    g0 = torch.FloatTensor((c0[0::2] / s12)).view(1, 1, -1).to(device)
                    g1 = torch.FloatTensor((c1[1::2] / s12)).view(1, 1, -1).to(device)
                    xin = stage12(g0, g1).detach().cpu().numpy().flatten() * s12
                else:
                    x1 = torch.FloatTensor(_stage12_prepare_input(c1) / s12).view(1, 1, -1).to(device)
                    y1c = stage12(x1).detach().cpu().numpy().flatten() * s12
                    xin = interleave(c0, y1c)
            if yr is not None:
                yt = yr[: len(xin)]
                s0 = max(float(np.max(np.abs(xin))), float(np.max(np.abs(yt))), 1e-12)
            else:
                s0 = max(float(np.max(np.abs(xin))), 1e-12)
            x = torch.FloatTensor(xin / s0).view(1, 1, -1).to(device)
            y = None
            if yr is not None:
                y = torch.FloatTensor(yt / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            if y is not None:
                y = torch.clamp(y, -1.0, 1.0)
            yhat = post_qe(x)
            win = torch.hann_window(int(n), device=device, dtype=x.dtype).view(1, 1, -1)
            reg = sum(torch.mean(p**2) for p in post_qe.parameters())
            ld = torch.mean((yhat - x) ** 2)
            if mode == "ref" and y is not None:
                # 旧目标：拟合 ref + 压 residual-harm
                # 注意：此目标对“真实场景”不一定可行（抖动/噪声不可逆），仅用于仿真 sanity check。
                lh = residual_harmonic_loss(yhat - y, y, int(k1), int(guard))
                lt = torch.mean((yhat - y) ** 2)
                loss_acc = loss_acc + (float(w_time) * lt + 1800.0 * lh + float(w_delta) * ld + float(ridge) * reg)
            else:
                # self-supervised：压输出 HD/IMD（以基波能量归一化），并保持基波不被“压没”
                Y = torch.fft.rfft((yhat - torch.mean(yhat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
                P = (Y.real**2 + Y.imag**2)
                Xf = torch.fft.rfft((x - torch.mean(x, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
                Px = (Xf.real**2 + Xf.imag**2)
                fund_bins = [int(k1)] if (k2 is None) else [int(k1), int(k2)]
                harm_bins: list[int] = []
                imd_bins: list[int] = []
                for kk in fund_bins:
                    harm_bins += _harm_bins_folded(int(kk), int(n), max_h=5)
                if k2 is not None:
                    imd_bins = _imd3_bins_folded(int(k1), int(k2), int(n))
                harm_bins = sorted(set(harm_bins) - set(fund_bins))
                imd_bins = sorted(set(imd_bins) - set(fund_bins))
                p_fund = _band_energy_sum(P, fund_bins, guard=int(guard)) + 1e-12
                p_hd = _band_energy_sum(P, harm_bins, guard=int(guard)) + 1e-12
                p_imd = _band_energy_sum(P, imd_bins, guard=int(guard)) + 1e-12
                p_fund_in = _band_energy_sum(Px, fund_bins, guard=int(guard)) + 1e-12
                p_hd_in = _band_energy_sum(Px, harm_bins, guard=int(guard)) + 1e-12
                p_imd_in = _band_energy_sum(Px, imd_bins, guard=int(guard)) + 1e-12
                r_hd = p_hd / p_fund
                r_hd_in = p_hd_in / p_fund_in
                r_imd = p_imd / p_fund
                r_imd_in = p_imd_in / p_fund_in
                beta = _env_float("POST_QE_HINGE_BETA", 0.2)
                l_hd = torch.mean(torch.relu(r_hd - r_hd_in) + float(beta) * r_hd)
                if len(imd_bins):
                    l_imd = torch.mean(torch.relu(r_imd - r_imd_in) + float(beta) * r_imd)
                else:
                    l_imd = torch.zeros_like(l_hd)
                l_fund = _funds_hold_loss(yhat, x, fund_bins=fund_bins, guard=int(guard))
                loss_acc = loss_acc + (float(w_hd) * l_hd + float(w_imd) * l_imd + float(w_fund) * l_fund + float(w_delta) * ld + float(ridge) * reg)
        loss_acc = loss_acc / float(max(int(batch), 1))
        opt.zero_grad()
        loss_acc.backward()
        torch.nn.utils.clip_grad_norm_(post_qe.parameters(), max_norm=1.0)
        opt.step()
        if step % 150 == 0:
            print(f"Post-QE step {step:4d}/{steps} | loss={loss_acc.item():.3e}")
    return post_qe


def train_post_nl(sim: TIADCSimulator, *, stage12: nn.Module, post_qe: Optional[PhysicalNonlinearityLayer], device: str, p0: dict, p1: dict, pref: dict) -> DifferentiableMemoryPolynomial:
    taps = _env_int("STAGE3_TAPS", 21)
    order = _env_int("STAGE3_ORDER", 3)
    steps = _env_int("STAGE3_STEPS", 1200)
    lr = _env_float("STAGE3_LR", 4e-4)
    batch = _env_int("STAGE3_BATCH", 9)
    n = _env_int("STAGE3_TONE_N", 16384)
    fmin = _env_float("STAGE3_FMIN_HZ", 0.1e9)
    fmax = _env_float("STAGE3_FMAX_HZ", 6.6e9)
    guard = _env_int("STAGE3_HARM_GUARD", 3)
    mode = os.getenv("STAGE3_MODE", "self").strip().lower()  # self | ref
    use_twotone = _env_bool("STAGE3_USE_TWOTONE", True)
    p_twotone = _env_float("STAGE3_P_TWOTONE", 0.65)
    min_sep = _env_int("STAGE3_TONE_MIN_SEP", 48)
    w_hd = _env_float("STAGE3_W_HD", 2600.0)
    w_imd = _env_float("STAGE3_W_IMD", 2600.0)
    w_fund = _env_float("STAGE3_W_FUND", 6.0)
    w_delta = _env_float("STAGE3_W_DELTA", 0.6)
    # ref 模式下保留旧目标
    w_time = _env_float("STAGE3_W_TIME", 60.0)
    w_harm = _env_float("STAGE3_W_HARM", 2400.0)
    w_reg = _env_float("STAGE3_W_REG", 5e-5)

    corr_lpf_hz = _env_float("POST_NL_CORR_LPF_HZ", 7.0e9)
    corr_lpf_taps = _env_int("POST_NL_CORR_LPF_TAPS", 33)
    post_nl = DifferentiableMemoryPolynomial(taps, order, fs=float(sim.fs), corr_lpf_hz=float(corr_lpf_hz), corr_lpf_taps=int(corr_lpf_taps)).to(device)
    opt = optim.Adam(post_nl.parameters(), lr=float(lr), betas=(0.5, 0.9))
    fs = float(sim.fs)
    t = torch.arange(n, device=device, dtype=torch.float32) / fs
    k_min = max(int(np.ceil(fmin * n / fs)), 8)
    k_max = min(int(np.floor(fmax * n / fs)), n // 2 - 8)

    print(f">> Post-NL: mode={mode} twotone={use_twotone} taps={taps} order={order} steps={steps} lr={lr:g} batch={batch} N={n}")
    for step in range(int(steps)):
        loss_acc = 0.0
        for _ in range(int(batch)):
            do_twotone = (mode != "ref") and bool(use_twotone) and (float(np.random.rand()) < float(p_twotone))
            if mode == "ref" or (not do_twotone):
                k1 = int(np.random.randint(k_min, k_max + 1))
                k2 = None
                bins = [int(k1)]
            else:
                bins = _sample_k_set(int(k_min), int(k_max), n_tones=2, min_sep=int(min_sep), n=int(n), avoid_image_collision=True)
                k1 = int(bins[0])
                k2 = int(bins[1]) if len(bins) > 1 else int(min(k1 + int(min_sep), k_max))
                bins = [int(k1), int(k2)]
            amp = float(np.random.uniform(0.45, 0.92))
            src_np = _multisine_np(n=int(n), bins=bins, amp=amp)
            c0, c1, y_in = sim.capture_two_way(src_np, p0=p0, p1=p1, n_bits=ADC_NBITS)
            yr = None
            if mode == "ref":
                _, _, yr = sim.capture_two_way(src_np, p0=pref, p1=pref, n_bits=ADC_NBITS)
            # Stage12 前的数值归一化基准：
            # - self：只能用“实际采样得到的 y_in”
            # - ref：用 y_in 与 yr 的最大值，避免尺度不一致
            s12 = max(float(np.max(np.abs(y_in))), 1e-12)
            if yr is not None:
                s12 = max(s12, float(np.max(np.abs(yr))), 1e-12)
            with torch.no_grad():
                if isinstance(stage12, PolyphaseStage12Model):
                    g0 = torch.FloatTensor((c0[0::2] / s12)).view(1, 1, -1).to(device)
                    g1 = torch.FloatTensor((c1[1::2] / s12)).view(1, 1, -1).to(device)
                    xin = stage12(g0, g1).detach().cpu().numpy().flatten() * s12
                else:
                    x1 = torch.FloatTensor(_stage12_prepare_input(c1) / s12).view(1, 1, -1).to(device)
                    y1c = stage12(x1).detach().cpu().numpy().flatten() * s12
                    xin = interleave(c0, y1c)
            # 注意：self 模式下没有理想 ref，因此以 xin 自身做归一化基准（避免把“不可逆噪声”当目标去拟合）
            if yr is not None:
                yt = yr[: len(xin)]
                s0 = max(float(np.max(np.abs(xin))), float(np.max(np.abs(yt))), 1e-12)
            else:
                yt = None
                s0 = max(float(np.max(np.abs(xin))), 1e-12)
            x = torch.FloatTensor(xin / s0).view(1, 1, -1).to(device)
            x = torch.clamp(x, -1.0, 1.0)
            y = None
            if yt is not None:
                y = torch.FloatTensor(yt / s0).view(1, 1, -1).to(device)
                y = torch.clamp(y, -1.0, 1.0)
            if post_qe is not None:
                with torch.no_grad():
                    x = post_qe(x)
            yhat = post_nl(x)
            ld = torch.mean((yhat - x) ** 2)
            reg = sum(torch.mean(p**2) for p in post_nl.parameters())
            if mode == "ref" and y is not None:
                lh = residual_harmonic_loss(yhat - y, y, int(k1), int(guard))
                win = torch.hann_window(int(n), device=device, dtype=x.dtype).view(1, 1, -1)
                Yh = torch.fft.rfft(yhat * win, dim=-1, norm="ortho")
                Yr = torch.fft.rfft(y * win, dim=-1, norm="ortho")
                lf = torch.mean((torch.log10(torch.abs(Yh[..., int(k1)]) + 1e-12) - torch.log10(torch.abs(Yr[..., int(k1)]) + 1e-12)) ** 2)
                lt = torch.mean((yhat - y) ** 2)
                loss_acc = loss_acc + (float(w_time) * lt + float(w_harm) * lh + float(w_fund) * lf + float(w_delta) * ld + float(w_reg) * reg)
            else:
                win = torch.hann_window(int(n), device=device, dtype=x.dtype).view(1, 1, -1)
                Y = torch.fft.rfft((yhat - torch.mean(yhat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
                P = (Y.real**2 + Y.imag**2)
                Xf = torch.fft.rfft((x - torch.mean(x, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
                Px = (Xf.real**2 + Xf.imag**2)
                fund_bins = [int(k1)] if (k2 is None) else [int(k1), int(k2)]
                harm_bins: list[int] = []
                imd_bins: list[int] = []
                for kk in fund_bins:
                    harm_bins += _harm_bins_folded(int(kk), int(n), max_h=5)
                if k2 is not None:
                    imd_bins = _imd3_bins_folded(int(k1), int(k2), int(n))
                harm_bins = sorted(set(harm_bins) - set(fund_bins))
                imd_bins = sorted(set(imd_bins) - set(fund_bins))
                p_fund = _band_energy_sum(P, fund_bins, guard=int(guard)) + 1e-12
                p_hd = _band_energy_sum(P, harm_bins, guard=int(guard)) + 1e-12
                p_imd = _band_energy_sum(P, imd_bins, guard=int(guard)) + 1e-12
                p_fund_in = _band_energy_sum(Px, fund_bins, guard=int(guard)) + 1e-12
                p_hd_in = _band_energy_sum(Px, harm_bins, guard=int(guard)) + 1e-12
                p_imd_in = _band_energy_sum(Px, imd_bins, guard=int(guard)) + 1e-12
                r_hd = p_hd / p_fund
                r_hd_in = p_hd_in / p_fund_in
                r_imd = p_imd / p_fund
                r_imd_in = p_imd_in / p_fund_in
                beta = _env_float("STAGE3_HINGE_BETA", 0.2)
                l_hd = torch.mean(torch.relu(r_hd - r_hd_in) + float(beta) * r_hd)
                if len(imd_bins):
                    l_imd = torch.mean(torch.relu(r_imd - r_imd_in) + float(beta) * r_imd)
                else:
                    l_imd = torch.zeros_like(l_hd)
                l_fund = _funds_hold_loss(yhat, x, fund_bins=fund_bins, guard=int(guard))
                loss_acc = loss_acc + (float(w_hd) * l_hd + float(w_imd) * l_imd + float(w_fund) * l_fund + float(w_delta) * ld + float(w_reg) * reg)
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


def _spectral_breakdown_db(sig: np.ndarray, *, fs: float, fin: float, guard: int = 5) -> dict:
    """
    用于“先定位瓶颈”的频域分解报告（dBc）：
    - fund：基波功率
    - img：2-way 交织镜像（fs/2 - fin）附近能量
    - harm：2~5 次谐波能量（折叠到 0..fs/2）
    - noise：剩余能量（除 DC、fund、img、harm 外）
    """
    x = np.asarray(sig, dtype=np.float64)
    n = len(x)
    if n < 64:
        return {"fund_db": 0.0, "img_db": 0.0, "harm_db": 0.0, "noise_db": 0.0}
    win = np.blackman(n)
    cg = float(np.mean(win))
    S = np.fft.rfft((x - np.mean(x)) * win)
    mag = np.abs(S) / (n / 2 * cg + 1e-20)
    p = mag**2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    # coherent-ish pick: local max around fin
    idx = int(np.argmin(np.abs(freqs - float(fin))))
    span = int(guard)
    s0 = max(0, idx - span)
    e0 = min(len(p), idx + span + 1)
    idxp = s0 + int(np.argmax(p[s0:e0]))
    p_fund = float(np.sum(p[max(0, idxp - span) : min(len(p), idxp + span + 1)])) + 1e-30

    # image: fs/2 - fin (bin: N/2 - k) in rfft
    k_fund = int(idxp)
    k_img = abs((n // 2) - k_fund)
    k_img = max(1, min(k_img, len(p) - 2))
    p_img = float(np.sum(p[max(0, k_img - span) : min(len(p), k_img + span + 1)]))

    # harmonics (2..5), folded
    p_harm = 0.0
    for h in range(2, 6):
        hf = (float(fin) * h) % float(fs)
        if hf > float(fs) / 2:
            hf = float(fs) - hf
        if hf < 1e6 or hf > float(fs) / 2 - 1e6:
            continue
        k = int(np.argmin(np.abs(freqs - hf)))
        ss = max(0, k - span)
        ee = min(len(p), k + span + 1)
        kk = ss + int(np.argmax(p[ss:ee]))
        p_harm += float(np.sum(p[max(0, kk - span) : min(len(p), kk + span + 1)]))

    # noise floor: everything else excluding DC bins, fund/img/harm windows
    mask = np.ones_like(p, dtype=bool)
    mask[:5] = False
    mask[max(0, idxp - span) : min(len(mask), idxp + span + 1)] = False
    mask[max(0, k_img - span) : min(len(mask), k_img + span + 1)] = False
    for h in range(2, 6):
        hf = (float(fin) * h) % float(fs)
        if hf > float(fs) / 2:
            hf = float(fs) - hf
        if hf < 1e6 or hf > float(fs) / 2 - 1e6:
            continue
        k = int(np.argmin(np.abs(freqs - hf)))
        ss = max(0, k - span)
        ee = min(len(p), k + span + 1)
        kk = ss + int(np.argmax(p[ss:ee]))
        mask[max(0, kk - span) : min(len(mask), kk + span + 1)] = False
    p_noise = float(np.sum(p[mask])) + 1e-30

    to_dbc = lambda pc: 10.0 * np.log10((float(pc) + 1e-30) / p_fund)
    return {
        "fund_db": 0.0,
        "img_db": float(to_dbc(p_img)),
        "harm_db": float(to_dbc(p_harm)),
        "noise_db": float(to_dbc(p_noise)),
        "k_fund": int(k_fund),
        "k_img": int(k_img),
    }

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
    stage12: nn.Module,
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
            re = post_qe.raw_even.detach().cpu().numpy()
            ro = post_qe.raw_odd.detach().cpu().numpy()
            print(
                f"\n[effect] Post-QE | alpha={float(post_qe.alpha):g} | "
                f"raw_even_abs_mean={float(np.mean(np.abs(re))):.2e} raw_even_abs_max={float(np.max(np.abs(re))):.2e} | "
                f"raw_odd_abs_mean={float(np.mean(np.abs(ro))):.2e} raw_odd_abs_max={float(np.max(np.abs(ro))):.2e}"
            )
        if post_nl is not None:
            re = post_nl.raw_even.detach().cpu().numpy()
            ro = post_nl.raw_odd.detach().cpu().numpy()
            print(
                f"[effect] Post-NL | alpha={float(post_nl.alpha):g} | "
                f"raw_even_abs_mean={float(np.mean(np.abs(re))):.2e} raw_even_abs_max={float(np.max(np.abs(re))):.2e} | "
                f"raw_odd_abs_mean={float(np.mean(np.abs(ro))):.2e} raw_odd_abs_max={float(np.max(np.abs(ro))):.2e}"
            )
    except Exception:
        pass

    fs = float(sim.fs)
    n = _env_int("EFFECT_N", 16384)
    amps = [float(x) for x in os.getenv("EFFECT_AMPS", "0.3,0.6,0.9").split(",")]
    freqs_ghz = [float(x) for x in os.getenv("EFFECT_FREQS_GHZ", "0.5,2.5,5.5").split(",")]

    print("\n[effect] A/B 消融（同一输入，同一 Stage1/2）")
    for fghz in freqs_ghz:
        # 相干采样：把目标频点吸附到 FFT bin，避免 spur/SFDR 指标被谱泄漏污染
        _, fin = _coherent_bin(fs, n, fghz * 1e9)
        for amp in amps:
            src = sim.tone(fin, n, amp)
            c0, c1, _ = sim.capture_two_way(src, p0=p0, p1=p1, n_bits=ADC_NBITS)
            with torch.no_grad():
                if isinstance(stage12, PolyphaseStage12Model):
                    g0 = torch.FloatTensor((c0[0::2] / float(scale))).view(1, 1, -1).to(device)
                    g1 = torch.FloatTensor((c1[1::2] / float(scale))).view(1, 1, -1).to(device)
                    post_lin = stage12(g0, g1).cpu().numpy().flatten() * float(scale)
                else:
                    x1 = torch.FloatTensor(_stage12_prepare_input(c1) / float(scale)).view(1, 1, -1).to(device)
                    y1c = stage12(x1).cpu().numpy().flatten() * float(scale)
                    post_lin = interleave(c0, y1c)
            margin = 500
            c0 = c0[margin:-margin]
            post = post_lin[margin:-margin]

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


def evaluate(sim: TIADCSimulator, *, stage12: nn.Module, post_qe: Optional[PhysicalNonlinearityLayer], post_nl: Optional[DifferentiableMemoryPolynomial], scale: float, device: str, p0: dict, p1: dict) -> tuple[np.ndarray, dict]:
    # 评估扫频也必须相干采样，否则高频 spur 会被泄漏摊平，出现“4G后没效果”的假象
    fs = float(sim.fs)
    n_eval = 16384
    f_targets = np.arange(0.1e9, 6.6e9, 0.2e9)
    freqs = []
    for f in f_targets:
        _, fc = _coherent_bin(fs, n_eval, float(f))
        freqs.append(fc)
    freqs = np.array(freqs, dtype=np.float64)
    m = {k: [] for k in ["sinad_pre", "sinad_post", "sinad_qe", "sinad_nl",
                         "enob_pre", "enob_post", "enob_qe", "enob_nl",
                         "thd_pre", "thd_post", "thd_qe", "thd_nl",
                         "sfdr_pre", "sfdr_post", "sfdr_qe", "sfdr_nl"]}
    for f in freqs:
        src = sim.tone(float(f), n_eval, 0.9)
        c0, c1, _ = sim.capture_two_way(src, p0=p0, p1=p1, n_bits=ADC_NBITS)
        with torch.no_grad():
            if isinstance(stage12, PolyphaseStage12Model):
                g0 = torch.FloatTensor((c0[0::2] / float(scale))).view(1, 1, -1).to(device)
                g1 = torch.FloatTensor((c1[1::2] / float(scale))).view(1, 1, -1).to(device)
                post_lin = stage12(g0, g1).cpu().numpy().flatten() * float(scale)
            else:
                x1 = torch.FloatTensor(_stage12_prepare_input(c1) / float(scale)).view(1, 1, -1).to(device)
                y1c = stage12(x1).cpu().numpy().flatten() * float(scale)
                post_lin = interleave(c0, y1c)
        margin = 500
        c0 = c0[margin:-margin]
        c1r = c1[margin:-margin]
        pre = interleave(c0, c1r)
        post = post_lin[margin:-margin]
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
    # 颜色调整：Pre-Calib 与 Post-QE 互换，便于肉眼区分
    plt.plot(fghz, m["sinad_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["sinad_post"], "g-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sinad_qe"], "r-o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["sinad_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("SINAD Improvement"); plt.ylabel("dB"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["enob_post"], "m-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["enob_qe"], "r-o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["enob_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("ENOB Improvement"); plt.ylabel("Bits"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["thd_post"], "b-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["thd_qe"], "r-o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["thd_nl"], "k--o", alpha=0.85, label="Post-NL")
    plt.title("THD Comparison"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["sfdr_post"], "c-o", linewidth=2, label="Post-Linear")
    plt.plot(fghz, m["sfdr_qe"], "r-o", linewidth=2, label="Post-QE")
    plt.plot(fghz, m["sfdr_nl"], "b--o", alpha=0.85, label="Post-NL")
    plt.title("SFDR Improvement"); plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    savefig("metrics_vs_freq")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== v0126_v1_slim ===")
    print(f"device={device} | ENABLE_POST_QE={_env_bool('ENABLE_POST_QE', True)} | ENABLE_STAGE3_NONLINEAR={_env_bool('ENABLE_STAGE3_NONLINEAR', True)}")

    sim = TIADCSimulator(fs=20e9)
    # 更贴近“两片 ADC 实际采样到的数据”的参数：
    # - 每路只在各自相位采样点产生数据（由 capture_two_way() 决定）
    # - 增益/带宽/固定时延失配 + 直流偏置 + 通道抖动
    # - 轻微静态非线性（用于后级 Post-QE/Post-NL 的价值验证）
    p0 = {
        "cutoff_freq": 8.0e9,
        "delay_samples": 0.00,
        "gain": 1.00,
        "offset": 0.0e-3,
        "jitter_std": ADC_JITTER_STD,
        "hd2": 1.0e-3,
        "hd3": 0.5e-3,
        "snr_target": None,
    }
    p1 = {
        "cutoff_freq": 7.75e9,
        "delay_samples": 0.22,
        "gain": 0.985,
        "offset": 0.8e-3,
        "jitter_std": ADC_JITTER_STD,
        "hd2": 3.5e-3,
        "hd3": 2.0e-3,
        "snr_target": None,
    }
    # 参考：两路一致（同带宽、零偏置、零时延、无线性/非线性失配）
    pref = {
        "cutoff_freq": p0["cutoff_freq"],
        "delay_samples": 0.0,
        "gain": 1.0,
        "offset": 0.0,
        "jitter_std": 0.0,
        "hd2": 0.0,
        "hd3": 0.0,
        "snr_target": None,
    }

    stage12, scale = train_stage12(sim, device=device, p0=p0, p1=p1, pref=pref)

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

    # 频域瓶颈定位：分解 fund / image / harmonics / noise floor（dBc）
    if _env_bool("DIAG_BOTTLENECK", True):
        print("\n=== 频域瓶颈定位（dBc，相对基波；越低越好）===")
        for g in [4.5, 5.5, 6.5]:
            fin = float(g) * 1e9
            k, fin_c = _coherent_bin(float(sim.fs), 16384, float(fin))
            src = np.sin(2.0 * np.pi * fin_c * (np.arange(16384, dtype=np.float64) / float(sim.fs))) * 0.85
            c0, c1, pre = sim.capture_two_way(src, p0=p0, p1=p1, n_bits=ADC_NBITS)
            # Stage12
            with torch.no_grad():
                if isinstance(stage12, PolyphaseStage12Model):
                    s = max(float(np.max(np.abs(pre))), 1e-12)
                    g0 = torch.tensor((c0[0::2] / s), dtype=torch.float32, device=device).view(1, 1, -1)
                    g1 = torch.tensor((c1[1::2] / s), dtype=torch.float32, device=device).view(1, 1, -1)
                    post = (stage12(g0, g1).detach().cpu().numpy().flatten() * s).astype(np.float64, copy=False)
                else:
                    post = pre.copy()
            qe = post.copy()
            if post_qe is not None:
                with torch.no_grad():
                    s = max(float(np.max(np.abs(post))), 1e-12)
                    xt = torch.tensor(post / s, dtype=torch.float32, device=device).view(1, 1, -1)
                    qe = (post_qe(xt).detach().cpu().numpy().flatten() * s).astype(np.float64, copy=False)
            nl = qe.copy()
            if post_nl is not None:
                with torch.no_grad():
                    s = max(float(np.max(np.abs(qe))), 1e-12)
                    xt = torch.tensor(qe / s, dtype=torch.float32, device=device).view(1, 1, -1)
                    nl = (post_nl(xt).detach().cpu().numpy().flatten() * s).astype(np.float64, copy=False)

            b_pre = _spectral_breakdown_db(pre, fs=float(sim.fs), fin=fin_c, guard=5)
            b_lin = _spectral_breakdown_db(post, fs=float(sim.fs), fin=fin_c, guard=5)
            b_qe = _spectral_breakdown_db(qe, fs=float(sim.fs), fin=fin_c, guard=5)
            b_nl = _spectral_breakdown_db(nl, fs=float(sim.fs), fin=fin_c, guard=5)
            print(
                f"f={fin_c/1e9:>4.2f}GHz | "
                f"img {b_pre['img_db']:>6.1f}->{b_lin['img_db']:>6.1f}->{b_qe['img_db']:>6.1f}->{b_nl['img_db']:>6.1f} | "
                f"harm {b_pre['harm_db']:>6.1f}->{b_lin['harm_db']:>6.1f}->{b_qe['harm_db']:>6.1f}->{b_nl['harm_db']:>6.1f} | "
                f"noise {b_pre['noise_db']:>6.1f}->{b_lin['noise_db']:>6.1f}->{b_qe['noise_db']:>6.1f}->{b_nl['noise_db']:>6.1f}  (k_img={b_pre['k_img']})"
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
