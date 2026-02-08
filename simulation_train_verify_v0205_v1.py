"""
v0205_tf_hybrid
基于“可微时频混合处理（Differentiable Time-Frequency Hybrid Processing）”的 TIADC 盲校准仿真：
- Analysis Filter Bank（STFT/DFT 滤波器组）
- Sub-band Complex Calibration（每子带×通道复权重）
- Synthesis Filter Bank（ISTFT 重构）

训练：频谱泄漏 + 图像杂散抑制 + 能量保持（盲校准，无理想参考）。
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


ADC_JITTER_STD = float(os.getenv("ADC_JITTER_STD", "20e-15"))
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


def _next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _hermitianize_rfft_bins(W_rfft: np.ndarray, n_fft: int) -> np.ndarray:
    """
    将 rfft（0..N/2）频响补齐成完整 N 点 FFT 频响，满足 Hermitian 对称，
    以便 ifft 得到（近似）实系数时域响应。
    W_rfft: [..., N/2+1] complex
    return: [..., N] complex
    """
    n_fft = int(n_fft)
    if n_fft < 2:
        raise ValueError("n_fft must be >=2")
    if W_rfft.shape[-1] != (n_fft // 2 + 1):
        raise ValueError("W_rfft last dim must be n_fft//2+1")
    # [0 .. N/2] + conj([N/2-1 .. 1])
    tail = np.conj(W_rfft[..., -2:0:-1])
    return np.concatenate([W_rfft, tail], axis=-1)


def solve_stage12_mimo_wiener_fir(
    sim: "TIADCSimulator",
    *,
    p0: dict,
    p1: dict,
    pref: dict,
    n_full: int,
    n_records: int,
    fir_taps: int,
    guard_drop: int = 256,
    reg_mu: float = 1e-6,
    wls_alpha: float = 0.0,
    wls_power: float = 2.0,
) -> np.ndarray:
    """
    2×2 MIMO（半速域）维纳/加权最小二乘闭式解：
    - 令 x0[n]=ideal even samples, x1[n]=ideal odd samples（由 pref 产生）
    - 令 g0[n], g1[n] 为实际两路采样子序列（由 p0/p1 产生）
    - 频域上求 W(ω)=S_xg(ω)·(S_gg(ω)+μI)^{-1}，得到最小均方误差线性估计器 x̂=Wg
    - 再将 W(ω) IFFT 截断为长度 fir_taps 的 2×2 FIR（写入 mix_fir）

    这是“通用且有理论支撑”的 TIADC 多相校准范式，避免单音在 fs/4 附近不可辨识导致的掉点。

    返回：w_time shape=(2,2,fir_taps) float32，其中 out=[x0,x1], in=[g0,g1]
    """
    n_full = int(n_full)
    n_records = int(n_records)
    fir_taps = int(fir_taps)
    if fir_taps < 5:
        fir_taps = 5
    if fir_taps % 2 == 0:
        fir_taps += 1
    n2 = n_full // 2
    if n2 < 512:
        raise ValueError("n_full too small for MIMO solver")
    # 使用 rfft 以简化 Hermitian 处理
    n_fft = _next_pow2(n2)
    n_rfft = n_fft // 2 + 1

    # 频率加权：w(ω)=1+alpha*(ω/π)^p ，属于 WLS 的标准做法
    k = np.arange(n_rfft, dtype=np.float64)
    w = 1.0 + float(wls_alpha) * ((k / max(1.0, (n_rfft - 1))) ** float(max(0.0, wls_power)))
    w = w.astype(np.float64)

    # 累计谱矩阵：S_xg(ω) 和 S_gg(ω)  (2x2 for each rfft bin)
    S_xg = np.zeros((2, 2, n_rfft), dtype=np.complex128)
    S_gg = np.zeros((2, 2, n_rfft), dtype=np.complex128)

    # 为了减少边缘效应，丢弃头尾 guard_drop（相当于有效采样窗口）
    gd = int(max(0, guard_drop))
    for _ in range(n_records):
        # 用相干多音作为持久激励（满秩更好，且与评估口径一致）
        # tone 数和分离度取“够用且稳”的默认值
        k_bins = _sample_k_set(8, n_full // 2 - 8, n_tones=8, min_sep=48, n=n_full, avoid_image_collision=True)
        src = _multisine_np(n=n_full, bins=k_bins, amp=float(np.random.uniform(0.55, 0.92)))
        # 实际采样（含失配）
        c0, c1, _ = sim.capture_two_way(src, p0=p0, p1=p1, n_bits=ADC_NBITS)
        g0 = c0[0::2].astype(np.float64, copy=False)
        g1 = c1[1::2].astype(np.float64, copy=False)
        # 理想参考（同样只在相位点采样，且不加抖动/非线性；量化也建议关掉）
        c0r, c1r, _ = sim.capture_two_way(src, p0=pref, p1=pref, n_bits=None)
        x0 = c0r[0::2].astype(np.float64, copy=False)
        x1 = c1r[1::2].astype(np.float64, copy=False)

        # 对齐长度（安全起见）
        L = min(len(g0), len(g1), len(x0), len(x1))
        g0 = g0[:L]
        g1 = g1[:L]
        x0 = x0[:L]
        x1 = x1[:L]
        if gd * 2 + 16 < L:
            g0 = g0[gd:-gd]
            g1 = g1[gd:-gd]
            x0 = x0[gd:-gd]
            x1 = x1[gd:-gd]
            L = len(g0)
        # 归一化（避免不同 record 幅度差异影响谱矩阵条件数）
        s = max(float(np.max(np.abs([g0, g1, x0, x1]))), 1e-12)
        g0 = (g0 / s).astype(np.float64, copy=False)
        g1 = (g1 / s).astype(np.float64, copy=False)
        x0 = (x0 / s).astype(np.float64, copy=False)
        x1 = (x1 / s).astype(np.float64, copy=False)

        # 加窗 rfft
        win = np.hanning(L).astype(np.float64)
        # zero-pad 到 n_fft
        G0 = np.fft.rfft((g0 - np.mean(g0)) * win, n=n_fft)
        G1 = np.fft.rfft((g1 - np.mean(g1)) * win, n=n_fft)
        X0 = np.fft.rfft((x0 - np.mean(x0)) * win, n=n_fft)
        X1 = np.fft.rfft((x1 - np.mean(x1)) * win, n=n_fft)

        # 组装向量
        G = np.stack([G0, G1], axis=0)  # [2, K]
        X = np.stack([X0, X1], axis=0)  # [2, K]

        # S_gg += w * (G G^H), S_xg += w * (X G^H)
        # 向量化：对所有频点一次性做 2x2 外积（比 Python for-loop 快很多）
        # GGh[o,i,k] = G[o,k] * conj(G[i,k])
        GGh = G[:, None, :] * np.conj(G[None, :, :])  # [2,2,K]
        XGh = X[:, None, :] * np.conj(G[None, :, :])  # [2,2,K]
        wk = w.reshape(1, 1, -1)  # [1,1,K]
        S_gg += GGh * wk
        S_xg += XGh * wk

    # 频点上求解 W(ω)=S_xg(ω)·(S_gg(ω)+μI)^{-1}
    # 2x2 可用解析逆做向量化（比逐点 np.linalg.inv 快，且更可控）
    mu = float(max(0.0, reg_mu))
    a00 = S_gg[0, 0, :] + mu
    a01 = S_gg[0, 1, :]
    a10 = S_gg[1, 0, :]
    a11 = S_gg[1, 1, :] + mu
    det = a00 * a11 - a01 * a10
    det_abs = np.abs(det)
    # 数值稳健：det 过小就做一点对角加载（等价于更强的 μ）
    det = np.where(det_abs < 1e-12, det + (mu + 1e-4), det)
    inv00 = a11 / det
    inv01 = -a01 / det
    inv10 = -a10 / det
    inv11 = a00 / det

    # W = S_xg @ inv(A)  (2x2 matmul for each k)
    sx00, sx01, sx10, sx11 = S_xg[0, 0, :], S_xg[0, 1, :], S_xg[1, 0, :], S_xg[1, 1, :]
    W_rfft = np.zeros((2, 2, n_rfft), dtype=np.complex128)
    W_rfft[0, 0, :] = sx00 * inv00 + sx01 * inv10
    W_rfft[0, 1, :] = sx00 * inv01 + sx01 * inv11
    W_rfft[1, 0, :] = sx10 * inv00 + sx11 * inv10
    W_rfft[1, 1, :] = sx10 * inv01 + sx11 * inv11

    # IFFT -> 截断 FIR taps
    # 先补齐全谱，再 ifft，得到循环时域响应，然后 fftshift 取中心 taps
    W_full = _hermitianize_rfft_bins(W_rfft, n_fft)  # [2,2,N]
    w_time_full = np.fft.ifft(W_full, axis=-1)  # complex, [2,2,N]
    w_time_full = np.real(w_time_full)
    w_time_full = np.fft.fftshift(w_time_full, axes=-1)  # 让“零时刻”居中

    mid = n_fft // 2
    half = fir_taps // 2
    w_cut = w_time_full[:, :, mid - half: mid + half + 1]
    # 轻微加窗减少截断纹波
    win_t = np.hamming(fir_taps).astype(np.float64)
    w_cut = (w_cut * win_t.reshape(1, 1, -1)).astype(np.float32, copy=False)
    return w_cut


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


# ============================================================================
# v0205: Differentiable Time-Frequency Hybrid Processing (Analysis-Calib-Synth)
# ============================================================================


def _deinterleave_two_way(y_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y_full -> (g0, g1) half-rate sequences (even/odd samples)."""
    y_full = np.asarray(y_full, dtype=np.float64)
    g0 = y_full[0::2].astype(np.float64, copy=False)
    g1 = y_full[1::2].astype(np.float64, copy=False)
    L = min(len(g0), len(g1))
    return g0[:L], g1[:L]


class STFTFilterBank(nn.Module):
    """
    用 STFT/ISTFT 实现的“可微分析-综合滤波器组”。

    说明：STFT 本质上是均匀 DFT 滤波器组（analysis FB），ISTFT 是其综合（synthesis FB）。
    只要 (window, hop) 满足 COLA 条件，就能做到（近似）完全重构（PR）。
    """

    def __init__(self, *, n_fft: int, hop_length: int, win: str = "hann"):
        super().__init__()
        n_fft = int(n_fft)
        hop_length = int(hop_length)
        if n_fft < 16:
            raise ValueError("n_fft too small")
        if hop_length < 1 or hop_length > n_fft:
            raise ValueError("invalid hop_length")
        self.n_fft = n_fft
        self.hop_length = hop_length
        if str(win).strip().lower() == "hann":
            w = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
        else:
            w = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
        self.register_buffer("window", w)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] real
        return: [B, F, T] complex (F=n_fft//2+1)
        """
        if x.dim() != 2:
            raise ValueError("analysis expects x shape [B,L]")
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=x.device, dtype=x.dtype),
            center=True,
            pad_mode="reflect",
            normalized=True,
            onesided=True,
            return_complex=True,
        )
        return X

    def synthesis(self, X: torch.Tensor, *, length: int) -> torch.Tensor:
        """
        X: [B, F, T] complex
        return: [B, L] real
        """
        if X.dim() != 3:
            raise ValueError("synthesis expects X shape [B,F,T]")
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(device=X.device, dtype=torch.float32),
            center=True,
            normalized=True,
            onesided=True,
            length=int(length),
        )
        return x


class ControllerMLP(nn.Module):
    """
    Meta-parameter generator:
    输入：粗 PSD 特征 + 温度
    输出：每个子带×通道的 (gain_log, phase)。
    """

    def __init__(self, *, feat_dim: int, n_subbands: int, n_ch: int, out_per_subband: int, hidden: int = 128):
        super().__init__()
        self.n_subbands = int(n_subbands)
        self.n_ch = int(n_ch)
        self.out_per_subband = int(out_per_subband)
        out_dim = 2 * self.n_subbands * int(self.out_per_subband)
        h = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(int(feat_dim), h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, out_dim),
        )
        # 关键稳定性：让 controller 初始输出为 0 => (gain_log,phase)=(0,0) => 权重为 identity。
        # 否则随机初始化会让一开始的 W[k] 在 0.3~3.3 倍范围乱飞，直接把 SINAD/SFDR 拉爆，
        # 训练也更容易陷入“压镜像但抬噪声/抬谐波”的坏局部最优。
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.zero_()
                if last.bias is not None:
                    last.bias.zero_()

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        feat: [B, feat_dim]
        return: gain_log, phase  both [B, K, out_per_subband]
        """
        z = self.net(feat)
        B = int(z.shape[0])
        z = z.view(B, 2, self.n_subbands, int(self.out_per_subband))
        gain_log = z[:, 0, :, :]
        phase = z[:, 1, :, :]
        # 数值稳定：限制相位范围，增益用 log-domain
        # Modified: Relax phase limit to allow unwrapped phase (crucial for interpolation and large delay)
        phase = phase # No tanh limit, let it grow if needed (linear phase)
        gain_log = torch.tanh(gain_log) * 1.2  # ~ +/-10%~20% 的可学习范围（可通过 loss 再放开）
        return gain_log, phase


class DifferentiableSubbandCalibrator(nn.Module):
    """
    分析-校准-综合：对每个子带 & 每个 TIADC 通道学一个复权重 W[k,m]。

    当前脚本是 2-way TIADC（M=2），但结构支持 M>=2（只要你把输入改成 M 路序列）。
    """

    def __init__(
        self,
        *,
        n_ch: int = 2,
        n_fft: int = 512,
        hop_length: int = 256,
        n_subbands: int = 64,
        psd_feat_bins: int = 64,
        use_controller: bool = True,
        use_mimo: bool = False,
        use_img_cancel: bool = False,
    ):
        super().__init__()
        self.n_ch = int(n_ch)
        self.fb = STFTFilterBank(n_fft=int(n_fft), hop_length=int(hop_length))
        self.n_subbands = int(n_subbands)
        self.psd_feat_bins = int(psd_feat_bins)
        self.use_controller = bool(use_controller)
        self.use_mimo = bool(use_mimo)
        self.use_img_cancel = bool(use_img_cancel)
        if int(self.n_ch) != 2 and self.use_mimo:
            raise NotImplementedError("use_mimo currently supports n_ch=2 only")
        self.out_per_subband = int(self.n_ch * self.n_ch) if self.use_mimo else int(self.n_ch)

        # 将 STFT 频点映射到 K 个子带（均匀划分）
        F = int(self.fb.n_fft // 2 + 1)
        bin2sb = (torch.arange(F, dtype=torch.long) * int(self.n_subbands) // max(1, F)).clamp(0, self.n_subbands - 1)
        self.register_buffer("bin2sb", bin2sb)

        if self.use_controller:
            # feat = [coarse_PSD(psd_feat_bins), temperature] -> (gain_log, phase)
            self.controller = ControllerMLP(
                feat_dim=self.psd_feat_bins + 1,
                n_subbands=self.n_subbands,
                n_ch=self.n_ch,
                out_per_subband=int(self.out_per_subband),
                hidden=_env_int("CTRL_HIDDEN", 128),
            )
        else:
            self.gain_log = nn.Parameter(torch.zeros(self.n_subbands, int(self.out_per_subband)))
            self.phase = nn.Parameter(torch.zeros(self.n_subbands, int(self.out_per_subband)))

        # image-canceller branch (wide-linear style): y = y_main + (-1)^n * H( (-1)^n * y )
        # 这个分支对 2-way TIADC 的 fs/2 镜像 spur 更“对症”，避免只靠主路 W[k] 难以压住最高频镜像。
        if self.use_img_cancel:
            self.img_gain_log = nn.Parameter(torch.zeros(self.n_subbands, int(self.out_per_subband)))
            self.img_phase = nn.Parameter(torch.zeros(self.n_subbands, int(self.out_per_subband)))

    def _extract_feat(self, y_full: torch.Tensor, temp_c: torch.Tensor) -> torch.Tensor:
        """
        y_full: [B, L] real
        temp_c: [B] or [B,1]
        return feat: [B, psd_feat_bins+1]
        """
        B, L = y_full.shape
        # PSD (coarse)
        win = torch.hann_window(L, periodic=True, dtype=y_full.dtype, device=y_full.device)
        Y = torch.fft.rfft((y_full - torch.mean(y_full, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
        P = (Y.real * Y.real + Y.imag * Y.imag).clamp_min(0.0)
        Pn = P / (torch.mean(P, dim=-1, keepdim=True) + 1e-12)
        feat_psd = F.adaptive_avg_pool1d(Pn.view(B, 1, -1), int(self.psd_feat_bins)).view(B, -1)
        if temp_c.dim() == 1:
            t = temp_c.view(B, 1)
        else:
            t = temp_c.view(B, 1)
        # 简单归一化到 [-1,1]（假设温度范围大致在 [0,100]℃）
        t = (t - 50.0) / 50.0
        return torch.cat([feat_psd, t.to(dtype=feat_psd.dtype)], dim=-1)

    def _weights_per_bin(self, gain_log: torch.Tensor, phase: torch.Tensor, *, B: int, Fbins: int) -> torch.Tensor:
        """
        gain_log, phase:
        - controller: [B,K,out_per_subband]
        - static: [K,out_per_subband]

        return complex weights:
        - diag mode: [B, M, F, 1]
        - mimo mode: [B, F, M, M]
        """
        # --- NEW: Linear Interpolation for Smooth Frequency Response ---
        # Instead of step-wise index_select (which causes time-domain ringing),
        # we interpolate the K subband parameters to Fbins smoothly.
        # This allows accurate fitting of linear phase (delay) without steps.
        
        # Prepare input for interpolate: [Batch, Channels, Length]
        # We treat 'K' as Length.
        
        # Helper to interpolate [..., K, C] -> [..., F, C]
        def _interp(x: torch.Tensor, target_len: int) -> torch.Tensor:
            # x: [..., K, C]
            # We need [N, C, K] for interpolate
            
            # 1. Permute to move K to the end: [..., C, K]
            dims = list(range(x.dim()))
            dims[-1], dims[-2] = dims[-2], dims[-1]
            x_perm = x.permute(*dims)
            
            # 2. Flatten batch dims: [N, C, K]
            shape = x_perm.shape
            C = shape[-2]
            K = shape[-1]
            x_flat = x_perm.reshape(-1, C, K)
            
            # 3. Interpolate
            out = F.interpolate(x_flat, size=target_len, mode='linear', align_corners=True) # [N, C, F]
            
            # 4. Reshape back: [..., C, F]
            out_shape = list(shape)
            out_shape[-1] = target_len
            out = out.view(*out_shape)
            
            # 5. Permute back: [..., C, F] -> [..., F, C]
            out = out.permute(*dims)
            return out

        # Interpolate gain/phase from K subbands to Fbins
        # Note: gain_log/phase can be [K, out] or [B, K, out]
        g_interp = _interp(gain_log, Fbins)
        p_interp = _interp(phase, Fbins)

        # Now use g_interp/p_interp (size Fbins) directly
        # No need for bin2sb mapping anymore
        
        mag_scale = float(os.getenv("TF_MAG_SCALE", "0.5"))
        if not self.use_mimo:
            # g_interp: [..., F, M]
            # Need: [B, M, F, 1]
            
            if g_interp.dim() == 2: # [F, M]
                g = g_interp[:, : int(self.n_ch)]
                p = p_interp[:, : int(self.n_ch)]
                # [F, M] -> [M, F] -> [1, M, F, 1] -> [B, M, F, 1]
                g = g.transpose(0, 1).unsqueeze(0).unsqueeze(-1).expand(int(B), -1, -1, 1)
                p = p.transpose(0, 1).unsqueeze(0).unsqueeze(-1).expand(int(B), -1, -1, 1)
            else: # [B, F, M]
                g = g_interp.permute(0, 2, 1).unsqueeze(-1) # [B, M, F, 1]
                p = p_interp.permute(0, 2, 1).unsqueeze(-1)
            
            mag = 1.0 + float(mag_scale) * torch.tanh(g)
            return mag * torch.exp(1j * p)

        # mimo: out_per_subband = M*M
        M = int(self.n_ch)
        if g_interp.dim() == 2: # [F, M*M]
            g = g_interp.view(Fbins, M, M).unsqueeze(0).expand(int(B), -1, -1, -1) # [B, F, M, M]
            p = p_interp.view(Fbins, M, M).unsqueeze(0).expand(int(B), -1, -1, -1)
        else: # [B, F, M*M]
            g = g_interp.view(int(B), Fbins, M, M)
            p = p_interp.view(int(B), Fbins, M, M)

        # 关键稳定性：MIMO 的 off-diagonal（通道耦合）必须默认接近 0，否则一开始就会把两路混在一起，
        # 导致单音/扫频指标直接崩掉、训练也更容易发散。
        #
        # Modified: REMOVED the frequency-dependent gating (off_gate).
        # The previous logic (blocking MIMO below 4.5GHz) prevented the model from resolving
        # the 4GHz/6GHz aliasing pair, which appears at the 4GHz bin (f_norm=0.4).
        # We now allow MIMO coupling across the full band and rely on the loss function.
        
        off_max = float(os.getenv("TF_MIMO_OFF_MAX", "0.5"))
        # off_gate = 1.0 (effectively removed)
        off_scale = float(off_max) 
        
        ejp = torch.exp(1j * p)
        diag = torch.eye(M, dtype=ejp.real.dtype, device=ejp.device).view(1, 1, M, M)
        mag = 1.0 + float(mag_scale) * torch.tanh(g)
        W_diag = mag * ejp * diag
        W_off = off_scale * torch.tanh(g) * ejp * (1.0 - diag)
        return W_diag + W_off

    def _weights_raw_per_bin(self, gain_log: torch.Tensor, phase: torch.Tensor, *, B: int, Fbins: int) -> torch.Tensor:
        """
        与 _weights_per_bin 类似，但 **不做** 单位阵混合（用于 image-canceller 分支，要求初始为 0）。
        return:
        - diag mode: [B, M, F, 1]
        - mimo mode: [B, F, M, M]
        """
        # --- NEW: Linear Interpolation for Smooth Frequency Response ---
        def _interp(x: torch.Tensor, target_len: int) -> torch.Tensor:
            # x: [..., K, C]
            # We need [N, C, K] for interpolate
            
            # 1. Permute to move K to the end: [..., C, K]
            # Note: permute expects dimension indices
            dims = list(range(x.dim()))
            # Swap last two dims: ..., K, C -> ..., C, K
            dims[-1], dims[-2] = dims[-2], dims[-1]
            x_perm = x.permute(*dims)
            
            # 2. Flatten batch dims: [N, C, K]
            shape = x_perm.shape
            C = shape[-2]
            K = shape[-1]
            x_flat = x_perm.reshape(-1, C, K)
            
            # 3. Interpolate
            out = F.interpolate(x_flat, size=target_len, mode='linear', align_corners=True) # [N, C, F]
            
            # 4. Reshape back: [..., C, F]
            out_shape = list(shape)
            out_shape[-1] = target_len
            out = out.view(*out_shape)
            
            # 5. Permute back: [..., C, F] -> [..., F, C]
            out = out.permute(*dims)
            return out

        g_interp = _interp(gain_log, Fbins)
        p_interp = _interp(phase, Fbins)

        mag_scale = float(os.getenv("TF_MAG_SCALE", "0.25"))
        if not self.use_mimo:
            if g_interp.dim() == 2:
                g = g_interp[:, : int(self.n_ch)]
                p = p_interp[:, : int(self.n_ch)]
                g = g.transpose(0, 1).unsqueeze(0).unsqueeze(-1).expand(int(B), -1, -1, 1)
                p = p.transpose(0, 1).unsqueeze(0).unsqueeze(-1).expand(int(B), -1, -1, 1)
            else:
                g = g_interp.permute(0, 2, 1).unsqueeze(-1)
                p = p_interp.permute(0, 2, 1).unsqueeze(-1)
            mag = 1.0 + float(mag_scale) * torch.tanh(g)
            return mag * torch.exp(1j * p)

        M = int(self.n_ch)
        if g_interp.dim() == 2:
            g = g_interp.view(Fbins, M, M).unsqueeze(0).expand(int(B), -1, -1, -1)
            p = p_interp.view(Fbins, M, M).unsqueeze(0).expand(int(B), -1, -1, -1)
        else:
            g = g_interp.view(int(B), Fbins, M, M)
            p = p_interp.view(int(B), Fbins, M, M)
        
        # 与 _weights_per_bin 相同的“对角=1、非对角=0”初始化友好参数化
        # Modified: REMOVED frequency gating here too.
        off_max = float(os.getenv("TF_MIMO_OFF_MAX", "0.5"))
        off_scale = float(off_max)
        
        ejp = torch.exp(1j * p)
        diag = torch.eye(M, dtype=ejp.real.dtype, device=ejp.device).view(1, 1, M, M)
        mag = 1.0 + float(mag_scale) * torch.tanh(g)
        W_diag = mag * ejp * diag
        W_off = off_scale * torch.tanh(g) * ejp * (1.0 - diag)
        return W_diag + W_off

    def forward(
        self,
        g_list: List[torch.Tensor],
        *,
        y_full_for_feat: torch.Tensor,
        temp_c: torch.Tensor,
        return_params: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        g_list: list of M tensors, each [B, Lsub] real (half-rate sequences)
        y_full_for_feat: [B, Lfull] real (for controller features)
        temp_c: [B]

        return calibrated full-rate: [B, Lfull] real
        """
        if len(g_list) != int(self.n_ch):
            raise ValueError(f"expected {self.n_ch} channels, got {len(g_list)}")
        B = int(g_list[0].shape[0])
        Lsub = int(g_list[0].shape[1])

        if self.use_controller:
            feat = self._extract_feat(y_full_for_feat, temp_c=temp_c)
            gain_log, phase = self.controller(feat)
        else:
            gain_log, phase = self.gain_log, self.phase

        # optional: clamp band-edge subbands to identity (avoid hurting extreme low/high)
        n_edge = int(_env_int("TF_CLAMP_EDGE_SBS", 0))
        if n_edge > 0:
            K = int(self.n_subbands)
            n_edge = max(0, min(n_edge, K // 2))
            if gain_log.dim() == 3:
                gain_log = gain_log.clone()
                phase = phase.clone()
                gain_log[:, :n_edge, :] = 0.0
                phase[:, :n_edge, :] = 0.0
                gain_log[:, K - n_edge :, :] = 0.0
                phase[:, K - n_edge :, :] = 0.0
            else:
                gain_log = gain_log.clone()
                phase = phase.clone()
                gain_log[:n_edge, :] = 0.0
                phase[:n_edge, :] = 0.0
                gain_log[K - n_edge :, :] = 0.0
                phase[K - n_edge :, :] = 0.0

        specs = []
        for m in range(int(self.n_ch)):
            specs.append(self.fb.analysis(g_list[m]))  # [B,F,T] complex
        S = torch.stack(specs, dim=1)  # [B,M,F,T]
        Fbins = int(S.shape[2])
        W = self._weights_per_bin(gain_log, phase, B=B, Fbins=Fbins)
        if not self.use_mimo:
            # [B,M,F,1]
            S2 = S * W  # broadcast over T
        else:
            # W: [B,F,M,M],  apply per bin: [B,F,T,M] @ [B,F,M,M] -> [B,F,T,M]
            Sp = S.permute(0, 2, 3, 1)  # [B,F,T,M]
            Spp = torch.einsum("bftm,bfmn->bftn", Sp, W)  # [B,F,T,M]
            S2 = Spp.permute(0, 3, 1, 2)  # [B,M,F,T]

        ys = []
        if not self.use_img_cancel:
            for m in range(int(self.n_ch)):
                ys.append(self.fb.synthesis(S2[:, m, :, :], length=Lsub))  # [B,Lsub]
        else:
            # (-1)^n sign for modulation/demodulation (shift by fs/2 in subrate)
            n_idx = torch.arange(Lsub, device=g_list[0].device, dtype=g_list[0].dtype).view(1, -1)
            sign = torch.where((n_idx.to(torch.int64) % 2) == 0, torch.ones_like(n_idx), -torch.ones_like(n_idx))
            # build img-branch weights (static parameters), use delta form so init=0
            img_g = getattr(self, "img_gain_log")
            img_p = getattr(self, "img_phase")
            Wimg_raw = self._weights_raw_per_bin(img_g, img_p, B=B, Fbins=Fbins)
            if not self.use_mimo:
                Wimg = Wimg_raw - 1.0  # [B,M,F,1]
            else:
                M = int(self.n_ch)
                I = torch.eye(M, dtype=Wimg_raw.real.dtype, device=Wimg_raw.device).view(1, 1, M, M)
                Wimg = Wimg_raw - I  # [B,F,M,M]

            # analysis of modulated signals
            specs_i = []
            for m in range(int(self.n_ch)):
                xm = g_list[m] * sign
                specs_i.append(self.fb.analysis(xm))
            Si = torch.stack(specs_i, dim=1)  # [B,M,F,T]

            if not self.use_mimo:
                Si2 = Si * Wimg  # [B,M,F,T]
            else:
                Sp_i = Si.permute(0, 2, 3, 1)  # [B,F,T,M]
                Si_pp = torch.einsum("bftm,bfmn->bftn", Sp_i, Wimg)  # [B,F,T,M]
                Si2 = Si_pp.permute(0, 3, 1, 2)

            for m in range(int(self.n_ch)):
                y_main = self.fb.synthesis(S2[:, m, :, :], length=Lsub)
                y_img = self.fb.synthesis(Si2[:, m, :, :], length=Lsub) * sign  # demod
                ys.append(y_main + y_img)

        # synthesis back to full-rate by interleaving (2-way TIADC)
        if int(self.n_ch) != 2:
            raise NotImplementedError("current script only implements 2-way interleave synthesis")
        y0 = ys[0]
        y1 = ys[1]
        L = min(int(y0.shape[-1]), int(y1.shape[-1]))
        y0 = y0[:, :L]
        y1 = y1[:, :L]
        out = torch.zeros((B, 2 * L), dtype=y0.dtype, device=y0.device)
        out[:, 0::2] = y0
        out[:, 1::2] = y1
        if not bool(return_params):
            return out
        return out, {"gain_log": gain_log, "phase": phase}


def _band_energy_from_rfft_power(P: torch.Tensor, k: torch.Tensor, *, guard: int) -> torch.Tensor:
    """
    P: [B, F] (rfft power)
    k: [B, K] bins
    return: [B] sum energy over each bin +/- guard
    """
    B, F = P.shape
    K = int(k.shape[-1])
    kk = k.clamp(0, F - 1)
    out = torch.zeros((B,), dtype=P.dtype, device=P.device)
    g = int(max(0, guard))
    for d in range(-g, g + 1):
        out = out + torch.gather(P, dim=-1, index=(kk + d).clamp(0, F - 1)).sum(dim=-1)
    return out


def _band_energy_ring_from_rfft_power(P: torch.Tensor, k: torch.Tensor, *, inner_guard: int, outer_guard: int) -> torch.Tensor:
    """
    P: [B, F] (rfft power)
    k: [B, K] bins
    return: [B] sum energy over (inner_guard, outer_guard] around each k (i.e. skirt ring)

    用途：约束“基波裙边/旁瓣”（常由时变校正/窗效应导致），这是 SINAD/SFDR 里非常关键但仅靠 leak/img 不容易压住的部分。
    """
    B, F = P.shape
    kk = k.clamp(0, F - 1)
    gi = int(max(0, inner_guard))
    go = int(max(gi + 1, outer_guard))
    out = torch.zeros((B,), dtype=P.dtype, device=P.device)
    for d in range(-go, go + 1):
        if abs(int(d)) <= gi:
            continue
        out = out + torch.gather(P, dim=-1, index=(kk + d).clamp(0, F - 1)).sum(dim=-1)
    return out


def _estimate_fund_bins_from_input(y_in: torch.Tensor, *, n_funds: int, exclude_dc: int = 6) -> torch.Tensor:
    """
    从输入 y_in 的谱峰估计“基波 bin”，作为盲校准的锚点（避免从 y_hat 自举选错 spur）。

    y_in: [B, L] real
    return: [B, n_funds] int64
    """
    B, L = y_in.shape
    win = torch.hann_window(L, periodic=True, dtype=y_in.dtype, device=y_in.device)
    Y = torch.fft.rfft((y_in - torch.mean(y_in, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    P = (Y.real * Y.real + Y.imag * Y.imag).clamp_min(0.0)
    if int(exclude_dc) > 0:
        P[:, : int(exclude_dc)] = 0.0
    kk = torch.topk(P, k=int(max(1, n_funds)), dim=-1).indices
    return kk.to(dtype=torch.long)


def blind_tiadc_anchor_loss(
    y_hat: torch.Tensor,
    y_in: torch.Tensor,
    *,
    n_funds: int = 6,
    guard: int = 3,
    exclude_dc: int = 6,
    skirt_outer: int = 24,
    noise_drop_topk: int = 64,
    k_fund_override: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    盲校准（Anchor 版本）：用输入 y_in 的谱峰做锚点。

    - leakage: 带外功率 / 带内功率  (带内=锚点±guard)
    - image:   图像功率 / 基波功率  (k_img = N/2 - k_fund)
    - energy:  输出能量保持（避免投机解）
    """
    B, L = y_hat.shape
    win = torch.hann_window(L, periodic=True, dtype=y_hat.dtype, device=y_hat.device)

    # anchor bins
    if k_fund_override is None:
        # from input PSD peaks (blind)
        k_fund = _estimate_fund_bins_from_input(y_in.detach(), n_funds=int(n_funds), exclude_dc=int(exclude_dc))
    else:
        k_fund = k_fund_override.to(device=y_hat.device, dtype=torch.long)

    Yh = torch.fft.rfft((y_hat - torch.mean(y_hat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    Ph = (Yh.real * Yh.real + Yh.imag * Yh.imag).clamp_min(0.0)  # [B,F]
    Yi = torch.fft.rfft((y_in - torch.mean(y_in, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    Pi = (Yi.real * Yi.real + Yi.imag * Yi.imag).clamp_min(0.0)  # [B,F]
    if int(exclude_dc) > 0:
        Ph0 = Ph.clone()
        Ph0[:, : int(exclude_dc)] = 0.0
    else:
        Ph0 = Ph

    p_keep = _band_energy_from_rfft_power(Ph, k_fund, guard=int(guard)) + 1e-12
    p_total = torch.sum(Ph0, dim=-1) + 1e-12
    p_leak = torch.clamp(p_total - p_keep, min=0.0)
    loss_leak = torch.mean(p_leak / p_keep)

    half = int(L // 2)
    k_img = torch.abs(half - k_fund)
    p_f = _band_energy_from_rfft_power(Ph, k_fund, guard=int(guard)) + 1e-12
    p_i = _band_energy_from_rfft_power(Ph, k_img, guard=int(guard)) + 1e-12
    loss_img = torch.mean(p_i / p_f)

    # fund fidelity: keep anchor-bin energy close to input (avoid distorting good bands)
    p_f_in = _band_energy_from_rfft_power(Pi, k_fund, guard=int(guard)) + 1e-12
    loss_fund = torch.mean((torch.log(p_f) - torch.log(p_f_in)) ** 2)

    # -----------------------
    # NEW: skirt + noise floor
    # -----------------------
    # skirt（裙边/旁瓣）：基波附近（但不包含基波窗口内）的能量，常见来源：
    # - 校正权重随帧变化 => AM/FM 调制 sidebands
    # - STFT/ISTFT overlap-add 的边界效应
    # 这项不压住，SINAD/SFDR 会很难全带宽变好。
    so = int(max(int(guard) + 2, int(skirt_outer)))
    p_sk = _band_energy_ring_from_rfft_power(Ph, k_fund, inner_guard=int(guard), outer_guard=int(so)) + 1e-12
    loss_skirt = torch.mean(p_sk / p_f)

    # noise floor：在排除 fund±skirt_outer、image±guard 之后的剩余功率。
    # 用“去掉 top-k 最大 spur”来获得更接近“噪声底”的估计（避免被少量 spur 主导）。
    Pout = Ph0.clone()
    Fbins = int(Pout.shape[-1])
    # exclude fund neighborhood
    for d in range(-so, so + 1):
        idx = (k_fund + int(d)).clamp(0, Fbins - 1)
        Pout.scatter_(dim=-1, index=idx, src=torch.zeros_like(idx, dtype=Pout.dtype))
    # exclude image neighborhood (narrow)
    for d in range(-int(guard), int(guard) + 1):
        idx = (k_img + int(d)).clamp(0, Fbins - 1)
        Pout.scatter_(dim=-1, index=idx, src=torch.zeros_like(idx, dtype=Pout.dtype))
    p_out_total = torch.sum(Pout, dim=-1).clamp_min(0.0) + 1e-12
    k_drop = int(max(0, noise_drop_topk))
    if k_drop > 0 and k_drop < Fbins:
        topv, _ = torch.topk(Pout, k=int(min(k_drop, Fbins - 1)), dim=-1)
        p_out_noise = torch.clamp(p_out_total - torch.sum(topv, dim=-1), min=0.0)
    else:
        p_out_noise = p_out_total
    loss_noise = torch.mean(p_out_noise / p_f)

    e_in = torch.mean(y_in * y_in, dim=-1).clamp_min(1e-12)
    e_hat = torch.mean(y_hat * y_hat, dim=-1).clamp_min(1e-12)
    loss_energy = torch.mean((torch.log(e_hat) - torch.log(e_in)) ** 2)

    parts = {
        "loss_leak": loss_leak,
        "loss_img": loss_img,
        "loss_fund": loss_fund,
        "loss_energy": loss_energy,
        "loss_skirt": loss_skirt,
        "loss_noise": loss_noise,
    }
    return loss_leak + loss_img + loss_fund + loss_energy, parts


def blind_spectral_purity_loss(
    y_hat: torch.Tensor,
    y_ref_energy: torch.Tensor,
    *,
    n_funds: int = 4,
    img_guard: int = 3,
    sparse_exclude_dc: int = 5,
) -> tuple[torch.Tensor, dict]:
    """
    盲校准损失（适合“稀疏载波/多音”）：
    - **泄漏最小化**：把能量集中到 top-N 频点附近（抑制 spur/泄漏）
    - **图像杂散抑制**：2-way 交织镜像 (fs/2 - fin)
    - **能量保持**：避免“把信号压扁”得到的投机解

    y_hat: [B, L]
    y_ref_energy: [B]  (例如输入 y 的 RMS^2)
    """
    B, L = y_hat.shape
    win = torch.hann_window(L, periodic=True, dtype=y_hat.dtype, device=y_hat.device)
    Y = torch.fft.rfft((y_hat - torch.mean(y_hat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
    P = (Y.real * Y.real + Y.imag * Y.imag).clamp_min(0.0)  # [B,F]

    # ---- pick top-N bins as "funds" (exclude DC) ----
    P0 = P.clone()
    if int(sparse_exclude_dc) > 0:
        P0[:, : int(sparse_exclude_dc)] = 0.0
    kk = torch.topk(P0, k=int(max(1, n_funds)), dim=-1).indices  # [B,K]

    # ---- leakage: minimize power outside selected bins ----
    p_keep = _band_energy_from_rfft_power(P, kk, guard=int(img_guard)) + 1e-12
    p_total = torch.sum(P0, dim=-1) + 1e-12
    p_leak = torch.clamp(p_total - p_keep, min=0.0)
    loss_sparse = torch.mean(p_leak / p_keep)

    # ---- image spur suppression (2-way: k_img = N/2 - k_fund in rfft) ----
    half = int(L // 2)
    k_img = torch.abs(half - kk)
    p_f = _band_energy_from_rfft_power(P, kk, guard=int(img_guard)) + 1e-12
    p_i = _band_energy_from_rfft_power(P, k_img, guard=int(img_guard)) + 1e-12
    loss_img = torch.mean(p_i / p_f)

    # ---- energy keep ----
    e_hat = torch.mean(y_hat * y_hat, dim=-1).clamp_min(1e-12)
    loss_energy = torch.mean((torch.log(e_hat) - torch.log(y_ref_energy.clamp_min(1e-12))) ** 2)

    parts = {"loss_sparse": loss_sparse, "loss_img": loss_img, "loss_energy": loss_energy}
    return (loss_sparse + loss_img + loss_energy), parts



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
            # MIMO 系统辨识模式下，2×2 FIR 需要更长 taps 才能稳定拟合（尤其是接近 fs/4 的幅相失配）。
            # 默认关闭 MIMO（恢复修改前方案），需用时设 STAGE12_MIMO_LS=1。
            _mimo = _env_bool("STAGE12_MIMO_LS", False)
            mtaps = _env_int("STAGE12_XCOUPLE_TAPS", 257 if _mimo else 49)
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


def _mimo_fir_gain_band_loss(
    w2: torch.Tensor,
    *,
    n_fft: int = 1024,
    diag_min_gain: float = 0.8,
    diag_max_gain: float = 1.25,
    x_max_gain: float = 0.35,
) -> torch.Tensor:
    """
    2×2 FIR 频响约束（MIMO 系统辨识里非常关键）：
    - 对角支路（self terms）幅度应接近 1：限制在 [diag_min_gain, diag_max_gain]
    - 交叉支路（cross terms）应更小：限制在 [0, x_max_gain]

    目的：防止“求逆/欠定解”把噪声或镜像 spur 一起放大，出现校准后更差。
    """
    if w2.ndim != 3 or int(w2.shape[0]) != 2 or int(w2.shape[1]) != 2:
        return torch.zeros((), device=w2.device, dtype=w2.dtype)
    loss = 0.0
    for i in range(2):
        loss = loss + _fir_gain_band_loss(w2[i, i, :].contiguous(), n_fft=int(n_fft), min_gain=float(diag_min_gain), max_gain=float(diag_max_gain))
    loss = loss + _fir_gain_band_loss(w2[0, 1, :].contiguous(), n_fft=int(n_fft), min_gain=0.0, max_gain=float(x_max_gain))
    loss = loss + _fir_gain_band_loss(w2[1, 0, :].contiguous(), n_fft=int(n_fft), min_gain=0.0, max_gain=float(x_max_gain))
    return loss / 4.0


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
    # 2×2 MIMO（系统辨识）模式：用 pref 提供的参考采样做监督，让问题变成可辨识的 MIMO 最小二乘系统辨识。
    # 这是更通用、理论闭环的解法（持久激励 + 监督辨识 + 正则化/增益约束），避免 fs/4 附近病态导致掉点。
    # 默认使用修改前的方案：polyphase + 自监督镜像抑制（效果更稳、低频/高频整体更好）。
    # 若需尝试 2×2 MIMO 监督系统辨识，可显式设置 STAGE12_MIMO_LS=1。
    mimo_mode = _env_bool("STAGE12_MIMO_LS", False)
    if mimo_mode:
        if pref is None:
            raise RuntimeError("STAGE12_MIMO_LS=1 requires pref (reference ADC) in simulation")
        use_ref = True
        # 工程默认：固定 ADC0(even) 为参考，只训练 ADC1(odd)+widely-linear 分支，更稳且避免欠定漂移
        train_both = _env_bool("STAGE12_MIMO_TRAIN_BOTH", False)
    # 工程建议：高频想要有效，FIR taps 往往需要 >=129（默认提高）
    taps = _env_int("STAGE12_TAPS", 257)
    n = _env_int("STAGE12_TONE_N", 16384)
    # continuation：先训练“镜像抑制分支”（B(ω)），再训练主路 A(ω)
    steps0 = _env_int("STAGE12_S0_STEPS", 8000)
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
    # 监督系统辨识时：镜像 spur 主要由 ref 对齐自然压制，这里只给小权重当正则
    w_img = _env_float("STAGE12_W_IMG", 40.0 if mimo_mode else (600.0 if use_ref else 5500.0))
    w_ref_time = _env_float("STAGE12_W_REF_TIME", 35.0)
    w_ref_spec = _env_float("STAGE12_W_REF_SPEC", 10.0 if mimo_mode else 6.0)
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

    # 2×2 MIMO（半速域）闭式解：理论上更“通用”
    # - 需要 pref（参考 ADC）提供 x0/x1 的理想采样
    # - 求得的 2×2 FIR 直接写入 model.mix_fir（xcouple 分支），并将 even/odd 主路固定为 identity
    # 现实落地：在实测里 pref 等价于“已知激励/参考通道”，若拿不到就不能做监督闭式解，只能走自监督镜像 spur 目标。
    # 注意：闭式解对窗函数/截断/Conv1d 互相关口径非常敏感，这里保留为实验开关，默认关闭。
    use_mimo_ls = _env_bool("STAGE12_MIMO_CLOSED_FORM", False) and use_polyphase and (pref is not None)
    if use_mimo_ls:
        if not isinstance(model, PolyphaseStage12Model):
            raise RuntimeError("STAGE12_MIMO_LS requires STAGE12_POLYPHASE=1")
        # 强制启用 2×2 FIR
        if not getattr(model, "use_xcouple", False):
            raise RuntimeError("STAGE12_MIMO_LS requires STAGE12_USE_XCOUPLE=1 (mix_fir enabled)")
        mtaps = int(getattr(model, "mix_fir").weight.shape[-1])
        # 求解参数
        n_rec = _env_int("STAGE12_MIMO_REC", 24)
        mu = _env_float("STAGE12_MIMO_MU", 3e-6)
        wls_a = _env_float("STAGE12_MIMO_WLS_ALPHA", 1.0)
        wls_p = _env_float("STAGE12_MIMO_WLS_POWER", 2.0)
        gd = _env_int("STAGE12_MIMO_GUARD_DROP", 256)
        print(
            f"=== Stage1/2 (2x2 MIMO-Wiener LS) mtaps={mtaps} N={n} rec={n_rec} mu={mu:g} "
            f"WLS(alpha={wls_a:g},p={wls_p:g}) ==="
        )
        w_mimo = solve_stage12_mimo_wiener_fir(
            sim,
            p0=p0,
            p1=p1,
            pref=pref,
            n_full=int(n),
            n_records=int(n_rec),
            fir_taps=int(mtaps),
            guard_drop=int(gd),
            reg_mu=float(mu),
            wls_alpha=float(wls_a),
            wls_power=float(wls_p),
        )
        with torch.no_grad():
            # 固定两路主路为 identity（把所有线性校正都交给 2×2 MIMO FIR）
            model.train_both = False
            # even：identity
            model.even.gain.fill_(1.0)
            model.even.delay.delay.fill_(0.0)
            model.even.fir.weight.zero_()
            model.even.fir.weight[0, 0, int(model.even.fir.weight.shape[-1] // 2)] = 1.0
            for p in model.even.parameters():
                p.requires_grad_(False)
            # odd：identity
            model.odd.gain.fill_(1.0)
            model.odd.delay.delay.fill_(0.0)
            model.odd.fir.weight.zero_()
            model.odd.fir.weight[0, 0, int(model.odd.fir.weight.shape[-1] // 2)] = 1.0
            for p in model.odd.parameters():
                p.requires_grad_(False)
            # 写入 2×2 FIR：out=[x0,x1], in=[g0,g1]
            # 注意：PyTorch Conv1d 做的是“互相关”(cross-correlation)，不是数学意义的卷积。
            # 我们用频域/卷积口径求得的 FIR 需要在写入权重前做 time-reverse，否则会等价于用反向滤波器，性能会严重恶化。
            w_mimo_cc = w_mimo[..., ::-1].copy()
            model.mix_fir.weight.copy_(torch.tensor(w_mimo_cc, dtype=model.mix_fir.weight.dtype, device=model.mix_fir.weight.device))
            for p in model.mix_fir.parameters():
                p.requires_grad_(False)
            # 关闭 image-canceller（可选）；MIMO 已能处理大部分镜像抑制，避免重复自由度造成不可控增益
            if getattr(model, "use_img_cancel", False) and _env_bool("STAGE12_MIMO_DISABLE_IMG_CANCEL", True):
                model.img_fir.weight.zero_()
                for p in model.img_fir.parameters():
                    p.requires_grad_(False)
        # 返回：闭式解不需要后续 S0/S1/S2 迭代
        return model, float(scale)

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

    # 预先缓存窗函数，避免在每次 loss 里重复构造（加速）
    win_full = torch.hann_window(int(n), device=device, dtype=torch.float32).view(1, 1, -1)

    # 监督(ref/MIMO)训练时的“干净采样口径”：
    # 线性 Stage1/2 只能校正线性失配（增益/时延/频响），不应强行拟合 jitter/量化/静态非线性。
    # 否则优化会产生“投机”滤波：局部压 spur 但整体带内谱形被破坏（典型表现：低频崩盘、高频看似变好）。
    if use_ref:
        p0_train = dict(p0)
        p1_train = dict(p1)
        if _env_bool("STAGE12_TRAIN_DISABLE_NONLINEAR", True):
            p0_train["hd2"] = 0.0
            p0_train["hd3"] = 0.0
            p1_train["hd2"] = 0.0
            p1_train["hd3"] = 0.0
        if _env_bool("STAGE12_TRAIN_DISABLE_JITTER", True):
            p0_train["jitter_std"] = 0.0
            p1_train["jitter_std"] = 0.0
        n_bits_train: Optional[int] = None if _env_bool("STAGE12_TRAIN_DISABLE_QUANT", True) else ADC_NBITS
    else:
        p0_train = p0
        p1_train = p1
        n_bits_train = ADC_NBITS

    def _make_batch_loss(
        *,
        kmin_override: Optional[int] = None,
        kmax_override: Optional[int] = None,
        img_scale: float = 1.0,
        stratify_mid_k: Optional[int] = None,
        spec_scale: float = 1.0,
        ref_scale: float = 1.0,
        low_protect_scale: float = 0.0,
    ) -> torch.Tensor:
        loss_acc = 0.0
        kmin_use = int(k_min if kmin_override is None else kmin_override)
        kmax_use = int(k_max if kmax_override is None else kmax_override)

        def _loss_for_src(
            src_np: np.ndarray,
            *,
            k_funds_local: list[int],
            img_scale_local: float,
            spec_scale_local: float,
            ref_scale_local: float,
            low_protect_scale_local: float,
        ) -> torch.Tensor:
            """把“给定激励 + tone bins”跑一遍前向并计算 loss（核心目标不变）。"""
            # 更贴近现实：直接模拟两片 ADC 的采样（只在各自相位产生样本）
            c0, c1, pre = sim.capture_two_way(src_np, p0=p0_train, p1=p1_train, n_bits=n_bits_train)
            if use_ref and pref is not None:
                # 参考应尽量“理想”：不量化（否则会把量化误差当作拟合目标，导致噪声放大）
                c0r, c1r, ref = sim.capture_two_way(src_np, p0=pref, p1=pref, n_bits=None)
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
            # 主目标的 FFT：复用缓存的窗函数（win_full）
            win = win_full.to(dtype=pre_t.dtype)
            Y = torch.fft.rfft((yhat - torch.mean(yhat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
            P = (Y.real**2 + Y.imag**2)
            # 低频保护（主要用于 Stage0 训练 B(ω) 时避免破坏低频主带）：
            # 在低频段约束 yhat 的频谱不要偏离 pre_t（校正前），让 Stage1/2 再去负责低频对齐。
            l_low_protect = 0.0
            if float(low_protect_scale_local) > 0:
                lp_hz = _env_float("STAGE12_S0_LOW_PROTECT_HZ", 3.0e9)
                w_lp = _env_float("STAGE12_S0_W_LOW_PROTECT", 25.0)
                k_lp = int(np.ceil(float(lp_hz) * float(n) / float(fs)))
                k_lp = max(4, min(k_lp, int(Y.shape[-1]) - 1))
                Yi_lp = torch.fft.rfft((pre_t - torch.mean(pre_t, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
                num = torch.mean(torch.abs(Y[..., : k_lp + 1] - Yi_lp[..., : k_lp + 1]) ** 2)
                den = torch.mean(torch.abs(Yi_lp[..., : k_lp + 1]) ** 2) + 1e-12
                l_low_protect = float(w_lp) * (num / den)
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
                crop = 256
                l_ref_time = torch.mean((yhat[..., crop:-crop] - ref_t[..., crop:-crop]) ** 2)
                # ref_spec：只在“激励 tone 附近”对齐复谱，避免用 log-power MSE 去拟合噪声底（会导致低频崩盘）
                win = win_full.to(dtype=pre_t.dtype)
                Yh = torch.fft.rfft((yhat - torch.mean(yhat, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")
                Yr = torch.fft.rfft((ref_t - torch.mean(ref_t, dim=-1, keepdim=True)) * win, dim=-1, norm="ortho")

                def _band_sum(Yc: torch.Tensor, k0: int, g: int) -> torch.Tensor:
                    s0 = max(0, int(k0) - int(g))
                    e0 = min(int(Yc.shape[-1]), int(k0) + int(g) + 1)
                    return torch.sum(Yc[..., s0:e0], dim=-1)

                gspec = int(_env_int("STAGE12_REF_SPEC_GUARD", 2))
                # per-tone complex MSE (with optional high-frequency emphasis)
                use_wls = _env_bool("STAGE12_REF_SPEC_WLS", True)
                alpha_w = _env_float("STAGE12_REF_SPEC_WLS_ALPHA", 0.7)
                pow_w = _env_float("STAGE12_REF_SPEC_WLS_POWER", 2.0)
                Kmax = float(max(1, (n // 2 - 1)))
                errs = []
                for kk0 in k_funds_local:
                    kk0 = int(kk0)
                    bh = _band_sum(Yh, kk0, gspec)
                    br = _band_sum(Yr, kk0, gspec)
                    w = 1.0
                    if use_wls:
                        w = 1.0 + float(alpha_w) * ((float(kk0) / Kmax) ** float(max(0.0, pow_w)))
                    errs.append(torch.mean(torch.abs((bh - br) * float(w)) ** 2))
                l_ref_spec = torch.mean(torch.stack(errs)) if len(errs) else torch.mean(torch.abs(Yh - Yr) ** 2)
                l_delta = torch.mean((yhat - pre_t) ** 2)
                # 额外保护：全带谱形保持（排除 fund/img 附近），防止 B(ω) 为压镜像而“抬噪声/造 spur”
                k_rep = int(k_funds_local[len(k_funds_local) // 2]) if len(k_funds_local) else int(max(k_min, 8))
                l_spec_hold = _spec_hold_excluding_bins_loss(
                    yhat,
                    pre_t,
                    fund_bin=int(k_rep),
                    guard=int(guard),
                    extra_exclude=(list(k_funds_local) + list(k_imgs_local)),
                )
                loss_main = (
                    float(ref_scale_local) * (float(w_ref_time) * l_ref_time + float(w_ref_spec) * l_ref_spec)
                    + float(w_img) * float(img_scale_local) * l_img
                    + 0.2 * float(w_delta) * l_delta
                    + float(w_spec_hold) * float(spec_scale_local) * l_spec_hold
                )
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

            if float(low_protect_scale_local) > 0:
                loss_main = loss_main + float(low_protect_scale_local) * l_low_protect

            # FIR 正则：平滑 + 频响增益约束（防止噪声放大导致某段频率崩盘）
            if isinstance(model, PolyphaseStage12Model):
                if mimo_mode and getattr(model, "use_xcouple", False):
                    # MIMO 模式：核心就是 2×2 mix_fir，本段对其做平滑 + 频响约束，避免噪声放大/发散
                    wx = model.mix_fir.weight  # [2,2,T]
                    l_reg = 0.0
                    for oo in range(2):
                        for ii in range(2):
                            woi = wx[oo, ii, :].view(-1)
                            l_reg = l_reg + torch.mean((woi[1:] - woi[:-1]) ** 2)
                    l_reg = l_reg / 4.0
                    l_band = _mimo_fir_gain_band_loss(wx, n_fft=1024, diag_min_gain=fir_min_gain, diag_max_gain=fir_max_gain, x_max_gain=0.55)
                    w_x_l2 = _env_float("STAGE12_W_XCOUPLE_L2", 2e-3)
                    l_reg = l_reg + float(w_x_l2) * torch.mean(wx * wx)
                else:
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
                    _loss_for_src(
                        src_lo,
                        k_funds_local=k_lo,
                        img_scale_local=float(img_scale),
                        spec_scale_local=float(spec_scale),
                        ref_scale_local=float(ref_scale),
                        low_protect_scale_local=float(low_protect_scale),
                    )
                    + _loss_for_src(
                        src_hi,
                        k_funds_local=k_hi,
                        img_scale_local=float(img_scale),
                        spec_scale_local=float(spec_scale),
                        ref_scale_local=float(ref_scale),
                        low_protect_scale_local=float(low_protect_scale),
                    )
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
                            _loss_for_src(
                                src_hi0,
                                k_funds_local=[int(k_hi0)],
                                img_scale_local=float(img_scale),
                                spec_scale_local=float(spec_scale),
                                ref_scale_local=float(ref_scale),
                                low_protect_scale_local=float(low_protect_scale),
                            )
                            + _loss_for_src(
                                src_lo0,
                                k_funds_local=[int(k_lo0)],
                                img_scale_local=float(img_scale),
                                spec_scale_local=float(spec_scale),
                                ref_scale_local=float(ref_scale),
                                low_protect_scale_local=float(low_protect_scale),
                            )
                        )
                continue

            # fallback：原来的单次 multisine / 单音路径
            if use_multi:
                k_funds = _sample_k_set(int(kmin_use), int(kmax_use), n_tones=int(n_tones), min_sep=min_sep, n=int(n), avoid_image_collision=True)
                src_np = _multisine_np(n=n, bins=k_funds, amp=float(np.random.uniform(0.55, 0.92)))
                loss_acc = loss_acc + _loss_for_src(
                    src_np,
                    k_funds_local=k_funds,
                    img_scale_local=float(img_scale),
                    spec_scale_local=float(spec_scale),
                    ref_scale_local=float(ref_scale),
                    low_protect_scale_local=float(low_protect_scale),
                )
            else:
                k = _next_k()
                src = (torch.sin(2.0 * np.pi * (float(k) * fs / float(n)) * t) * float(np.random.uniform(0.4, 0.95))).detach().cpu().numpy().astype(np.float64)
                loss_acc = loss_acc + _loss_for_src(
                    src,
                    k_funds_local=[int(k)],
                    img_scale_local=float(img_scale),
                    spec_scale_local=float(spec_scale),
                    ref_scale_local=float(ref_scale),
                    low_protect_scale_local=float(low_protect_scale),
                )

        return loss_acc / float(max(int(batch), 1))

    print(
        f"=== Stage1/2 (polyphase={use_polyphase} | ref={use_ref} | train_both={train_both}) taps={taps} N={n} "
        f"f=[{fmin/1e9:.2f},{fmax/1e9:.2f}]GHz | sampler={sampler} | HIGH_BIAS={high_bias:g} ==="
    )
    # MIMO(ref-supervised)推荐走“Stage0->Stage1->Stage2”的 continuation：
    # - Stage0 先把 widely-linear 分支(B(ω)：img_cancel/mix_fir)压到位
    # - Stage1/2 再做主路 A(ω) 的 delay/gain/FIR
    # 这样职责更清晰，且不会出现“为了压某个镜像 spur 把低频主带谱形搞崩”的投机解。
    #
    # 如果确实想做“一锅端联合训练”（风险更高），可显式开启 STAGE12_MIMO_JOINT_TRAIN=1。
    if mimo_mode and isinstance(model, PolyphaseStage12Model) and _env_bool("STAGE12_MIMO_JOINT_TRAIN", False):
        if not getattr(model, "use_xcouple", False):
            raise RuntimeError("STAGE12_MIMO_LS=1 requires STAGE12_USE_XCOUPLE=1 (mix_fir enabled)")
        if getattr(model, "use_img_cancel", False) and _env_bool("STAGE12_MIMO_DISABLE_IMG_CANCEL", False):
            with torch.no_grad():
                model.img_fir.weight.zero_()
            for p in model.img_fir.parameters():
                p.requires_grad_(False)

        mimo_steps = _env_int("STAGE12_MIMO_STEPS", 900)
        mimo_lr = _env_float("STAGE12_MIMO_LR", 8e-4)
        mimo_clip = _env_float("STAGE12_MIMO_CLIP", 1.0)
        print(f"=== StageMIMO(JOINT): train A(ω)+B(ω) | steps={mimo_steps} lr={mimo_lr:g} ===")
        mimo_params: list[nn.Parameter] = []
        if getattr(model, "train_both", False):
            mimo_params += list(model.even.delay.parameters()) + [model.even.gain] + list(model.even.fir.parameters())
        mimo_params += list(model.odd.delay.parameters()) + [model.odd.gain] + list(model.odd.fir.parameters())
        if getattr(model, "use_xcouple", False):
            mimo_params += list(model.mix_fir.parameters())
        if getattr(model, "use_img_cancel", False):
            mimo_params += list(model.img_fir.parameters())
        optm = optim.Adam(mimo_params, lr=float(mimo_lr), betas=(0.5, 0.9))
        for step in range(int(mimo_steps)):
            optm.zero_grad()
            loss = _make_batch_loss(img_scale=1.0, spec_scale=1.0, stratify_mid_k=(int(n) // 4))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mimo_params, max_norm=float(mimo_clip))
            optm.step()
            if step % 150 == 0:
                print(f"StageMIMO(JOINT) step {step:4d}/{mimo_steps} | loss={loss.item():.3e}")
        return model, float(scale)
    # Phase-0：镜像抑制分支（B(ω)）
    # - grad：用自监督 loss 迭代训练 img_fir/mix_fir（原方案）
    # - mimo：用 2×2 MIMO-Wiener 闭式解快速估计 mix_fir，作为 Phase-0 的“强初始化/替代”
    phase0_mode = os.getenv("STAGE12_PHASE0_MODE", "grad").strip().lower()  # grad | mimo
    if phase0_mode not in ("grad", "mimo"):
        phase0_mode = "grad"

    if phase0_mode == "mimo" and isinstance(model, PolyphaseStage12Model) and (pref is not None) and getattr(model, "use_xcouple", False):
        # 路线2：快速闭式解（只写 mix_fir）。默认禁用 img_cancel，避免重复自由度导致不稳。
        mtaps = int(getattr(model, "mix_fir").weight.shape[-1])
        n_rec = _env_int("STAGE12_PHASE0_MIMO_REC", 12)
        mu = _env_float("STAGE12_PHASE0_MIMO_MU", 3e-6)
        wls_a = _env_float("STAGE12_PHASE0_MIMO_WLS_ALPHA", 1.0)
        wls_p = _env_float("STAGE12_PHASE0_MIMO_WLS_POWER", 2.0)
        gd = _env_int("STAGE12_PHASE0_MIMO_GUARD_DROP", 256)
        print(f"=== Stage0 (MIMO-Wiener init) mtaps={mtaps} rec={n_rec} mu={mu:g} WLS(alpha={wls_a:g},p={wls_p:g}) ===")
        w_mimo = solve_stage12_mimo_wiener_fir(
            sim,
            p0=p0,
            p1=p1,
            pref=pref,
            n_full=int(n),
            n_records=int(n_rec),
            fir_taps=int(mtaps),
            guard_drop=int(gd),
            reg_mu=float(mu),
            wls_alpha=float(wls_a),
            wls_power=float(wls_p),
        )
        with torch.no_grad():
            # Conv1d 是互相关口径，需要 time-reverse
            w_mimo_cc = w_mimo[..., ::-1].copy()
            model.mix_fir.weight.copy_(torch.tensor(w_mimo_cc, dtype=model.mix_fir.weight.dtype, device=model.mix_fir.weight.device))
            if getattr(model, "use_img_cancel", False) and _env_bool("STAGE12_PHASE0_MIMO_DISABLE_IMG_CANCEL", True):
                model.img_fir.weight.zero_()
                for p in model.img_fir.parameters():
                    p.requires_grad_(False)
            # 冻结 mix_fir：把它当作 B(ω) 固定基底，后续只训 A(ω)
            if _env_bool("STAGE12_FREEZE_IMG_AFTER_S0", True):
                for p in model.mix_fir.parameters():
                    p.requires_grad_(False)

    else:
        # 原 Phase-0：梯度训练 img_fir/mix_fir
        img_params: list[nn.Parameter] = []
        if isinstance(model, PolyphaseStage12Model):
            if getattr(model, "use_img_cancel", False):
                img_params += list(model.img_fir.parameters())
            if getattr(model, "use_xcouple", False):
                img_params += list(model.mix_fir.parameters())
        if len(img_params) > 0 and int(steps0) > 0:
            print("=== Stage0: Image-canceller / X-couple (grad) ===")
            s0_img_scale = _env_float("STAGE12_S0_IMG_SCALE", 2.6)
            s0_spec_scale = _env_float("STAGE12_S0_SPEC_SCALE", 0.25)
            # Stage0 主要训练 B(ω)（镜像抑制分支），ref 对齐只作为弱约束（否则容易牺牲低频谱形）
            s0_ref_scale = _env_float("STAGE12_S0_REF_SCALE", 0.15 if use_ref else 1.0)
            # Stage0：低频保护（默认开启），避免 B(ω) 为压高频镜像而破坏低频主带
            s0_low_protect = _env_float("STAGE12_S0_LOW_PROTECT_SCALE", 1.0 if use_ref else 0.6)
            opt0 = optim.Adam(img_params, lr=float(lr_img), betas=(0.5, 0.9))
            for step in range(int(steps0)):
                opt0.zero_grad()
                loss = _make_batch_loss(
                    img_scale=float(s0_img_scale),
                    spec_scale=float(s0_spec_scale),
                    ref_scale=float(s0_ref_scale),
                    low_protect_scale=float(s0_low_protect),
                    stratify_mid_k=(int(n) // 4),
                )
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


def train_tf_hybrid(
    sim: TIADCSimulator,
    *,
    device: str,
    p0_base: dict,
    p1_base: dict,
) -> DifferentiableSubbandCalibrator:
    """
    v0205 新方案训练入口：
    Analysis(STFT FB) -> Subband complex weights -> Synthesis(ISTFT FB)

    训练目标：盲校准（无参考），最小化 频谱稀疏 + 镜像杂散 + 能量保持 (+ 子带平滑正则)。

    -------------------------------------------------------------------------
    重要说明（为什么 v0205 “看起来容易跑差”）
    -------------------------------------------------------------------------
    这套 TF-Hybrid 是“无参考”的自监督校准：没有 ideal ref 波形可对齐，因此 loss 只能基于频谱结构来约束。
    这带来两个直接后果：
    1) loss 和最终我们关心的 SINAD/SFDR/ENOB 并不完全等价：
       - 本函数的 loss 主打“把能量集中在基波附近 + 压 2-way 镜像 spur + 能量保持”
       - 但 SINAD/SFDR 还高度受“噪声底/窗泄漏/谐波折叠”等影响
       => 训练可能学到“压镜像但抬噪声底”的解，导致 SINAD/SFDR 反而变差。
    2) 2-way TIADC 有一个很关键的“镜像 bin”关系：
       - 对 full-rate FFT 的基波 bin k，镜像近似在 k_img = |N/2 - k|（折叠到 rfft）
       - 如果 multisine 同时包含 k 与 k_img，那么“基波”和“镜像”目标会互相撞车（不可辨识/自相矛盾）
       => 训练会出现你看到的“保住一个频点、牺牲另一个频点”的掉点/尖峰现象。
       所以这里在采样 tone 集合时显式避免镜像撞车（见 _ok_add）。
    """
    # --------------------
    # 训练主超参（训练时长/步数/批量/学习率）
    # --------------------
    n = _env_int("TF_N", 16384)  # 每条训练样本长度（full-rate，20G 采样）
    # 默认值改为更“稳且全频段均匀”的配置（配合两阶段 fine-tune 使用）：
    # - 更小 lr + 更长 steps：避免后段频率被“学坏/反向优化”
    steps = _env_int("TF_STEPS", 1500)  # 训练迭代步数
    batch = _env_int("TF_BATCH", 4)  # batch size（每步生成 batch 条独立 multisine 样本）
    lr = _env_float("TF_LR", 1e-3)  # Adam 学习率
    grad_clip = _env_float("TF_CLIP", 1.0)  # 梯度裁剪（防止早期不稳定/爆炸）

    # --------------------
    # STFT 滤波器组配置（Analysis/Synthesis）
    # --------------------
    # n_fft/hop 决定了时频分辨率与重构误差（需满足 COLA 才能近似 PR）
    # - n_fft 越大：频率分辨率越高，但时域更长、对非平稳更敏感
    # - hop 越大：计算更快，但 overlap-add 误差与“块效应”风险增大
    n_fft = _env_int("TF_NFFT", 512)
    hop = _env_int("TF_HOP", 256)

    # --------------------
    # 子带参数化（把 STFT 的 F 个 bin 映射到 K 个子带）
    # --------------------
    # n_subbands 越大：每个子带更窄，能更精细地做幅相校正；但自由度更高，训练更难、也更容易过拟合/抬噪声。
    # 子带数默认提高到 128：中高频更容易做细粒度校正，避免“2G 后没效果/中频掉点”
    n_subbands = _env_int("TF_SUBBANDS", 256)

    # Controller 特征用到的粗 PSD bins（越大越能“看清”谱形，但 controller 更难学/更慢）
    psd_bins = _env_int("TF_PSD_BINS", 64)

    # --------------------
    # 架构开关
    # --------------------
    # use_controller=True：
    # - 不直接优化每个子带的权重 W[k]，而是用一个 MLP 从“当前谱形 + 温度”预测 W[k]（元参数生成）
    # - 好处：可以让 W 对温度/状态自适应；坏处：更容易不稳定，因此 controller 需要 identity 初始化（见 ControllerMLP）
    # controller 默认关闭：静态子带权重更稳，更容易做全带宽均匀校准；需要温漂自适应时再开启
    use_controller = _env_bool("TF_USE_CONTROLLER", False)

    # use_mimo=True：每个频点/子带用 M×M 复矩阵（允许通道间耦合）
    # - 对 2-way TIADC 来说能表达更一般的线性关系，但自由度更高，常需要更强约束与更长训练
    # - 【关键】：对于 >fs/4 的宽带信号，必须开启 MIMO 才能利用通道间相位差进行解混叠（区分 4G 和 6G）
    use_mimo = _env_bool("TF_USE_MIMO", True)

    # use_img_cancel=True：增加 wide-linear style 的 image-canceller 分支
    # - 对 2-way 的 fs/2 镜像 spur 往往更“对症”，比单纯调 W[k] 更容易压镜像
    # Modified: Disable to avoid interference with MIMO and simplify optimization.
    use_img_cancel = _env_bool("TF_USE_IMG_CANCEL", False)

    # loss weights
    # --------------------
    # loss 的权重（非常重要：决定“压镜像” vs “抬噪声” 的取舍）
    # --------------------
    # w_sparse（=leak）：频谱稀疏/泄漏损失。希望能量集中在基波附近，减少 spur/泄漏。
    # w_img：镜像 spur 抑制损失（2-way: k_img = |N/2 - k_fund|）
    # w_fund：基波能量保持/一致性（避免“把信号压扁”投机降低 leak/img）
    # w_energy：全局能量保持（避免投机解）
    # w_smooth：子带权重沿频率轴平滑正则（防止 W[k] 抖动导致时域振铃/抬噪声）
    # w_id：权重接近 identity 的正则（越大越保守，越不容易破坏原信号；但也更难学到强校正）
    w_sparse = _env_float("TF_W_SPARSE", 1.0)  # leak
    w_img = _env_float("TF_W_IMG", 10.0)       # Strong penalty on image
    w_fund = _env_float("TF_W_FUND", 10.0)     # EQUALLY Strong penalty on signal loss (Protect the signal!)
    w_energy = _env_float("TF_W_ENERGY", 2.0)  # Global energy preservation
    w_smooth = _env_float("TF_W_SMOOTH", 1e-5) # Slight smoothing to prevent overfitting to noise
    # identity 正则默认稍弱一些，避免中高频被“锁死”而看起来没效果
    w_id = _env_float("TF_W_ID", 1e-4)
    # NEW：噪声底/裙边损失（直接对齐 SINAD/SFDR 的关键项）
    # - skirt：基波附近（但不含基波窗内）的能量，常对应“裙边/旁瓣/调制边带”
    # - noise：排除基波与镜像邻域后的剩余功率（去掉 top-k spur 后近似噪声底）
    # Reduced significantly: Don't force the model to fix physical jitter noise (it can't).
    w_skirt = _env_float("TF_W_SKIRT", 0.1)
    w_noise = _env_float("TF_W_NOISE", 0.1)
    skirt_outer = _env_int("TF_SKIRT_OUTER", 24)
    noise_drop_topk = _env_int("TF_NOISE_DROP_TOPK", 64)

    # MIMO off-diagonal 约束（更物理）：低频强抑制耦合，高频放开
    # - 解决“fs/4 附近镜像对只能顾一头”的问题，同时避免低频被无谓耦合破坏
    # Modified: Set to 0.0 because we removed the off_gate and now rely on loss to determine coupling.
    w_mimo_off_reg = _env_float("TF_W_MIMO_OFF_REG", 0.0)
    mimo_off_f0 = _env_float("TF_MIMO_OFF_F0_NORM", 0.45)
    mimo_off_sharp = _env_float("TF_MIMO_OFF_SHARP", 14.0)

    # --------------------
    # 可选：加入“参考信号”的监督项（仅仿真可用；实测一般拿不到 ideal ref）
    # --------------------
    # use_ref=True 时，会额外生成一条“理想参考输出” y_ref（两路完全一致、无失配/无非线性/默认无抖动/无量化），
    # 并在 loss 中加入对齐到 y_ref 的时域/频域误差项，从而把欠定的盲校准问题变成“可辨识的系统辨识”。
    #
    # 经验上这会显著提升整体指标（尤其是 SINAD/SFDR），因为它直接约束了“噪声底/泄漏/相位”而不只是镜像比。
    # 默认开启参考监督（仿真里可用）：显著提升全带宽的稳定性与可辨识性。
    # 如需回到纯盲校准，设置 TF_USE_REF=0。
    use_ref = _env_bool("TF_USE_REF", True)
    w_ref_time = _env_float("TF_W_REF_TIME", 25.0)
    w_ref_spec = _env_float("TF_W_REF_SPEC", 8.0)
    ref_crop = _env_int("TF_REF_CROP", 512)  # 参考监督时丢弃边缘，避免 STFT center/pad 的边界影响
    # 参考监督时，盲校准项（leak/img/fund/energy）往往会与 ref 目标“重复甚至冲突”，
    # 训练更稳的做法通常是：
    # - 要么只用 ref（TF_REF_ONLY=1）
    # - 要么把盲校准项作为弱正则（TF_REF_BLEND_BLIND<1）
    ref_only = _env_bool("TF_REF_ONLY", False)
    # 参考监督下，盲校准项只做弱正则更稳
    # Modified: Increased to 1.0 to let explicit image/leak loss work at full strength alongside ref loss.
    ref_blind_scale = _env_float("TF_REF_BLEND_BLIND", 1.0)

    # reference ADC parameters (ideal-ish)
    # - cutoff 默认取两路中较大者（避免把 ref 变成额外的带宽限制）
    # - delay/gain/offset/hd2/hd3/jitter 默认全为 0（理想）
    # - n_bits 强制 None（float），作为“理想参考”
    ref_cut = _env_float(
        "TF_REF_CUTOFF_HZ",
        float(max(float(p0_base.get("cutoff_freq", 8.0e9)), float(p1_base.get("cutoff_freq", 8.0e9)))),
    )
    ref_jitter = _env_float("TF_REF_JITTER_STD", 0.0)
    pref = {
        "cutoff_freq": float(ref_cut),
        "delay_samples": 0.0,
        "gain": 1.0,
        "offset": 0.0,
        "jitter_std": float(ref_jitter),
        "hd2": 0.0,
        "hd3": 0.0,
        "snr_target": None,
    }

    # identity 正则的“边缘增强”：
    # - 频带两端（极低频/极高频）更容易被 STFT 边界效应/窗泄漏影响
    # - 也更容易出现“为了压某个 spur 把边缘整段弄坏”
    # => 在 band-edge 子带上更强地拉回 identity（edge_w 越大约束越强）
    id_edge_alpha = _env_float("TF_ID_EDGE_ALPHA", 2.0)
    id_edge_p = _env_float("TF_ID_EDGE_P", 2.0)

    # temperature simulation (for controller conditioning)
    # --------------------
    # 温漂模型（只用于训练时生成“条件变化”的数据，让 controller 学会补偿）
    # --------------------
    # 注意：这是一个“轻量线性”温漂模型，目的是提供变化维度，而不是完全物理准确。
    t_lo = _env_float("TF_TEMP_MIN_C", 20.0)
    t_hi = _env_float("TF_TEMP_MAX_C", 80.0)
    t_ref = _env_float("TF_TEMP_REF_C", 25.0)
    d_delay = _env_float("TF_TEMP_DELAY_DRIFT", 0.06)  # 满速采样点的延时漂移（samples @ full-rate）
    d_cut = _env_float("TF_TEMP_CUTOFF_DRIFT", -0.015)  # 截止频率相对漂移（relative）
    d_gain = _env_float("TF_TEMP_GAIN_DRIFT", 0.004)  # 增益相对漂移（relative）

    def _with_temp(p: dict, temp_c: float) -> dict:
        # very lightweight drift model: linear w.r.t temperature
        r = (float(temp_c) - float(t_ref)) / max(1e-9, float(t_hi) - float(t_lo))
        q = dict(p)
        if "delay_samples" in q:
            q["delay_samples"] = float(q["delay_samples"]) + float(d_delay) * r
        if "cutoff_freq" in q:
            q["cutoff_freq"] = float(q["cutoff_freq"]) * (1.0 + float(d_cut) * r)
        if "gain" in q:
            q["gain"] = float(q["gain"]) * (1.0 + float(d_gain) * r)
        return q

    model = DifferentiableSubbandCalibrator(
        n_ch=2,
        n_fft=int(n_fft),
        hop_length=int(hop),
        n_subbands=int(n_subbands),
        psd_feat_bins=int(psd_bins),
        use_controller=bool(use_controller),
        use_mimo=bool(use_mimo),
        use_img_cancel=bool(use_img_cancel),
    ).to(device)

    # 优化参数：
    # - 静态权重模式下（use_controller=0）：优化的是 (gain_log, phase) 这两张表
    # - controller 模式下（use_controller=1）：优化的是 controller MLP 的权重（由它生成 gain_log/phase）
    params = list(model.parameters())
    opt = optim.Adam(params, lr=float(lr), betas=(0.5, 0.9))

    # --------------------
    # 训练数据生成：相干 multisine + TIADC 仿真采样
    # --------------------
    # n_bits_train：
    # - True(默认)：训练用 float（不加量化噪声），让模型先学“主要结构”（线性失配/镜像）
    # - False：训练包含量化（更贴近真实，但更难学，且更容易被噪声主导导致目标不稳定）
    n_bits_train = None if _env_bool("TF_TRAIN_FLOAT", True) else ADC_NBITS

    # multisine 的 tone 数量：越多越“持久激励”，越能约束 W；但也越容易出现 tone 之间/镜像之间的冲突，需要避撞车。
    # tone 数默认提高：让 multisine 更接近“持久激励”，同时配合 low/mid/high 分层覆盖中频
    n_tones = _env_int("TF_N_TONES", 9)

    # tone 的 bin 范围（rfft 侧 [0..N/2]）；保留 guard 避免 DC/边缘数值问题
    kmin = _env_int("TF_KMIN", 8)
    kmax = _env_int("TF_KMAX", n // 2 - 8)

    # 高频偏置：<1 会偏向高频（u**high_bias 更靠近 1），用于让高频也经常被训练到
    high_bias = _env_float("TF_HIGH_BIAS", 0.3)

    # 盲校准 anchor：选择多少个“基波”去计算 k_img、keep-band 等（通常与 n_tones 一致即可）
    n_funds = _env_int("TF_TOP_FUNDS", max(1, n_tones))

    # 图像/keep-band 的 guard（以 bin 为单位，实际等价于“基波附近 +/- guard 的窗口”）
    img_guard = _env_int("TF_IMG_GUARD", 3)

    # 排除 DC/极低频 bins（用于 anchor 的 top-k 选择，避免 DC/低频噪声干扰）
    exclude_dc = _env_int("TF_EXCLUDE_DC", 6)

    # 训练激励模式：
    # - multisine（默认）：多音持久激励，适合“整体辨识”
    # - tone：单音（更贴近 evaluate 的单音扫频口径），通常更容易把全带宽指标拉到更均匀
    src_mode = os.getenv("TF_SRC_MODE", "multisine").strip().lower()  # multisine | tone
    tone_amp_lo = _env_float("TF_TONE_AMP_MIN", 0.55)
    tone_amp_hi = _env_float("TF_TONE_AMP_MAX", 0.92)
    # 单音采样偏置：<1 偏向高频；=1 近似均匀；>1 偏向低频
    tone_high_bias = _env_float("TF_TONE_HIGH_BIAS", float(high_bias))
    # 单音采样器：
    # - random：随机采样（可用 tone_high_bias 做高频偏置）
    # - cycle：循环覆盖整个 tone_pool，强制覆盖全频段（更利于“全带宽均匀”）
    # fine-tune 默认使用 cycle：强制覆盖全频段（尤其是 2~5GHz 这段最容易欠训练）
    tone_sampler = os.getenv("TF_TONE_SAMPLER", "cycle").strip().lower()  # random | cycle
    tone_match_eval = _env_bool("TF_TONE_MATCH_EVAL", True)
    tone_shuffle = _env_bool("TF_TONE_CYCLE_SHUFFLE", True)

    # tone_pool：用于 cycle 模式
    # 默认构造为“评估扫频点”对应的 coherent bins（与 evaluate() 口径一致），这样 fine-tune 更对齐最终指标。
    # 另外可通过 TF_TONE_MIN_HZ/TF_TONE_MAX_HZ 将 fine-tune 的频段聚焦在某一段（例如只补 2GHz 以上）。
    # fine-tune 的默认频段仍覆盖全带宽；如需“只补 2GHz 以上”，可显式设置 TF_TONE_MIN_HZ=2e9。
    tone_min_hz = _env_float("TF_TONE_MIN_HZ", 0.1e9)
    tone_max_hz = _env_float("TF_TONE_MAX_HZ", 6.6e9)

    tone_pool: np.ndarray
    _tone_idx = 0

    def _build_tone_pool(*, fmin_hz: float, fmax_hz: float) -> np.ndarray:
        f0 = float(max(0.0, fmin_hz))
        f1 = float(max(f0, fmax_hz))
        # 对 “中高频更难校正” 做采样加权（cycle 时等价于重复出现更多次）
        os_mid_min = _env_float("TF_TONE_OS_MID_MIN_HZ", 2.0e9)
        os_mid_max = _env_float("TF_TONE_OS_MID_MAX_HZ", 5.0e9)
        os_mid_fac = _env_int("TF_TONE_OS_MID_FACTOR", 3)
        os_hi_min = _env_float("TF_TONE_OS_HI_MIN_HZ", 5.0e9)
        os_hi_fac = _env_int("TF_TONE_OS_HI_FACTOR", 5)
        if bool(tone_match_eval):
            # 与 evaluate() 默认一致：0.1~6.6GHz step 0.2GHz
            f_targets = np.arange(0.1e9, 6.6e9, 0.2e9)
            ks_eval = []
            for f in f_targets:
                if not (f0 <= float(f) <= f1):
                    continue
                k0, _ = _coherent_bin(float(sim.fs), int(n), float(f))
                if int(kmin) <= int(k0) <= int(kmax):
                    ks_eval.append(int(k0))
            ks_eval = sorted(set(ks_eval))
            if len(ks_eval) == 0:
                return np.arange(int(kmin), int(kmax) + 1, dtype=np.int32)
            # oversample by band difficulty
            ks2: list[int] = []
            for kk in ks_eval:
                f = float(kk) * float(sim.fs) / float(n)
                rep = 1
                if float(os_mid_min) <= f <= float(os_mid_max):
                    rep = max(rep, int(os_mid_fac))
                if f >= float(os_hi_min):
                    rep = max(rep, int(os_hi_fac))
                for _ in range(int(max(1, rep))):
                    ks2.append(int(kk))
            return np.array(ks2, dtype=np.int32) if len(ks2) else np.array(ks_eval, dtype=np.int32)
        # 非 match_eval：直接用 bin 范围，并按 min/max Hz 过滤
        ks = []
        for kk in range(int(kmin), int(kmax) + 1):
            f = float(kk) * float(sim.fs) / float(n)
            if f0 <= f <= f1:
                ks.append(int(kk))
        return np.array(ks, dtype=np.int32) if len(ks) else np.arange(int(kmin), int(kmax) + 1, dtype=np.int32)

    def _reset_tone_pool(*, fmin_hz: float, fmax_hz: float) -> None:
        nonlocal _tone_idx, tone_pool
        tone_pool = _build_tone_pool(fmin_hz=float(fmin_hz), fmax_hz=float(fmax_hz))
        _tone_idx = 0
        if bool(tone_shuffle) and len(tone_pool) > 1:
            np.random.shuffle(tone_pool)

    # init tone_pool with current default range
    _reset_tone_pool(fmin_hz=float(tone_min_hz), fmax_hz=float(tone_max_hz))

    def _next_tone_k(*, bias: float) -> int:
        """tone 频点采样：random 或 cycle。"""
        nonlocal _tone_idx, tone_pool
        if str(tone_sampler) == "cycle" and len(tone_pool) > 0:
            kk = int(tone_pool[int(_tone_idx) % int(len(tone_pool))])
            _tone_idx += 1
            # 每轮覆盖一次 pool 后可重洗牌，避免每次都按同一顺序
            if bool(tone_shuffle) and (int(_tone_idx) % int(len(tone_pool)) == 0) and len(tone_pool) > 1:
                np.random.shuffle(tone_pool)
            return kk
        # random fallback
        return _sample_k_biased(int(kmin), int(kmax), high_bias=float(bias))

    print(
        f"=== v0205 TF-Hybrid | controller={use_controller} mimo={use_mimo} | N={n} steps={steps} batch={batch} "
        f"STFT(n_fft={n_fft},hop={hop}) subbands={n_subbands} psd_bins={psd_bins} ==="
    )

    stratify = _env_bool("TF_STRATIFY_BANDS", True)
    lo_max = _env_int("TF_LO_MAX_BIN", max(int(kmin), int(n) // 8))
    hi_min = _env_int("TF_HI_MIN_BIN", max(int(kmin), int(n) // 4))

    t_np = np.arange(int(n), dtype=np.float64)

    # --------------------
    # reference supervision helpers (scale-invariant)
    # --------------------
    def _crop_edges(x: torch.Tensor) -> torch.Tensor:
        c = int(max(0, ref_crop))
        if c <= 0:
            return x
        if x.shape[-1] <= 2 * c + 32:
            return x
        return x[..., c:-c]

    def _si_alpha(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        最小二乘标量对齐：alpha = argmin ||a - alpha*b||^2
        返回 alpha shape [B,1]
        """
        eps = 1e-12
        am = a - torch.mean(a, dim=-1, keepdim=True)
        bm = b - torch.mean(b, dim=-1, keepdim=True)
        num = torch.sum(am * bm, dim=-1, keepdim=True)
        den = torch.sum(bm * bm, dim=-1, keepdim=True).clamp_min(eps)
        return num / den

    def _si_time_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """scale-invariant time-domain MSE."""
        aa = _crop_edges(a)
        bb = _crop_edges(b)
        am = aa - torch.mean(aa, dim=-1, keepdim=True)
        bm = bb - torch.mean(bb, dim=-1, keepdim=True)
        alpha = _si_alpha(am, bm)
        e = am - alpha * bm
        return torch.mean(e * e)

    def _si_spec_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """scale-invariant spectral loss (relative complex L2)."""
        aa = _crop_edges(a)
        bb = _crop_edges(b)
        am = aa - torch.mean(aa, dim=-1, keepdim=True)
        bm = bb - torch.mean(bb, dim=-1, keepdim=True)
        alpha = _si_alpha(am, bm)  # [B,1]
        Lc = int(am.shape[-1])
        win = torch.hann_window(Lc, periodic=True, dtype=am.dtype, device=am.device).view(1, -1)
        A = torch.fft.rfft(am * win, dim=-1, norm="ortho")
        Bc = torch.fft.rfft(bm * win, dim=-1, norm="ortho")
        # alpha is real scalar; broadcast to complex
        alpha_c = alpha.to(dtype=A.real.dtype)
        E = A - alpha_c * Bc
        num = torch.mean((E.real * E.real + E.imag * E.imag))
        den = torch.mean((Bc.real * Bc.real + Bc.imag * Bc.imag)).clamp_min(1e-12)
        return num / den

    # ---------------------------------------------------------------------
    # 两阶段训练（可选）：先用 multisine 学整体，再用 tone fine-tune 贴合单音扫频
    # ---------------------------------------------------------------------
    # 默认开启单音 fine-tune：把训练分布对齐 evaluate() 的单音扫频口径，显著改善“2G 后没效果/掉点”
    finetune_steps = _env_int("TF_FINETUNE_TONE_STEPS", 800)
    finetune_lr = _env_float("TF_FINETUNE_LR", 1e-4)
    finetune_bias = _env_float("TF_FINETUNE_TONE_HIGH_BIAS", 1.0)  # fine-tune 默认更“均匀扫频”

    # 额外的高频补偿 fine-tune（可选；默认关闭）。
    # 通常先靠“cycle + oversample 高频”的单音 fine-tune 就能把 5~6.6GHz 拉起来；
    # 若还不够，再开 FT2 做更激进的高频补偿。
    finetune2_steps = _env_int("TF_FINETUNE2_TONE_STEPS", 800)
    finetune2_lr = _env_float("TF_FINETUNE2_LR", 5e-5)
    finetune2_fmin = _env_float("TF_FINETUNE2_TONE_MIN_HZ", 3.0e9)
    finetune2_fmax = _env_float("TF_FINETUNE2_TONE_MAX_HZ", 6.6e9)
    # 高频阶段的权重调整：更强调镜像抑制、适当放松 identity（尤其是 band-edge）。
    finetune2_w_img_scale = _env_float("TF_FINETUNE2_W_IMG_SCALE", 2.8)
    finetune2_w_id_scale = _env_float("TF_FINETUNE2_W_ID_SCALE", 0.7)
    finetune2_edge_alpha_scale = _env_float("TF_FINETUNE2_ID_EDGE_ALPHA_SCALE", 0.35)

    def _train_loop(
        *,
        n_steps: int,
        mode: str,
        tag: str,
        tone_fmin_hz: Optional[float] = None,
        tone_fmax_hz: Optional[float] = None,
        w_img_scale: float = 1.0,
        w_id_scale: float = 1.0,
        edge_alpha_scale: float = 1.0,
        lr_override: Optional[float] = None,
    ) -> None:
        # “保存最佳权重”以避免训练后段偶发发散导致指标变差：
        # - 尤其是 tone fine-tune，偶尔会遇到某些频点/幅度组合导致梯度把权重推离最佳点
        # - 这会让你看到“前面都很好，最后输出却变差/2G后没效果”的现象
        save_best = _env_bool("TF_SAVE_BEST", True)
        save_every = _env_int("TF_SAVE_BEST_EVERY", 25 if str(tag) == "main" else 10)
        best_loss = float("inf")
        best_state: Optional[dict] = None

        # set lr (in-place)
        for pg in opt.param_groups:
            if lr_override is not None:
                pg["lr"] = float(lr_override)
            else:
                pg["lr"] = float(lr) if str(tag) == "main" else float(finetune_lr)

        # reset tone_pool for this stage if requested
        if tone_fmin_hz is not None and tone_fmax_hz is not None:
            _reset_tone_pool(fmin_hz=float(tone_fmin_hz), fmax_hz=float(tone_fmax_hz))
        for step in range(int(n_steps)):
            # sample batch temperatures
            temps = np.random.uniform(float(t_lo), float(t_hi), size=(int(batch),)).astype(np.float64)

            y_full_list = []
            y_ref_list = []
            g0_list = []
            g1_list = []
            k_fund_list = []
            for b in range(int(batch)):
                # 生成激励 src（multisine 或 tone）
                if str(mode) == "tone":
                    # tone 模式：建议用 cycle+match_eval 强制覆盖全频段，提升 worst-case
                    kk = _next_tone_k(bias=float(finetune_bias))
                    amp = float(np.random.uniform(float(tone_amp_lo), float(tone_amp_hi)))
                    fin = float(kk) * float(sim.fs) / float(n)
                    src = sim.tone(fin, int(n), float(amp)).astype(np.float64, copy=False)
                    ks = [int(kk)]
                else:
                    ks = []
                    tries = 0
                    half = int(n) // 2
                    # 是否避免“镜像撞车”（同时选到 k 与 |N/2-k|）：
                    # - 纯盲校准时：必须避免，否则 loss_img/keep-band 会自相矛盾，训练掉点
                    # - 有参考监督时：可以允许（ref 提供可辨识目标），并且**建议允许**，
                    #   这样能让模型学会同时处理 (4.5,5.5) 这种 fs/4 附近的镜像成对频点，避免“修好一个、弄坏另一个”。
                    avoid_img_collision = _env_bool("TF_AVOID_IMAGE_COLLISION", (not bool(use_ref)))

                    def _ok_add(kk2: int) -> bool:
                        kk2 = int(kk2)
                        if not (int(kmin) <= kk2 <= int(kmax)):
                            return False
                        if any(abs(kk2 - int(x)) < 24 for x in ks):
                            return False
                        if bool(avoid_img_collision):
                            ki = abs(int(half) - kk2)
                            ki = max(1, min(int(ki), int(half) - 1))
                            if ki in ks:
                                return False
                            for x in ks:
                                kix = abs(int(half) - int(x))
                                kix = max(1, min(int(kix), int(half) - 1))
                                if kix == kk2:
                                    return False
                        return True

                    if stratify and int(n_tones) >= 2:
                        # 3 段分层采样（low / mid / high），避免出现“2~5GHz 训练不到”的频段空洞：
                        # - low: [kmin .. lo_max]
                        # - mid: (lo_max .. hi_min)  （若 hi_min > lo_max+1）
                        # - high:[hi_min .. kmax]
                        #
                        # 注意：旧版只做 low+high，会导致 lo_max~hi_min 之间的大段频率（例如 2.5~5GHz）欠训练，
                        # 从而出现你说的“2G 后几乎没效果/掉点”的现象。
                        lo0 = int(kmin)
                        lo1 = int(min(kmax, lo_max))
                        mid0 = int(max(kmin, lo1 + 1))
                        mid1 = int(min(kmax, hi_min - 1))
                        hi0 = int(max(kmin, hi_min))
                        hi1 = int(kmax)

                        # 分配 tone 数（尽量 low/mid/high 都有覆盖）
                        nt = int(n_tones)
                        if nt >= 3 and mid0 <= mid1 and hi0 <= hi1 and lo0 <= lo1:
                            n_lo = max(1, nt // 3)
                            n_mid = max(1, nt // 3)
                            n_hi = max(1, nt - n_lo - n_mid)
                            # low
                            while len(ks) < int(n_lo) and tries < 4000:
                                tries += 1
                                k_try = int(np.random.randint(int(lo0), int(lo1) + 1))
                                if _ok_add(k_try):
                                    ks.append(int(k_try))
                            # mid（尽量均匀，不必强高频偏置）
                            while len(ks) < int(n_lo + n_mid) and tries < 4000:
                                tries += 1
                                k_try = int(np.random.randint(int(mid0), int(mid1) + 1))
                                if _ok_add(k_try):
                                    ks.append(int(k_try))
                            # high（用 high_bias 偏向更高频）
                            while len(ks) < int(nt) and tries < 4000:
                                tries += 1
                                k_try = _sample_k_biased(int(hi0), int(hi1), high_bias=float(high_bias))
                                if _ok_add(k_try):
                                    ks.append(int(k_try))
                        else:
                            # fallback：两段分层（low+high）
                            n_hi = max(1, nt // 2)
                            n_lo = max(1, nt - n_hi)
                            while len(ks) < int(n_lo) and tries < 4000:
                                tries += 1
                                k_try = int(np.random.randint(int(lo0), int(lo1) + 1))
                                if _ok_add(k_try):
                                    ks.append(int(k_try))
                            while len(ks) < int(nt) and tries < 4000:
                                tries += 1
                                k_try = _sample_k_biased(int(hi0), int(hi1), high_bias=float(high_bias))
                                if _ok_add(k_try):
                                    ks.append(int(k_try))
                    else:
                        while len(ks) < int(n_tones) and tries < 4000:
                            tries += 1
                            k_try = _sample_k_biased(int(kmin), int(kmax), high_bias=float(high_bias))
                            if _ok_add(k_try):
                                ks.append(int(k_try))
                    if not ks:
                        ks = [int(kmin)]
                    amp = float(np.random.uniform(float(tone_amp_lo), float(tone_amp_hi)))
                    src = np.zeros(int(n), dtype=np.float64)
                    for k_tone in ks:
                        phi = float(np.random.uniform(0.0, 2.0 * np.pi))
                        src += np.sin(2.0 * np.pi * float(k_tone) * t_np / float(n) + phi)
                    src = (src / max(1, len(ks)) * float(amp)).astype(np.float64, copy=False)

                p0 = _with_temp(p0_base, float(temps[b]))
                p1 = _with_temp(p1_base, float(temps[b]))
                _, _, y_full = sim.capture_two_way(src, p0=p0, p1=p1, n_bits=n_bits_train)
                g0, g1 = _deinterleave_two_way(y_full)

                if bool(use_ref):
                    _, _, y_ref = sim.capture_two_way(src, p0=pref, p1=pref, n_bits=None)
                    s_ref = max(float(np.max(np.abs(y_ref))), 1e-12)
                    y_ref_list.append((y_ref / s_ref).astype(np.float32, copy=False))

                s = max(float(np.max(np.abs(y_full))), 1e-12)
                y_full_list.append((y_full / s).astype(np.float32, copy=False))
                g0_list.append((g0 / s).astype(np.float32, copy=False))
                g1_list.append((g1 / s).astype(np.float32, copy=False))
                k_fund_list.append(np.array(ks[: int(n_funds)], dtype=np.int64))

            y_in = torch.tensor(np.stack(y_full_list, axis=0), dtype=torch.float32, device=device)  # [B,L]
            y_ref_t = None
            if bool(use_ref) and len(y_ref_list) == len(y_full_list) and len(y_ref_list) > 0:
                y_ref_t = torch.tensor(np.stack(y_ref_list, axis=0), dtype=torch.float32, device=device)  # [B,L]
            g0 = torch.tensor(np.stack(g0_list, axis=0), dtype=torch.float32, device=device)
            g1 = torch.tensor(np.stack(g1_list, axis=0), dtype=torch.float32, device=device)
            temp_t = torch.tensor(temps.astype(np.float32), dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            y_hat, info = model([g0, g1], y_full_for_feat=y_in, temp_c=temp_t, return_params=True)

            k_fund_t = torch.tensor(np.stack(k_fund_list, axis=0), dtype=torch.long, device=device)  # [B,K]
            _, parts = blind_tiadc_anchor_loss(
                y_hat,
                y_in,
                n_funds=int(n_funds),
                guard=int(img_guard),
                exclude_dc=int(exclude_dc),
                skirt_outer=int(skirt_outer),
                noise_drop_topk=int(noise_drop_topk),
                k_fund_override=k_fund_t,
            )
            loss_leak = parts["loss_leak"]
            loss_img = parts["loss_img"]
            loss_fund = parts["loss_fund"]
            loss_energy = parts["loss_energy"]
            loss_skirt = parts.get("loss_skirt", torch.zeros((), device=device, dtype=torch.float32))
            loss_noise = parts.get("loss_noise", torch.zeros((), device=device, dtype=torch.float32))
            blind_loss = float(w_sparse) * loss_leak + float(w_img) * float(w_img_scale) * loss_img + float(w_fund) * loss_fund + float(w_energy) * loss_energy
            blind_loss = blind_loss + float(w_skirt) * loss_skirt + float(w_noise) * loss_noise
            if bool(use_ref):
                blind_loss = float(ref_blind_scale) * blind_loss
                if bool(ref_only):
                    blind_loss = blind_loss * 0.0
            loss = blind_loss

            loss_ref_t = torch.zeros((), device=device, dtype=torch.float32)
            loss_ref_f = torch.zeros((), device=device, dtype=torch.float32)
            if y_ref_t is not None:
                loss_ref_t = _si_time_loss(y_hat, y_ref_t)
                loss_ref_f = _si_spec_loss(y_hat, y_ref_t)
                loss = loss + float(w_ref_time) * loss_ref_t + float(w_ref_spec) * loss_ref_f

            gain_log = info["gain_log"]
            phase = info["phase"]
            if gain_log.dim() == 3:
                dg = gain_log[:, 1:, :] - gain_log[:, :-1, :]
                dp = phase[:, 1:, :] - phase[:, :-1, :]
                # Second derivative for phase (allow linear slope/delay, penalize curvature)
                d2p = dp[:, 1:, :] - dp[:, :-1, :]
            else:
                dg = gain_log[1:, :] - gain_log[:-1, :]
                dp = phase[1:, :] - phase[:-1, :]
                d2p = dp[1:, :] - dp[:-1, :]
            
            # Penalize gain slope (smooth gain) and phase curvature (linear phase)
            # Note: We relax phase 1st-derivative penalty because delay = constant slope.
            loss_smooth = torch.mean(dg * dg) + 0.5 * torch.mean(d2p * d2p)
            loss = loss + float(w_smooth) * loss_smooth

            # MIMO off-diagonal 正则：用 gain_log 的 off 项做“低频强抑制、高频放开”
            # 注意：这里用 subband index 的归一化近似频率（足够通用，且不依赖真实 fs）
            if bool(getattr(model, "use_mimo", False)) and float(w_mimo_off_reg) > 0.0:
                Ksb = int(model.n_subbands)
                kk = torch.linspace(0.0, 1.0, steps=Ksb, device=device, dtype=gain_log.dtype).view(1, Ksb, 1)
                gate = torch.sigmoid(float(mimo_off_sharp) * (kk - float(mimo_off_f0)))  # [1,K,1]
                allow = gate  # high-freq allow
                penal = (1.0 - allow)  # low-freq penalize
                # reshape gain_log to [...,K,M,M]
                Mch = int(getattr(model, "n_ch", 2))
                if gain_log.dim() == 3:
                    gmat = gain_log.view(int(gain_log.shape[0]), Ksb, Mch, Mch)
                else:
                    gmat = gain_log.view(1, Ksb, Mch, Mch)
                eye = torch.eye(Mch, device=device, dtype=gmat.dtype).view(1, 1, Mch, Mch)
                goff = torch.tanh(gmat) * (1.0 - eye)  # off-diagonal only
                loss_off = torch.mean((goff * goff) * (penal.view(1, Ksb, 1, 1) ** 2))
                loss = loss + float(w_mimo_off_reg) * loss_off

            Ksb = int(model.n_subbands)
            kk_sb = torch.arange(Ksb, device=device, dtype=torch.float32)
            c = max(1.0, float(Ksb - 1) / 2.0)
            edge_w = 1.0 + (float(id_edge_alpha) * float(edge_alpha_scale)) * (torch.abs(kk_sb - c) / c) ** float(max(0.0, id_edge_p))  # [K]
            if bool(getattr(model, "use_mimo", False)):
                # MIMO mode: gain_log is [..., K, M*M]
                # We want to penalize diagonal terms towards 0 (gain=1) strongly
                # But penalize off-diagonal terms towards 0 (gain=0) weakly or not at all
                # because off-diagonal terms are needed for de-aliasing.
                Mch = int(getattr(model, "n_ch", 2))
                if gain_log.dim() == 3:
                    # [B, K, M*M]
                    gmat = gain_log.view(int(gain_log.shape[0]), Ksb, Mch, Mch)
                    pmat = phase.view(int(phase.shape[0]), Ksb, Mch, Mch)
                else:
                    # [K, M*M]
                    gmat = gain_log.view(Ksb, Mch, Mch)
                    pmat = phase.view(Ksb, Mch, Mch)
                
                eye = torch.eye(Mch, device=device, dtype=gmat.dtype)
                if gmat.dim() == 4:
                    eye = eye.view(1, 1, Mch, Mch)
                else:
                    eye = eye.view(1, Mch, Mch)
                
                # Diagonal mask
                mask_diag = eye
                mask_off = 1.0 - eye
                
                # Apply edge weighting
                if gmat.dim() == 4:
                    ew = edge_w.view(1, Ksb, 1, 1)
                else:
                    ew = edge_w.view(Ksb, 1, 1)

                # Loss ID: strong on diagonal, weak on off-diagonal
                # Modified: REMOVED phase penalty. Absolute phase penalty prevents learning delay (linear phase).
                # We rely on loss_smooth to keep phase clean, and ref/blind loss to set the correct delay.
                loss_id_diag = torch.mean((gmat * mask_diag * ew) ** 2)
                loss_id_off = torch.mean((gmat * mask_off * ew) ** 2)
                
                # Use a much smaller weight for off-diagonal terms to allow coupling to grow
                loss_id = loss_id_diag + 0.01 * loss_id_off
            else:
                # Diag mode: standard logic
                if gain_log.dim() == 3:
                    gw = gain_log * edge_w.view(1, Ksb, 1)
                else:
                    gw = gain_log * edge_w.view(Ksb, 1)
                # Modified: Removed phase penalty here too
                loss_id = torch.mean(gw * gw)
            
            loss = loss + float(w_id) * float(w_id_scale) * loss_id

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))
            opt.step()

            # track best checkpoint (cheap cadence; copy to CPU to avoid GPU bloat)
            if bool(save_best) and int(save_every) > 0 and (step % int(save_every) == 0 or step == int(n_steps) - 1):
                ls_now = float(loss.detach().cpu())
                if ls_now < float(best_loss):
                    best_loss = float(ls_now)
                    st = model.state_dict()
                    best_state = {k: v.detach().cpu().clone() for k, v in st.items()}

            if step % 150 == 0 or step == int(n_steps) - 1:
                ls = float(loss.detach().cpu())
                if y_ref_t is None:
                    print(
                        f"TF[{tag}] step {step:4d}/{int(n_steps)} | loss={ls:.3e} | "
                        f"leak={float(loss_leak.detach().cpu()):.3e} img={float(loss_img.detach().cpu()):.3e} fund={float(loss_fund.detach().cpu()):.3e} "
                        f"energy={float(loss_energy.detach().cpu()):.3e} smooth={float(loss_smooth.detach().cpu()):.3e} id={float(loss_id.detach().cpu()):.3e}"
                    )
                else:
                    print(
                        f"TF[{tag}] step {step:4d}/{int(n_steps)} | loss={ls:.3e} | "
                        f"leak={float(loss_leak.detach().cpu()):.3e} img={float(loss_img.detach().cpu()):.3e} fund={float(loss_fund.detach().cpu()):.3e} "
                        f"energy={float(loss_energy.detach().cpu()):.3e} skirt={float(loss_skirt.detach().cpu()):.3e} noise={float(loss_noise.detach().cpu()):.3e} "
                        f"ref_t={float(loss_ref_t.detach().cpu()):.3e} ref_f={float(loss_ref_f.detach().cpu()):.3e} "
                        f"smooth={float(loss_smooth.detach().cpu()):.3e} id={float(loss_id.detach().cpu()):.3e}"
                    )

        # restore best weights for this stage
        if bool(save_best) and best_state is not None:
            model.load_state_dict(best_state, strict=True)

    # main stage
    _train_loop(n_steps=int(steps), mode=str(src_mode), tag="main")
    # fine-tune stage (optional)
    if int(finetune_steps) > 0:
        _train_loop(
            n_steps=int(finetune_steps),
            mode="tone",
            tag="ft",
            tone_fmin_hz=float(tone_min_hz),
            tone_fmax_hz=float(tone_max_hz),
        )
    # high-band fine-tune (optional)
    if int(finetune2_steps) > 0:
        _train_loop(
            n_steps=int(finetune2_steps),
            mode="tone",
            tag="ft2",
            tone_fmin_hz=float(finetune2_fmin),
            tone_fmax_hz=float(finetune2_fmax),
            w_img_scale=float(finetune2_w_img_scale),
            w_id_scale=float(finetune2_w_id_scale),
            edge_alpha_scale=float(finetune2_edge_alpha_scale),
            lr_override=float(finetune2_lr),
        )

    return model


def evaluate(sim: TIADCSimulator, *, calibrator: DifferentiableSubbandCalibrator, device: str, p0: dict, p1: dict) -> tuple[np.ndarray, dict]:
    # 评估扫频也必须相干采样，否则高频 spur 会被泄漏摊平，出现“4G后没效果”的假象
    fs = float(sim.fs)
    n_eval = 16384
    f_targets = np.arange(0.1e9, 6.6e9, 0.2e9)
    freqs = []
    for f in f_targets:
        _, fc = _coherent_bin(fs, n_eval, float(f))
        freqs.append(fc)
    freqs = np.array(freqs, dtype=np.float64)
    m = {k: [] for k in ["sinad_pre", "sinad_post", "enob_pre", "enob_post", "thd_pre", "thd_post", "sfdr_pre", "sfdr_post"]}

    calibrator.eval()
    for f in freqs:
        src = sim.tone(float(f), n_eval, 0.9)
        _, _, y_full = sim.capture_two_way(src, p0=p0, p1=p1, n_bits=ADC_NBITS)

        margin = 500
        pre = y_full[margin:-margin].astype(np.float64, copy=False)

        g0, g1 = _deinterleave_two_way(y_full)
        s = max(float(np.max(np.abs(y_full))), 1e-12)
        y_in = torch.tensor((y_full / s).astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
        g0t = torch.tensor((g0 / s).astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
        g1t = torch.tensor((g1 / s).astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
        temp0 = torch.tensor([25.0], dtype=torch.float32, device=device)  # eval默认固定温度
        with torch.no_grad():
            post_full = calibrator([g0t, g1t], y_full_for_feat=y_in, temp_c=temp0).detach().cpu().numpy().flatten() * float(s)
        post = post_full[margin:-margin].astype(np.float64, copy=False)

        s0, e0, t0, sf0 = calc_metrics(pre, sim.fs, float(f))
        s1, e1, t1, sf1 = calc_metrics(post, sim.fs, float(f))
        m["sinad_pre"].append(s0); m["sinad_post"].append(s1)
        m["enob_pre"].append(e0); m["enob_post"].append(e1)
        m["thd_pre"].append(t0); m["thd_post"].append(t1)
        m["sfdr_pre"].append(sf0); m["sfdr_post"].append(sf1)
    return freqs, m


def plot_metrics(freqs: np.ndarray, m: dict) -> None:
    fghz = freqs / 1e9
    plt.figure(figsize=(10, 13))
    plt.subplot(4, 1, 1)
    plt.plot(fghz, m["sinad_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["sinad_post"], "g-o", linewidth=2, label="Post-Calib (TF-Hybrid)")
    plt.title("SINAD Improvement"); plt.ylabel("dB"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 2)
    plt.plot(fghz, m["enob_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["enob_post"], "m-o", linewidth=2, label="Post-Calib (TF-Hybrid)")
    plt.title("ENOB Improvement"); plt.ylabel("Bits"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(fghz, m["thd_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["thd_post"], "b-o", linewidth=2, label="Post-Calib (TF-Hybrid)")
    plt.title("THD Comparison"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    plt.subplot(4, 1, 4)
    plt.plot(fghz, m["sfdr_pre"], color="#ff7f0e", linestyle="--", marker="o", alpha=0.55, label="Pre-Calib")
    plt.plot(fghz, m["sfdr_post"], "c-o", linewidth=2, label="Post-Calib (TF-Hybrid)")
    plt.title("SFDR Improvement"); plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    savefig("metrics_vs_freq")


def _summarize_metrics(freqs: np.ndarray, m: dict) -> None:
    def _arr(x):
        return np.asarray(x, dtype=np.float64)

    ds = _arr(m["sinad_post"]) - _arr(m["sinad_pre"])
    df = _arr(m["sfdr_post"]) - _arr(m["sfdr_pre"])
    de = _arr(m["enob_post"]) - _arr(m["enob_pre"])
    # THD: more negative is better, so "improvement" = pre - post
    dt = _arr(m["thd_pre"]) - _arr(m["thd_post"])
    print("\n=== 全频段汇总（Post - Pre）===")
    print(f"SINAD  mean={float(np.mean(ds)):+.2f} dB | worst={float(np.min(ds)):+.2f} dB")
    print(f"SFDR   mean={float(np.mean(df)):+.2f} dB | worst={float(np.min(df)):+.2f} dB")
    print(f"ENOB   mean={float(np.mean(de)):+.2f} bits | worst={float(np.min(de)):+.2f} bits")
    print(f"THD    mean={float(np.mean(dt)):+.2f} dB | worst={float(np.min(dt)):+.2f} dB  (正数表示更好)")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== v0205 TF-Hybrid (Analysis-Calib-Synth Filter Bank) ===")
    print(f"device={device} | TF_USE_CONTROLLER={_env_bool('TF_USE_CONTROLLER', False)}")

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

    calibrator = train_tf_hybrid(sim, device=device, p0_base=p0, p1_base=p1)

    freqs, m = evaluate(sim, calibrator=calibrator, device=device, p0=p0, p1=p1)

    print("\n=== 关键频点（Pre -> TF-Hybrid）===")
    for g in [0.1, 1.5, 3.0, 4.5, 5.5, 6.0, 6.5]:
        i = int(np.argmin(np.abs(freqs - g * 1e9)))
        print(
            f"f={freqs[i]/1e9:>4.1f}GHz | "
            f"SINAD {m['sinad_pre'][i]:>6.2f}->{m['sinad_post'][i]:>6.2f} dB | "
            f"THD {m['thd_pre'][i]:>7.2f}->{m['thd_post'][i]:>7.2f} dBc | "
            f"SFDR {m['sfdr_pre'][i]:>6.2f}->{m['sfdr_post'][i]:>6.2f} dBc"
        )

    # 频域瓶颈定位：分解 fund / image / harmonics / noise floor（dBc）
    if _env_bool("DIAG_BOTTLENECK", True):
        print("\n=== 频域瓶颈定位（dBc，相对基波；越低越好）===")
        for g in [4.5, 5.5, 6.5]:
            fin = float(g) * 1e9
            _, fin_c = _coherent_bin(float(sim.fs), 16384, float(fin))
            src = np.sin(2.0 * np.pi * fin_c * (np.arange(16384, dtype=np.float64) / float(sim.fs))) * 0.85
            _, _, pre_full = sim.capture_two_way(src, p0=p0, p1=p1, n_bits=ADC_NBITS)

            g0, g1 = _deinterleave_two_way(pre_full)
            s = max(float(np.max(np.abs(pre_full))), 1e-12)
            y_in = torch.tensor((pre_full / s).astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
            g0t = torch.tensor((g0 / s).astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
            g1t = torch.tensor((g1 / s).astype(np.float32), dtype=torch.float32, device=device).view(1, -1)
            temp0 = torch.tensor([25.0], dtype=torch.float32, device=device)
            with torch.no_grad():
                post_full = calibrator([g0t, g1t], y_full_for_feat=y_in, temp_c=temp0).detach().cpu().numpy().flatten() * float(s)

            b_pre = _spectral_breakdown_db(pre_full, fs=float(sim.fs), fin=fin_c, guard=5)
            b_post = _spectral_breakdown_db(post_full, fs=float(sim.fs), fin=fin_c, guard=5)
            print(
                f"f={fin_c/1e9:>4.2f}GHz | "
                f"img {b_pre['img_db']:>6.1f}->{b_post['img_db']:>6.1f} | "
                f"harm {b_pre['harm_db']:>6.1f}->{b_post['harm_db']:>6.1f} | "
                f"noise {b_pre['noise_db']:>6.1f}->{b_post['noise_db']:>6.1f}  (k_img={b_pre['k_img']})"
            )
    plot_metrics(freqs, m)
    _summarize_metrics(freqs, m)


if __name__ == "__main__":
    main()
