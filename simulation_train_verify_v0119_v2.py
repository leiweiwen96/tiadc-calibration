import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")  # 保存图片到本地（兼容无显示环境）
import matplotlib.pyplot as plt
import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# ==============================================================================
# Plot 保存工具（替代 plt.show）
# ==============================================================================
_PLOT_DIR: Optional[Path] = None


def _get_plot_dir() -> Path:
    """
    统一的图片输出目录：<脚本目录>/plots/<运行时间戳>/
    """
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
    # Windows/跨平台安全：只保留常见字符，其余替换为下划线
    name = re.sub(r"[^0-9a-zA-Z._-]+", "_", name)
    return name.strip("_") or "figure"


def save_current_figure(name: str, *, dpi: int = 200, ext: str = "png") -> Path:
    """
    保存当前 figure 到本地，并自动 close 防止内存累积。

    文件名规则：YYYYMMDD_HHMMSS_mmm_<name>.<ext>
    目录规则：<脚本目录>/plots/<运行时间戳>/
    """
    out_dir = _get_plot_dir()
    ts_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒
    stem = _safe_stem(name)
    out_path = out_dir / f"{ts_ms}_{stem}.{ext}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved: {out_path}")
    return out_path

# ==============================================================================
# Post-EQ（交织后 LTI 均衡器）：把整体 TIADC 输出往“单位冲激/恒等系统”校准
# ------------------------------------------------------------------------------
# 背景：
# - 你希望移除 Stage3 的“非线性残差修整”，改成：
#   先 Stage1/2 做通道对齐（Ch1->Ch0），再对交织后输出做一次整体校准，
#   目标逼近“恒等系统”（等价于冲激响应≈δ[n]、相位/群延时对齐）。
#
# 这里的实现思路：
# - 在全速率交织输出 y 上设计一个 LTI FIR g，使得 (g * y) ≈ x_target
# - x_target 不是“全频带硬反卷积”，而是：
#   先把理想输入限制在带宽内（POST_EQ_BAND_HZ），带外不强行追求恒等，
#   否则会引起噪声放大/带外崩坏（典型 Wiener 反卷积问题）。
# - 使用频域维纳反卷积（一次闭式求解，不走 Adam），并给带内高频更大权重。
#
# 重要：
# - Post-EQ 是“交织后的整体校准”，是 LTI；它只能校整体幅相/群延时，
#   不会像 PTV/Volterra 那样区分 even/odd 非线性。但它足够用于“单位冲激/恒等”的线性目标验证。
# ==============================================================================


def tiadc_interleave_from_fullrate(c0_full: np.ndarray, c1_full: np.ndarray) -> np.ndarray:
    """
    把两路 20GSps 全速率序列交织成 20GSps TIADC 输出（包含抽取）：
    - 偶数点来自 c0_full[0::2]
    - 奇数点来自 c1_full[1::2]
    """
    s0 = c0_full[0::2]
    s1 = c1_full[1::2]
    L = min(len(s0), len(s1))
    out = np.zeros(2 * L, dtype=np.float64)
    out[0::2] = s0[:L]
    out[1::2] = s1[:L]
    return out


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

    # 1) 构造“干净”的交织输入
    y0_clean = simulator.apply_channel_effect(base_sig, jitter_std=0, n_bits=None, **params_ch0)
    y1_clean = simulator.apply_channel_effect(base_sig, jitter_std=0, n_bits=None, **params_ch1)
    with torch.no_grad():
        x1 = torch.FloatTensor(y1_clean / scale).view(1, 1, -1).to(device)
        y1_cal = model(x1).cpu().numpy().flatten() * scale

    y_tiadc = tiadc_interleave_from_fullrate(y0_clean, y1_cal)

    # 2) 构造 target
    # - 恒等口径：x_target = base_sig(带宽内)；更像“单位冲激/恒等系统校准”
    # - 参考口径：x_target = reference instrument capture；更像“有参考的 foreground calibration”
    if use_reference_target:
        params_ref = {
            "cutoff_freq": float(params_ch0["cutoff_freq"]),
            "delay_samples": 0.0,
            "gain": 1.0,
            "hd2": 0.0,
            "hd3": 0.0,
            "snr_target": float(reference_snr_db),
        }
        y_ref_clean = simulator.apply_channel_effect(base_sig, jitter_std=0, n_bits=None, **params_ref)
        x_target = y_ref_clean[: len(y_tiadc)].astype(np.float64, copy=False)

        # 让指标绘图里也能看到 reference baseline（与 Stage3 的 reference_params 复用同一字段）
        model.reference_params = params_ref
    else:
        b, a = signal.butter(6, float(band_hz) / (simulator.fs / 2), btype="low")
        x_ideal = signal.lfilter(b, a, base_sig.astype(np.float64))
        x_ideal = np.clip(x_ideal, -1.0, 1.0)
        x_target = x_ideal[: len(y_tiadc)]

    # 3) 设计 FIR 并挂载
    taps = int(taps)
    if taps % 2 == 0:
        taps += 1
    g = design_post_eq_fir_wiener(
        y=y_tiadc,
        x_target=x_target,
        fs_hz=simulator.fs,
        taps=taps,
        band_hz=band_hz,
        hf_weight=hf_weight,
        ridge=ridge,
    )
    attach_post_eq_fir(model, g=g, device=device)
    model.post_fir_meta = {
        "taps": int(taps),
        "band_hz": float(band_hz),
        "hf_weight": float(hf_weight),
        "ridge": float(ridge),
        "use_reference_target": bool(use_reference_target),
        "reference_snr_db": float(reference_snr_db),
    }
    print(f">> Post-EQ FIR ready: taps={taps}, band={band_hz/1e9:.2f}GHz, ridge={ridge:g}, hf_w={hf_weight:g}")

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
STAGE3_SCHEME = "odd_volterra_ls"  # 推荐：odd_poly_ls/odd_volterra_ls（更稳）；也可用 ptv_* 系列
STAGE3_SWEEP = False            # True 时：自动对比多种 scheme，选最优挂载
STAGE3_SWEEP_SCHEMES = ("odd_poly_ls", "odd_volterra_ls", "ptv_poly_ls", "ptv_volterra_ls", "ptv_poly", "ptv_volterra")
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
# [新增] Post-EQ 的训练目标：是否使用“参考仪器输出”作为 target
# - False: target = 低通后的 base_sig（恒等/单位冲激口径），容易出现“整体指标不动/带外恶化”
# - True : target = reference instrument capture（更符合“交织后给个参考，再做校正”的验证）
POST_EQ_USE_REFERENCE_TARGET = True
POST_EQ_REFERENCE_SNR_DB = 120.0  # 参考通道目标 SNR（dB），越大越接近“理想参考”

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
POST_EQ_TAPS = _env_int("POST_EQ_TAPS", POST_EQ_TAPS)
POST_EQ_RIDGE = _env_float("POST_EQ_RIDGE", POST_EQ_RIDGE)
POST_EQ_BAND_HZ = _env_float("POST_EQ_BAND_HZ", POST_EQ_BAND_HZ)
POST_EQ_HF_WEIGHT = _env_float("POST_EQ_HF_WEIGHT", POST_EQ_HF_WEIGHT)
POST_EQ_USE_REFERENCE_TARGET = _env_bool("POST_EQ_USE_REFERENCE_TARGET", POST_EQ_USE_REFERENCE_TARGET)
POST_EQ_REFERENCE_SNR_DB = _env_float("POST_EQ_REFERENCE_SNR_DB", POST_EQ_REFERENCE_SNR_DB)

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
    
    # 模拟真实的物理通道非线性
    params_ch0 = {'cutoff_freq': 6.0e9, 'delay_samples': 0.0, 'gain': 1.0, 'hd2': 1e-3, 'hd3': 1e-3}
    params_ch1 = {'cutoff_freq': 5.9e9, 'delay_samples': 0.42, 'gain': 0.98, 'hd2': 2e-3, 'hd3': 2e-3}
    
    print(f"开始相对校准训练: Mapping Ch1 -> Ch0")

    np.random.seed(42)
    white_noise = np.random.randn(N_train)
    b_src, a_src = signal.butter(6, 9.0e9/(simulator.fs/2), btype='low')
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

    def _apply_stage3_post(sig_post_lin: np.ndarray) -> np.ndarray:
        if not (hasattr(model, "post_linearizer") and model.post_linearizer is not None):
            return sig_post_lin
        s = max(float(np.max(np.abs(sig_post_lin))), 1e-12)
        with torch.no_grad():
            x = torch.FloatTensor(sig_post_lin / s).view(1, 1, -1).to(device)
            y = model.post_linearizer(x).cpu().numpy().flatten() * s
        return y

    def _quick_eval(in_freqs_hz, *, tone_n: int):
        # 返回 dict：线性与非线性后的“逐频点”与汇总（均值/最差下降）
        sin_lin, sf_lin, th_lin = [], [], []
        sin_nl, sf_nl, th_nl = [], [], []

        for f in in_freqs_hz:
            src = simulator.generate_tone_data(f, N=int(tone_n))
            y0 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch0)
            y1 = simulator.apply_channel_effect(src, jitter_std=0, n_bits=None, **params_ch1)
            with torch.no_grad():
                x1 = torch.FloatTensor(y1 / scale).view(1, 1, -1).to(device)
                y1c = model(x1).cpu().numpy().flatten() * scale

            sig_post_lin = _interleave_for_metric(y0, y1c)
            sig_post_nl = _apply_stage3_post(sig_post_lin)

            s1, _, t1, sf1 = _calc_spectrum_metrics_fast(sig_post_lin, simulator.fs, f)
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

        if base == "ptv_volterra":
            ddsp_model = DDSPVolterraNetwork(taps=STAGE3_TAPS).to(device)
        elif base == "ptv_poly":
            ddsp_model = PTVMemorylessPoly23().to(device)
        elif base == "odd_volterra":
            ddsp_model = OddOnlyVolterraNetwork(taps=STAGE3_TAPS).to(device)
        elif base == "odd_poly":
            ddsp_model = OddOnlyMemorylessPoly23().to(device)
        else:
            raise ValueError(f"Unknown scheme={scheme!r} (base={base!r})")

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
    # 若启用 Post-EQ，则默认不再训练 Stage3（避免两套后级互相“打架”）
    if (not ENABLE_POST_EQ) and ENABLE_STAGE3_NONLINEAR:
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

    # 模拟真实的物理通道（待校准对象）
    params_ch0 = {'cutoff_freq': 6.0e9, 'delay_samples': 0.0, 'gain': 1.0, 'hd2': 1e-3, 'hd3': 1e-3}
    params_ch1 = {'cutoff_freq': 5.9e9, 'delay_samples': 0.42, 'gain': 0.98, 'hd2': 2e-3, 'hd3': 2e-3}

    # 更接近真实仪器的参考通道（同带宽，低非线性、低抖动、高分辨率，但不是“完美无噪声”）
    params_ref = {
        'cutoff_freq': params_ch0['cutoff_freq'],
        'delay_samples': 0.0,
        'gain': 1.0,
        'hd2': 0.0,
        'hd3': 0.0,
        'snr_target': REF_SNR_DB,
    }

    print("开始绝对校准训练: Mapping (Ch0, Ch1) -> Reference Instrument")

    np.random.seed(42)
    white_noise = np.random.randn(N_train)
    b_src, a_src = signal.butter(6, 9.0e9 / (simulator.fs / 2), btype='low')
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
    
    test_freqs = np.arange(0.1e9, 9.6e9, 0.5e9)
    
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
        y_ch0 = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch0)
        # Ch1 (Bad)
        y_ch1_raw = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch1)
        
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

        # 先算 Post-EQ（交织后线性均衡）：只影响 *_post_eq，不与 Stage3 “后级非线性”混淆
        if hasattr(model, "post_fir") and model.post_fir is not None:
            sig_post_eq = apply_post_eq_fir(sig_post_lin, post_fir=model.post_fir, device=device)
        else:
            sig_post_eq = sig_post_lin

        # 再算 Post-NL（交织后非线性修整）：保持与训练口径一致（默认作用于 sig_post_lin）
        if hasattr(model, 'post_linearizer') and model.post_linearizer is not None:
            scale_pl = getattr(model, 'post_linearizer_scale', None)
            if scale_pl is None:
                scale_pl = max(np.max(np.abs(sig_post_lin)), 1e-12)
            sig_post_t = torch.FloatTensor(sig_post_lin / scale_pl).view(1, 1, -1).to(device)
            with torch.no_grad():
                sig_final_t = model.post_linearizer(sig_post_t)
            sig_post_nl = sig_final_t.cpu().numpy().flatten() * scale_pl
        else:
            sig_post_nl = sig_post_eq

        # 参考基线（可选）
        if p_ref is not None:
            y_ref = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ref)
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

    test_freqs = np.arange(0.1e9, 9.6e9, 0.5e9)
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
    
    ch0_real = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch0)
    ch1_bad  = sim.apply_channel_effect(src, jitter_std=0, n_bits=None, **p_ch1)
    
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
    print(f"ENABLE_POST_EQ={ENABLE_POST_EQ} | POST_EQ_USE_REFERENCE_TARGET={POST_EQ_USE_REFERENCE_TARGET} | POST_EQ_BAND_HZ={POST_EQ_BAND_HZ/1e9:.2f}GHz | POST_EQ_TAPS={POST_EQ_TAPS} | POST_EQ_RIDGE={POST_EQ_RIDGE:g}")
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
    for i in _pick([0.1, 2.1, 4.1, 5.1, 6.1, 8.1, 9.1]):
        fghz = test_freqs[i] / 1e9
        ref_str = ""
        if 'thd_ref' in m:
            ref_str = (
                f" | Ref: SINAD {m['sinad_ref'][i]:>6.2f} dB"
                f", THD {m['thd_ref'][i]:>7.2f} dBc"
                f", SFDR {m['sfdr_ref'][i]:>6.2f} dBc"
            )
        eq_str = ""
        if hasattr(model, "post_fir") and model.post_fir is not None:
            eq_str = (
                f" | Post-EQ: SINAD {m['sinad_post_eq'][i]:>6.2f} dB"
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
    if hasattr(model, "post_fir") and model.post_fir is not None:
        plt.plot(test_freqs/1e9, m['sinad_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-EQ')
    if 'sinad_post_nl' in m:
        plt.plot(test_freqs/1e9, m['sinad_post_nl'], 'b--o', alpha=0.85, label='Post-NL')
    plt.title("SINAD Improvement")
    plt.ylabel("dB"); plt.legend(); plt.grid(True)
    
    # 2. ENOB 对比
    plt.subplot(4,1,2)
    plt.plot(test_freqs/1e9, m['enob_pre'], 'r--o', alpha=0.5, label='Pre-Calib')
    plt.plot(test_freqs/1e9, m['enob_post'], 'm-o', linewidth=2, label='Post-Linear')
    if hasattr(model, "post_fir") and model.post_fir is not None:
        plt.plot(test_freqs/1e9, m['enob_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-EQ')
    if 'enob_post_nl' in m:
        plt.plot(test_freqs/1e9, m['enob_post_nl'], 'b--o', alpha=0.85, label='Post-NL')
    plt.title("ENOB Improvement")
    plt.ylabel("Bits"); plt.legend(); plt.grid(True)
    
    # 3. THD 对比
    plt.subplot(4,1,3)
    plt.plot(test_freqs/1e9, m['thd_pre'], 'r--o', alpha=0.5, label='Pre-Calib (Ch1 Raw)')
    plt.plot(test_freqs/1e9, m['thd_post'], 'b-o', linewidth=2, label='Post-Linear')
    if hasattr(model, "post_fir") and model.post_fir is not None:
        plt.plot(test_freqs/1e9, m['thd_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-EQ')
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
    if hasattr(model, "post_fir") and model.post_fir is not None:
        plt.plot(test_freqs/1e9, m['sfdr_post_eq'], color='#ff7f0e', marker='o', linewidth=2, label='Post-EQ')
    if 'sfdr_post_nl' in m:
        plt.plot(test_freqs/1e9, m['sfdr_post_nl'], 'b--o', alpha=0.85, label='Post-NL')
    plt.title("SFDR Improvement")

    plt.xlabel("Frequency (GHz)"); plt.ylabel("dBc"); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    save_current_figure("metrics_vs_freq")