"""
postfx_anchor.py — r13 banded 参数驱动版。

LAB 分频 + 行带锚定 + 直方图锚定 + ΔE 守卫 + 边缘 wrap + 热区回写。
输入 576x864 原生, 输出同尺寸 + bbox; 放大交 SSR (leafer-node / CPU Lanczos)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class PostfxSettings:
    ab_mix: float = 0.80
    band_sobel_ksize: int = 3
    band_gain: float = 0.30
    band_max: float = 0.55
    feather_px: int = 8
    edge_wrap_px: int = 3
    hist_anchor: bool = True
    chroma_guard_de: int = 6

    @classmethod
    def from_env(cls) -> PostfxSettings:
        return cls(
            ab_mix=float(os.environ.get("POSTFX_LAB_AB_MIX", "0.80")),
            band_sobel_ksize=int(os.environ.get("POSTFX_BAND_SOBEL_KSIZE", "3")),
            band_gain=float(os.environ.get("POSTFX_BAND_GAIN", "0.30")),
            band_max=float(os.environ.get("POSTFX_BAND_MAX", "0.55")),
            feather_px=int(os.environ.get("POSTFX_MASK_FEATHER_PX", "8")),
            edge_wrap_px=int(os.environ.get("POSTFX_EDGE_WRAP_PX", "3")),
            hist_anchor=os.environ.get("POSTFX_HIST_ANCHOR", "true").lower() in ("true", "1", "yes"),
            chroma_guard_de=int(os.environ.get("POSTFX_CHROMA_GUARD_DELTA_E", "6")),
        )


@dataclass
class PfResult:
    img: np.ndarray
    qc_flags: list[str] = field(default_factory=list)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


def _hist_anchor(src: np.ndarray, ref: np.ndarray, m: np.ndarray) -> np.ndarray:
    """白场/肤色分位对齐 — 治 r12 灰罩暗斑的核心一步。"""
    out = src.astype(np.float32).copy()
    for c in range(3):
        mask = m > 0.1
        s = src[:, :, c][mask]
        r = ref[:, :, c][mask]
        if s.size < 100:
            continue
        lo_s, hi_s = np.percentile(s, [2, 98])
        lo_r, hi_r = np.percentile(r, [2, 98])
        gain = (hi_r - lo_r) / max(hi_s - lo_s, 1e-3)
        out[:, :, c] = (out[:, :, c] - lo_s) * gain + lo_r
    return np.clip(out, 0, 255)


def _band_weight(orig_gray: np.ndarray, ksize: int = 3, gain: float = 0.30, max_val: float = 0.55) -> np.ndarray:
    """Vertical-stripe weight: only trust vertical-over-horizontal gradients, avoid mistaking buttons for stripes."""
    sx = np.abs(cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=ksize))
    sy = np.abs(cv2.Sobel(orig_gray, cv2.CV_32F, 0, 1, ksize=ksize))
    band = np.clip(sx - 0.6 * sy, 0, None)
    band = cv2.GaussianBlur(band, (0, 0), 1.2)
    band = cv2.normalize(band, None, 0, 1, cv2.NORM_MINMAX)
    return np.clip(band * gain, 0, max_val)[..., None]


def postfx_anchor(
    raw: np.ndarray,
    orig: np.ndarray,
    mask: np.ndarray,
    settings: PostfxSettings | None = None,
) -> PfResult:
    """
    r13 banded POSTFX:
    ① 白场/肤色锚定 → ② LAB 分频 (L 取模型, ab 主要取原图)
    ③ 行带锚定 (竖纹强的像素更信任原图) → ④ ΔE 守卫
    ⑤ 边缘 wrap 阴影 → ⑥ 热区 + 质检旁路
    """
    s = settings or PostfxSettings.from_env()
    H, W = raw.shape[:2]
    flags: list[str] = []

    orig = cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur((mask.astype(np.float32) / 255), (0, 0), s.feather_px * 2 / 3)[..., None]
    m = np.clip(m, 0, 1)

    # ① 白场/肤色锚定
    raw_a = _hist_anchor(raw, orig, m) if s.hist_anchor else raw.astype(np.float32)

    # ② LAB 分频
    lab_r = cv2.cvtColor(np.uint8(np.clip(raw_a, 0, 255)), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_o = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab_mix = lab_o.copy()
    ab_mix[:, :, 1:] = lab_o[:, :, 1:] * s.ab_mix + lab_r[:, :, 1:] * (1 - s.ab_mix)
    base = cv2.cvtColor(np.uint8(np.clip(ab_mix, 0, 255)), cv2.COLOR_LAB2BGR).astype(np.float32)
    base[:, :, 0] = lab_r[:, :, 0]
    base = cv2.cvtColor(np.uint8(np.clip(base, 0, 255)), cv2.COLOR_LAB2BGR).astype(np.float32)

    # ③ 行带锚定
    w = _band_weight(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), s.band_sobel_ksize, s.band_gain, s.band_max)
    blend = raw_a * (1 - m) + (base * (1 - w) + orig.astype(np.float32) * w) * m

    # ④ ΔE 守卫
    dE = float(
        np.mean(np.sqrt(((lab_r[:, :, 1:] - lab_o[:, :, 1:]) ** 2).sum(axis=2))[m[:, :, 0] > 0.5])
        if (m[:, :, 0] > 0.5).any()
        else 0.0
    )
    if dE > s.chroma_guard_de:
        blend = raw_a * (1 - m) + base * m
        flags.append("CHROMA_GUARD")

    # ⑤ 边缘 wrap 阴影
    erode_kernel = np.ones((s.edge_wrap_px * 2 + 1,) * 2)
    er = cv2.erode((m[:, :, 0] > 0.5).astype(np.uint8), erode_kernel)
    ring = cv2.GaussianBlur(
        ((m[:, :, 0] > 0.5).astype(np.float32) - er),
        (0, 0),
        1.5,
    )[..., None]
    blend *= 1 - 0.18 * ring

    # ⑥ 热区 + 质检
    ys, xs = np.where(m[:, :, 0] > 0.5)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if xs.size else (0, 0, 0, 0)
    if dE > s.chroma_guard_de * 1.5:
        flags.append("COLOR_DRIFT")

    return PfResult(
        img=np.uint8(np.clip(blend, 0, 255)),
        qc_flags=flags,
        bbox=bbox,
    )
