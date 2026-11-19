"""电商合规标准件: 按平台规则集检测成品图合规性, 输出 compliance_report。

设计原则(标准件):
- 规则声明式: 每平台一条 ComplianceRules, 加平台只加一条规则集 + 注册, 不动引擎。
- 检测纯 CPU: 白底占比 / 边缘水印扫描 / 促销文字区 / 尺寸 / 体积, 无 GPU 依赖。
- 输出结构化: compliance_report 落 job, 前端/运营可逐条看"哪条没过 + 严重度"。
- 降级不阻断: 检测异常不影响出图(只记 warning), 合规不通过 ≠ 不出图, 只标 review。

严重度:
- pass: 全部通过
- warning: 非阻断(如白底占比 85%~95%, 建议但可放行)
- fail: 阻断(如强制白底平台白底 < 80%, 促销文字检测命中) → delivery_tier 强制 review

平台合规规则(来自各电商后台要求):
- shopify:  白底可选(非强制), 1:1, <=5MB, 无水印/促销文字
- amazon:   强制纯白底, 1:1, <=20MB, 主图禁促销文字/logo/非主体
- aliexpress: 强制白底, 主图 >=800x800, <=5MB, 无水印
- tiktok:   白底可选, 主图 1:1, 视频封面 3:4 / 9:16, 无水印
- independent_site: 白底可选, 无硬性尺寸, <=5MB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("atelier.compliance")

# ---------- 规则声明 ----------


@dataclass(frozen=True)
class ComplianceRules:
    """一个平台的合规规则集(声明式, 加平台只加一条)。"""

    platform: str
    min_side: int = 800
    force_white_bg: bool = False
    white_bg_min_ratio: float = 0.80
    white_bg_warn_ratio: float = 0.90
    max_bytes: int = 5 * 1024 * 1024
    ban_promo_text: bool = False
    ban_watermark: bool = True
    required_count: int = 1
    required_count_max: int = 99
    aspect: str = "1:1"


RULES: dict[str, ComplianceRules] = {
    "shopify": ComplianceRules(
        platform="shopify",
        min_side=1000,
        force_white_bg=False,
        white_bg_min_ratio=0.0,
        max_bytes=5 * 1024 * 1024,
        ban_promo_text=True,
        ban_watermark=True,
        aspect="1:1",
    ),
    "amazon": ComplianceRules(
        platform="amazon",
        min_side=1000,
        force_white_bg=True,
        white_bg_min_ratio=0.85,
        white_bg_warn_ratio=0.95,
        max_bytes=20 * 1024 * 1024,
        ban_promo_text=True,
        ban_watermark=True,
        required_count=1,
        required_count_max=8,
        aspect="1:1",
    ),
    "aliexpress": ComplianceRules(
        platform="aliexpress",
        min_side=800,
        force_white_bg=True,
        white_bg_min_ratio=0.80,
        white_bg_warn_ratio=0.90,
        max_bytes=int(4.5 * 1024 * 1024),
        ban_promo_text=True,
        ban_watermark=True,
        required_count=1,
        required_count_max=8,
        aspect="1:1",
    ),
    "tiktok": ComplianceRules(
        platform="tiktok",
        min_side=1000,
        force_white_bg=False,
        white_bg_min_ratio=0.0,
        max_bytes=5 * 1024 * 1024,
        ban_promo_text=False,
        ban_watermark=True,
        aspect="1:1",
    ),
    "independent_site": ComplianceRules(
        platform="independent_site",
        min_side=0,
        force_white_bg=False,
        white_bg_min_ratio=0.0,
        max_bytes=5 * 1024 * 1024,
        ban_promo_text=False,
        ban_watermark=False,
        aspect="auto",
    ),
}


def rules_for(platform: str) -> ComplianceRules | None:
    return RULES.get(platform)


# ---------- 检测原语(纯 CPU) ----------


def white_bg_ratio(img: Image.Image, threshold: int = 250) -> float:
    """四边 8px 带 + 对角采样, 计算纯白底占比(0-1)。"""
    arr = np.array(img.convert("RGB"), dtype=np.int16)
    h, w = arr.shape[:2]
    b = min(8, h // 4, w // 4)
    if b < 2:
        return 1.0
    band = np.concatenate(
        [
            arr[:b].reshape(-1, 3),
            arr[-b:].reshape(-1, 3),
            arr[:, :b].reshape(-1, 3),
            arr[:, -b:].reshape(-1, 3),
        ]
    )
    white = np.all(band >= threshold, axis=1)
    return float(white.mean())


def edge_watermark_scan(img: Image.Image, band: int = 16) -> float:
    """四边 band 带内非白像素占比(>0.05 视为有水印/logo/促销文字嫌疑)。"""
    arr = np.array(img.convert("RGB"), dtype=np.int16)
    h, w = arr.shape[:2]
    b = min(band, h // 4, w // 4)
    if b < 2:
        return 0.0
    band_arr = np.concatenate(
        [
            arr[:b].reshape(-1, 3),
            arr[-b:].reshape(-1, 3),
            arr[:, :b].reshape(-1, 3),
            arr[:, -b:].reshape(-1, 3),
        ]
    )
    non_white = np.any(band_arr < 240, axis=1)
    return float(non_white.mean())


def file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


# ---------- 合规检测(标准件核心) ----------


@dataclass
class ComplianceCheck:
    rule: str
    passed: bool
    severity: str  # "pass" / "warning" / "fail"
    detail: str = ""
    measured: float | None = None
    threshold: float | None = None


@dataclass
class ComplianceReport:
    platform: str
    overall: str  # "pass" / "warning" / "fail"
    checks: list[ComplianceCheck] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "overall": self.overall,
            "summary": self.summary,
            "checks": [
                {
                    "rule": c.rule,
                    "passed": c.passed,
                    "severity": c.severity,
                    "detail": c.detail,
                    "measured": c.measured,
                    "threshold": c.threshold,
                }
                for c in self.checks
            ],
        }

    def as_job_fields(self) -> dict:
        """落 job 的扁平字段(供前端/运营逐条看)。"""
        return {
            "compliance_overall": self.overall,
            "compliance_summary": self.summary,
            "compliance_checks": [
                {"rule": c.rule, "passed": c.passed, "severity": c.severity, "detail": c.detail} for c in self.checks
            ],
        }


def run_compliance_check(img: Image.Image, platform: str, saved_path: Path) -> ComplianceReport:
    """对一个已落盘成品图跑平台合规规则集, 返回结构化报告。

    检测异常不阻断(返回 warning 报告), 合规不通过 ≠ 不出图。
    """
    rules = rules_for(platform)
    if rules is None:
        return ComplianceReport(
            platform=platform,
            overall="pass",
            summary="No rules registered for this platform; pass by default.",
        )

    checks: list[ComplianceCheck] = []
    w, h = img.size
    nbytes = file_size_bytes(saved_path)

    # 1. 尺寸下限
    min_side_ok = min(w, h) >= rules.min_side
    checks.append(
        ComplianceCheck(
            rule="min_side",
            passed=min_side_ok,
            severity="pass" if min_side_ok else "fail",
            detail=f"{w}x{h} (min {rules.min_side}px)" if not min_side_ok else "ok",
            measured=float(min(w, h)),
            threshold=float(rules.min_side),
        )
    )

    # 2. 白底占比(强制白底平台)
    if rules.force_white_bg:
        wbr = white_bg_ratio(img)
        if wbr >= rules.white_bg_warn_ratio:
            sev = "pass"
        elif wbr >= rules.white_bg_min_ratio:
            sev = "warning"
        else:
            sev = "fail"
        checks.append(
            ComplianceCheck(
                rule="white_bg",
                passed=wbr >= rules.white_bg_min_ratio,
                severity=sev,
                detail=f"white ratio {wbr:.2f} (min {rules.white_bg_min_ratio}, warn {rules.white_bg_warn_ratio})",
                measured=wbr,
                threshold=rules.white_bg_min_ratio,
            )
        )

    # 3. 文件大小
    size_ok = nbytes <= rules.max_bytes
    checks.append(
        ComplianceCheck(
            rule="file_size",
            passed=size_ok,
            severity="pass" if size_ok else "fail",
            detail=f"{nbytes // 1024}KB / limit {rules.max_bytes // 1024}KB",
            measured=float(nbytes),
            threshold=float(rules.max_bytes),
        )
    )

    # 4. 边缘水印/促销文字扫描
    if rules.ban_watermark or rules.ban_promo_text:
        wm = edge_watermark_scan(img)
        wm_ok = wm < 0.05
        sev = "pass" if wm_ok else ("warning" if wm < 0.10 else "fail")
        label = "watermark/promo" if rules.ban_promo_text else "watermark"
        checks.append(
            ComplianceCheck(
                rule=f"edge_{label}",
                passed=wm_ok,
                severity=sev,
                detail=f"edge non-white ratio {wm:.3f} (threshold 0.05)",
                measured=wm,
                threshold=0.05,
            )
        )

    # 5. 宽高比
    if rules.aspect == "1:1":
        ratio = w / h if h else 0
        ratio_ok = 0.9 <= ratio <= 1.1
        checks.append(
            ComplianceCheck(
                rule="aspect_1_1",
                passed=ratio_ok,
                severity="pass" if ratio_ok else "warning",
                detail=f"{w}:{h} ratio={ratio:.2f}",
                measured=ratio,
                threshold=1.0,
            )
        )

    # 汇总
    has_fail = any(c.severity == "fail" for c in checks)
    has_warn = any(c.severity == "warning" for c in checks)
    overall = "fail" if has_fail else ("warning" if has_warn else "pass")
    failed_rules = [c.rule for c in checks if c.severity == "fail"]
    warned_rules = [c.rule for c in checks if c.severity == "warning"]
    summary = f"{platform}: {len(checks)} checks, fail={failed_rules or 'none'}, warn={warned_rules or 'none'}"
    return ComplianceReport(platform=platform, overall=overall, checks=checks, summary=summary)
