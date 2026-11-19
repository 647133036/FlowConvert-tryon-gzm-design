"""合规标准件测试: 平台规则集 / 检测原语 / 报告结构 / worker 落库。"""

from pathlib import Path

import compliance as comp
import numpy as np
from PIL import Image
from platform_presets import PLATFORM_PRESETS


def test_rules_for_registered_platforms():
    for p in ("shopify", "amazon", "aliexpress", "tiktok", "independent_site"):
        r = comp.rules_for(p)
        assert r is not None, f"no rules for {p}"
        assert r.platform == p


def test_rules_for_unknown_platform():
    assert comp.rules_for("nope") is None


def test_white_bg_ratio_pure_white():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    assert comp.white_bg_ratio(img) > 0.99


def test_white_bg_ratio_non_white():
    img = Image.new("RGB", (100, 100), (100, 150, 200))
    assert comp.white_bg_ratio(img) < 0.10


def test_edge_watermark_scan_clean():
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    assert comp.edge_watermark_scan(img) < 0.01


def test_edge_watermark_scan_dirty():
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[:20, :, :] = 50  # top 20 rows dark = watermark
    img = Image.fromarray(arr)
    assert comp.edge_watermark_scan(img) > 0.05


def test_compliance_report_pass():
    """纯白底 + 小体积 + 1:1 → amazon 合规通过。"""
    img = Image.new("RGB", (1000, 1000), (255, 255, 255))
    buf = Path("/tmp/_comp_test_white.jpg")
    img.save(buf, "JPEG", quality=94)
    report = comp.run_compliance_check(img, "amazon", buf)
    assert report.overall in ("pass", "warning"), f"expected pass/warning, got {report.overall}: {report.summary}"
    d = report.to_dict()
    assert d["platform"] == "amazon"
    assert isinstance(d["checks"], list)
    assert len(d["checks"]) >= 3


def test_compliance_report_fail_non_white():
    """非白底图 + 强制白底平台 → fail。"""
    img = Image.new("RGB", (1000, 1000), (50, 50, 50))
    buf = Path("/tmp/_comp_test_dark.jpg")
    img.save(buf, "JPEG", quality=94)
    report = comp.run_compliance_check(img, "amazon", buf)
    assert report.overall == "fail"
    failed = [c for c in report.checks if c.severity == "fail"]
    assert any(c.rule == "white_bg" for c in failed)


def test_compliance_report_no_rules_passes():
    """未注册平台 → pass(不阻断)。"""
    img = Image.new("RGB", (500, 500), (255, 255, 255))
    buf = Path("/tmp/_comp_test_unknown.jpg")
    img.save(buf, "JPEG", quality=94)
    report = comp.run_compliance_check(img, "unknown_platform", buf)
    assert report.overall == "pass"


def test_compliance_as_job_fields():
    img = Image.new("RGB", (1000, 1000), (255, 255, 255))
    buf = Path("/tmp/_comp_test_jobs.jpg")
    img.save(buf, "JPEG", quality=94)
    report = comp.run_compliance_check(img, "shopify", buf)
    fields = report.as_job_fields()
    assert "compliance_overall" in fields
    assert "compliance_summary" in fields
    assert "compliance_checks" in fields
    assert isinstance(fields["compliance_checks"], list)


def test_tiktok_independent_in_presets():
    assert "tiktok" in PLATFORM_PRESETS
    assert "independent_site" in PLATFORM_PRESETS
    assert PLATFORM_PRESETS["tiktok"]["force_white"] is False
    assert PLATFORM_PRESETS["independent_site"]["force_white"] is True


def test_amazon_force_white_stricter():
    r_amz = comp.rules_for("amazon")
    r_tiktok = comp.rules_for("tiktok")
    assert r_amz is not None and r_tiktok is not None
    assert r_amz.force_white_bg is True
    assert r_amz.white_bg_min_ratio > r_tiktok.white_bg_min_ratio
    assert r_tiktok.force_white_bg is False
