from __future__ import annotations

import io
import json
import time
import urllib.error
from pathlib import Path

import pytest
import tryon_provider as tp
from PIL import Image


def _img(size=(320, 480), color=(120, 60, 60)) -> Image.Image:
    return Image.new("RGB", size, color)


def _jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: dict | bytes):
        self._payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


# ---------- 幂等键(§03) ----------


def test_idem_key_deterministic_and_no_timestamp(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    for p in (a, b):
        _img().save(p, "JPEG")
    k1 = tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42])
    k2 = tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42])
    assert k1 == k2


def test_idem_key_changes_with_content_mode_seeds(tmp_path, monkeypatch):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _img().save(a, "JPEG")
    _img().save(b, "JPEG")
    base = tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42])
    # 内容指纹: 换图内容 → 换键
    _img(color=(10, 10, 10)).save(b, "JPEG")
    assert tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42]) != base
    # mode: 改 steps → 换键
    _img().save(b, "JPEG")
    monkeypatch.setenv("VTON_FASHN_STEPS", "50")
    assert tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42]) != base
    monkeypatch.setenv("VTON_FASHN_STEPS", "30")
    assert tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42]) == base
    # seed / 参数 → 换键
    assert tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [7]) != base
    assert tp.idem_key(a, b, "men", "upper", "white", 2048, "none", "jpg", [42]) != base


def test_pipeline_version_in_key(tmp_path, monkeypatch):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _img().save(a, "JPEG")
    _img().save(b, "JPEG")
    base = tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42])
    monkeypatch.setattr(tp, "PIPELINE_VERSION", "0.3.0")
    assert tp.idem_key(a, b, "women", "upper", "white", 2048, "none", "jpg", [42]) != base


def test_file_sha256_memo_and_change(tmp_path):
    p = tmp_path / "x.jpg"
    _img().save(p, "JPEG")
    s1 = tp.file_sha256(p)
    assert tp.file_sha256(p) == s1
    _img(color=(1, 2, 3)).save(p, "JPEG")
    assert tp.file_sha256(p) != s1


# ---------- 路由(§07) ----------


def test_active_provider_default_local(monkeypatch):
    monkeypatch.delenv("VTON_PROVIDER", raising=False)
    monkeypatch.delenv("AIDGE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("AIDGE_PUBLIC_BASE_URL", raising=False)
    tp._BREAKERS.clear()
    assert tp.active_provider() == "fashn_local"


def test_active_provider_aidge_when_configured(monkeypatch):
    tp._BREAKERS.clear()
    monkeypatch.setenv("VTON_PROVIDER", "aidge")
    monkeypatch.setenv("AIDGE_API_KEY", "sk-test")
    monkeypatch.setenv("AIDGE_PUBLIC_BASE_URL", "https://demo.example.com")
    assert tp.active_provider() == "aidge"


def test_active_provider_aidge_unconfigured_falls_back(monkeypatch):
    tp._BREAKERS.clear()
    monkeypatch.setenv("VTON_PROVIDER", "aidge")
    monkeypatch.delenv("AIDGE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("AIDGE_PUBLIC_BASE_URL", raising=False)
    assert tp.active_provider() == "fashn_local"


# ---------- Aidge 客户端 ----------


def _enable_aidge(monkeypatch, tmp_path):
    tp._BREAKERS.clear()
    monkeypatch.setenv("AIDGE_API_KEY", "sk-test")
    monkeypatch.setenv("AIDGE_PUBLIC_BASE_URL", "https://demo.example.com")


def test_aidge_constrain_size():
    big = tp._constrain_for_aidge(_img(size=(3000, 4000)))
    assert max(big.size) <= 2049
    small = tp._constrain_for_aidge(_img(size=(100, 150)))
    assert min(small.size) >= 256


def test_aidge_generate_happy_path(monkeypatch, tmp_path):
    import store

    monkeypatch.setattr(store, "UPLOADS", tmp_path)
    _enable_aidge(monkeypatch, tmp_path)
    out_img = _img(size=(512, 768), color=(9, 9, 9))
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            assert req.get_header("X-acs-action") == "AiTryOn"
            assert req.get_header("X-acs-version") == "2026-04-28"
            body = json.loads(req.data.decode("utf-8"))
            assert body["ClothType"] == "tops"
            assert body["Resolution"] == "1K"
            assert body["ModelImageUrl"].startswith("https://demo.example.com/files/uploads/aidge_person_")
            assert body["ClothImageUrl"][0].startswith("https://demo.example.com/files/uploads/aidge_cloth_")
            return _FakeResponse({"Data": {"ImageUrl": "https://oss.example.com/signed.jpg"}, "RequestId": "rid-1"})
        return _FakeResponse(_jpeg_bytes(out_img))

    monkeypatch.setattr(tp.urllib.request, "urlopen", fake_urlopen)
    img = tp.aidge_generate(_img(), _img(), "upper")
    assert img.mode == "RGB"
    assert img.info["provider"] == "aidge"
    assert img.info["provider_cost_cent"] == tp.PROVIDER_PRICES["aidge"]
    assert img.info["provider_req_id"] == "rid-1"
    staged = list(Path(tmp_path).glob("aidge_*.jpg"))
    assert len(staged) == 2


def test_aidge_429_honors_retry_after(monkeypatch, tmp_path):
    import store

    monkeypatch.setattr(store, "UPLOADS", tmp_path)
    _enable_aidge(monkeypatch, tmp_path)
    sleeps = []
    monkeypatch.setattr(tp.time, "sleep", lambda s: sleeps.append(s))
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if getattr(req, "data", None) is None:
            return _FakeResponse(_jpeg_bytes(_img(size=(512, 768), color=(5, 5, 5))))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                "u", 429, "Too Many Requests", {"Retry-After": "3"}, io.BytesIO(b"rate")
            )
        return _FakeResponse({"Data": {"ImageUrl": "https://oss.example.com/x.jpg"}})

    monkeypatch.setattr(tp.urllib.request, "urlopen", fake_urlopen)
    img = tp.aidge_generate(_img(), _img(), "lower")
    assert img.mode == "RGB"
    assert sleeps == [3.0]


def test_aidge_402_trips_breaker_no_retry(monkeypatch, tmp_path):
    import store

    monkeypatch.setattr(store, "UPLOADS", tmp_path)
    _enable_aidge(monkeypatch, tmp_path)
    attempts = []

    def fake_urlopen(req, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError("u", 402, "Payment Required", {}, io.BytesIO(b"no credit"))

    monkeypatch.setattr(tp.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(tp.time, "sleep", lambda s: pytest.fail("402 must not sleep/retry"))
    with pytest.raises(tp.ProviderError) as e:
        tp.aidge_generate(_img(), _img(), "upper")
    assert e.value.http_status == 402
    assert len(attempts) == 1
    ok, reason = tp.aidge_available()
    assert not ok and "402" in reason
    tp._BREAKERS.clear()


def test_aidge_5xx_backoff_retries_then_raises(monkeypatch, tmp_path):
    import store

    monkeypatch.setattr(store, "UPLOADS", tmp_path)
    _enable_aidge(monkeypatch, tmp_path)
    sleeps = []
    monkeypatch.setattr(tp.time, "sleep", lambda s: sleeps.append(s))

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError("u", 503, "Service Unavailable", {}, io.BytesIO(b"down"))

    monkeypatch.setattr(tp.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(tp.ProviderError):
        tp.aidge_generate(_img(), _img(), "dress")
    assert sleeps == [1.0, 2.0, 4.0]


def test_aidge_cloth_type_mapping():
    assert tp.AIDGE_CLOTH_TYPE["upper"] == "tops"
    assert tp.AIDGE_CLOTH_TYPE["tunic"] == "tops"
    assert tp.AIDGE_CLOTH_TYPE["lower"] == "bottoms"
    assert tp.AIDGE_CLOTH_TYPE["dress"] == "dresses"
    assert tp.AIDGE_CLOTH_TYPE["onesie"] == "dresses"


def test_estimate_cost_cents(monkeypatch):
    monkeypatch.delenv("VTON_PROVIDER", raising=False)
    monkeypatch.delenv("AIDGE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("AIDGE_PUBLIC_BASE_URL", raising=False)
    tp._BREAKERS.clear()
    est = tp.estimate_cost_cents(3, "fashn_local")
    assert est == {
        "provider": "fashn_local",
        "images": 3,
        "cost_cent": tp.PROVIDER_PRICES["fashn_local"] * 3,
        "eta_sec": tp.PROVIDER_ETA_SEC["fashn_local"] * 3,
    }
    est2 = tp.estimate_cost_cents(2, "aidge")
    assert est2["cost_cent"] == tp.PROVIDER_PRICES["aidge"] * 2


def test_breaker_expires():
    tp._BREAKERS.clear()
    tp._trip_breaker("aidge", "test", seconds=0.05)
    ok1, r1 = tp._breaker_state("aidge")
    assert ok1 and "test" in r1
    time.sleep(0.08)
    ok2, _r2 = tp._breaker_state("aidge")
    assert not ok2


def test_provider_status_shape(monkeypatch):
    monkeypatch.delenv("VTON_PROVIDER", raising=False)
    monkeypatch.delenv("AIDGE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("AIDGE_PUBLIC_BASE_URL", raising=False)
    tp._BREAKERS.clear()
    st = tp.provider_status()
    assert st["active"] == "fashn_local"
    assert "aidge" in st and "fashn_local" in st
    assert "unimplemented" in st and "fashn_api" in st["unimplemented"]
    assert st["pipeline_version"] == tp.PIPELINE_VERSION
