"""生成源 Provider 层(架构 §00 L3 / §03 幂等 / §06 成本 / §07 路由)。

模型是耗材, 路由是资产: 换 v1.6 或别家 v2, 业务表/审核台/成本曲线
一行不动, 只加一个 Provider 实现 + 一行注册 + 一条价格表(§07)。

- 幂等键(§03): 内容指纹(sha256) + 生成参数 + pipeline_ver。不含
  provider(可互换, 缓存对生成意图生效)、不含时间戳(含了等于没有)。
- Aidge(阿里 AiTryOn) Provider: 402 直落失败+熔断(§11 R5: 没钱≠抖动);
  429 读 Retry-After 退避; 5xx/超时 2^n 退避 <=3 次; 签名结果 URL
  立刻下载落盘, 不存 URL(§11 R3: 账在图没在)。
- 成本(§06): 价格表 env 可覆盖; 缓存命中 0 元(HIT 免单)。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

logger = logging.getLogger("atelier.tryon_provider")

PIPELINE_VERSION = os.environ.get("VTON_PIPELINE_VERSION", "0.2.0")

# §06 价格表(分/张, env 可覆盖): fashn_local 为 GPU 摊薄估算值。
PROVIDER_PRICES = {
    "fashn_local": int(os.environ.get("VTON_FASHN_LOCAL_COST_CENT", "5")),
    "aidge": int(os.environ.get("AIDGE_COST_CENT", "15")),
}
PROVIDER_ETA_SEC = {
    "fashn_local": 9,
    "aidge": 6,
}
# §07 注册表: 未实现的 Provider 也注册, 给出不可用原因(换引擎不动业务)。
UNIMPLEMENTED_PROVIDERS = {
    "fashn_api": "FASHN hosted API provider not implemented (local + aidge active)",
    "leap": "Leap provider not implemented",
    "volc": "Volc provider not implemented (scene/background swap)",
}

AIDGE_ENDPOINT = os.environ.get("AIDGE_ENDPOINT", "https://aidge.aliyuncs.com")
AIDGE_ACTION = "AiTryOn"
AIDGE_API_VERSION = os.environ.get("AIDGE_API_VERSION", "2026-04-28")
AIDGE_TIMEOUT = float(os.environ.get("AIDGE_TIMEOUT", "120"))
AIDGE_BREAKER_SECONDS = float(os.environ.get("AIDGE_BREAKER_SECONDS", "1800"))
AIDGE_MAX_RETRIES = max(0, int(os.environ.get("AIDGE_MAX_RETRIES", "3")))

# 内部 garment_type(upper/tunic/lower/dress/onesie) → Aidge ClothType 枚举。
# onesie/连体衣 Aidge 无对应枚举, 就近映射 dresses(整件单衣语义)。
AIDGE_CLOTH_TYPE = {
    "upper": "tops",
    "tunic": "tops",
    "tops": "tops",
    "lower": "bottoms",
    "bottoms": "bottoms",
    "dress": "dresses",
    "dresses": "dresses",
    "onesie": "dresses",
}

# ---------- 错误 ----------


class ProviderError(RuntimeError):
    def __init__(self, provider, message, *, http_status=None, retryable=False, retry_after=None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.http_status = http_status
        self.retryable = retryable
        self.retry_after = retry_after


# ---------- 熔断(§05: 402 直落 DEAD + 熔断; §11 R5) ----------

_BREAKERS: dict[str, dict] = {}


def _breaker_state(name: str) -> tuple[bool, str]:
    st = _BREAKERS.get(name)
    if not st:
        return False, ""
    if time.time() >= st["open_until"]:
        _BREAKERS.pop(name, None)
        return False, ""
    mins = int((st["open_until"] - time.time()) / 60) + 1
    return True, f"circuit open since {st['reason']} (~{mins}min left)"


def _trip_breaker(name: str, reason: str, seconds: float | None = None) -> None:
    _BREAKERS[name] = {
        "open_until": time.time() + (AIDGE_BREAKER_SECONDS if seconds is None else seconds),
        "reason": reason,
    }
    logger.error("provider %s circuit OPEN: %s", name, reason)


# ---------- 内容指纹(§02 garment_asset.sha256 思路) ----------

_SHA_MEMO: dict[tuple, str] = {}


def file_sha256(path: Path) -> str:
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    hit = _SHA_MEMO.get(key)
    if hit:
        return hit
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    dig = h.hexdigest()
    _SHA_MEMO[key] = dig
    return dig


# ---------- 幂等键(§03) ----------


def generation_mode() -> str:
    """mode = steps/guidance; 改生成参数 = 换出图, 必须进幂等键。"""
    steps = os.environ.get("VTON_FASHN_STEPS", "30")
    guidance = os.environ.get("VTON_FASHN_GUIDANCE", "1.5")
    return f"steps{steps}-cfg{guidance}"


def idem_key(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    background: str,
    long_side: int,
    platform: str,
    ext: str,
    seeds: list[int],
) -> str:
    """同人同衣同参数 ⇒ 同键。

    任何一个"变了就该重出"的要素都进键; provider 与时间戳刻意不进:
    provider 可互换(§03), 时间戳进键等于没有幂等。
    """
    parts = [
        "vton",
        PIPELINE_VERSION,
        file_sha256(base_path),
        file_sha256(garment_path),
        audience,
        garment_type,
        background,
        str(int(long_side)),
        platform or "none",
        ext,
        generation_mode(),
        "segfree1",
        ",".join(str(int(s)) for s in seeds),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# ---------- 路由(§07) ----------


def select_provider() -> str:
    v = (os.environ.get("VTON_PROVIDER", "auto") or "auto").lower().strip()
    return v or "auto"


def _aidge_api_key() -> str:
    return os.environ.get("AIDGE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""


def aidge_available() -> tuple[bool, str]:
    open_, reason = _breaker_state("aidge")
    if open_:
        return False, f"aidge {reason}"
    if not _aidge_api_key():
        return False, "AIDGE_API_KEY not set"
    if not os.environ.get("AIDGE_PUBLIC_BASE_URL", "").strip():
        return False, "AIDGE_PUBLIC_BASE_URL not set (Aidge fetches images via public URL)"
    return True, ""


def active_provider() -> str:
    sel = select_provider()
    if sel in ("aidge", "auto"):
        ok, reason = aidge_available()
        if ok:
            return "aidge"
        if sel == "aidge":
            logger.warning("VTON_PROVIDER=aidge but unavailable (%s); falling back to fashn_local", reason)
    return "fashn_local"


def provider_status() -> dict:
    aok, areason = aidge_available()
    return {
        "selected": select_provider(),
        "active": active_provider(),
        "pipeline_version": PIPELINE_VERSION,
        "aidge": {
            "available": aok,
            "reason": "" if aok else areason,
            "cost_cent": PROVIDER_PRICES["aidge"],
            "eta_sec": PROVIDER_ETA_SEC["aidge"],
        },
        "fashn_local": {
            "cost_cent": PROVIDER_PRICES["fashn_local"],
            "eta_sec": PROVIDER_ETA_SEC["fashn_local"],
        },
        "unimplemented": dict(UNIMPLEMENTED_PROVIDERS),
    }


def estimate_cost_cents(n_images: int = 1, provider: str | None = None) -> dict:
    p = provider if provider in PROVIDER_PRICES else active_provider()
    return {
        "provider": p,
        "images": int(n_images),
        "cost_cent": PROVIDER_PRICES.get(p, 0) * int(n_images),
        "eta_sec": PROVIDER_ETA_SEC.get(p, 0) * int(n_images),
    }


# ---------- Aidge 客户端 ----------


def _constrain_for_aidge(img: Image.Image) -> Image.Image:
    """Aidge 约束: 两边 256..2049px(超边缩到 2049, 短边不足 256 放大), <=10MB。"""
    w, h = img.size
    if max(w, h) > 2049:
        s = 2049 / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
        w, h = img.size
    if min(w, h) < 256:
        s = 256 / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
    return img


def _stage_for_aidge(img: Image.Image, tag: str) -> str:
    """落盘 uploads/ 并回公网 URL。

    平铺文件名(内容 sha1 前缀): 同图重投同文件, 秒传/去重(§02); 不建
    子目录, 兼容既有 uploads 清理逻辑。
    """
    from store import UPLOADS

    UPLOADS.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    raw = buf.getvalue()
    name = f"aidge_{tag}_{hashlib.sha1(raw).hexdigest()[:16]}.jpg"
    target = UPLOADS / name
    if not target.is_file():
        target.write_bytes(raw)
    base = os.environ.get("AIDGE_PUBLIC_BASE_URL", "").strip().rstrip("/")
    return f"{base}/files/uploads/{name}"


def _aidge_post(payload: dict) -> dict:
    req = urllib.request.Request(
        AIDGE_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_aidge_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-acs-action": AIDGE_ACTION,
            "x-acs-version": AIDGE_API_VERSION,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=AIDGE_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = e.read(512).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        msg = f"HTTP {e.code}: {body or e.reason}"
        if e.code == 429:
            ra = e.headers.get("Retry-After") or ""
            try:
                wait = float(ra)
            except (TypeError, ValueError):
                wait = 2.0
            raise ProviderError(
                "aidge", msg, http_status=429, retryable=True, retry_after=min(max(wait, 0.5), 30.0)
            ) from e
        if e.code >= 500:
            raise ProviderError("aidge", msg, http_status=e.code, retryable=True) from e
        raise ProviderError("aidge", msg, http_status=e.code) from e
    except urllib.error.URLError as e:
        raise ProviderError("aidge", f"network error: {e.reason}", retryable=True) from e
    except TimeoutError as e:
        raise ProviderError("aidge", "timeout", retryable=True) from e


def _aidge_request_with_retry(payload: dict) -> dict:
    attempts = 0
    while True:
        try:
            return _aidge_post(payload)
        except ProviderError as e:
            if e.http_status == 402:
                # §05/§11 R5: 没钱≠抖动, 直落失败 + 熔断, 绝不重试。
                _trip_breaker("aidge", "HTTP 402 payment required")
                raise
            if not e.retryable or attempts >= AIDGE_MAX_RETRIES:
                raise
            wait = e.retry_after if e.retry_after else float(2**attempts)
            wait = min(wait, 30.0)
            logger.warning("aidge retry %d/%d after %.1fs: %s", attempts + 1, AIDGE_MAX_RETRIES, wait, e)
            time.sleep(wait)
            attempts += 1


def _download_image(url: str) -> Image.Image:
    with urllib.request.urlopen(url, timeout=AIDGE_TIMEOUT) as resp:
        raw = resp.read()
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def aidge_generate(base_img: Image.Image, garment_img: Image.Image, garment_type: str) -> Image.Image:
    """阿里 AiTryOn 单图试穿(同步 1K)。

    返回 RGB 图, info 带 provider/provider_cost_cent; 引擎层共用同一条
    POSTFX 链(preserve/protect), 生成源无感(§08)。
    """
    cloth = AIDGE_CLOTH_TYPE.get((garment_type or "").lower())
    if not cloth:
        logger.warning("aidge: no ClothType mapping for %r; defaulting to tops", garment_type)
        cloth = "tops"
    person = _constrain_for_aidge(base_img.convert("RGB"))
    garment = _constrain_for_aidge(garment_img.convert("RGB"))
    payload = {
        "ModelImageUrl": _stage_for_aidge(person, "person"),
        "ClothImageUrl": [_stage_for_aidge(garment, "cloth")],
        "ClothType": cloth,
        "Resolution": "1K",
        "Mode": "lite",
    }
    data = _aidge_request_with_retry(payload)
    image_url = ((data.get("Data") or {}).get("ImageUrl") or "").strip()
    if not image_url:
        raise ProviderError("aidge", f"response missing Data.ImageUrl: {json.dumps(data)[:256]}")
    # §11 R3: 签名 URL 有过期时间, 必须立刻下载; 只存 URL = 账在图没在。
    img = _download_image(image_url).convert("RGB")
    img.info["provider"] = "aidge"
    img.info["provider_cost_cent"] = PROVIDER_PRICES["aidge"]
    img.info["provider_req_id"] = str(data.get("RequestId") or "")
    return img
