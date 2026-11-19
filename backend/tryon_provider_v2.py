"""
TryOnProvider dual impl: fashn_local + fal API.

L3 garment-swap routing. L2 orchestrator calls L3 via Provider interface.
Frontend never touches Provider SDK; only recognizes the 6 S09 endpoints.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    success: bool
    output_url: str = ""
    raw_url: str = ""
    latency_ms: int = 0
    error: str = ""
    provider: str = ""
    req_id: str = ""


@dataclass
class ProviderConfig:
    fashn_local_url: str = ""
    fashn_local_timeout_s: int = 60
    fal_api_key: str = ""
    fal_model: str = "fal-ai/fashn/tryon/v1.5"
    fal_timeout_s: int = 90
    provider_default: str = "fashn_local"

    @classmethod
    def from_env(cls) -> ProviderConfig:
        return cls(
            fashn_local_url=os.environ.get("FASHN_LOCAL_URL", "http://127.0.0.1:7868"),
            fashn_local_timeout_s=int(os.environ.get("FASHN_LOCAL_TIMEOUT_S", "60")),
            fal_api_key=os.environ.get("FAL_API_KEY", ""),
            fal_model=os.environ.get("FAL_MODEL", "fal-ai/fashn/tryon/v1.5"),
            fal_timeout_s=int(os.environ.get("FAL_TIMEOUT_S", "90")),
            provider_default=os.environ.get("PROVIDER_DEFAULT", "fashn_local"),
        )


class TryOnProvider:
    """Base provider interface."""

    name: str = "base"

    def generate(
        self,
        person_image: str,
        garment_image: str,
        *,
        category: str = "tops",
        steps: int = 30,
        guidance: float = 1.5,
        seg_free: bool = True,
        seed: int = 42,
    ) -> ProviderResult:
        raise NotImplementedError


class FashnLocalProvider(TryOnProvider):
    """Local FASHN VTON v1.5 on :7868 (127.0.0.1 only, never exposed)."""

    name = "fashn_local"

    def __init__(self, config: ProviderConfig):
        self._url = config.fashn_local_url
        self._timeout = config.fashn_local_timeout_s

    def healthz(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self._url}/healthz", timeout=5) as r:
                return r.status == 200
        except OSError:
            return False

    def generate(
        self,
        person_image: str,
        garment_image: str,
        *,
        category: str = "tops",
        steps: int = 30,
        guidance: float = 1.5,
        seg_free: bool = True,
        seed: int = 42,
    ) -> ProviderResult:
        import json
        import urllib.request

        t0 = time.monotonic()
        payload = json.dumps(
            {
                "person": person_image,
                "garment": garment_image,
                "category": category,
                "steps": steps,
                "guidance": guidance,
                "seg_free": seg_free,
                "seed": seed,
            }
        ).encode()
        try:
            req = urllib.request.Request(
                f"{self._url}/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                body = json.loads(r.read())
                latency = int((time.monotonic() - t0) * 1000)
                return ProviderResult(
                    success=True,
                    output_url=str(body.get("output_url", "")),
                    raw_url=str(body.get("raw_url", "")),
                    latency_ms=latency,
                    provider=self.name,
                    req_id=str(body.get("req_id", "")),
                )
        except OSError as exc:
            latency = int((time.monotonic() - t0) * 1000)
            return ProviderResult(
                success=False,
                error=str(exc),
                latency_ms=latency,
                provider=self.name,
            )


class FalProvider(TryOnProvider):
    """fal.ai hosted FASHN ($0.075/img, 24h URL expiry — store files, not URLs)."""

    name = "fal"

    def __init__(self, config: ProviderConfig):
        self._key = config.fal_api_key
        self._model = config.fal_model
        self._timeout = config.fal_timeout_s

    def generate(
        self,
        person_image: str,
        garment_image: str,
        *,
        category: str = "tops",
        steps: int = 30,
        guidance: float = 1.5,
        seg_free: bool = True,
        seed: int = 42,
    ) -> ProviderResult:
        import json
        import urllib.request

        if not self._key:
            return ProviderResult(
                success=False,
                error="FAL_API_KEY not set",
                provider=self.name,
            )
        t0 = time.monotonic()
        payload = json.dumps(
            {
                "model": self._model,
                "person": person_image,
                "garment": garment_image,
                "category": category,
                "steps": steps,
                "guidance": guidance,
                "seg_free": seg_free,
                "seed": seed,
            }
        ).encode()
        try:
            req = urllib.request.Request(
                "https://fal.run/fal-ai/fashn/tryon/v1.5",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Key {self._key}",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                body = json.loads(r.read())
                latency = int((time.monotonic() - t0) * 1000)
                return ProviderResult(
                    success=True,
                    output_url=str(body.get("url", body.get("image", {}).get("url", ""))),
                    latency_ms=latency,
                    provider=self.name,
                    req_id=str(body.get("request_id", "")),
                )
        except OSError as exc:
            latency = int((time.monotonic() - t0) * 1000)
            return ProviderResult(
                success=False,
                error=str(exc),
                latency_ms=latency,
                provider=self.name,
            )


class ProviderRouter:
    """Routes to local first, falls back to fal on OOM/timeout/429."""

    def __init__(self, config: ProviderConfig | None = None):
        self._config = config or ProviderConfig.from_env()
        self._local = FashnLocalProvider(self._config)
        self._fal = FalProvider(self._config)
        self._fallback_reason: str = ""

    @property
    def fallback_reason(self) -> str:
        return self._fallback_reason

    def healthz(self) -> dict[str, bool]:
        return {"fashn_local": self._local.healthz(), "fal": bool(self._config.fal_api_key)}

    def generate(
        self,
        person_image: str,
        garment_image: str,
        *,
        category: str = "tops",
        steps: int = 30,
        guidance: float = 1.5,
        seg_free: bool = True,
        seed: int = 42,
        force_provider: str = "",
    ) -> ProviderResult:
        if force_provider == "fal":
            return self._fal.generate(
                person_image,
                garment_image,
                category=category,
                steps=steps,
                guidance=guidance,
                seg_free=seg_free,
                seed=seed,
            )

        result = self._local.generate(
            person_image,
            garment_image,
            category=category,
            steps=steps,
            guidance=guidance,
            seg_free=seg_free,
            seed=seed,
        )
        if result.success:
            return result

        # 402 is fatal — never retry
        if "402" in result.error:
            return result

        self._fallback_reason = result.error
        logger.warning("fashn_local failed (%s), falling back to fal", result.error)
        return self._fal.generate(
            person_image,
            garment_image,
            category=category,
            steps=steps,
            guidance=guidance,
            seg_free=seg_free,
            seed=seed,
        )
