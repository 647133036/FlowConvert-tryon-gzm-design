"""
aiBindings template schema — parses & validates the §02 design-doc JSON.

This is the L1 "排" layer's data contract. GzmDesign's `api.ts` is the
only fetch entry point; the orchestrator (L2) consumes these structures
to build idempotency keys, route providers, and compute cost estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AiDefaults:
    mode: str = "Balanced"
    seed_lock: bool = True
    seg_free: bool = True
    postfx: str = "auto_if_risk"
    provider_policy: str = "cheapest_capable"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AiDefaults:
        return cls(
            mode=str(raw.get("mode", "Balanced")),
            seed_lock=bool(raw.get("seedLock", True)),
            seg_free=bool(raw.get("segFree", True)),
            postfx=str(raw.get("postfx", "auto_if_risk")),
            provider_policy=str(raw.get("providerPolicy", "cheapest_capable")),
        )


@dataclass
class LayerBinding:
    role: str
    lock_ai: bool = False
    asset_id: str = ""
    model_id: str = ""
    ver_expr: str = ""
    sku_expr: str = ""
    colorway: str = ""
    src_expr: str = ""
    category_from: str = ""
    text_expr: str = ""
    text: str = ""
    required: bool = False
    on_missing: str = ""
    fallback: str = ""
    min_clear_space: int = 0
    tap_area_min: int = 0
    link_expr: str = ""
    emit: str = ""
    font: str = ""
    hide_if_no_discount: bool = False
    force_postfx_if: list[str] = field(default_factory=list)
    gen_area: dict[str, float] = field(default_factory=dict)
    from_key: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, key: str, raw: dict[str, Any]) -> LayerBinding:
        binding = cls(
            role=str(raw.get("role", key)),
            raw=raw,
        )
        if "asset" in raw:
            asset = raw["asset"]
            binding.model_id = str(asset.get("modelId", ""))
            binding.ver_expr = str(asset.get("verExpr", ""))
        if "bind" in raw:
            bind = raw["bind"]
            binding.sku_expr = str(bind.get("skuExpr", ""))
            binding.colorway = str(bind.get("colorway", ""))
        binding.src_expr = str(raw.get("srcExpr", ""))
        binding.category_from = str(raw.get("categoryFrom", ""))
        binding.lock_ai = bool(raw.get("lockAI", False))
        binding.asset_id = str(raw.get("assetId", ""))
        binding.text_expr = str(raw.get("textExpr", ""))
        binding.text = str(raw.get("text", ""))
        binding.required = bool(raw.get("required", False))
        binding.on_missing = str(raw.get("onMissing", ""))
        binding.fallback = str(raw.get("fallback", ""))
        binding.min_clear_space = int(raw.get("minClearSpace", 0))
        binding.tap_area_min = int(raw.get("tapAreaMin", 0))
        binding.link_expr = str(raw.get("linkExpr", ""))
        binding.emit = str(raw.get("emit", ""))
        rules = raw.get("rules")
        if isinstance(rules, dict):
            binding.font = str(rules.get("font", ""))
            binding.hide_if_no_discount = bool(rules.get("hideIfNoDiscount", False))
        binding.force_postfx_if = [str(x) for x in raw.get("forcePostfxIf", [])]
        binding.gen_area = {str(k): float(v) for k, v in raw.get("genArea", {}).items()}
        binding.from_key = str(raw.get("from", ""))
        return binding


@dataclass
class PostfxConfig:
    align: str = "tps+mask"
    feather_px: int = 9
    blend: str = ""
    edge_wrap_shadow: int = 3
    delta_e_guard: int = 6

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PostfxConfig:
        return cls(
            align=str(raw.get("align", "tps+mask")),
            feather_px=int(raw.get("featherPx", 9)),
            blend=str(raw.get("blend", "")),
            edge_wrap_shadow=int(raw.get("edgeWrapShadow", 3)),
            delta_e_guard=int(raw.get("deltaE_guard", 6)),
        )


@dataclass
class ExportConfig:
    engine: str = "leafer-node"
    scale: int = 2
    outputs: list[dict[str, Any]] = field(default_factory=list)
    aigc_mark: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExportConfig:
        return cls(
            engine=str(raw.get("engine", "leafer-node")),
            scale=int(raw.get("scale", 2)),
            outputs=[dict(x) for x in raw.get("outputs", [])],
            aigc_mark=dict(raw.get("aigcMark", {})),
        )


@dataclass
class AiBindings:
    defaults: AiDefaults = field(default_factory=AiDefaults)
    layers: dict[str, LayerBinding] = field(default_factory=dict)
    postfx: PostfxConfig = field(default_factory=PostfxConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AiBindings:
        defaults = AiDefaults.from_dict(raw.get("defaults", {}))
        layers = {}
        for key, val in raw.get("layers", {}).items():
            layers[key] = LayerBinding.from_dict(key, val)
        postfx = PostfxConfig.from_dict(raw.get("postfx", {}))
        export = ExportConfig.from_dict(raw.get("export", {}))
        return cls(defaults=defaults, layers=layers, postfx=postfx, export=export)


@dataclass
class Template:
    v: int = 2
    template_ver: str = ""
    pipeline_ver: str = ""
    boards: list[dict[str, Any]] = field(default_factory=list)
    ai_bindings: AiBindings = field(default_factory=AiBindings)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Template:
        ai_raw = raw.get("aiBindings", {})
        return cls(
            v=int(raw.get("v", 2)),
            template_ver=str(raw.get("templateVer", "")),
            pipeline_ver=str(raw.get("pipelineVer", "")),
            boards=[dict(b) for b in raw.get("boards", [])],
            ai_bindings=AiBindings.from_dict(ai_raw),
            raw=raw,
        )

    def validate(self) -> list[str]:
        """Return a list of validation errors; empty = OK."""
        errors: list[str] = []
        ab = self.ai_bindings
        if not ab.layers:
            errors.append("aiBindings.layers is empty")
        for key, layer in ab.layers.items():
            if layer.role not in ("model", "garment.top", "garment.bottom", "price", "logo", "badge", "hotspot"):
                errors.append(f"layer '{key}': unknown role '{layer.role}'")
            if layer.role in ("model", "garment.top", "garment.bottom") and not layer.gen_area:
                errors.append(f"layer '{key}': role={layer.role} requires genArea")
            if layer.role == "badge" and layer.required and not layer.text:
                errors.append(f"layer '{key}': required badge has no text")
        for board in self.boards:
            if "key" not in board or "w" not in board or "h" not in board:
                errors.append(f"board missing key/w/h: {board}")
        return errors


def compute_visual_hash(template: Template) -> str:
    """Hash only the model+garment subtree; price/logo edits don't break cache."""
    import hashlib
    import json

    subtree = {
        key: layer.raw
        for key, layer in template.ai_bindings.layers.items()
        if layer.role in ("model", "garment.top", "garment.bottom")
    }
    canonical = json.dumps(subtree, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_idempotency_key(template: Template, row: dict[str, Any], seed_ns: str) -> str:
    """
    §02: idem_key = tenant|template_ver|layer_key|sku_expr|colorway|seed_ns|pipeline_ver
    Only model/garment layers contribute to visual_hash (price edits don't invalidate).
    """
    vh = compute_visual_hash(template)
    sku = str(row.get("sku", ""))
    color = str(row.get("color", "STD"))
    parts = [
        template.template_ver,
        sku,
        color,
        seed_ns,
        template.pipeline_ver,
        vh[:16],
    ]
    return "|".join(parts)


def estimate_cost(template: Template, rows: list[dict[str, Any]], provider: str = "fashn_local") -> dict[str, Any]:
    """
    §09 POST /api/jobs/estimate — pure computation, no DB, no charge.
    Returns {total, hits, run, cent, eta_sec, provider, blocked[]}
    """
    per_image_cent = {
        "fashn_local": 5,
        "fashn_api": 7.5,
        "leap": 10.5,
        "aidge": 20,
        "viapi": 20,
        "third_party": 50,
    }
    per_sec = {
        "fashn_local": 8,
        "fashn_api": 12,
        "leap": 15,
        "aidge": 30,
        "viapi": 30,
        "third_party": 60,
    }
    rate = per_image_cent.get(provider, 5)
    eta = per_sec.get(provider, 8)

    total = len(rows)
    blocked: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if not row.get("sku"):
            blocked.append({"row": i, "reason": "sku 为空"})
        if not row.get("img_top") and not row.get("img_bottom"):
            blocked.append({"row": i, "reason": "无上衣图也无下装图"})

    run = total - len(blocked)
    cache_hits = 0
    cent = run * rate
    eta_sec = run * eta

    return {
        "total": total,
        "hits": cache_hits,
        "run": run,
        "cent": cent,
        "eta_sec": eta_sec,
        "provider": provider,
        "blocked": blocked,
    }
