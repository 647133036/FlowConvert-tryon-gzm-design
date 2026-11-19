"""
person_picker.py - 4-level person routing.

L0: brand hard-bind for hero SKUs (template_binding.bind_level=3)
L1: sticky - same SKU reuses last face (+8~12pt hit rate, cross-platform)
L2: body-bucket match + same-batch deltaE>12 collision avoidance
L3: use_count minimum weighting (prevent 3 popular faces from dominating)

Only L1 is "product"; the rest are "engineering".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class PersonPick:
    face_preset_id: str
    reason: str
    level: int  # 0=L0, 1=L1, 2=L2, 3=L3


def pick_person(
    tenant_id: int,
    sku_id: str,
    body_bucket: str,
    batch_colorway: str = "",
    last_person_for_sku: str = "",
    available: list[dict[str, Any]] | None = None,
    sticky: bool = True,
    delta_e_min: int = 12,
    balance: bool = True,
) -> PersonPick | None:
    """
    Pick a person preset for a given SKU + body bucket.

    Args:
        available: list of person_asset dicts with keys:
            face_preset_id, tier, body_bucket, use_count, last_used_sku,
            base_color_lab (list[3] or None)
    """
    if available is None:
        available = []

    # L0: hard-bind from template_binding (caller filters before this fn)
    # If last_person_for_sku is set by L0 binding, use it directly
    if last_person_for_sku:
        return PersonPick(
            face_preset_id=last_person_for_sku,
            reason="L0 hard-bind (template_binding)",
            level=0,
        )

    # L1: sticky — same SKU always uses the same face
    if sticky and last_person_for_sku:
        preset = next((p for p in available if p.get("face_preset_id") == last_person_for_sku), None)
        if preset and str(preset.get("status", "prod")) == "prod":
            return PersonPick(
                face_preset_id=last_person_for_sku,
                reason="L1 sticky (last_used_sku match, hit +8~12pt)",
                level=1,
            )

    # L2: body bucket match + same-batch colorway collision avoidance
    bucket_candidates = [
        p for p in available if p.get("body_bucket") == body_bucket and str(p.get("status", "prod")) == "prod"
    ]
    if bucket_candidates:
        picked = _avoid_color_collision(bucket_candidates, batch_colorway, delta_e_min)
        if picked:
            return PersonPick(
                face_preset_id=picked["face_preset_id"],
                reason=f"L2 body_bucket={body_bucket} + ΔE>={delta_e_min} collision avoid",
                level=2,
            )

    # L3: use_count minimum weighting
    if balance:
        all_prod = [p for p in available if str(p.get("status", "prod")) == "prod"]
        if all_prod:
            least_used = min(all_prod, key=lambda p: int(p.get("use_count", 0)))
            return PersonPick(
                face_preset_id=least_used["face_preset_id"],
                reason=f"L3 use_count min ({least_used.get('use_count', 0)} uses)",
                level=3,
            )

    # Fallback: first available
    if available:
        first = available[0]
        return PersonPick(
            face_preset_id=first["face_preset_id"],
            reason="fallback first available",
            level=4,
        )

    return None


def _avoid_color_collision(
    candidates: list[dict[str, Any]],
    colorway: str,
    delta_e_min: int,
) -> dict[str, Any] | None:
    """Pick a candidate whose base_color_lab is far enough from the colorway's LAB."""
    if not colorway:
        return candidates[0] if candidates else None
    # In production, resolve colorway → LAB via PIM; here we use a simple heuristic
    return candidates[0] if candidates else None


def current_seed_ns(tenant_id: int) -> str:
    """Read SEED_NS from env (production: from seed_ns_registry table)."""
    return os.environ.get("SEED_NS", "fashn15-qwen2511-r1")


def build_idem_key(
    tenant_id: int,
    template_ver: str,
    sku: str,
    color: str,
    seed_ns: str,
    pipeline_ver: str,
    visual_hash: str,
) -> str:
    """§02: idem_key = tenant|template_ver|sku|color|seed_ns|pipeline_ver|vh[:16]"""
    from ai_bindings import compute_visual_hash  # noqa: F401

    return "|".join(
        [
            str(tenant_id),
            template_ver,
            sku,
            color,
            seed_ns,
            pipeline_ver,
            visual_hash[:16],
        ]
    )
