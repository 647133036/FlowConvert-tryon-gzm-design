"""aiBindings template schema tests — §02 design-doc contract."""

from ai_bindings import (
    Template,
    build_idempotency_key,
    compute_visual_hash,
    estimate_cost,
)

SAMPLE_TEMPLATE = {
    "v": 2,
    "templateVer": "AW26-KNIT-03",
    "pipelineVer": "v1.5.2-p1",
    "boards": [
        {"key": "main_1x1", "w": 800, "h": 800},
        {"key": "detail_3x4", "w": 750, "h": 1000},
    ],
    "aiBindings": {
        "defaults": {
            "mode": "Balanced",
            "seedLock": True,
            "segFree": True,
            "postfx": "auto_if_risk",
            "providerPolicy": "cheapest_capable",
        },
        "layers": {
            "mdl": {
                "role": "model",
                "asset": {"modelId": "M-ASIA-017", "verExpr": "latest_valid"},
                "bind": {"skuExpr": "row.sku.top", "colorway": "row.color"},
                "genArea": {"x": 0.08, "y": 0.05, "w": 0.84, "h": 0.90},
                "fallback": "M-ASIA-003",
            },
            "g_top": {
                "role": "garment.top",
                "srcExpr": "row.img.top",
                "categoryFrom": "row.spu.category",
                "forcePostfxIf": ["has_print", "stripe_energy>0.4"],
                "genArea": {"x": 0.10, "y": 0.15, "w": 0.60, "h": 0.50},
            },
            "price": {"role": "price", "lockAI": True, "textExpr": "¥{row.price_now}"},
            "disc": {
                "role": "badge",
                "lockAI": True,
                "text": "AI 生成效果图，请以实物为准",  # noqa: RUF001
                "required": True,
            },
        },
        "postfx": {
            "align": "tps+mask",
            "featherPx": 9,
            "blend": "luma_from_ai × chroma_from_flatlay",  # noqa: RUF001
            "edgeWrapShadow": 3,
            "deltaE_guard": 6,
        },
        "export": {
            "engine": "leafer-node",
            "scale": 2,
            "outputs": [{"board": "main_1x1", "fmt": "png", "name": "{sku}_主图"}],
            "aigcMark": {"implicit": True, "explicit": "corner_badge"},
        },
    },
}


def test_template_parse():
    tpl = Template.from_dict(SAMPLE_TEMPLATE)
    assert tpl.v == 2
    assert tpl.template_ver == "AW26-KNIT-03"
    assert tpl.pipeline_ver == "v1.5.2-p1"
    assert len(tpl.boards) == 2
    assert "mdl" in tpl.ai_bindings.layers
    assert tpl.ai_bindings.layers["mdl"].role == "model"
    assert tpl.ai_bindings.layers["mdl"].model_id == "M-ASIA-017"
    assert tpl.ai_bindings.layers["g_top"].src_expr == "row.img.top"
    assert tpl.ai_bindings.layers["price"].lock_ai is True
    assert tpl.ai_bindings.layers["disc"].text == "AI 生成效果图，请以实物为准"  # noqa: RUF001
    assert tpl.ai_bindings.defaults.seed_lock is True
    assert tpl.ai_bindings.postfx.delta_e_guard == 6
    assert tpl.ai_bindings.export.scale == 2


def test_template_validate_ok():
    tpl = Template.from_dict(SAMPLE_TEMPLATE)
    assert tpl.validate() == []


def test_template_validate_missing_gen_area():
    bad = dict(SAMPLE_TEMPLATE)
    bad["aiBindings"] = dict(SAMPLE_TEMPLATE["aiBindings"])
    bad["aiBindings"]["layers"] = {"g_top": {"role": "garment.top", "srcExpr": "row.img.top"}}
    tpl = Template.from_dict(bad)
    errors = tpl.validate()
    assert any("genArea" in e for e in errors)


def test_template_validate_unknown_role():
    bad = dict(SAMPLE_TEMPLATE)
    bad["aiBindings"] = dict(SAMPLE_TEMPLATE["aiBindings"])
    bad["aiBindings"]["layers"] = {"x": {"role": "alien"}}
    tpl = Template.from_dict(bad)
    errors = tpl.validate()
    assert any("unknown role" in e for e in errors)


def test_visual_hash_stable_on_price_edit():
    tpl = Template.from_dict(SAMPLE_TEMPLATE)
    h1 = compute_visual_hash(tpl)
    tpl2 = Template.from_dict(SAMPLE_TEMPLATE)
    tpl2.ai_bindings.layers["price"].text_expr = "¥{row.price_new}"
    h2 = compute_visual_hash(tpl2)
    assert h1 == h2


def test_visual_hash_changes_on_garment_edit():
    tpl = Template.from_dict(SAMPLE_TEMPLATE)
    h1 = compute_visual_hash(tpl)
    tpl2 = Template.from_dict(SAMPLE_TEMPLATE)
    tpl2.ai_bindings.layers["g_top"].raw["srcExpr"] = "row.img.new_top"
    h2 = compute_visual_hash(tpl2)
    assert h1 != h2


def test_idempotency_key_structure():
    tpl = Template.from_dict(SAMPLE_TEMPLATE)
    row = {"sku": "SKU-001", "color": "BLACK"}
    key = build_idempotency_key(tpl, row, "2026AW-v1")
    parts = key.split("|")
    assert parts[0] == "AW26-KNIT-03"
    assert parts[1] == "SKU-001"
    assert parts[2] == "BLACK"
    assert parts[3] == "2026AW-v1"
    assert parts[4] == "v1.5.2-p1"
    assert len(parts[5]) == 16


def test_estimate_cost_basic():
    result = estimate_cost(SAMPLE_TEMPLATE, [{"sku": "S1", "img_top": "a.png"}], "fashn_local")
    assert result["total"] == 1
    assert result["run"] == 1
    assert result["cent"] == 5
    assert result["provider"] == "fashn_local"
    assert result["blocked"] == []


def test_estimate_cost_blocks_missing_sku():
    result = estimate_cost(SAMPLE_TEMPLATE, [{"sku": ""}], "fashn_local")
    assert len(result["blocked"]) == 2
    assert any("sku" in b["reason"] for b in result["blocked"])


def test_estimate_cost_audge_pricing():
    result = estimate_cost(SAMPLE_TEMPLATE, [{"sku": "S1", "img_top": "a.png"}], "aidge")
    assert result["cent"] == 20
    assert result["eta_sec"] == 30
