import sys
import types

from PIL import Image


def test_fashn_category_mapping():
    from engine_fashn import fashn_category

    assert fashn_category("upper") == "tops"
    assert fashn_category("tunic") == "tops"
    assert fashn_category("lower") == "bottoms"
    assert fashn_category("dress") == "one-pieces"
    assert fashn_category("onesie") == "one-pieces"
    assert fashn_category("auto") == "tops"


def test_fashn_available_missing_weights(monkeypatch, tmp_path):
    import engine_fashn

    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path / "fashn")
    monkeypatch.setattr(engine_fashn, "_REPO_ROOT", tmp_path)
    avail, reason = engine_fashn.fashn_available()
    assert avail is False
    assert "Missing FASHN weights" in reason


def test_parser_stub_skips_real_human_parser(monkeypatch):
    import engine_fashn

    sys.modules.pop("fashn_human_parser", None)
    engine_fashn._ensure_parser_stub()
    stub = sys.modules["fashn_human_parser"]
    assert getattr(stub, "_atelier_commercial_stub", False) is True
    pred = stub.FashnHumanParser().predict(Image.new("RGB", (8, 12), (1, 2, 3)))
    assert pred.shape == (12, 8)
    assert int(pred.sum()) == 0


def test_parser_stub_satisfies_fashn_vton_v15_import_surface():
    """fashn_vton v1.5 模块导入期需要的全部名字(GPU 冒烟曾因缺 3 个名字炸在 import)。"""
    import engine_fashn

    sys.modules.pop("fashn_human_parser", None)
    engine_fashn._ensure_parser_stub()
    stub = sys.modules["fashn_human_parser"]
    # pipeline.py 导入: CATEGORY_TO_BODY_COVERAGE, FashnHumanParser
    for name in ("CATEGORY_TO_BODY_COVERAGE", "FashnHumanParser"):
        assert hasattr(stub, name), name
    # preprocessing/agnostic.py 导入: BODY_COVERAGE_TO_LABELS, IDENTITY_LABELS, LABELS_TO_IDS
    bctl = stub.BODY_COVERAGE_TO_LABELS
    assert set(bctl) == {"upper", "lower", "full"}
    assert set(bctl) == set(stub.CATEGORY_TO_BODY_COVERAGE.values())
    identity = tuple(stub.IDENTITY_LABELS)
    assert identity and all(isinstance(x, str) for x in identity)
    l2i = stub.LABELS_TO_IDS
    assert l2i and all(isinstance(v, int) for v in l2i.values())
    # 每个覆盖档的 label 必须都在 LABELS_TO_IDS 里(结构自洽)
    for labels in bctl.values():
        for label in labels:
            assert label in l2i, label
    for label in identity:
        assert label in l2i, label


def test_get_fashn_pipe_uses_stub_and_weights_dir(monkeypatch, tmp_path):
    import engine_fashn

    calls = {}

    class FakePipeline:
        def __init__(self, weights_dir):
            calls["weights_dir"] = weights_dir

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            img = Image.new("RGB", (16, 16), (4, 5, 6))
            return types.SimpleNamespace(images=[img])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})

    pipe = engine_fashn._get_fashn_pipe()
    assert calls["weights_dir"] == str(tmp_path)
    assert pipe is engine_fashn._get_fashn_pipe()
    assert getattr(sys.modules["fashn_human_parser"], "_atelier_commercial_stub", False)


def test_compose_tryon_fashn_calls_segmentation_free(monkeypatch, tmp_path):
    import engine_fashn
    from PIL import Image as PILImage

    calls = {}

    class FakePipeline:
        def __init__(self, weights_dir):
            self.weights_dir = weights_dir

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            return types.SimpleNamespace(images=[PILImage.new("RGB", (32, 32), (9, 9, 9))])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    monkeypatch.setenv("VTON_PATTERN_LOCK", "0")
    monkeypatch.setenv("VTON_WARP", "0")

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)
    PILImage.new("RGB", (40, 60), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 64, "white", "western_europe")
    kwargs = calls["kwargs"]
    assert kwargs["category"] == "one-pieces"
    assert kwargs["garment_photo_type"] == "flat-lay"
    assert kwargs["segmentation_free"] is True
    assert out.info["engine"] == "fashn-vton"
    assert out.info["fashn_category"] == "one-pieces"
    assert max(out.size) == 64
    assert out.info["postfx"] == "full"


def test_compose_tryon_fashn_postfx_raw_bypass(monkeypatch, tmp_path):
    """VTON_POSTFX=0: 跳过 protect/refit/anchor/ghost/inpaint/reapply 全
    POSTFX 链路 + 印花核心直贴, 交付 FASHN 原始生成但做 CPU Lanczos 放大到
    long_side(SKU 像素下限), 元数据 postfx=raw + delivery_tier=review(人工
    视觉验收采用才落 PROD)。r14 GPU 视觉验收: s0_fashn_raw 比 final_2048
    好, POSTFX 净负, 须支持 RAW 直出 + CPU 放大交付。
    """
    import engine_fashn
    import engine_sdxl
    import platform_presets
    import prewhiten as prewhiten_mod
    from PIL import Image as PILImage

    class FakePipeline:
        def __init__(self, weights_dir):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[PILImage.new("RGB", (32, 32), (9, 9, 9))])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    monkeypatch.setenv("VTON_POSTFX", "0")
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")
    monkeypatch.setattr(engine_sdxl, "foreground_mask", lambda img: None)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda img, subject_mask=None: img)
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (32, 32), (10, 20, 30)).save(person)
    PILImage.new("RGB", (32, 32), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 64, "white", "western_europe")
    # RAW 档: CPU Lanczos 放大到 long_side=64(满足 SKU 像素下限), 不跑印花核心直贴
    assert max(out.size) == 64, f"POSTFX=0 RAW 档须放大到 long_side=64, 实测 {out.size}"
    assert out.info["postfx"] == "raw"
    assert out.info["delivery_tier"] == "review", "RAW 档无印花保真分, 须走 review 人工验收"
    assert out.info["detail_applied"] is False
    assert out.info["lock_identity"] is False


def test_compose_tryon_fashn_raw_no_downsize_when_at_long_side(monkeypatch, tmp_path):
    """RAW 档只做放大不做缩小: FASHN 原生 64x64 已达 long_side=64, 保持原样,
    不触发放大, 像素级不变(验证放大只在长边不足时发生)。"""
    import engine_fashn
    import engine_sdxl
    import numpy as np
    import platform_presets
    import prewhiten as prewhiten_mod
    from PIL import Image as PILImage

    class FakePipeline:
        def __init__(self, weights_dir):
            pass

        def __call__(self, **kwargs):
            arr = np.full((64, 64, 3), 128, dtype=np.uint8)
            return types.SimpleNamespace(images=[PILImage.fromarray(arr)])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    monkeypatch.setenv("VTON_POSTFX", "0")
    monkeypatch.setattr(engine_sdxl, "foreground_mask", lambda img: None)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda img, subject_mask=None: img)
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (64, 64), (10, 20, 30)).save(person)
    PILImage.new("RGB", (64, 64), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 64, "white", "western_europe")
    assert out.size == (64, 64), f"已达 long_side 不放大, 实测 {out.size}"
    assert out.info["postfx"] == "raw"
    assert out.info["delivery_tier"] == "review"


def test_compose_tryon_fashn_fidelity_survives_bg_and_resize(monkeypatch, tmp_path):
    """GPU 冒烟实测: 白底转换/缩放替换 Image 后 .info 丢失, print_fidelity 变 null。

    保真分在内容阶段计算, 但必须写入尾段元数据块(背景/缩放之后)才能落库。
    """
    import engine_fashn
    import engine_sdxl
    import numpy as np
    import platform_presets
    import preserve_refit
    import prewhiten as prewhiten_mod
    from PIL import Image as PILImage

    class FakePipeline:
        def __init__(self, weights_dir):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[PILImage.new("RGB", (32, 32), (9, 9, 9))])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")
    monkeypatch.setenv("VTON_PROTECT_FEATHER", "4")

    ref = PILImage.new("RGBA", (32, 32), (200, 10, 10, 255))
    alpha = PILImage.new("L", (32, 32), 255)
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))
    monkeypatch.setattr(engine_sdxl, "crop_garment_to_content", lambda img: img)
    monkeypatch.setattr(engine_sdxl, "infer_auto_garment_type", lambda ext: "dress")
    monkeypatch.setattr(engine_sdxl, "align_garment_to_pose", lambda *a, **k: (ref, alpha))
    monkeypatch.setattr(engine_sdxl, "intersect_mask_with_garment_alpha", lambda m, a: (m, 1))
    monkeypatch.setattr(platform_presets, "detect_garment_extent", lambda img: None)
    monkeypatch.setattr(
        platform_presets,
        "foreground_mask",
        lambda img: np.full((img.size[1], img.size[0]), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        platform_presets,
        "clothing_mask",
        lambda result, **k: np.full((result.size[1], result.size[0]), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(platform_presets, "limit_mask_to_subject", lambda m, s: m)
    monkeypatch.setattr(platform_presets, "reapply_garment_detail", lambda result, ref, mask, **k: result)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda result, subject_mask=None: result)
    monkeypatch.setattr(preserve_refit, "protect_inverse_refit", lambda r, b, m, feather_px=4: r)

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)
    PILImage.new("RGB", (40, 60), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 48, "white", "western_europe")
    assert out.info["engine"] == "fashn-vton"
    assert out.info["detail_applied"] is True
    # long_side=48 强制缩放(32->48), 白底路径换新 Image; 保真分必须仍在
    assert max(out.size) == 48
    assert "print_fidelity" in out.info, "fidelity lost across background/resize transforms"
    assert out.info["print_fidelity"] == 1.0
    assert out.info["delivery_tier"] == "auto"


def _fake_fashn(monkeypatch, tmp_path, size=(32, 32)):
    """共享夹具: FakePipeline 输出可控尺寸, 全部平台钩子可控。"""
    import engine_fashn
    from PIL import Image as PILImage

    class FakePipeline:
        def __init__(self, weights_dir):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[PILImage.new("RGB", size, (9, 9, 9))])

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    return engine_fashn, PILImage


def test_compose_fashn_drawn_align_uses_refit_and_decoupled_masks(monkeypatch, tmp_path):
    """VTON_FASHN_PRINT_ALIGN=drawn(默认): 贴印几何走 TPS 重投, 三 mask 解耦。

    贴印区(paste_mask)=印花核心 ∩ keep; 保护(protect)拿 keep_mask(纯裁剪);
    模板路径(align_garment_to_pose)必须不被触碰。退化 drawn_mask(全画布)
    自动回退模板路径由既有 fidelity 测试覆盖。
    """
    import engine_sdxl
    import numpy as np
    import platform_presets
    import preserve_refit
    import prewhiten as prewhiten_mod

    engine_fashn, PILImage = _fake_fashn(monkeypatch, tmp_path, size=(96, 96))
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")
    monkeypatch.setenv("VTON_PROTECT_FEATHER", "4")

    def _no_template(*a, **k):
        raise AssertionError("drawn path must not call align_garment_to_pose")

    monkeypatch.setattr(engine_sdxl, "align_garment_to_pose", _no_template)
    monkeypatch.setattr(engine_sdxl, "crop_garment_to_content", lambda img: img)
    monkeypatch.setattr(engine_sdxl, "infer_auto_garment_type", lambda ext: "dress")
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))
    monkeypatch.setattr(platform_presets, "detect_garment_extent", lambda img: None)
    monkeypatch.setattr(
        platform_presets, "foreground_mask", lambda img: np.full((img.size[1], img.size[0]), 255, dtype=np.uint8)
    )

    drawn = np.zeros((96, 96), dtype=np.uint8)
    drawn[16:80, 16:64] = 255
    monkeypatch.setattr(platform_presets, "clothing_mask", lambda result, **k: drawn.copy())

    reapply_masks = []

    def _reapply(result, ref, mask, **k):
        reapply_masks.append(np.asarray(mask).copy())
        return ref.convert("RGB")

    monkeypatch.setattr(platform_presets, "reapply_garment_detail", _reapply)

    protect_masks = []

    def _protect(result, base, mask, feather_px=4):
        protect_masks.append(np.asarray(mask).copy())
        return result

    monkeypatch.setattr(preserve_refit, "protect_inverse_refit", _protect)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda result, subject_mask=None: result)

    garment = tmp_path / "g.png"
    # 仿真平铺结构(与真实 SKU 同构): 白底 + 芥末黄衣 + 微灰白印花。
    # 尺寸按真实比例: 羽化/腐蚀等像素级常数在微型图上占比失真。
    flat = PILImage.new("RGB", (64, 80), (255, 255, 255))
    from PIL import ImageDraw

    d = ImageDraw.Draw(flat)
    d.rectangle((8, 8, 56, 72), fill=(216, 198, 151))
    d.rectangle((22, 28, 42, 48), fill=(238, 234, 226))
    flat.save(garment)
    person = tmp_path / "p.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 144, "white", "western_europe")
    assert out.info["engine"] == "fashn-vton"
    assert out.info["detail_applied"] is True
    # 三 mask 解耦: 贴印区=印花核心, 贴印推迟到最终分辨率(96→144)执行;
    # protect 拿到原生 96 的整件衣料 keep(裁到 alpha)。
    assert len(reapply_masks) == 1 and len(protect_masks) == 1
    paste = reapply_masks[0]
    keep = protect_masks[0]
    assert paste.sum() > 0
    assert paste.shape[0] == 144, "print paste must run at final resolution"
    keep_full = np.array(PILImage.fromarray(keep).resize((paste.shape[1], paste.shape[0]), PILImage.Resampling.NEAREST))
    assert np.asarray((paste > 0) & (keep_full > 0)).sum() == (paste > 0).sum()
    assert keep.sum() < 255 * 96 * 96, "keep must be clipped, not full canvas"
    # keep(protect 区)在 drawn bbox 内, 原生 96 分辨率
    ys, xs = np.where(keep > 0)
    assert xs.min() >= 16 and xs.max() < 64 and ys.min() >= 16 and ys.max() < 80
    # 贴印核心落在白印花 warp 后的位置(96 尺度 core bbox (26,32)-(53,60) → 144 尺度 x1.5)
    pys, pxs = np.where(paste > 0)
    assert pxs.min() >= 39 - 10 and pxs.max() <= 80 + 10, (pxs.min(), pxs.max())
    assert pys.min() >= 48 - 10 and pys.max() <= 90 + 10, (pys.min(), pys.max())
    # NCC 浮点舍入: 贴印区=参考原样 → 分数 ≈ 1.0
    assert out.info["print_fidelity"] > 0.999, out.info["print_fidelity"]
    assert out.info["delivery_tier"] == "auto"
    assert out.info["lock_identity"] is True


def _setup_ghost_inpaint_scene(monkeypatch, tmp_path, resid_return):
    """公共场景: fake FASHN 输出=暗芥末衣身+上方白墨鬼影块; 平铺参考
    亮衣料+下半部白印花(核心场)。返回 (engine_fashn, reapply/inpaint/
    resid 三个 spy 的记录列表)。r12 起鬼影擦除 = ghost_inpaint_pass
    局部修补(Telea 只从笔画周边取色, 袖褶光影全保) + strict 残余亮块
    小范围恒等贴回色调匹配参考兜底, 不再整场凸包贴回(r11 GPU 视觉
    验收"衣袖失真")。"""
    import sys as _sys

    import engine_sdxl
    import numpy as np
    import platform_presets
    import preserve_refit
    import prewhiten as prewhiten_mod
    from PIL import Image as PILImage
    from PIL import ImageDraw

    # 生成图衣身比平铺参考暗 ~25 灰阶 → 兜底贴回必须用色调匹配 ref
    fake_out = PILImage.new("RGB", (96, 96), (216, 198, 151))
    ImageDraw.Draw(fake_out).rectangle((24, 18, 44, 38), fill=(255, 255, 255))

    class GhostPipeline:
        def __init__(self, weights_dir):
            pass

        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[fake_out])

    import engine_fashn

    fake_mod = types.ModuleType("fashn_vton")
    fake_mod.TryOnPipeline = GhostPipeline
    monkeypatch.setitem(_sys.modules, "fashn_vton", fake_mod)
    monkeypatch.setattr(engine_fashn, "FASHN_DIR", tmp_path)
    monkeypatch.setattr(engine_fashn, "_fashn_state", {"pipe": None})
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")
    monkeypatch.setenv("VTON_PROTECT_FEATHER", "4")
    monkeypatch.setattr(engine_sdxl, "crop_garment_to_content", lambda img: img)
    monkeypatch.setattr(engine_sdxl, "infer_auto_garment_type", lambda ext: "dress")
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))
    monkeypatch.setattr(platform_presets, "detect_garment_extent", lambda img: None)
    monkeypatch.setattr(
        platform_presets, "foreground_mask", lambda img: np.full((img.size[1], img.size[0]), 255, dtype=np.uint8)
    )
    drawn = np.zeros((96, 96), dtype=np.uint8)
    drawn[16:80, 16:64] = 255
    monkeypatch.setattr(platform_presets, "clothing_mask", lambda result, **k: drawn.copy())

    reapply_calls = []

    def _reapply(result, ref, mask, **k):
        reapply_calls.append((np.asarray(mask).copy(), dict(k), ref))
        return ref.convert("RGB")

    monkeypatch.setattr(platform_presets, "reapply_garment_detail", _reapply)
    monkeypatch.setattr(preserve_refit, "protect_inverse_refit", lambda r, b, m, feather_px=4: r)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda result, subject_mask=None: result)

    inpaint_calls = []
    _real_inpaint = preserve_refit.ghost_inpaint_pass

    def _inpaint(img, mask, **k):
        inpaint_calls.append(np.asarray(mask).copy())
        return _real_inpaint(img, mask, **k)

    resid_calls = []

    def _resid(img, cloth_ref, keep, core, **k):
        resid_calls.append(np.asarray(img).copy())
        return resid_return

    monkeypatch.setattr(preserve_refit, "ghost_inpaint_pass", _inpaint)
    monkeypatch.setattr(preserve_refit, "ghost_strict_residual", _resid)

    garment = tmp_path / "g.png"
    flat = PILImage.new("RGB", (64, 80), (255, 255, 255))
    d = ImageDraw.Draw(flat)
    d.rectangle((8, 8, 56, 72), fill=(240, 226, 180))
    # 印花在下半部: 核心场与上方鬼影拉开 >dilate9 的距离
    d.rectangle((22, 48, 42, 66), fill=(255, 255, 255))
    flat.save(garment)
    person = tmp_path / "p.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)
    return engine_fashn, reapply_calls, inpaint_calls, resid_calls


def test_compose_fashn_ghost_erase_inpaint_no_residual(monkeypatch, tmp_path):
    """鬼影擦除=局部修补优先: inpaint 填笔画后 strict 残余为空 → 不再有
    鬼场整块贴回, 只剩单次核心恒等贴印(r11 GPU 整场贴回把袖区 26.5k
    自然袖料拍平 → 视觉验收"衣袖失真")。"""
    import numpy as np

    engine_fashn, reapply_calls, inpaint_calls, resid_calls = _setup_ghost_inpaint_scene(
        monkeypatch, tmp_path, resid_return=None
    )
    out = engine_fashn.compose_tryon_fashn(
        str(tmp_path / "p.png"), str(tmp_path / "g.png"), "women", "dress", 144, "white", "western_europe"
    )
    assert out.info["detail_applied"] is True
    assert "ghost" not in out.info.get("degraded_reason", ""), out.info.get("degraded_reason")
    # inpaint 恰好一次, 输入掩膜盖住鬼影本体(96 (24,18)-(44,38) → 144 中心 [42,51])
    assert len(inpaint_calls) == 1
    gmask = inpaint_calls[0]
    assert gmask.shape[0] == 144
    assert gmask[42, 51] > 0, "鬼影本体中心必须在修补掩膜内"
    # 残余检测收到了 inpaint 后的图, 鬼影中心已被填成非白(Telea 从周边取色)
    assert len(resid_calls) == 1
    s1 = resid_calls[0]
    assert s1[42, 51].mean() < 250, "inpaint 后鬼影中心不得残留纯白"
    # 无残余 → 无兜底 reapply, 只剩核心恒等贴印(细笔触 1px 羽化)
    assert len(reapply_calls) == 1, f"expect core paste only, got {len(reapply_calls)}"
    core_mask, core_kw, _core_ref = reapply_calls[0]
    assert core_mask.shape[0] == 144
    assert core_kw.get("light_strength") == 0.0 and core_kw.get("fold_gain") == 0.0
    assert core_kw.get("feather_px") == 1.0, "核心场细笔触贴印必须 1px 羽化(5px 把笔触 alpha 压到 0.2-0.4)"
    # 鬼影(上方)与核心场(下方)必须分离
    gys, _gxs = np.where(gmask > 0)
    cys, _cxs = np.where(core_mask > 0)
    assert gys.max() < cys.min()


def test_compose_fashn_ghost_residual_fallback_pastes_matched_ref(monkeypatch, tmp_path):
    """密集笔画网 inpaint 填不净(r11 实测残余 8957px): strict 残余检测
    非空 → 小范围恒等贴回色调匹配参考兜底(恒等参数防 gen 鬼影区明暗
    场/褶皱带通把鬼影形状搬回), 再核心恒等贴印。"""
    import numpy as np

    resid_mask = np.zeros((144, 144), dtype=np.uint8)
    resid_mask[27:66, 36:57] = 255
    engine_fashn, reapply_calls, inpaint_calls, _resid_calls = _setup_ghost_inpaint_scene(
        monkeypatch, tmp_path, resid_return=resid_mask
    )
    out = engine_fashn.compose_tryon_fashn(
        str(tmp_path / "p.png"), str(tmp_path / "g.png"), "women", "dress", 144, "white", "western_europe"
    )
    assert out.info["detail_applied"] is True
    assert "ghost" not in out.info.get("degraded_reason", ""), out.info.get("degraded_reason")
    assert len(inpaint_calls) == 1, "兜底前必须先做局部修补"
    # 兜底贴回(先) + 核心贴印(后)
    assert len(reapply_calls) == 2, f"expect residual fallback + core paste, got {len(reapply_calls)}"
    fb_mask, fb_kw, fb_ref = reapply_calls[0]
    core_mask, core_kw, _core_ref = reapply_calls[1]
    assert fb_mask.shape[0] == 144 and core_mask.shape[0] == 144
    assert (
        fb_kw.get("light_strength", 1.0) == 0.0
        and fb_kw.get("hf_strength", 1.0) == 0.0
        and fb_kw.get("fold_gain", 1.0) == 0.0
    ), "兜底贴回必须恒等参数(gen 鬼影区明暗场/褶皱带通会把鬼影形状搬回)"
    assert core_kw.get("feather_px") == 1.0
    # 兜底贴回的 ref 已平移到生成图衣身色调(暗芥末 ~188 均亮)
    _fr = np.asarray(fb_ref.convert("RGB"), np.float32)
    _fm = np.asarray(fb_mask) > 0
    _gen_garment_lum = (216 + 198 + 151) / 3.0
    assert abs(_fr[_fm].mean() - _gen_garment_lum) < 8, (
        f"兜底贴回 ref 均亮 {_fr[_fm].mean():.0f} 应贴近生成图衣身 {_gen_garment_lum:.0f}"
    )
    # 鬼影(上方)与核心场(下方)必须分离
    gys, _gxs = np.where(fb_mask > 0)
    cys, _cxs = np.where(core_mask > 0)
    assert gys.max() < cys.min()


def test_compose_fashn_template_align_env_restores_legacy(monkeypatch, tmp_path):
    """VTON_FASHN_PRINT_ALIGN=template: 显式回退模板路径, 语义与旧版一致。"""
    import engine_sdxl
    import numpy as np
    import platform_presets
    import preserve_refit
    import prewhiten as prewhiten_mod

    engine_fashn, PILImage = _fake_fashn(monkeypatch, tmp_path)
    monkeypatch.setenv("VTON_FASHN_PRINT_ALIGN", "template")
    monkeypatch.setenv("VTON_PATTERN_LOCK", "1")
    monkeypatch.setenv("VTON_WARP", "1")

    ref = PILImage.new("RGBA", (32, 32), (200, 10, 10, 255))
    alpha = PILImage.new("L", (32, 32), 255)
    calls = []

    def _align(*a, **k):
        calls.append(1)
        return ref, alpha

    monkeypatch.setattr(engine_sdxl, "align_garment_to_pose", _align)
    monkeypatch.setattr(engine_sdxl, "crop_garment_to_content", lambda img: img)
    monkeypatch.setattr(engine_sdxl, "infer_auto_garment_type", lambda ext: "dress")
    monkeypatch.setattr(prewhiten_mod, "prewhiten_input", lambda img: (img, {"status": "already_white"}))
    monkeypatch.setattr(platform_presets, "detect_garment_extent", lambda img: None)
    monkeypatch.setattr(
        platform_presets, "foreground_mask", lambda img: np.full((img.size[1], img.size[0]), 255, dtype=np.uint8)
    )
    drawn = np.zeros((32, 32), dtype=np.uint8)
    drawn[10:26, 8:24] = 255
    monkeypatch.setattr(platform_presets, "clothing_mask", lambda result, **k: drawn.copy())
    monkeypatch.setattr(engine_sdxl, "intersect_mask_with_garment_alpha", lambda m, a: (m, 1))
    monkeypatch.setattr(platform_presets, "reapply_garment_detail", lambda result, r, mask, **k: r.convert("RGB"))
    monkeypatch.setattr(preserve_refit, "protect_inverse_refit", lambda r, b, m, feather_px=4: r)
    monkeypatch.setattr(platform_presets, "force_white_background", lambda result, subject_mask=None: result)

    person = tmp_path / "p.png"
    garment = tmp_path / "g.png"
    PILImage.new("RGB", (40, 60), (10, 20, 30)).save(person)
    PILImage.new("RGB", (24, 32), (200, 10, 10)).save(garment)

    out = engine_fashn.compose_tryon_fashn(person, garment, "women", "dress", 48, "white", "western_europe")
    assert calls, "template env must route through align_garment_to_pose"
    assert out.info["print_fidelity"] == 1.0
    assert out.info["delivery_tier"] == "auto"
