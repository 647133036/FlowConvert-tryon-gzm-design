import os
import threading
import time
import traceback
from pathlib import Path

from engine import compose_tryon, engine_status, guess_audience, guess_garment_type
from store import BASES, RESULTS, UPLOADS, claim_next_job, find_in_dir, patch_job
from store import settings as load_settings


def _default_workers() -> int:
    from os import cpu_count, getenv

    return max(1, min(int(getenv("WORKERS", "4")), cpu_count() or 4))


def _base_file(base: dict) -> Path:
    if base.get("path"):
        return Path(base["path"])
    return BASES / base["file"]


def process_job(job: dict, bases: list[dict]) -> None:
    patch_job(job["id"], {"progress": 8, "message": "Reading base and garment"})
    base = next((b for b in bases if b["id"] == job["base_id"]), None)
    if not base:
        raise RuntimeError("Base model not found")
    src = UPLOADS / Path(job["source_url"]).name if job.get("source_url") else None
    if not (src and src.exists()):
        src = UPLOADS / job["source_name"] if job.get("source_name") else None
    if not (src and src.exists()):
        name = Path(job.get("source_url") or job.get("source_name", "")).name
        src = find_in_dir(UPLOADS, name)
    if not (src and src.exists()):
        raise RuntimeError("Garment source image missing")
    base_path = _base_file(base)
    if not base_path.exists():
        raise RuntimeError("Base model image missing; upload a base first")

    audience = job.get("audience") or "auto"
    if audience == "auto":
        audience = base.get("audience") or guess_audience(job.get("source_name", ""), "women")
    garment_type = job.get("garment_type") or "auto"
    if garment_type == "auto":
        garment_type = guess_garment_type(job.get("source_name", ""))

    patch_job(
        job["id"],
        {
            "progress": 28,
            "message": "Segmenting garment and estimating drape",
            "audience": audience,
            "garment_type": garment_type,
        },
    )
    cfg = load_settings()
    long_side = int(cfg.get("output_long_side", 2048))
    background = job.get("background") or cfg.get("background", "white")
    platform = job.get("platform", "none")
    ext = "png" if job.get("output_format") == "png" or background == "transparent" else "jpg"

    # Provider 层(§00 L3/§07): 生成源路由。aidge 无 seed 参数,
    # 多 seed = 多花钱同结果 → 单发控成本。
    from tryon_provider import PROVIDER_PRICES, active_provider, estimate_cost_cents, idem_key

    provider = active_provider()
    try:
        _n_env = int(os.environ.get("VTON_MULTI_SEED", "3"))
    except ValueError:
        _n_env = 3
    n_seeds = max(1, min(_n_env, 8))
    seed_intent = int(os.environ.get("VTON_SEED", "42"))
    if n_seeds == 1 or provider == "aidge":
        seeds = [seed_intent]
    else:
        from preserve_refit import refit_seeds_default

        seeds = refit_seeds_default()[:n_seeds]
        # 纯色款单 seed 控成本: job 未显式标 has_print=False 时按印花处理
        if not job.get("has_print", True):
            seeds = seeds[:1]
    n_runs = len(seeds)

    # 幂等(§03): 同人同衣同参数 → 复用上一张成图, 命中 0 元(§06 HIT 免单)。
    idem = idem_key(base_path, src, audience, garment_type, background, long_side, platform, ext, seeds)
    est = estimate_cost_cents(n_runs, provider)
    cached = _find_cached_output(idem)
    if cached is not None:
        _finish_from_cache(job, cached, idem)
        return
    patch_job(job["id"], {"idem_key": idem, "cost_est_cent": est["cost_cent"], "provider": provider})

    engine_info = engine_status()
    engine_label = (
        "IDM-VTON diffusion engine" if engine_info["idm_vton_available"] else "composite engine (no-GPU fallback)"
    )
    patch_job(job["id"], {"progress": 35, "message": f"Calling {engine_label}"})
    ethnicity = base.get("ethnicity") or ""
    # 多 seed 出图: 每个 seed 跑一次 compose_tryon, 按质检分排序取最高。
    # VTON_MULTI_SEED=0 强制单 seed; 非法值回退默认 3; 上限 8(控成本)。
    if n_runs == 1:
        result = compose_tryon(base_path, src, audience, garment_type, long_side, background, ethnicity)
    else:
        from preserve_refit import run_multiseed
        from quality_gate import score_for_sku

        def _task(seed):
            os.environ["VTON_SEED"] = str(seed)
            try:
                return compose_tryon(base_path, src, audience, garment_type, long_side, background, ethnicity)
            finally:
                os.environ.pop("VTON_SEED", None)

        def _scorer(img, seed):
            # 首选 compose 引擎写入的印花保真分(贴回区内 warp 参考 vs 结果,
            # 跨 seed 可比); 无分时回退平铺参考 vs 结果的粗 SSIM(弱信号)。
            pf = img.info.get("print_fidelity")
            if pf is not None:
                return float(pf)
            try:
                from PIL import Image

                ref_arr = Image.open(src).convert("RGB")
                out_arr = img.convert("RGB")
                if out_arr.size != ref_arr.size:
                    out_arr = out_arr.resize(ref_arr.size)
                h, w = ref_arr.size[1], ref_arr.size[0]
                box = (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))
                score, _tier = score_for_sku(ref_arr.crop(box), out_arr.crop(box), sku_has_text=False)
                return score
            except Exception:  # noqa: BLE001
                return 0.0

        result, _all_scores, _chosen = run_multiseed(_task, seeds, _scorer)
        result.info["multiseed"] = {
            "n_seeds": n_seeds,
            "seeds": seeds,
            "chosen_seed": _chosen,
            "scores": {str(s): sc for s, sc in _all_scores},
        }
        if len(seeds) > 1:
            patch_job(job["id"], {"message": f"Multi-seed ({n_seeds}) done, chosen seed={_chosen}"})
    engine_used = result.info.get("engine") or engine_info["engine"]
    # 成本记账(§06 ACT): aidge 逐张实价; local 按 GPU 摊薄估算 x 实跑次数;
    # composite 兜底不烧 GPU 记 0。
    if "aidge" in engine_used:
        provider_used = "aidge"
        cost_act = int(result.info.get("provider_cost_cent") or PROVIDER_PRICES["aidge"]) * n_runs
    elif engine_used.startswith("composite"):
        provider_used, cost_act = "fashn_local", 0
    else:
        provider_used = "fashn_local"
        _per_run = result.info.get("provider_cost_cent")
        if _per_run is None:
            _per_run = PROVIDER_PRICES.get("fashn_local", 5)
        cost_act = int(_per_run) * n_runs
    degraded = bool(result.info.get("engine_degraded", False))
    degraded_reason = result.info.get("degraded_reason") or ""
    patch_job(
        job["id"],
        {
            "progress": 78,
            "message": "Rendering final image",
            "engine": engine_used,
            "engine_degraded": degraded,
            "degraded_reason": degraded_reason,
            "cost_cent": cost_act,
            "cache_hit": False,
        },
    )
    # 印花保真分 + 交付档位落库: auto 直接放行, review 须人工确认后才推店。
    _tier_fields = {}
    if "print_fidelity" in result.info:
        _tier_fields["print_fidelity"] = result.info["print_fidelity"]
    if "delivery_tier" in result.info:
        _tier_fields["delivery_tier"] = result.info["delivery_tier"]
    if "multiseed" in result.info:
        _tier_fields["multiseed"] = result.info["multiseed"]
    if _tier_fields:
        patch_job(job["id"], _tier_fields)

    out_path = RESULTS / f"{job['id']}.{ext}"
    if ext == "png" and platform == "none":
        result.save(out_path, "PNG")
    else:
        # 平台模板(shopify/amazon/aliexpress):1:1 居中裁剪 + 平台尺寸 + 纯白底 + JPEG 限体积。
        from platform_presets import PLATFORM_PRESETS, save_platform_result

        out_jpg = RESULTS / f"{job['id']}.jpg"
        if platform in PLATFORM_PRESETS:
            _, nbytes = save_platform_result(result, platform, out_jpg)
            out_path = out_jpg
            ext = "jpg"
            if nbytes:
                patch_job(job["id"], {"message": f"Done ({platform} template, {nbytes // (1024 * 1024)}MB)"})
        else:
            result.convert("RGB").save(out_jpg, "JPEG", quality=94, optimize=True)
            out_path = out_jpg
            ext = "jpg"
    preview = RESULTS / f"{job['id']}_preview.jpg"
    thumb = result.convert("RGB")
    thumb.thumbnail((640, 860))
    thumb.save(preview, "JPEG", quality=88)
    patch_job(
        job["id"],
        {
            "status": "done",
            "progress": 100,
            "message": "Done; download or push to product page",
            "result_url": f"/files/results/{out_path.name}",
            "preview_url": f"/files/results/{preview.name}",
            "provider": provider_used,
        },
    )


def _find_cached_output(idem: str) -> dict | None:
    """同 idem_key 的已完成 job, 且成图文件在盘才算命中(§11 R3: 文件才是事实)。"""
    from store import list_jobs

    for j in list_jobs():
        if j.get("idem_key") != idem or j.get("status") != "done":
            continue
        name = Path(j.get("result_url") or "").name
        if not name:
            continue
        p = RESULTS / name
        if p.is_file():
            return {"result": p, "job": j}
    return None


def _finish_from_cache(job: dict, cached: dict, idem: str) -> None:
    """缓存命中(§06 HIT 免单): 复制成图文件, 0 元秒回, 不碰生成引擎。"""
    import shutil

    src_path = cached["result"]
    out_path = RESULTS / f"{job['id']}{src_path.suffix}"
    shutil.copyfile(src_path, out_path)
    preview = RESULTS / f"{job['id']}_preview.jpg"
    try:
        from PIL import Image

        thumb = Image.open(out_path).convert("RGB")
        thumb.thumbnail((640, 860))
        thumb.save(preview, "JPEG", quality=88)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    patch_job(
        job["id"],
        {
            "status": "done",
            "progress": 100,
            "message": "Cache hit: same person/garment/params, reused prior output",
            "result_url": f"/files/results/{out_path.name}",
            "preview_url": f"/files/results/{preview.name}",
            "idem_key": idem,
            "provider": cached["job"].get("provider", ""),
            "cache_hit": True,
            "cost_cent": 0,
            "cost_est_cent": 0,
        },
    )


def _worker_loop() -> None:
    while True:
        job = claim_next_job()
        if job is None:
            time.sleep(0.4)
            continue
        try:
            from store import load_state

            bases = load_state()["bases"]
            process_job(job, bases)
        except Exception as exc:  # noqa: BLE001
            try:
                patch_job(
                    job["id"],
                    {
                        "status": "failed",
                        "progress": 100,
                        "message": "Generation failed",
                        "error": str(exc),
                    },
                )
            except Exception:  # noqa: BLE001
                traceback.print_exc()


class WorkerPool:
    def __init__(self, workers: int | None = None) -> None:
        self._workers = max(1, int(workers if workers is not None else _default_workers()))
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for i in range(self._workers):
            t = threading.Thread(target=_worker_loop, name=f"tryon-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def join(self) -> None:
        for t in self._threads:
            t.join(timeout=0.1)


def start_worker(workers: int | None = None) -> WorkerPool:
    pool = WorkerPool(workers if workers is not None else _default_workers())
    pool.start()
    return pool
