"""3-concurrent GPU parallelism VRAM probe.

Run on GPU server:
    PYTHONPATH=/root/vton/repo:/root/autodl-tmp/fashn-vton-1.5/src \
    CUDA_VISIBLE_DEVICES=0 python3 scripts/gpu/vram_probe.py

Reports peak VRAM per task and total, no OOM expected at 3 concurrent.
If OOM: reports which task failed, VRAM at time of failure.
"""

import os
import sys
import threading
import time
import traceback

os.environ.setdefault("LICENSE_MODE", "commercial")

import torch

device = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_mem // 1024**3} GB VRAM")

from fashn_vton import TryOnPipeline
from PIL import Image
import numpy as np

FASHN_DIR = os.environ.get("FASHN_DIR", "/root/vton/repo/models/fashn")
pipe = TryOnPipeline(weights_dir=FASHN_DIR)
torch.cuda.empty_cache()

base_bytes = 32 * 1024 * 1024
base_arr = np.random.randint(0, 255, (768, 576, 3), dtype=np.uint8)
garment_arr = np.random.randint(0, 255, (768, 576, 3), dtype=np.uint8)
base_img = Image.fromarray(base_arr)
garment_img = Image.fromarray(garment_arr)

results: list = [None] * 3
errors: list = [None] * 3
barrier = threading.Barrier(3)
lock = threading.Lock()


def run_task(idx: int) -> None:
    try:
        vram_before = torch.cuda.memory_allocated() // 1024**2
        barrier.wait()
        out = pipe(
            person_image=base_img,
            garment_image=garment_img,
            category="tops",
            garment_photo_type="flat-lay",
            num_timesteps=2,
            guidance_scale=1.0,
            seed=idx,
            segmentation_free=True,
        )
        vram_after = torch.cuda.memory_allocated() // 1024**2
        vram_peak = torch.cuda.max_memory_allocated() // 1024**2
        with lock:
            results[idx] = out.images[0]
            errors[idx] = f"OK (vram delta={vram_after - vram_before}MB, peak={vram_peak}MB)"
            print(f"  task {idx}: done, {errors[idx]}")
    except Exception as e:
        vram_now = torch.cuda.memory_allocated() // 1024**2
        vram_peak = torch.cuda.max_memory_allocated() // 1024**2
        with lock:
            errors[idx] = f"FAIL {type(e).__name__}: {e} (vram_now={vram_now}MB, vram_peak={vram_peak}MB)"
            print(f"  task {idx}: FAIL, {errors[idx]}", file=sys.stderr)
        traceback.print_exc()


torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

t0 = time.monotonic()
threads = [threading.Thread(target=run_task, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.monotonic() - t0

vram_total = torch.cuda.max_memory_allocated() // 1024**2
print(f"\nTotal wall time: {elapsed:.1f}s")
print(f"Peak VRAM (all 3 concurrent): {vram_total} MB / {props.total_mem // 1024**2} MB")
ok = sum(1 for r in results if r is not None)
fail = sum(1 for e in errors if e and e.startswith("FAIL"))
print(f"Tasks: {ok} OK, {fail} FAIL")
if ok == 3:
    print("PASS: 3-concurrent FASHN inference within VRAM budget")
else:
    print("FAIL: concurrent tasks did not all complete")
    sys.exit(1)
