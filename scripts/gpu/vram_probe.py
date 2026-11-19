import os, sys, threading, time
os.environ.setdefault('LICENSE_MODE', 'commercial')

import torch

device = torch.device('cuda')
props = torch.cuda.get_device_properties(0)
print(f'GPU: {props.name}, {props.total_memory // 1024**3} GB VRAM', flush=True)

from pathlib import Path
repo_root = Path('/root/vton/repo')
sys.path.insert(0, str(repo_root / 'backend'))
sys.path.insert(0, '/root/autodl-tmp/fashn-vton-1.5/src')

from engine_fashn import _ensure_parser_stub
_ensure_parser_stub()
print('parser stub injected', flush=True)

from fashn_vton import TryOnPipeline
import numpy as np
from PIL import Image

FASHN_DIR = os.environ.get('FASHN_DIR', '/root/vton/repo/models/fashn')
print(f'loading FASHN pipe from {FASHN_DIR}...', flush=True)
t0 = time.monotonic()
pipe = TryOnPipeline(weights_dir=FASHN_DIR)
print(f'pipe loaded in {time.monotonic()-t0:.1f}s', flush=True)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

base_arr = np.random.randint(0, 255, (768, 576, 3), dtype=np.uint8)
garment_arr = np.random.randint(0, 255, (768, 576, 3), dtype=np.uint8)
base_img = Image.fromarray(base_arr)
garment_img = Image.fromarray(garment_arr)

results = [None] * 3
errors = [None] * 3
barrier = threading.Barrier(3)
lock = threading.Lock()

def run_task(idx):
    try:
        vram_before = torch.cuda.memory_allocated() // 1024**2
        barrier.wait()
        out = pipe(person_image=base_img, garment_image=garment_img,
                   category='tops', garment_photo_type='flat-lay',
                   num_timesteps=4, guidance_scale=1.0, seed=idx,
                   segmentation_free=True)
        vram_after = torch.cuda.memory_allocated() // 1024**2
        vram_peak = torch.cuda.max_memory_allocated() // 1024**2
        with lock:
            results[idx] = out.images[0]
            errors[idx] = f'OK (delta={vram_after-vram_before}MB peak={vram_peak}MB)'
            print(f'  task {idx}: {errors[idx]}', flush=True)
    except Exception as e:
        vram_peak = torch.cuda.max_memory_allocated() // 1024**2
        with lock:
            errors[idx] = f'FAIL {type(e).__name__}: {e} (peak={vram_peak}MB)'
            print(f'  task {idx}: {errors[idx]}', file=sys.stderr, flush=True)

print('starting 3 concurrent tasks...', flush=True)
t1 = time.monotonic()
threads = [threading.Thread(target=run_task, args=(i,)) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
elapsed = time.monotonic() - t1

vram_peak = torch.cuda.max_memory_allocated() // 1024**2
ok = sum(1 for r in results if r is not None)
fail = sum(1 for e in errors if e and e.startswith('FAIL'))
print(f'Wall time: {elapsed:.1f}s', flush=True)
print(f'Peak VRAM (3 concurrent): {vram_peak} MB / {props.total_memory // 1024**2} GB', flush=True)
print(f'Tasks: {ok} OK, {fail} FAIL', flush=True)
if ok == 3:
    print('PASS: 3-concurrent FASHN within VRAM budget', flush=True)
else:
    print('FAIL: see errors above', flush=True)
    sys.exit(1)
