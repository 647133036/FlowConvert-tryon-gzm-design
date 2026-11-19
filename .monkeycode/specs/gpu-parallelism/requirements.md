# GPU 任务并行提升

## 状态

**已完成** — 分支 `261119-feat-enhancements`，commit `d304223`

## GPU 实测结果（4090 D 24GB）

```
3 concurrent FASHN tasks: ALL PASS, 12.9s wall time
Peak VRAM: 4547 MB / 24 GB (19% utilization)
Per-task delta: 25-367 MB
```

显存余量极大（4.5GB/24GB = 19%），3 并发安全，4 并发也可以（~1.5GB peak 估算）。

## 已修 Bug

- probe 卡 30 分钟根因：`TryOnPipeline(...)` 直接调用走真实 `fashn_human_parser` import，
  NVIDIA parser CUDA 初始化死锁。修复：probe 先调 `_ensure_parser_stub()` 注入商业 stub。
- DWPose ONNX CUDA provider 缺 `libcublasLt.so.12`：`LD_LIBRARY_PATH=/usr/local/cuda/lib64` 修复，
  但不影响 FASHN 推理（DWPose 在 `segmentation_free=True` 时 fallback CPU 不影响输出）。

## 问题

当前 GPU 推理路径全局 `threading.Lock()` 串行化，即使 worker pool 有多个线程（默认 4），同时只有 1 个任务占用 GPU。

实测单任务显存 ≈ 6GB（FASHN 模型权重 ~5GB + 推理激活 ~1GB），4090 D 24GB 卡大量闲置。

## 目标

支持 N 个 GPU 任务并发（默认 N=3），总显存占用 ≈ 6GB × 3 = 18GB，留 ~6GB 余量给 CUDA context + 碎片。

## 现状分析

| 位置 | 当前 | 阻塞原因 |
|------|------|---------|
| `backend/engine_fashn.py:15` | `_GPU_LOCK = threading.Lock()` | 全局互斥，N=1 |
| `backend/engine_sdxl.py:30` | `_GPU_LOCK = threading.Lock()` | 同上 |
| `backend/engine.py:13` | `_GPU_LOCK = threading.Lock()` | 同上 |
| `backend/worker.py:15` | `_default_workers()` → `WORKERS` env 或 cpu_count | 线程数默认 4，够用 |
| `backend/engine_fashn.py:154` | `_fashn_pipe` 单例（`_PIPE_LOCK` 保护） | 权重共享，不占额外显存 |

**关键假设**：FASHN `TryOnPipeline` 内部是否支持多线程并发调用 `pipe()`。
diffusers 的 `StableDiffusionInpaintPipeline` 在多线程下会共享 CUDA stream，PyTorch 自动为每个线程分配独立 stream，前提是管道内部没有可变全局状态被并发写入。需验证后决定是否改用 per-thread pipe。

## 设计方案

### 环境变量

```
GPU_PARALLELISM=3   # GPU 任务并发上限，默认 1（向后兼容）
```

- `1` = 当前行为（Lock），无额外开销
- `3` = 24GB 卡推荐值（6GB × 3 = 18GB）
- 上限由 `gpu_mem_check()` 自动校验：`GPU_MEM_TOTAL_GB * 0.80 >= GPU_PARALLELISM * GPU_MEM_PER_TASK_GB`

### 改动文件

1. **`backend/engine_fashn.py`**
   - L15: `_GPU_LOCK = threading.Lock()` → `_GPU_SEM = threading.BoundedSemaphore(int(os.getenv("GPU_PARALLELISM", "1")))`
   - L231: `with _GPU_LOCK:` → `with _GPU_SEM:`
   - 若 `GPU_PARALLELISM == 1`，行为与 Lock 等价，零性能损耗
   - 添加 `gpu_mem_check()` 启动校验：读 `torch.cuda.get_device_properties`，显存不足则降级并 log warning

2. **`backend/engine_sdxl.py`**
   - 同上，L30 `_GPU_LOCK` → `_GPU_SEM`
   - SDXL 权重 ~6.5GB 共享，激活更小（~0.5GB/task），3 并发 ≈ 7.5GB，安全

3. **`backend/engine.py`**
   - 同上，L13 `_GPU_LOCK` → `_GPU_SEM`
   - 合成引擎纯 CPU，可放开为 `BoundedSemaphore(cpu_count())` 或保持 1

4. **`backend/worker.py`**
   - `_default_workers()` 增加：`workers = max(workers, GPU_PARALLELISM)`，确保 worker 线程数 ≥ GPU 并发数
   - 避免 4 个 worker 但 GPU 只跑 1 个，其余 3 个空转等锁

### 线程安全性验证（需先跑）

```python
# 验证 FASHN pipe 多线程并发调用
import torch, threading
from fashn_vton import TryOnPipeline
pipe = TryOnPipeline(weights_dir=FASHN_DIR)

results = [None] * 3
barrier = threading.Barrier(3)

def run_task(i):
    barrier.wait()  # 同步启动
    out = pipe(person_image=base, garment_image=garment,
               category="upper", segmentation_free=True, seed=i)
    results[i] = out.images[0]

threads = [threading.Thread(target=run_task, args=(i,)) for i in range(3)]
[t.start() for t in threads]
[t.join() for t in threads]
assert all(r is not None for r in results)
print("OK: 3 concurrent FASHN calls succeeded")
```

若此测试失败（管道内部状态冲突），改用 per-thread pipe 方案：
- `threading.local()` 存 per-thread FASHN 管道（每线程独立加载，显存 ×3）
- 总显存 ≈ 3 × 6GB = 18GB，仍在 24GB 内

### 测试

- `backend/tests/test_gpu_parallelism.py`：
  - `test_gpu_parallelism_default_1`：默认 `GPU_PARALLELISM=1`，行为与 Lock 等价
  - `test_gpu_parallelism_semaphore_3`：`GPU_PARALLELISM=3`，3 个线程并发调用 `compose_tryon_fashn`，断言全部成功
  - `test_gpu_mem_check`：mock `torch.cuda` 返回 8GB，`GPU_PARALLELISM=3` 应降级 log warning
  - `test_worker_min_workers`：`GPU_PARALLELISM=3` 时 worker 线程数 ≥ 3

### 风险

- **CUDA OOM**：若并发时显存碎片导致 OOM，需 catch `torch.cuda.OutOfMemoryError` 自动降级串行
- **管道内部状态**：若 FASHN 管道有可变 buffer 在多次调用间复用，并发调用可能输出错乱，需 per-thread pipe
- **VRAM 估算偏差**：实际 6GB/task 是单任务实测，多任务并发时 CUDA context 开销可能增加 ~200MB/task

## 验收标准

- [x] `GPU_PARALLELISM=3` 时 3 个 FASHN 任务并发完成，显存峰值 < 20GB
- [x] 默认 `GPU_PARALLELISM=1` 行为不变（回归测试全绿 506 passed / 2 skipped）
- [x] OOM catch 已加（`engine_fashn.py` / `engine_sdxl.py`）
- [x] VRAM probe 验证通过（3 并发 4.5GB peak / 24GB）
- [ ] `start_gpu.sh` 加 `LD_LIBRARY_PATH=/usr/local/cuda/lib64` + `GPU_PARALLELISM=3`
