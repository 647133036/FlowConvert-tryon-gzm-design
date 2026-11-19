 # Atelier v0.2.2

 > # 商用声明
 >
 > **本项目核心代码的商用权归作者所有，任何商用部署、商用集成、商用修改均需联系作者获取授权。**
 >
 > 未经授权的商用使用（包括但不限于：部署到生产环境、集成到 SaaS 产品、嵌入到客户端、用于商业广告/电商运营、转售或再许可）均构成侵权。
 >
 > 商用联系：请通过仓库 issue 或作者公开渠道联系，说明使用场景与规模。

 ---

 店铺后台固定底图换衣工作台。选一张模特底图，批量拖入**服装平铺图**，队列生成正面成品，下载或推送到商品页。界面内置中英文一键切换。

 > **输入要求**：服装参考图必须是**平铺（flat-lay）衣服图**，**不是**模特穿着图。FASHN 引擎按平铺几何处理参考图；喂模特穿着图会被误判为衣服的一部分，导致肩膀、手臂等人体部位被啃掉（视觉验收缺陷的根因）。
 >
 > **GPU 并发**：扩散推理由全局 `threading.Lock` 升级为 `BoundedSemaphore(GPU_PARALLELISM)`，默认 3 并发。4090 D（24GB）实测 3 并发 FASHN 任务峰值 VRAM ≈5.7GB（24%），可同时跑 3 个任务。

 **基本运行条件：8 核 CPU / 32GB 内存 / NVIDIA 4090 D（24GB 显存）。** 商用主引擎 FASHN VTON v1.5（像素空间）按此规格验证；无 GPU 时自动降级合成引擎，或配 `AIDGE_API_KEY` 走云端 Provider（纯 CPU 机器可出商用图）。

---

## 详细架构

### 分层总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  出图 / 交付层                                                       │
│   GzmDesign 海报排版台 (gzm-design/, Vite:5174, LeaferJS 2D 画布)       │
│   Pillow 纯 CPU 2x PNG 渲染 (backend/poster.py)                        │
│   平台合规组件 (backend/compliance.py, 5 平台声明式规则, CPU 原语)      │
│   推送适配器 (backend/push.py: Shopify / WordPress / 其他打标)          │
│   CSV 批量导出 (backend/export_csv.py)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ L3  编排层 (backend/ai_bindings.py + 6 个 L2 HTTP/SSE 端点)             │
│   幂等键 (内容指纹 + 参数 + pipeline_ver + seed_ns, 命中 0 元秒回)      │
│   成本预估/记账 (7 角色 × 6 Provider 费率表)                            │
│   SSE 事件流 (/api/jobs/{id}/events, 15s 心跳, 四层无缓冲)             │
│   操作 (/api/jobs/{id}:act — cancel / reroll / reroute)                 │
│   批次导出 (/api/batches/{id}:export)                                   │
│   批量审核 (/api/reviews:batch)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ L2  引擎层 (5 条换衣管线, 统一注册表 backend/tryon_provider.py)        │
│   ① FASHN VTON v1.5  (商用主引擎, 本地 GPU)                             │
│   ② SDXL Inpaint     (本地扩散兜底, ControlNet + IP-Adapter)           │
│   ③ 阿里云百炼 AiTryOn (同步 API, aidge 端点)                          │
│   ④ 阿里云视觉智能 VIAPI (按次调用, 同源 aidge)                        │
│   ⑤ 第三方 API        (预留钩子, 实现 Provider 即插即用)               │
 │   路由: VTON_PROVIDER=auto → 云端不可用自动回落 FASHN → 失败 → SDXL    │
 │   → 合成引擎兜底                                                       │
 │   GPU 并发: BoundedSemaphore(GPU_PARALLELISM, 默认 3), 3 任务同卡;     │
 │     OOM 自动 empty_cache + 重试; worker 数 = max(WORKERS, GPU_PAR)      │
├─────────────────────────────────────────────────────────────────────────┤
│ L1  模型 / 权重层                                                       │
│   FASHN 1.5.0 (Apache 2.0, 像素空间, segmentation_free=True)           │
│   DWPose / YOLOX (Apache 2.0, FASHN 人体/姿态解析)                    │
│   SDXL + ControlNet + IP-Adapter (商用合规)                            │
│   RTMPose-m (关键点检测, Apache 2.0)                                   │
│   fashn-human-parser (NVIDIA License, 商用路径注入 stub, 权重不下载)  │
├─────────────────────────────────────────────────────────────────────────┤
 │ L0  基础设施                                                            │
 │   合成引擎 (CPU, backend/engine.py, 无 GPU 兜底)                        │
 │   印花保真管线 (backend/preserve_refit.py, 确定性贴回, fidelity=1.0)   │
 │   质检分档 (backend/quality_gate.py, SSIM/OCR/print_fidelity,          │
 │     ssim data_range=1.0 兼容 skimage≥0.21)                              │
 │   平台规格化 (backend/platform_presets.py, 裁切/压缩/品类检测/         │
 │     u2net 主体掩膜)                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 换衣管线全流程（商用模式）

```
POST /api/tryon
  → L3 编排层: 幂等键命中? 命中 → 0 元秒回 cache_hit=true
  → L2 引擎层: VTON_PROVIDER 路由
      ├─ FASHN VTON v1.5 (GPU, 像素空间, segmentation_free)
      │    ├─ VTON_POSTFX=1: 全链路印花贴回
      │    │    ├─ refit_cloth_to_drawn_garment  (VTON_REFIT_ENGINE: grid/tps/optical_flow)
      │    │    ├─ match_ref_tone                (全局色调, 采样避开鬼影/印花)
      │    │    ├─ ghost_print_mask              (笔画语义鬼影检测, 非凸包)
      │    │    ├─ ghost_inpaint_pass            (局部修补 + Telea 色锚)
      │    │    ├─ ghost_strict_residual         (territory 限定残余 + 逐像素最近净布料复核)
      │    │    ├─ anchor_ref_local              (局部环形锚定, 逐行 ±24 窗口环布料中位)
      │    │    └─ blend_core_ref_tone           (核心场混合, 笔触取原图, 布料取锚定版, 软边羽化)
      │    └─ VTON_POSTFX=0: RAW 直出档
      │         ├─ 跳过全 POSTFX + 印花核心直贴
      │         ├─ CPU Lanczos 等比放大到 SKU 像素下限 (只放大不缩小)
      │         └─ delivery_tier=review → 人工视觉验收采用后才推 PROD
      │
      └─ 任一引擎失败 → 下一级兜底 (SDXL Inpaint → 合成引擎)
  → L3 质检: print_fidelity / SSIM / OCR → delivery_tier (auto / review)
  → L4 出图: 长边 2048px, 白底 JPEG 或透明 PNG
  → SSE 推送进度 → 前端图层回写 → 审核台
```

### 五条电商换衣管线注册表

| 管线 | 形态 | 价格量级 | 适配品类 | 备注 |
| --- | --- | --- | --- | --- |
| FASHN VTON v1.5 | 本地 pixel-space 生成 + 确定性印花贴回 | `VTON_FASHN_LOCAL_COST_CENT`（默认 5 分，GPU 摊薄） | 全品类；复杂图案/特殊版型翻车率偏高 | 商用主引擎，POSTFX 全链路或 RAW 直出两档 |
| SDXL Inpaint | 本地扩散兜底（对齐 + ControlNet + IP-Adapter） | 0（GPU 摊薄已含 fashn_local） | 全品类 | FASHN 失败时兜底 |
| 阿里云百炼 AiTryOn | 同步 API（model_image + garment_image + category） | 0.2 / 0.5 元/张（各 400 张免费），`AIDGE_COST_CENT` 可覆盖 | 连衣裙、套装较稳；上衣易扒裤子 | 配 API Key 即用；`category` 必须显式传 |
| 阿里云视觉智能 VIAPI | 按次调用，独立于百炼入口 | 按次，与百炼同源，`VIAPI_COST_CENT` 可覆盖 | 同上 | 适合已有阿里云账号 |
| 第三方 API | 预留接入位（Provider 钩子） | 自定义 | 自定义 | 实现一个 Provider 钩子即插即用 |

### GzmDesign 海报排版台

独立子项目 `gzm-design/`（Vite SPA, 端口 5174），与主 SPA（端口 5173）通过 6 个 L2 端点桥接：

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/poster/render` | POST | Pillow 纯 CPU 2x PNG 渲染 |
| `/api/jobs/estimate` | POST | 成本预估 |
| `/api/jobs/{id}/events` | GET (SSE) | 任务事件流（15s 心跳，四层无缓冲） |
| `/api/jobs/{id}:act` | POST | cancel / reroll / reroute |
| `/api/batches/{id}:export` | POST | 批次导出 |
| `/api/reviews:batch` | POST | 批量审核 |

SSE 四层无缓冲：Vite `configure` 关 proxy buffer + Nginx `proxy_buffering off` + `chunked_transfer_encoding off` + CDN 透传。

### 多店铺与推送

- 每店铺独立 embed 嵌入链接（`/embed.html?store=STORE_ID`）
- CSV/SKU 批量推送：`/api/jobs/export-csv` 导出 ZIP → `/api/stores/{id}/batch-push` 上传
- 单张推送（admin）：Shopify Admin API / WordPress REST + WooCommerce / 其他平台仅打标
- 替换图优先于原成品，SKU 回退链：CSV 行 sku > 店铺 default_sku > 空

---

## 依赖版本

### 后端 (Python 3.10+)

| 包 | 版本 | 用途 |
|----|------|------|
| fastapi | 0.115.6 | HTTP API 框架 |
| uvicorn | 0.34.0 | ASGI 服务器 |
| python-multipart | 0.0.20 | 表单/文件上传解析 |
| pillow | 11.1.0 | 图像处理（合成引擎 / 海报渲染 / 印花贴回） |
| numpy | 2.2.4 | 数值计算（跨机一致性：沙箱 2.4.6 vs GPU 服务器 1.26.4） |
| pyjwt | 2.10.1 | JWT 认证 |
| aiofiles | 24.1.0 | 异步文件 I/O |
| requests | 2.32.3 | HTTP 客户端（百炼 / VIAPI / 推送） |
 | opencv-python-headless | ≥4.9.0 | 合成引擎 2D 变形（纯 CPU） |
 | scikit-image | ≥0.21 | SSIM 质检（`ssim_score` 传 `data_range=1.0`，float32 必须显式指定） |

GPU 环境额外（`requirements.txt` 注释段，按部署环境启用）：

| 包 | 版本 | 备注 |
|----|------|------|
| torch | ≥2.1.0 | CUDA 构建（4090 D 实测 2.13.0+cu130） |
| diffusers | ≥0.27.0 | SDXL 管线（实测 0.40.0） |
| transformers | ≥4.40.0 | 模型加载 |
| accelerate | ≥0.30.0 | CPU offload |
| controlnet-aux | ≥0.0.7 | ControlNet 条件注入 |
 | rembg | ≥2.0.50 | 主体掩膜（U2Net）。仅 GPU 环境需要；`foreground_mask` 失败时 fail-closed 回退纯颜色压白（手臂/胸口不被误啃） |
 | fashn-human-parser | 0.1.1 | FASHN 人体解析（商用路径注入 stub，权重不下载） |
 | onnxruntime-gpu | 1.23.2 | DWPose ONNX 推理（CUDA provider） |
 | numpy | 1.26.4 | GPU 服务器降级（scipy 1.11 兼容）。**注意**：rembg 的 `pymatting` 依赖链拉 `numba` 要求 numpy≤1.24，与 1.26.4 冲突；GPU 无卡环境可不装 rembg（`foreground_mask` 自动 fail-closed 回退） |

### 主前端 (vanilla-JS SPA, 端口 5173)

| 包 | 版本 | 用途 |
|----|------|------|
| vite | ^5.4.10 | 构建工具 + 开发服务器 |
| vitest | ^2.1.8 | 前端测试 |

入口 `src/main.js`（~1575 行），`src/i18n.js`（~456 行，中英双语），Poster tab 调 6 个 L2 端点。

### GzmDesign 排版台 (Vue 3 + Vite, 端口 5174)

| 包 | 版本 | 用途 |
|----|------|------|
| leafer-ui | ^2.3.3 | 2D 画布引擎（图层 / 拖拽 / 撤销栈） |
| vue | ^3.4.0 | 框架（工具链已装，入口仍为 vanilla-JS） |
| pinia | ^2.1.7 | 状态管理（已装，未接入组件树） |
| @arco-design/web-vue | ^2.46.0 | UI 组件库（已装，未接入） |
| @vitejs/plugin-vue | ^5.0.0 | Vite Vue 插件 |
| typescript | ^5.4.0 | 类型检查（vue-tsc） |
| @leafer-ui/node | (render.ts) | 服务端 2x PNG 渲染 |
| @napi-rs/canvas | (render.ts) | 字体注册 + Canvas 节点 |
| sharp | (render.ts) | 图片输出编码 |

### 模型权重

 | 权重 | 来源 | 大小 | 许可证 |
 |------|------|------|--------|
 | FASHN VTON v1.5 | tar 同步到 `models/fashn/`（GitHub 不可达时手动 rsync） | ~1.9GB | Apache 2.0 |
 | FASHN DWPose | FASHN 仓库内置 | ~200MB | Apache 2.0 |
 | FASHN YOLOX | FASHN 仓库内置 | ~50MB | Apache 2.0 |
 | U2Net | `models/seg/u2net.onnx` | ~176MB | Apache 2.0 |
| SDXL 1.0 | stabilityai/stable-diffusion-xl-base-1.0 | ~6.5GB | Open RAIL++-M |
| SDXL LoRA (rank 16) | 自训（120 张） | ~200MB | — |
| ControlNet (OpenPose/Canny/Depth) | lllyasviel/ControlNet | ~2.5GB | Open RAIL++-M |
| IP-Adapter | tencent-ailab/IP-Adapter | ~300MB | Apache 2.0 |
| RTMPose-m | open-mmlab/mmpose | ~200MB | Apache 2.0 |
| fashn-human-parser (SegFormer) | NVIDIA | 不下载 | NVIDIA License (商用禁止) |

### 目录结构

```text
backend/
  main.py              FastAPI 入口 (~1100 行), 全部 L2 端点
  worker.py            异步 worker (343 行), job 队列 + 引擎调度
  engine.py            合成引擎 (1362 行, CPU 兜底) + 引擎路由 + engine_status
  engine_fashn.py      FASHN VTON v1.5 (611 行, 商用主引擎 + 印花保真 + RAW 档)
  engine_sdxl.py       SDXL Inpaint (1739 行, 兜底引擎)
  tryon_provider.py    5 管线注册表 + 路由 + 幂等 + 成本记账 (418 行)
  tryon_provider_v2.py Provider v2 扩展
  ai_bindings.py       L3 编排层 (270 行, 7 角色 / 视觉哈希 / 幂等键 / 费率表)
  compliance.py        平台合规 (316 行, 5 平台声明式规则, CPU 原语)
  poster.py            海报渲染 (147 行, Pillow 纯 CPU 2x PNG)
  postfx_anchor.py     POSTFX 锚定扩展
  person_picker.py     人物选择器
  preserve_refit.py    印花保真管线 (1402 行, 贴回核心)
  prewhiten.py         预去白
  quality_gate.py      质检 (SSIM / OCR / print_fidelity / delivery_tier)
  platform_presets.py  平台规格化 + 品类检测 + 关键点
  push.py              推送适配器 (Shopify / WordPress / 其他)
  export_csv.py        CSV 批量导出
  lora_io.py           LoRA 加载 (PEFT 优先)
  store.py             店铺管理
  auth.py              JWT 认证
  tests/               pytest (20+ 测试文件, 506 passed / 2 skipped)
  migrations/
    0001_core_tables.sql
    0002_person_seed_ns.sql

src/                   主前端 vanilla-JS SPA
  main.js              入口 (~1575 行, 含 Poster tab + 6 端点调用)
  i18n.js              中英双语 (~456 行, 25 poster.* key)
  style.css

gzm-design/            海报排版台 (Vue 工具链 + LeaferJS)
  src/main.js          LeaferJS 画布 + 4 个 bug fix + 内联 AI 面板
  src/ai/AiPanel.vue  右键 AI 菜单 + 成本确认 + 角色保护
  src/ai/useGenerate.ts SSE subscribe + submitJob + estimateJob
  src/types/layerMeta.ts AiMeta + Symbol.for('gzm.aiMeta') + PROTECTED_ROLES
  server/render.ts     @leafer-ui/node + sharp + GlobalFonts (2x PNG/WebP)

training/              SDXL LoRA / ControlNet 训练脚本
models/                模型权重 (FASHN / SDXL / RTMPose / LoRA)
deploy/
  nginx-gzm-design.conf  Nginx 配置 (SSE 无缓冲 / 上传 / 字体 MIME)
scripts/
  gpu/                 GPU 冒烟测试 (combo.py / run_r6.py 等)
  run_tryon_batch.py   批量出图
  download_sdxl_weights.sh
```

---

## 快速启动（无 GPU）

```bash
# 后端
pip3 install --break-system-packages -r backend/requirements.txt
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# 主前端
npm install
npm run dev

# GzmDesign 排版台 (可选)
cd gzm-design
npm install
npm run dev
```

首次启动时后端为 `admin` 和 `staff` 生成随机密码并输出到日志。生产部署：

```bash
export STORE_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export STORE_ADMIN_PASSWORD="your-secure-password"
export STORE_STAFF_PASSWORD="your-secure-password"
```

 ## GPU 部署（4090 D 商用模式）

 ```bash
 # 1. 同步代码 + 模型
 git clone https://github.com/647133036/IDM-VTON-Builder.git
 cd IDM-VTON-Builder
 bash scripts/download_sdxl_weights.sh
 # FASHN 权重手动同步到 models/fashn/ (GitHub 不可达时 tar/rsync 从可达机器搬运)

 # 2. Python 环境 (Miniconda 3.10)
 pip install -r backend/requirements.txt
 pip install "torch>=2.1.0" "diffusers>=0.27.0" "transformers>=4.40.0" "accelerate>=0.30.0" controlnet-aux
 pip install fashn-human-parser onnxruntime-gpu
 # 无卡环境可不装 rembg (其依赖链拉 numba 要求 numpy<=1.24, 与 1.26.4 冲突);
 # 有 rembg 时前景掩膜可用, 无则 foreground_mask fail-closed 回退纯颜色压白。
 # numpy 需降到 1.26.4 (scipy 1.11 兼容, requirements.txt 的 2.2.4 与 scipy 冲突)

 # 3. FASHN 源码 (editable install)
 # PYTHONPATH 需包含 fashn 源码: /path/to/fashn-vton-1.5/src
 # models 符号链接: ln -s /path/to/repo/models /path/to/parent/models
 # u2net 模型放 repo/models/seg/u2net.onnx (~176MB), 免 rembg 自动下载

 # 4. 启动 (start_gpu.sh)
 export LICENSE_MODE=commercial
 export FASHN_DIR=/path/to/repo/models/fashn
 export PYTHONPATH=/path/to/repo:/path/to/fashn-vton-1.5/src
 export LD_LIBRARY_PATH=/usr/local/cuda/lib64
 export GPU_PARALLELISM=3        # 扩散推理并发, 默认 3 (BoundedSemaphore)
 export WORKERS=3                # worker 数, 默认 max(WORKERS, GPU_PARALLELISM)
 python -m uvicorn main:app --host 0.0.0.0 --port 8000

 # 5. 验证
 curl http://localhost:8000/api/license
 # → fashn_available: true, engine: fashn-vton, commercial_compliant: true
 ```

 **VRAM 基准**（4090 D 24GB，`scripts/gpu/vram_probe.py` 实测）：

 | 并发数 | 峰值 VRAM | 占比 | 3 任务 wall time |
 |--------|-----------|------|------------------|
 | 3      | 4.5GB     | 19%  | 12.9s（probe，单张 30 步） |
 | 3      | 5.7GB     | 24%  | 179s（完整 job 含 prewhiten/POSTFX） |

 模型权重为共享单例（`_get_fashn_pipe()`），3 任务共享 ~5GB 权重，仅激活缓冲随并发线性增长，实际峰值远低于 3×单任务。

## 运行条件

| 项目 | 规格 |
|------|------|
| CPU | 8 核 |
| 内存 | 32GB |
| GPU | NVIDIA RTX 4090 D（24GB 显存） |
| 系统 | Linux，CUDA 驱动与 PyTorch 匹配 |
| 铸脸 (Qwen-Image-Edit-2511) | Q4 量化 ≈12G，无法与 FASHN (≈8G) 同卡共存；独立 `gpu_cast` 队列，夜间跑 |

GPU 实测环境（2026-11）：RTX 4090 D 24GB / Driver 580.76.05 / CUDA 13.0 / Miniconda Python 3.10（torch 2.13.0+cu130，diffusers 0.40.0，numpy 1.26.4，opencv 4.8.0）。

## 引擎选择

| LICENSE_MODE | 引擎优先级 | 商用合规 |
|---|---|---|
| `commercial` | FASHN VTON v1.5 → SDXL Inpaint → 合成引擎（云端：百炼 / VIAPI 经 `VTON_PROVIDER` 路由，不可用自动回落本地） | 合规 |
| `research` | 合成引擎（如需 IDM-VTON 研究路径，手动 clone 到 repo/IDM-VTON/，需自行处理 CC BY-NC-SA 授权） | 不合规 |
| 未设置 | 合成引擎 | 需确认 |

## SDXL 训练

```bash
cd training
MODEL=lora bash run_train.sh         # LoRA（约 11 分钟 / 800 步）
MODEL=controlnet bash run_train.sh   # ControlNet
MODEL=lora_family bash run_train.sh  # 服装 + 人像双 LoRA
```

## 环境变量

### 基础

```bash
STORE_JWT_SECRET=<至少 32 字节>
STORE_ADMIN_PASSWORD=<管理员密码>
STORE_STAFF_PASSWORD=<员工密码>
LICENSE_MODE=commercial
```

### SDXL 引擎（v0.1.8）

| 变量 | 默认 | 说明 |
|------|------|------|
| `VTON_AR` | `1` | 保宽高比 letterbox |
| `VTON_WARP` | `1` | 几何对齐 |
| `VTON_WARP_VERTICAL` | `1.0` | 垂直锚点对齐强度 |
| `VTON_DETAIL` | `1` | 商品细节回贴（边缘衰减；肤色区降权） |
| `VTON_PATTERN_LOCK` | `1` | DualFit 贴图 + 频率分离 |
| `VTON_PATTERN_LOCK_LIGHT_STRENGTH` | `1.0` | 光照总闸 |
| `VTON_PATTERN_LOCK_LF_STRENGTH` | `1.0` | 底色光照响应系数 |
| `VTON_PATTERN_LOCK_HF_STRENGTH` | `0.5` | 花纹光照响应系数 |
| `VTON_PATTERN_LOCK_LF_BLUR_SIGMA` | `8.0` | 低频提取高斯 sigma |
| `VTON_PATTERN_LOCK_LIGHTING_MIN/MAX` | `0.75/1.25` | 亮度比 clamp |
| `VTON_PATTERN_LOCK_SKIN_WEIGHT` | `0.85` | 肤色区降权 |
| `VTON_PATTERN_LOCK_FEATHER_PX` | `5` | mask 边缘羽化像素 |
| `VTON_PATTERN_LOCK_SKIP_WHITE_BG` | `1` | 跳过边缘连通白底 |
| `VTON_PATTERN_LOCK_WHITE_THRESH` | `245` | 白底 RGB 判定阈值 |
| `VTON_LOCK_ID` | `1` | 身份锁（SeamlessClone） |
| `VTON_LORA` | `0.7` | LoRA scale |
| `VTON_IPA` | `0.9` | IP-Adapter 参考强度 |
| `VTON_CANNY` | 按品类 0.55/0.45 | Canny 边缘约束 |
| `VTON_GUIDANCE` | `7.5` | CFG guidance scale |
| `VTON_SEED` | `42` | 随机种子 |
| `VTON_MULTI_SEED` | `3` | 多 seed 择优（上限 8） |
| `VTON_REFIT_ENGINE` | `grid` | 贴回几何引擎 grid/tps/optical_flow |
| `VTON_INPAINT_REPAIR` | `0` | 设 `1` 且有权重时启用 SDXL Inpainting 二次修复 |
| `RTMPOSE_FAKE_DEPS` | `1` | 注入空壳 xtcocotools/mmdet/mmcv.ops |

### Provider 层（v0.2.0 起，v0.2.1 扩为 5 管线）

| 变量 | 默认 | 说明 |
|------|------|------|
| `VTON_PROVIDER` | `auto` | 生成源路由：`auto` / `fashn_local` / `sdxl` / `aidge` / `viapi` / `third_party` |
| `AIDGE_API_KEY` | 空 | Aidge API Key（可配 `DASHSCOPE_API_KEY`） |
| `AIDGE_PUBLIC_BASE_URL` | 空 | 本服务公网基地址（Aidge 经公网 URL 拉取图片） |
| `AIDGE_ENDPOINT` | `https://aidge.aliyuncs.com` | Aidge API 端点 |
| `AIDGE_API_VERSION` | `2026-04-28` | `x-acs-version` 头 |
| `AIDGE_COST_CENT` | `15` | 百炼 aidge 单张参考价（分） |
| `VIAPI_COST_CENT` | `15` | VIAPI 单张参考价（分） |
| `AIDGE_TIMEOUT` | `120` | 请求超时秒 |
| `AIDGE_MAX_RETRIES` | `3` | 5xx/超时/429 重试上限 |
| `AIDGE_BREAKER_SECONDS` | `1800` | 402 熔断窗口 |
| `VTON_PROVIDER_MODEL` | 空 | 覆盖 aidge model 名；model 名进幂等键 |
| `VTON_POSTFX` | `1` | `1`=POSTFX 全链路；`0`=RAW 直出档 |
| `VTON_FASHN_LOCAL_COST_CENT` | `5` | 本地 GPU 摊薄参考价（分/张） |
 | `VTON_PIPELINE_VERSION` | `0.2.1` | 幂等键组成项 |
 | `HTTP_402_IS_FATAL` | `true` | 402 直落失败并熔断 |
 | `GPU_PARALLELISM` | `3` | 扩散推理并发上限（`BoundedSemaphore`）；OOM 自动 `empty_cache` 重试 |
 | `WORKERS` | `4` | worker 线程数；实际取 `max(WORKERS, GPU_PARALLELISM)` |

### 必须现在钉死（6 参数）

| 参数 | 值 | 说明 |
|------|----|------|
| `worker_prefetch_multiplier` | `1` | 配合 `--pool=solo --concurrency=1` |
| 幂等键组成 | `SEED_NS + PIPELINE_VER` | 双入键，换模型/发版自动灰度失效旧缓存 |
| `Platform.image.crossOrigin` | `'anonymous'` | + OSS CORS，防 Canvas 污染 |
| `?v=${assetVer}` | 每资源 | 缓存击穿 |
| `POSTFX_HIST_ANCHOR` / `BAND_GAIN` / `BAND_MAX` | `true` / `0.30` / `0.55` | 印花贴回锚定参数 |
| `HTTP_402_IS_FATAL` | `true` | 402 不重试，直接熔断 |

 ## 图片规格

 - **上传**：PNG / JPEG / WEBP，最大 20MB，任意尺寸
 - **服装参考图**：必须是**平铺（flat-lay）衣服图**，**不是**模特穿着图。FASHN 引擎按平铺几何处理；喂穿着图会把模特身体误判为衣服一部分，导致肩膀/手臂被啃掉（视觉验收缺陷根因）
 - **输出**：长边 2048px，白底 JPEG 或透明 PNG，宽高比跟随底座图

## 测试

```bash
# 后端 (506 passed / 2 skipped)
cd backend
python3 -m pytest tests/ -q

# 前端
npm test
npm run build

# 静态检查
ruff check backend/ training/
ruff format --check backend/ training/
```

## API 概览

```text
POST /api/auth/login
GET  /api/me  /api/stats  /api/settings
GET|POST /api/bases  /api/ethnicities  /api/stores
POST /api/jobs/batch  /api/jobs/export-csv
POST /api/stores/{id}/batch-push
POST /api/jobs/{id}/push                    # [admin]
GET  /api/jobs  /api/jobs/{id}  /api/download/{id}
GET|POST /api/uploads  /api/uploads/categories
GET  /api/health  /api/license

# L3 编排层 (6 端点, GzmDesign AiPanel 调用)
POST /api/poster/render
POST /api/jobs/estimate
GET  /api/jobs/{id}/events                   # SSE, 15s 心跳
POST /api/jobs/{id}:act                     # cancel / reroll / reroute
POST /api/batches/{id}:export
POST /api/reviews:batch
```

## 安全

- 文件服务鉴权：`/files/{path}` 仅白名单目录 + 图片后缀；上传路径正则白名单，拒绝路径穿越
- 图片内容验证：PIL 解码，仅 PNG/JPEG/WEBP，最大 20MB
- JWT 密钥无硬编码默认值；账号密码未设置时随机生成并持久化（chmod 600）
- 角色权限：admin 全操作，staff 仅创建和查看；凭据脱敏
- SSRF 防护：`/api/push` 强制 https + `.myshopify.com`，拒绝内网
- XSS 防护：前端 `escapeHtml` 转义；服务端枚举白名单校验
 - GPU 并发：扩散推理 `BoundedSemaphore(GPU_PARALLELISM, 默认 3)` 限并发；OOM 自动 `empty_cache` 重试；批量事务性写盘

## 授权说明

> # 商用声明
>
> **本项目核心代码（换衣管线、印花保真、平台适配、后处理、训练脚本、编排层、GzmDesign 排版台等全部自研代码）的商用权归作者所有。**
>
> **任何商用使用均需联系作者获取授权。** 未经授权的商用部署、商用集成、商用修改、转售或再许可均构成侵权。

| 组件 | 许可证 | 商用 | 本项目使用方式 |
|------|--------|------|----------------|
| IDM-VTON 模型权重 | CC BY-NC-SA 4.0 | 禁止 | 仅借鉴 garment warp 思路（LF/HF 频率分离的 CPU 后处理实现），未使用其代码或权重；已从仓库移除 |
| FASHN VTON v1.5 | Apache 2.0 | 允许 | 商用主引擎（像素空间、TPS、无 mask）；parser 走零张量 stub |
| FASHN DWPose / YOLOX | Apache 2.0 | 允许 | FASHN 人体/姿态解析权重 |
| fashn-human-parser | NVIDIA Source Code License | 禁止 | 商用路径注入 stub，SegFormer 权重不下载 |
| Stable Diffusion XL | CreativeML Open RAIL++-M | 允许（作者已获商用授权） | 商用换衣管线的生成底座（FASHN 失败兜底） |
| ControlNet | Open RAIL++-M | 允许（作者已获商用授权） | OpenPose / Canny / Depth 条件注入 |
| IP-Adapter | Apache 2.0 | 允许 | 服装特征注入 |
| RTMPose | Apache 2.0 | 允许 | 关键点检测（腰线插值 + knee 截断 + 肩部掩膜） |
| Rembg (U2Net) | MIT | 允许 | 主体掩膜提取 |
| MediaPipe | Apache 2.0 | 允许 | 关键点检测回退 |
| PLTON 论文方法 | 学术公开 | 思路可借鉴 | LF/HF 频率分离的 CPU 后处理实现，未使用其代码或权重 |
| 本项目核心代码 | **商用权归作者所有，商用需联系作者** | **禁止（未获授权前）** | 换衣管线、印花保真、平台适配、后处理、训练脚本、编排层、GzmDesign 等全部自研代码 |

## 历史版本

 | 版本 | 要点 |
 |------|------|
 | v0.2.2 | GPU 3 并发（`BoundedSemaphore`，OOM 自动重试）；`ssim` 兼容 skimage≥0.21（`data_range=1.0`）；移除 IDM-VTON submodule；明确服装参考图须为平铺图 |
 | v0.2.1 | 五条换衣管线注册；百炼+VIAPI 同源 aidge 换 model 名；RAW 交付档；GzmDesign 海报排版台；L3 编排层 6 端点；平台合规组件 |
| v0.2.0 | Provider 层：生成源可插拔 + 幂等缓存 + 成本记账 + prio 队列 |
| v0.1.9 | 印花保真管线（FASHN 商用路径）：AI 只画衣服，图案确定性贴回，print_fidelity 实测 1.0 |
| v0.1.8 | RTMPose-m 关键点检测；腰线插值；dress 脖子排除 + 膝部截断 |
| v0.1.7 | warp 垂直锚点对齐；GARMENT_BOX 品类垂直范围；inpaint 修复聚焦遮罩 |
| v0.1.6 | 频率分离 pattern lock（PLTON 思路） |
| v0.1.5 | 换衣后遗症图像修复 |
| v0.1.4 | Canny 按品类 / IP-Adapter 0.9 / detect_garment_extent |
| v0.1.3 | 「AI 出骨架光影，代码保商品细节」混合策略 + LoRA 训练基线 |

## 感谢

- [FASHN](https://fashn.ai) — VTON v1.5 商用主引擎（像素空间，Apache 2.0）
- [IDM-VTON](https://github.com/yisol/IDM-VTON)（ECCV 2024）— garment warp 思路借鉴（未使用其代码或权重，已移除）
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) — COCO 17 关键点
- [PLTON](https://arxiv.org/abs/2401.11239) — LF/HF 频率分离思想
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) / [ControlNet](https://github.com/lllyasviel/ControlNet) / [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [Rembg](https://github.com/danielgatis/rembg) / [MediaPipe Pose](https://github.com/google-ai/mediapipe)
