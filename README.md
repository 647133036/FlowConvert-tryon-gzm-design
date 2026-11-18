# Atelier v0.1.9

店铺后台固定底图换衣工作台。选一张模特底图，批量拖入服装平铺图，队列生成正面成品，下载或推送到商品页。界面内置中英文一键切换。

**基本运行条件：8 核 CPU / 32GB 内存 / NVIDIA A10（24GB 显存）。** 商用管线（FASHN VTON v1.5 像素空间换装 + 确定性印花保真贴回）按此规格验证；无 GPU 时自动降级合成引擎，效果有限。

## v0.1.9 更新日志：印花保真管线（FASHN 商用路径）

### 问题定义

生成式换衣有一个天然矛盾：AI 负责把衣服"穿"上模特（褶皱、光影、垂坠感），但 SKU 平铺图上的 logo / 印花 / 文字一旦交给生成模型重画，字形必然漂移——电商场景里印花就是商品本身，错一笔就是详情页事故。

v0.1.9 的解法：**AI 只画衣服，图案确定性贴回。** FASHN VTON v1.5（Apache 2.0，像素空间，`segmentation_free=True`）生成整体上身效果后，印花区域从 SKU 参考图经几何 refit + 色调锚定后像素级贴回，`print_fidelity` 实测保持 1.0000。

### 贴回管线（engine_fashn.py + preserve_refit.py）

```
FASHN 生成
  └─ refit_cloth_to_drawn_garment   平铺图案 warp 缩放对齐到生成衣服区
       （VTON_REFIT_ENGINE = grid / tps / optical_flow）
  └─ match_ref_tone                 全局色调匹配（采样避开鬼影/印花笔画）
  └─ ghost_print_mask               鬼影检测（笔画语义，非凸包——
                                    袖区 30k px 布料不再被误贴平）
  └─ ghost_inpaint_pass             局部修补优先 + Telea 色锚：
                                    Telea 填充实测会白化（填充 240,234,219
                                    vs 净布料 203,181,116），逐像素向最近
                                    净布料平移局部色偏
  └─ ghost_strict_residual          territory 限定残余检测 + 逐像素最近
                                    净布料色复核（无 stroke_mask 时检测区
                                    排除自身，防自证）
  └─ anchor_ref_local               残余兜底 + 核心场贴回的局部环形锚定：
                                    全局色调按整件布料中位（含深色下摆 57-96）
                                    匹配，贴回区落在亮胸口（185-195）时会拍出
                                    深色矩形（"像贴上去的/整体下移/衣领断层"）；
                                    改为逐行 ±24 行窗口环布料中位，行分段
                                    加性平移，两个方向（亮贴暗/暗贴亮）收敛
  └─ blend_core_ref_tone            核心场混合：笔触取自原始参考（保真），
                                    布料底取锚定版，亮度 smoothstep 软边 +
                                    1px 羽化（笔触 AA 环亮度 109→181，ref 218）
  └─ quality_gate.print_fidelity_score  印花保真分（贴回区灰度 NCC 映射）
```

最近净布料色（`_nearest_fabric_color`）为纯 numpy 两阶段 Voronoi 传播：先 5x1 横向核（布料明暗沿行渐变，实测右袖 B 通道 159@y620→125@y920；多行核的 max 膨胀会把上方更亮布料逐行复合进来），再 3x3 等方核收尾竖洞。

### GPU 实测（RTX 4090 D，r12/r13 轮）

| 指标 | 旧管线 | 新管线 | 参考 |
|------|--------|--------|------|
| 袖口残留带亮度 | 224（残留白笔画）/ 84（贴平） | 182 | gen 袖料底 183 |
| 猴头区误伤 | 180→123（暗化） | 180→157 | 无改动 192 |
| 核心场布料底 | 107（深色矩形） | 178 | gen 胸口 185-195 |
| 袖褶高频相关 | 0.058 | 0.967 | 1.0 |
| 笔触 AA 环 | 109（暗圈） | 181 | ref 218 |
| print_fidelity | — | 1.0000 | 1.0 |
| 场外严格残影 | 24486 px（gen 本身） | 256 px | →0 |

肩部掩膜补全（`shoulder_complete_mask`）经 GPU 验证为正确 no-op（当前 SKU 无啃肩行），RTMPose 在 GPU 环境正常（肩点 conf 1.0）。

### 跨机环境实测（沙箱 vs GPU 服务器）

- 沙箱 cv2 5.0.0 非真 OpenCV：`DIST_LABEL_PIXEL` 标签恒 0；`resize` 压掉单通道维。GPU 服务器 cv2 4.8.0 为真实现。跨机语义一致的图像原语一律纯 numpy 实现。
- 沙箱 numpy 2.4.6 广播不前导补维（仅右对齐）：`(h,1)` 源对 `(h,w,3)` 会行列转置；服务器 numpy 1.23.5 的 `np.linalg.solve` 在良态 TPS 矩阵上返回垃圾（`tps_warp` 已带残差守卫 + lstsq 换道重试，r6-r8 fidelity 卡 0.85 平台的根因）。

### 质检与多 seed

- `delivery_tier`：SSIM（0.70）/ OCR（0.80）/ print_fidelity 分 `auto` / `review` 档
- `VTON_MULTI_SEED`（默认 3）：worker 按 quality_gate 择优，print_fidelity 跨 seed 可比
- 三引擎（FASHN / SDXL / 合成）统一写 `result.info.print_fidelity + delivery_tier`

## 换衣管线全流程

按电商换衣的五个阶段组织（商用模式）：

| 阶段 | 实现 |
|------|------|
| 1. 衣服抠底 | `garment_alpha` / `detect_garment_extent`：SKU 平铺图出衣服区域、袖长/衣长、品类推断 |
| 2. 模特抠底 | 生成结果上出 `keep_mask`（模特身上衣服区）+ RTMPose 肩点 + `shoulder_complete_mask`；反选区保护原模特像素 |
| 3. 衣服重绘在模特上 | FASHN VTON v1.5 生成整体上身效果（唯一的 AI 重绘步骤；SDXL Inpaint / 合成引擎兜底） |
| 4. 图案缩放贴合 | **确定性贴回，不交给生成模型**：refit warp 缩放 → 色调匹配/局部环形锚定 → 鬼影擦除 → 核心场贴印（花纹/文字/纽扣/袖口/领口原样贴回）→ AA 羽化 |
| 5. 电商规格 | 质检分档 + 多 seed 择优 + 2048px 白底输出；平台规格化处理（裁切模板/压缩规范）在路线图上 |

## 功能

- 女装 / 男装 / 女童 / 男童 / 婴幼儿正面底座，可随时替换或上传自家模特
- 批量拖入几十到上百张服装图，只出正面，长边 2048px，白底 JPEG 或透明 PNG
- 印花保真：logo / 文字 / 图案像素级贴回，`print_fidelity` 实测 1.0
- 已上传图片形成可复用图片库，支持服饰分类管理与批量调整分类
- 底座分类（受众 / 年龄带 / 人种）可管理、可批量调整；人种标签可自定义（默认 10 个地理区域），换衣引擎按底座人种修正体型
- 多店铺管理：每店铺独立 embed 嵌入链接 + CSV/SKU 批量推送
- 真实推送：Shopify Admin API 上传商品图，WordPress REST 媒体 + WooCommerce 商品关联
- 成品大图预览弹窗，左右对照原图与成品
- 界面中英文一键切换，语言偏好本地持久化
- 三层引擎体系：FASHN VTON v1.5（商用主引擎）/ SDXL Inpaint（兜底）/ 合成引擎（无 GPU 兜底）；研究模式仍可用 IDM-VTON（需授权）

## 快速启动（无 GPU）

```bash
# 后端
pip3 install --break-system-packages -r backend/requirements.txt
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# 前端
npm install
npm run dev
```

首次启动时后端为 `admin` 和 `staff` 生成随机密码并输出到日志。生产部署：

```bash
export STORE_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export STORE_ADMIN_PASSWORD="your-secure-password"
export STORE_STAFF_PASSWORD="your-secure-password"
```

## 运行条件

| 项目 | 规格 |
|------|------|
| CPU | 8 核 |
| 内存 | 32GB |
| GPU | NVIDIA A10（24GB 显存） |
| 系统 | Linux，CUDA 驱动与 PyTorch 匹配 |

GPU 实测环境（2026-11）：RTX 4090 D 24GB / Driver 580.76.05 / CUDA 13.0 / Miniconda Python 3.10（torch 2.13.0+cu130，diffusers 0.40.0，numpy 1.23.5，opencv 4.8.0）。SDXL 路径 `enable_model_cpu_offload` + VAE fp32 解码；训练 LoRA 按 A10 23GB 验证（约 11 分钟 / 800 步）。复现入口 `scripts/gpu/`（`combo.py` / `run_r6.py` 等）与 `scripts/run_tryon_batch.py`。

## GPU 部署（商用模式）

```bash
git clone https://github.com/647133036/IDM-VTON-Builder.git
cd IDM-VTON-Builder
git submodule update --init --recursive --depth 1
bash scripts/download_sdxl_weights.sh
cd backend
pip install -r requirements.txt
pip install "torch>=2.1.0" "diffusers>=0.27.0" "transformers>=4.40.0" "accelerate>=0.30.0" controlnet-aux rembg
export LICENSE_MODE=commercial
export STORE_JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export STORE_ADMIN_PASSWORD="your-secure-password"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/api/license
```

## 引擎选择

| LICENSE_MODE | 引擎优先级 | 商用合规 |
|---|---|---|
| `commercial` | FASHN VTON v1.5 -> SDXL Inpaint -> 合成引擎 | 合规 |
| `research` | IDM-VTON -> 合成引擎 | 不合规 |
| 未设置 | 合成引擎 | 需确认 |

## SDXL 训练

```bash
cd training
MODEL=lora bash run_train.sh         # LoRA（约 11 分钟 / 800 步）
MODEL=controlnet bash run_train.sh   # ControlNet
MODEL=lora_family bash run_train.sh  # 服装 + 人像双 LoRA
```

训练基线（v0.1.3+）：`bfloat16`（Ampere+ GPU）、双格式保存（PEFT + diffusers）、`time_ids` / `letterbox_rgb` 保宽高比、VAE fp32、min-SNR gamma=5.0。

## 多店铺与推送

### CSV/SKU 批量推送

- 任务队列页导出 CSV+图：POST `/api/jobs/export-csv` 下载 ZIP
- 店铺卡片批量推送：上传 CSV/ZIP，POST `/api/stores/{id}/batch-push`
- 替换图优先于原成品，SKU 回退链：CSV 行 sku > 店铺 default_sku > 空

### 单张推送

完成的任务勾选店铺推送（admin 权限）：Shopify Admin API / WordPress REST 媒体 + WooCommerce 关联 / 其他平台仅打标。

### 嵌入

```html
<iframe src="/embed.html?store=STORE_ID" style="width:100%;min-height:720px;border:0"></iframe>
```

## 环境变量

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
| `VTON_PATTERN_LOCK_LIGHT_STRENGTH` | `1.0` | 光照总闸：0=纯贴图，1=全效 |
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
| `RTMPOSE_FAKE_DEPS` | `1` | 注入空壳 xtcocotools/mmdet/mmcv.ops；真实依赖已装时设 `0` |

FASHN 印花保真管线默认开启（`engine_fashn.py` 内建，无开关；鬼影/残余/核心场各阶段独立 try/except 降级，任一失败保留生成像素）。

## 图片规格

**上传**：PNG / JPEG / WEBP，最大 20MB，任意尺寸

**输出**：长边 2048px，白底 JPEG 或透明 PNG，宽高比跟随底座图

## 测试

```bash
# 后端（448 passed / 2 skipped）
cd backend
python3 -m pytest tests/ -q

# 前端（14 passed）
npm test
npm run build

# 静态检查
ruff check backend/ training/
ruff format --check backend/ training/
```

覆盖：单元（引擎/人种/CRUD/JWT/推送/worker/LoRA/并发/训练几何）、平台适配（clothing_mask / blend_tryon / reapply / detect_garment_extent / 肤色 / 接缝 inpaint / pose 关键点 / dress 遮罩）、印花保真（preserve_refit 71 用例：TPS 残差守卫 / 鬼影擦除 / 残余兜底 / 局部环形锚定 / 行分段 delta / 同行布料取样 / blend 软边 / fidelity）、组件（i18n/render）、集成（全 API 链路）、系统（真实 worker 出图）。

## 目录结构

```text
backend/
  engine.py                合成引擎（CPU 兜底）
  engine_fashn.py          FASHN VTON v1.5 商用主引擎（parser stub + segmentation_free + 印花保真贴回接线）
  engine_sdxl.py           SDXL Inpaint 兜底引擎（对齐 + ControlNet + IP-Adapter + 动态遮罩 + 图案锁 + 接缝/肤色后处理）
  preserve_refit.py        印花保真管线（refit warp / 色调匹配 / 鬼影检测擦除 / 残余兜底 / 局部环形锚定 / 核心场混合）
  quality_gate.py          质检（SSIM / OCR / print_fidelity / delivery_tier）
  platform_presets.py      店规 + detect_garment_extent + clothing_mask + blend_tryon + reapply / pattern_lock + seam/skin + RTMPose/MediaPipe 关键点检测
  export_csv.py            CSV/SKU 批量导出
  lora_io.py               LoRA 加载（PEFT 优先）
  tests/                   pytest（448 passed）
src/                       前端 SPA + i18n
training/                  SDXL LoRA / ControlNet 训练脚本
models/sdxl/lora/          LoRA 权重（120 张训练，rank 16）
models/rtmpose/            RTMPose-m checkpoint（需手动下载）
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
```

## 安全

本项目经过两轮安全审查：

- 文件服务鉴权：`/files/{path}` 仅暴露白名单目录，仅允许图片后缀；上传路径正则白名单校验，拒绝路径穿越
- 图片内容验证：PIL 解码校验，仅接受 PNG/JPEG/WEBP，最大 20MB
- JWT 密钥无硬编码默认值；账号密码未设置时随机生成并持久化（chmod 600）
- 角色权限：admin 全操作，staff 仅创建任务和查看；凭据脱敏（`/api/stores` 不返回明文 api_key）
- SSRF 防护：`/api/push` 强制 https + `.myshopify.com`，拒绝内网地址
- XSS 防护：前端 `escapeHtml` 转义；服务端枚举白名单校验
- GPU 并发：扩散推理路径全局 `threading.Lock` 串行化；SDXL 管线缓存常驻；批量事务性（先全量校验再统一写盘）

已知限制：DNS 重绑定（TOCTOU）未完全消除；无 GPU 环境下合成引擎为启发式近似；去水印为浅色角部 best-effort。

## 授权说明

| 组件 | 许可证 | 商用 | 本项目使用方式 |
|------|--------|------|----------------|
| IDM-VTON 模型权重 | CC BY-NC-SA 4.0 | 禁止 | 仅借鉴 garment warp 思路，未使用其权重 |
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
| 本项目核心代码 | 商用权归作者所有 | 允许 | 换衣管线、印花保真、平台适配、后处理、训练脚本等全部自研代码 |

**商用部署**：设 `LICENSE_MODE=commercial`，引擎优先级 FASHN VTON v1.5 → SDXL Inpaint → 合成引擎，不加载 IDM-VTON。FASHN 主模型与 DWPose / YOLOX 权重为 Apache 2.0；fashn-human-parser（NVIDIA SegFormer）含商用限制，商用路径注入零张量 stub、权重永不下载。作者已获得 SDXL / ControlNet 商用授权，IP-Adapter / RTMPose / Rembg / MediaPipe 均为商用合规许可，可安全用于商业场景。

## 历史版本

| 版本 | 要点 |
|------|------|
| v0.1.8 | RTMPose-m 关键点检测替代 MediaPipe（侧/背面退化 13.7%→2.2%）；腰线 shoulder→hip 插值（0.544）；dress 脖子排除 + 膝部截断 |
| v0.1.7 | warp 垂直锚点对齐（logo 偏位修正）；GARMENT_BOX 品类垂直范围；肤色校正排除脸区；inpaint 修复聚焦遮罩 + 管线单例 |
| v0.1.6 | 频率分离 pattern lock（PLTON 思路）：底色全强度响应光照、花纹半强度——立体感与图案保真同时成立 |
| v0.1.5 | 换衣后遗症图像修复：胸口白块 / 手臂变白 / 上衣齐腰 / 图案被重画（见 `git log`） |
| v0.1.4 | Canny 按品类 / IP-Adapter 0.9 / detect_garment_extent / SeamlessClone 贴回 |
| v0.1.3 | 「AI 出骨架光影，代码保商品细节」混合策略 + LoRA 训练基线 |

## 感谢

- [FASHN](https://fashn.ai) — VTON v1.5 商用主引擎（像素空间，Apache 2.0）
- [IDM-VTON](https://github.com/yisol/IDM-VTON)（ECCV 2024）— garment warp 思路借鉴（未用其权重）
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)（arXiv:2306.06029）— COCO 17 关键点腰线插值 / 膝部截断 / 肩部掩膜
- [PLTON](https://arxiv.org/abs/2401.11239) — LF/HF 频率分离思想（CPU 后处理实现）
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) / [ControlNet](https://github.com/lllyasviel/ControlNet) / [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) — 商用管线生成底座与条件注入
- [Rembg](https://github.com/danielgatis/rembg) — 主体掩膜（U2Net）
- [MediaPipe Pose](https://github.com/google-ai/mediapipe) — 关键点检测回退
