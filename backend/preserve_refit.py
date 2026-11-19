from __future__ import annotations

import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger("atelier.preserve_refit")

# 贴印参考图长边下限: 禁止先缩到 1024 再 warp 丢掉印花高频。
REFIT_REF_LONG_SIDE_MIN = 1024


def protect_inverse_refit(
    result: Image.Image,
    source: Image.Image,
    recast_mask: Image.Image,
    feather_px: int = 4,
) -> Image.Image:
    """遮罩外保护贴回: 源图像素贴回结果图, 仅重绘区域内保留生成像素。

    替代 blend_tryon + _seam_repair 两步为一步:
    1. protect_mask = 全图 - (recast_mask 膨胀 feather_px)
    2. result[protect_mask] = source[protect_mask]
    3. 边界带(膨胀带 - 未膨胀带)用 cv2.inpaint TELEA 平滑过渡

    脸、发丝、手、露肤手臂、背景保持源图像素; 遮罩内保留生成像素。
    VTON_LOCK_ID=0 时调用方跳过本函数。

    Args:
        result: 生成图 (RGB)
        source: 源图 (RGB), 与 result 同尺寸
        recast_mask: 重绘区域掩膜 (L, 255=重绘, 0=保护)
        feather_px: 边界带宽度, 0=无羽化直接贴回

    Returns:
        RGB 贴回结果
    """
    res = result.convert("RGB")
    src = source.convert("RGB")
    if res.size != src.size:
        src = src.resize(res.size, Image.Resampling.LANCZOS)
    r_arr = np.array(res, dtype=np.uint8)
    s_arr = np.array(src, dtype=np.uint8)

    mask = np.asarray(recast_mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != r_arr.shape[:2]:
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize((r_arr.shape[1], r_arr.shape[0]), Image.Resampling.BILINEAR)
        )

    # 保护区域 = 全图 - recast_mask(膨胀羽化带)
    if feather_px > 0:
        try:
            import cv2

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_px * 2 + 1, feather_px * 2 + 1))
            dilated = cv2.dilate(mask, k)
            dilated = cv2.dilate(mask, k)
            # 边界带 = 膨胀带 - 原始遮罩 (薄环)。此前误把整块膨胀区传给
            # cv2.inpaint, TELEA 把整件衣服当待修复区, 从四周白背景重建
            # 出白色糊块 (GPU 实测: apply 区 0.37 画布全部变白)。
            boundary = ((dilated > 20) & (mask <= 20)).astype(np.uint8) * 255
            protect_mask = (mask <= 20).astype(np.uint8)
            protect_mask[boundary > 0] = 0  # 边界带不进保护, 留给 Telea
        except ImportError:
            protect_mask = (mask <= 20).astype(np.uint8)
            boundary = None
    else:
        protect_mask = (mask <= 20).astype(np.uint8)
        boundary = None

    # 贴回源图像素到保护区域
    out = r_arr.copy()
    out[protect_mask > 0] = s_arr[protect_mask > 0]

    # 边界带 Telea 接缝修复
    if boundary is not None and boundary.any():
        try:
            import cv2

            out = cv2.inpaint(out, boundary, 5, cv2.INPAINT_TELEA)
        except Exception:  # noqa: BLE001
            pass

    return Image.fromarray(out, "RGB")


def tps_warp(
    ref_img: Image.Image,
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
    canvas_size: tuple[int, int],
) -> Image.Image:
    """RBF TPS 几何 warp: 把 SKU 平铺图投到人体表面。

    控制点来自 RTMPose 关键点 + 服装轮廓采样。TPS 比网格 remap 更平滑,
    高频印花不随扩散重解释。

    Args:
        ref_img: SKU 平铺图 (RGBA)
        src_pts: 平铺图锚点 (领口/肩/下摆/袖口), 至少 4 个
        dst_pts: 人体对应位置, 与 src_pts 等长
        canvas_size: 输出画布 (w, h)

    Returns:
        RGBA warp canvas, 透明背景
    """
    if len(src_pts) < 4 or len(src_pts) != len(dst_pts):
        logger.warning("tps_warp: need >= 4 anchor pairs, got %d; returning original", len(src_pts))
        return ref_img.resize(canvas_size, Image.Resampling.LANCZOS)

    try:
        import cv2

        arr = np.array(ref_img, dtype=np.float32)
        h, w = ref_img.size[1], ref_img.size[0]
        canvas_w, canvas_h = canvas_size

        # 源锚点 (x, y) 在平铺图上, 目标锚点在画布上
        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)

        # 逆向 TPS: 以目标锚点为节点、源锚点为值, 求 g: 画布坐标 -> 平铺图坐标,
        # remap 时对每个画布像素取 g 求采样位置(逆向映射, 高频印花不走样)。
        # Bookstein 方程组: [[K, P^T], [P, 0]] (n+3)x(n+3), P = [1; x; y] (3, n)。
        #
        # 坐标归一化: 像素尺度下 K 项 ~ r^2 log r (1e6) 与 P 项 (~1e2) 混尺度,
        # cond(M) 实测 ~1e13, 同锚点跨 LAPACK 版本解发散(GPU r5 实测 warp 塌成
        # 斜条, fidelity 0.498)。统一缩到 [0,1]^2 求解再反缩放, 节点精确性不变。
        n = len(src)
        dst_min = dst.min(axis=0)
        dst_rng = np.maximum(dst.max(axis=0) - dst_min, 1e-9)
        src_min = src.min(axis=0)
        src_rng = np.maximum(src.max(axis=0) - src_min, 1e-9)
        dst_n = (dst - dst_min) / dst_rng
        src_n = (src - src_min) / src_rng

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = _rbf_kernel(dst_n[i], dst_n[j])
        P = np.vstack([np.ones((1, n)), dst_n.T])
        M = np.zeros((n + 3, n + 3))
        M[:n, :n] = K
        M[:n, n:] = P.T
        M[n:, :n] = P
        rhs = np.zeros((n + 3, 2))
        rhs[:n, :] = src_n
        sol = np.linalg.solve(M, rhs)
        # 求解守卫: 病态锚点(共线/重复)可能静默解出垃圾, 残差检查兜底,
        # 触发即抛错走 LANCZOS 回退, 不把烂映射交给贴印。
        _resid = float(np.abs(M @ sol - rhs).max())
        if not np.isfinite(_resid) or _resid > 1e-6:
            # 换道重试: GPU 服务器 OpenBLAS ILP64 的 LAPACK gesv 内核在
            # 良态矩阵上返回垃圾(实测 cond 20 的 11x11 残差 2.7117555,
            # 与冒烟日志逐位一致; r6-r8 GPU 因此全部静默回退 LANCZOS,
            # fidelity 卡 0.85 平台), 同库 SVD 路径(lstsq) 残差 1.3e-15
            # 正常。solve 失败先换 lstsq, 真病态锚点才抛错。
            sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
            _resid = float(np.abs(M @ sol - rhs).max())
            if not np.isfinite(_resid) or _resid > 1e-6:
                raise ValueError(f"TPS solve residual too large: {_resid}")
        w_coeff = sol[:n, :]
        aff = sol[n:, :]

        # 生成目标像素网格
        xg, yg = np.meshgrid(np.arange(canvas_w, dtype=np.float64), np.arange(canvas_h, dtype=np.float64))
        gx = np.stack([xg, yg], axis=-1).reshape(-1, 2)
        gx_n = (gx - dst_min) / dst_rng

        # TPS 映射: 目标画布 -> 源平铺图
        K_g = np.zeros((len(gx), n))
        for i in range(n):
            K_g[:, i] = _rbf_kernel_batch(gx_n, dst_n[i])
        G = np.hstack([np.ones((len(gx), 1)), gx_n])
        src_coords = (G @ aff + K_g @ w_coeff) * src_rng + src_min

        # 采样越界裁剪前先记 valid 掩膜: np.clip 会把画布外采样点硬夹到
        # 源图边缘(边缘涂抹), 对 alpha 全开的纯色衣会把整张画布涂满;
        # 越界像素必须显式清零(透明), 不能复用边缘值。
        # 半像素容差: TPS 数值噪声让边界锚点坐标出现 -1e-6 / (w-1)+1e-6,
        # 严格 >=0 会把本该有效的边界像素清零(锚点对齐回归)。
        valid = (
            (src_coords[:, 0] >= -0.5)
            & (src_coords[:, 0] <= w - 0.5)
            & (src_coords[:, 1] >= -0.5)
            & (src_coords[:, 1] <= h - 0.5)
        ).reshape(canvas_h, canvas_w)

        sx = np.clip(src_coords[:, 0], 0, w - 1).astype(np.float32).reshape(canvas_h, canvas_w)
        sy = np.clip(src_coords[:, 1], 0, h - 1).astype(np.float32).reshape(canvas_h, canvas_w)

        map_x = sx
        map_y = sy
        warped = cv2.remap(
            arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
        )
        warped[~valid] = 0
        return Image.fromarray(warped.astype(np.uint8), "RGBA")
    except Exception:
        logger.exception("tps_warp failed; falling back to LANCZOS resize")
        return ref_img.resize(canvas_size, Image.Resampling.LANCZOS)


def _rbf_kernel(p1: np.ndarray, p2: np.ndarray) -> float:
    """Thin-plate-spline RBF kernel: r^2 * log(r)."""
    r = float(np.linalg.norm(p1 - p2))
    if r < 1e-12:
        return 0.0
    return r * r * np.log(r)


def _rbf_kernel_batch(pts: np.ndarray, center: np.ndarray) -> np.ndarray:
    """向量化 RBF kernel: 多个点到单个中心."""
    d = pts - center
    r = np.linalg.norm(d, axis=1)
    out = np.zeros_like(r)
    valid = r > 1e-12
    r2 = r[valid]
    out[valid] = r2 * r2 * np.log(r2)
    return out


def detect_print_roi(
    cloth_ref: Image.Image,
    apply_mask: Image.Image,
    edge_px: int = 4,
    hf_thresh: float = 12.0,
    min_area_frac: float = 0.002,
) -> np.ndarray | None:
    """印花 ROI 边界带: warp 参考图在贴回区内的高频区(logo/字母/大图案)的边缘带。

    高频 = |灰度 - 高斯模糊| 超阈值的区域(印花/文字/缝合线), 收缩到贴回
    mask 内, 形态学去噪后取 dilate-erode 边界带。字母/图案内部像素由
    确定性贴印(pattern_lock/TPS)构成, 不进 inpaint——只有贴印区与生成
    衣料的边界带进局部 inpaint 做混合。

    Args:
        cloth_ref: warp 后的 SKU 平铺参考图 (RGB/RGBA)
        apply_mask: 贴回区域掩膜 (L, 255=重绘)
        edge_px: 边界带半宽(px)
        hf_thresh: 高频能量阈值(0-255 灰度差)
        min_area_frac: ROI 面积占比下限, 低于此返回 None(纯色款无印花)

    Returns:
        uint8 边界带掩膜(255=进 inpaint), 纯色/失败返回 None
    """
    try:
        import cv2

        ref = np.asarray(cloth_ref.convert("L"), dtype=np.float32)
        mask = np.asarray(apply_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != ref.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR)
            )

        blur = cv2.GaussianBlur(ref, (0, 0), 4.0)
        hf = np.abs(ref - blur)
        roi = ((hf > hf_thresh) & (mask > 127)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)

        if float(roi.mean()) / 255.0 < min_area_frac:
            return None

        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_px * 2 + 1, edge_px * 2 + 1))
        dil = cv2.dilate(roi, ke)
        ero = cv2.erode(roi, ke)
        band = cv2.subtract(dil, ero)
        return band
    except Exception:
        logger.exception("detect_print_roi failed")
        return None


def print_core_mask(
    cloth_ref: Image.Image,
    keep_mask: np.ndarray | Image.Image,
    color_thresh: float = 60.0,
    min_area_frac: float = 0.004,
    grow_px: int = 3,
) -> np.ndarray | None:
    """印花核心区: 与衣料基色色距大的近中性色/异色相实心块。

    v1 高频带把门襟/纽扣/褶皱全当印花(GPU r6 视觉验收: 门襟中线纵向
    拉丝); v2 基色色距过滤挡不住同色相褶皱阴影(真丝暗褶的色度偏移
    同样 >40)。v3 三轴联合, 商用印花语义:
    1) 色距 > color_thresh: 与 keep 区基色(中位 RGB)欧氏距离足够大;
    2) 近中性色(像素 RGB max-min < 40): 白/灰/黑印花, 面料同色相的
       褶皱阴影天然带面料色度, 不满足;
    3) 或色相偏移(像素色度向量与基色色度向量夹角 > 30 度): 彩色印花;
       褶皱阴影沿基色色相方向变暗/变亮, 色相守恒, 被排除。
    形态过滤: 3px 腐蚀存活 + 连通域面积 >= max(150, 8% 最大块) +
    bbox 短边 >= 8px(细缝线/纽扣碎片即便膨胀串联也会先在成分级被杀)。
    回膨胀(恢复腐蚀 + grow_px 外扩盖抗锯齿缝), 裁回 keep_mask。

    与 detect_print_roi(dilate-erode 边界带, 供局部 inpaint)互补,
    本函数返回印花本体实心区, 作为确定性贴印区域与印花保真分测区。
    纯色款/同色相暗褶面料(全中性灰底的深褶有残留风险)返回 None,
    调用方回退整件衣料贴回。

    Args:
        cloth_ref: warp 后的 SKU 参考图 (RGB/RGBA)
        keep_mask: 衣料保留区掩膜 (L/ndarray, 255=衣料)
        color_thresh: 与基色的欧氏色距阈值(0-255 RGB 尺度)
        min_area_frac: 候选/核心区面积占衣料区(keep)比下限, 低于此返回 None
        grow_px: 核心区外扩像素(边缘覆盖), 0=不扩

    Returns:
        uint8 核心区掩膜(255=贴印区), 纯色/失败返回 None
    """
    try:
        import cv2

        ref = np.asarray(cloth_ref.convert("RGB"), dtype=np.float32)
        mask = np.asarray(keep_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != ref.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype("uint8")).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR)
            )
        # keep 内缩 4px + cand 腐蚀 3px 合计清 warp 白边距环: keep 的
        # dilate(alpha,6) 外沿是纯白画布 RGB(近中性+大色距, 会成环状
        # cand, 其凸包=整件衣 → 恒等整件贴回), 环宽=6px; 内缩 4px 留
        # 2px 残环由 cand 腐蚀(3px)清零。不能用衣料 alpha 门槛: 白墨
        # 印花与画布同色, garment_alpha 把笔触判成 alpha=0 透明洞(实测
        # min 0), 满印墨迹网连通触边, 任何 alpha 拓扑内外判定无解;
        # 内缩 8px 会误杀贴边真印花且小画布占比失真。
        ka_img = (mask > 127).astype(np.uint8)
        ka_inner = cv2.erode(ka_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        ka = ka_inner.reshape(-1)
        if ka.sum() < 64:
            return None
        keep_area = float(np.asarray(mask > 127).sum())
        px = ref.reshape(-1, 3)
        base = np.median(px[ka], axis=0)
        dist = np.linalg.norm(px - base, axis=-1)
        neutral = (px.max(-1) - px.min(-1)) < 40.0
        cv_px = px - px.mean(-1, keepdims=True)
        cv_base = base - float(base.mean())
        n_px = np.linalg.norm(cv_px, axis=-1)
        cos = (cv_px @ cv_base) / np.maximum(n_px * float(np.linalg.norm(cv_base)), 1e-6)
        hue_shift = cos < 0.866
        # 灰基色(白/灰衣)的 cv_base≈0, 色相轴失效: 彩色印花改判
        # "像素自身带显著色度"(褶皱在灰衣上是纯亮度变化, 天然无色度)
        base_neutral = float(base.max() - base.min()) < 15.0
        px_chromatic = n_px > 25.0
        # 亮向约束(彩色基色): 近中性印花必须比基色亮。深褶阴影脱饱和后
        # 同样过 dist+neutral 轴(1536 实测腹部 188x193 暗块), 亮度方向是
        # 印花(亮)/阴影(暗)的本质分界; 黑印花款回退整件贴回仍交付。
        bright = np.ones_like(neutral) if base_neutral else px.mean(-1) >= float(base.mean()) + 10.0
        cand = ka & (dist > color_thresh) & ((neutral & bright) | hue_shift | (base_neutral & px_chromatic))
        cand = cand.reshape(ref.shape[:2])
        # 候选/核心门槛按衣料区(keep)占比而非画布占比: FASHN 原生 576x768 画布下
        # warp 把印花面积缩 16 倍, 画布占比门把真印花判成纯色(r8 GPU 实测 core=None
        # 回退整件贴回, 印花全程被 hf/folds 调制融化)。衣料占比跨工作分辨率不变;
        # 绝对下限只拦纯噪声, 实质过滤由腐蚀+成分级 150px 面积门承担。
        if float(cand.sum()) < max(16.0, min_area_frac * keep_area):
            return None
        c8 = cand.astype(np.uint8) * 255
        # cand 腐蚀只清 1px 噪声: 白墨笔触本身仅 3-4px 细, 半径 2 的腐蚀
        # 把笔触网打成碎末(实测 135 个 <160px 碎片, 全过不了面积门 → 误判
        # 纯色); keep 内缩 4px 残下的 2px 白环由成分级 bbox 跨度门拒绝。
        er = cv2.erode(c8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if er.max() == 0:
            return None
        num, lab, stats, _ = cv2.connectedComponentsWithStats((er > 0).astype(np.uint8))
        if num <= 1:
            return None
        # keep(dilated-alpha) 外沿的纯白画布残环: 环宽仅 2px(内缩 4px 后)
        # 但包围整件衣, 凸包=整件 → 恒等整件贴回(灾难)。成分 bbox 跨过
        # keep 尺寸 70% 即拒: 真印花(胸印花 bbox 39%)安全余量充足。
        k_ys, k_xs = np.where(ka_img)
        k_w = float(k_xs.max() - k_xs.min())
        k_h = float(k_ys.max() - k_ys.min())
        # 成分级纹理门(笔触/高光判别): 白色印花笔触与丝绸高光斑在颜色轴上
        # 不可分(同为亮中性+大色距, GPU 实测高光斑全过 v3.1 颜色轴), 只有
        # 纹理可分 —— 笔触=边缘高能, 高光=平滑梯度。测度带 = cand ∩ 成分
        # 支撑(dilate23) 减成分自身: 印花候选的边界笔触/实心 logo 的边界
        # 落在带内(canny 边缘线就在 cand 边界上, erode 成分中心距自身边界
        # 天然 ≥3px, 固定外环带会错过边缘线), 高光斑梯度平缓 canny 不点火
        # (大面积实心 logo 的边缘密度不会被面积稀释)。
        gray = np.asarray(cloth_ref.convert("L"), dtype=np.uint8)
        if gray.shape[:2] != ref.shape[:2]:
            gray = np.asarray(cloth_ref.convert("L").resize((ref.shape[1], ref.shape[0])), dtype=np.uint8)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90) > 0
        kd_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        cand_b = cand.copy()
        # 面积门用绝对 150px: 0.08x最大成分的相对门在有大型种子时
        # 把中尺寸印花片段(下腹印花腐蚀碎片)整批误杀(实测丢 2/4 印花);
        # 粉尘(≤20px)由绝对门拦, 大结构由 bbox 跨度门+纹理门+亮向轴拦。
        floor = 150.0
        seeds = np.zeros(ref.shape[:2], dtype=np.uint8)
        for i in range(1, num):
            if float(stats[i, cv2.CC_STAT_AREA]) < floor:
                continue
            if min(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])) < 8:
                continue
            if float(stats[i, cv2.CC_STAT_WIDTH]) > 0.7 * k_w or float(stats[i, cv2.CC_STAT_HEIGHT]) > 0.7 * k_h:
                continue
            comp = (lab == i).astype(np.uint8)
            band = cand_b & (cv2.dilate(comp, kd_out) > 0) & (lab != i)
            if band.sum() < 32:
                continue
            # canny 密度单判据: 笔触≥11%/实心logo边界≈49%(GPU+合成实测),
            # 高光斑≤2.1% —— 5% 门两侧各余量一倍; 拉普拉斯能量对边缘
            # 锐度敏感(LANCZOS 软化后 15.8 与高光 14 同档), 不做门槛。
            if float(edges[band].mean()) * 100.0 < 5.0:
                continue
            seeds[lab == i] = 1
        # 种子发芽: 笔触网被 cand 腐蚀(1px)碎成大量 <floor 片段, 但整个
        # 网络在 cand 上连通 —— 种子所在的 cand 成分整块并入, 碎片随
        # 种子复活(实测: 只用腐蚀成分会丢 60% 印花面积, 鬼影残留);
        # 噪声粉尘与种子不相交, 留在门外。凸包按 cand 成分逐个填充:
        # 细笔触印花只贴笔触会留 FASHN 鬼影(NCC -0.4), 凸包=印花场
        # 整场贴参考原样。
        if seeds.max() == 0:
            return None
        _cnum, clab, _, _ = cv2.connectedComponentsWithStats((cand_b > 0).astype(np.uint8))
        sel_ids = {int(v) for v in np.unique(clab[seeds > 0])} - {0}
        hulls = []
        for cid in sel_ids:
            pts = np.column_stack(np.where(clab == cid))
            if len(pts) < 32:
                continue
            hull = cv2.convexHull(pts[:, ::-1].astype(np.int32).reshape(-1, 1, 2))
            # 贴边印花 all-or-nothing 拒绝: 种子成分在 cand 层(keep 内缩 9px)
            # 永远全界内, 但凸包填充按凸性跨过 keep 硬边界 —— 跨界场贴回时
            # 被 content 门(keep>127)+roi 软边切成半张印花, 比不贴更糟
            # (r10 GPU 2048 实测下腹印花 84% 入界 → 半贴半留发糊, 视觉
            # "右手图案没保真"; 1536 下该印花不检出反而干净)。凸包填充
            # 骑 keep 硬区 ≥95% 才收, 否则整场拒绝。
            _pf = np.zeros(ref.shape[:2], dtype=np.uint8)
            cv2.fillPoly(_pf, [hull], 255)
            if float(ka_img[_pf > 0].mean()) < 0.95:
                continue
            hulls.append(hull)
        if not hulls:
            return None
        field = np.zeros(ref.shape[:2], dtype=np.uint8)
        cv2.fillPoly(field, hulls, 255)
        if float((field > 0).sum()) < max(32.0, min_area_frac * keep_area):
            return None
        roi = cv2.dilate(field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if grow_px > 0:
            roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_px * 2 + 1, grow_px * 2 + 1)))
        return np.minimum(roi, mask.astype(np.uint8))
    except Exception:
        logger.exception("print_core_mask failed")
        return None


def ghost_print_mask(
    gen_img: Image.Image,
    cloth_ref: Image.Image,
    keep_mask: np.ndarray | Image.Image,
    exclude_mask: np.ndarray | Image.Image | None = None,
) -> np.ndarray | None:
    """生成图上的印花鬼影笔画检测(FASHN 自绘印花与 TPS 贴印几何错位时擦除用)。

    FASHN 生成器按自己的理解摆放印花(实测比平铺几何偏上), 贴印核心只盖
    TPS 映射位置 → 生成图的鬼影整套暴露在核心场之外(视觉验收: 上方
    半透明猴头重影)。本函数在生成图上找鬼影笔画: 同色轴(亮中性+大色距,
    基色取参考图衣料区中位数 —— prewhiten 已证两图基色一致)+ bbox
    跨度门(整件环状结构); 无纹理门(生成图丝绸渲染处处带噪, 鬼影 4.3
    与高光 4.5 lapE 同档, 实测色轴零高光假阳性)。

    掩膜 = 笔画连通成分(+种子发芽)+9px margin, 非凸包填充(r11 GPU
    视觉验收"衣袖失真": 凸包按凸性把袖区 26.5k 自然袖料吞进恒等贴平
    区)。擦除由调用方执行: ghost_inpaint_pass 局部修补保袖褶 +
    ghost_strict_residual 兜底。

    Returns:
        uint8 掩膜(255=鬼影笔画+margin), 无检出/失败返回 None
    """
    try:
        import cv2

        gen = np.asarray(gen_img.convert("RGB"), dtype=np.float32)
        ref = np.asarray(cloth_ref.convert("RGB"), dtype=np.float32)
        mask = np.asarray(keep_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != gen.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype("uint8")).resize((gen.shape[1], gen.shape[0]), Image.Resampling.BILINEAR)
            )
        ka_img = (mask > 127).astype(np.uint8)
        ka_inner = cv2.erode(ka_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        ka = ka_inner.reshape(-1)
        if ka.sum() < 64:
            return None
        px_ref = ref.reshape(-1, 3)
        base = np.median(px_ref[ka], axis=0)
        px = gen.reshape(-1, 3)
        dist = np.linalg.norm(px - base, axis=-1)
        neutral = (px.max(-1) - px.min(-1)) < 40.0
        bright = px.mean(-1) >= float(base.mean()) + 10.0
        cand = (ka & (dist > 60.0) & (neutral & bright)).reshape(gen.shape[:2])
        # 排除环 9px(与核心场自身外扩对齐)而非 19: 场(hull+9)与鬼场
        # (≥core+9, 场末 dilate9 回填)必须无缝衔接, 否则 9-19px 环带内
        # FASHN 银白残影两头都不清理(r9b GPU 视觉验收: 印花轮廓被
        # 白色不规则边缘切割); 环带成分 ~10px 宽, 腐蚀 1px 后仍可成核。
        ex_d = None
        if exclude_mask is not None:
            ex = np.asarray(exclude_mask)
            if ex.ndim == 3:
                ex = ex[..., 0]
            if ex.shape[:2] != gen.shape[:2]:
                ex = np.array(
                    Image.fromarray(ex.astype("uint8")).resize((gen.shape[1], gen.shape[0]), Image.Resampling.BILINEAR)
                )
            ex_d = cv2.dilate((ex > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            cand = cand & (ex_d == 0)
        if float(cand.sum()) < 32:
            return None
        er = cv2.erode(cand.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if er.max() == 0:
            return None
        num, lab, stats, _ = cv2.connectedComponentsWithStats((er > 0).astype(np.uint8))
        if num <= 1:
            return None
        k_ys, k_xs = np.where(ka_img)
        k_w = float(k_xs.max() - k_xs.min())
        k_h = float(k_ys.max() - k_ys.min())
        # 无纹理门: 生成图丝绸渲染处处带噪(lapE 鬼影 4.3-5.2 vs 高光 4.5,
        # canny10/30 0.7-18% vs 7.8% 同档互叠), 纹理无法区分; 色轴已足:
        # 生成图暖调高光(230,215,175 类)不过中性轴, 实测 gen 侧 cand 只有
        # 鬼影+印花残迹, 零高光假阳性。
        # 掩膜语义 = 笔画本身 + 9px margin(非凸包填充): r11 GPU 视觉验收
        # "衣袖失真"—— 凸包按凸性把袖区 26.5k 自然袖料吞进恒等贴平区,
        # 擦除语义只需盖住笔画, 调用方(erase_ghost_prints 流程)对笔画
        # inpaint 局部修补 + 残余亮块小范围贴参考兜底, 袖褶光影全保。
        seeds = np.zeros(gen.shape[:2], dtype=np.uint8)
        for i in range(1, num):
            if float(stats[i, cv2.CC_STAT_AREA]) < 150:
                continue
            if min(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])) < 8:
                continue
            if float(stats[i, cv2.CC_STAT_WIDTH]) > 0.7 * k_w or float(stats[i, cv2.CC_STAT_HEIGHT]) > 0.7 * k_h:
                continue
            seeds[lab == i] = 1
        if seeds.max() == 0:
            return None
        # 种子发芽: 腐蚀(3px)碎片段经 cand 连通性复活(与 print_core_mask
        # 同构), margin 9px 覆盖笔画 AA + 半透晕边。掩膜不裁核心场: margin
        # 越过 core 边缘的部分由后续核心恒等贴印重贴(贴印 alpha=1 + feather
        # 1px, 不透出底层), 而 [core, core+4) 环内的鬼影笔画必须有人管
        # (r12 重放实测: 裁掉该环会让擦除带残留 +17 灰阶)。
        _gnum, glab, _, _ = cv2.connectedComponentsWithStats((cand > 0).astype(np.uint8))
        sel_ids = np.isin(glab, np.unique(glab[seeds > 0]))
        strokes = np.where(sel_ids & (cand > 0), np.uint8(255), np.uint8(0)).astype(np.uint8)
        roi = cv2.dilate(strokes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
        return np.minimum(roi, ka_img * 255)
    except Exception:
        logger.exception("ghost_print_mask failed")
        return None


def _coerce_mask_to_img(mask: np.ndarray | Image.Image, shape: tuple[int, int]) -> np.ndarray:
    """mask 对齐到图像尺寸(灰度化+bilinear 缩放), 供鬼影系列函数复用。"""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != tuple(shape):
        m = np.array(Image.fromarray(m.astype("uint8")).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))
    return m


def _nearest_fabric_color(arr_f: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """每像素**最近**净布料像素颜色(粗分辨率 Voronoi 近邻色传播)。

    全程纯 numpy 实现, 语义必须跨机一致: 沙箱 cv2 5.0.0 非真
    OpenCV(DIST_LABEL_PIXEL 标签实测恒 0; resize 会压掉单通道维,
    (10,20,1)→(5,5) 返回 2D), GPU 服务器 cv2 4.8.0 才是真实现 ——
    行为差异让沙箱测试不可信。另: 沙箱 numpy 2.4.6 广播不前导补维
    (右对齐), 构造行渐变测试数据须逐行显式赋值。1/8 粗尺度(切片
    降采样)上净布料色掩蔽膨胀外扩, 最近邻回填(np.repeat x8),
    基色采样精度 ±8px 对平滑布料渐变足够。
    高光袖料上鬼影印在高光带(局部 ~220,205,160), 单一全局/成分级基色
    会把与高光同色的合法填充误判; 最近净布料色让袖面像素自动对袖料
    高光、身面像素自动对衣身暗料。
    传播两阶段: ①纯横向核(5x1) —— 布料明暗沿行渐变(高光带在上/阴影
    在下, r13 GPU 实测右袖 B 159@y620 → 125@y920), 且值传播用膨胀
    (窗口取 max), 多行核会把上方更亮布料逐行复合进来(实测洞中心取到
    57 灰阶之外的顶行色) → 同行布料才是正确锚, 单行核无跨行复合;
    ②等方核(3x3)收尾 —— 完全被盖住的行(竖洞)横向无源, 竖向补齐。
    arr_f: float32 RGB; clean: bool。
    """
    step = 8
    h, w = arr_f.shape[:2]
    arr_s = arr_f[::step, ::step].astype(np.float32)
    clean_s = clean[::step, ::step]
    val = arr_s * clean_s[..., None]
    known = clean_s.copy()
    ch, cw = known.shape
    for kx, ky in ((5, 1), (3, 3)):
        py, px = ky // 2, kx // 2
        for _ in range(512):
            if known.all():
                break
            vp = np.pad(val, ((py, py), (px, px), (0, 0)))
            kp = np.pad(known, ((py, py), (px, px)), constant_values=False)
            gv = vp[py : py + ch, px : px + cw].copy()
            gk = kp[py : py + ch, px : px + cw].copy()
            for oy in range(ky):
                for ox in range(kx):
                    if oy or ox:
                        gv = np.maximum(gv, vp[oy : oy + ch, ox : ox + cw])
                        gk = np.maximum(gk, kp[oy : oy + ch, ox : ox + cw])
            fill = gk & ~known
            if not fill.any():
                break
            val[fill] = gv[fill]
            known = gk
    return np.repeat(np.repeat(val, step, axis=0), step, axis=1)[:h, :w]


def anchor_ref_local(
    gen_img: Image.Image | np.ndarray,
    matched_ref: Image.Image | np.ndarray,
    keep_mask: np.ndarray | Image.Image,
    region_mask: np.ndarray | Image.Image,
    exclude_masks: list | None = None,
    grow_px: int = 40,
) -> Image.Image | np.ndarray:
    """色调匹配参考的**局部**环形锚定: 贴回区内容对齐 gen 贴回区周边净布料。

    全局色调匹配按整件 keep 布料统计锚定; 生成图布料明暗跨区域大幅变化
    (r13 GPU 实测: 亮胸口 185-195 ↔ 深色下摆 57-96), 贴回区(核心场/
    残余兜底)落在亮胸口时, 全局匹配按中位(96)锚定的暗布料(94)贴上
    去成 413x300 深色矩形 —— "衣服像贴上去的/整体下移/衣领断层"
    (r13 GPU 视觉验收)。反向场景同理(r10: 平铺亮衣料 198+ 拍上暗
    衣身 96 一圈发光) —— 逐通道 delta = gen 周边净布料中位 - 匹配
    参考周边布料中位, 加性平移到两侧等价, 两个方向都收敛。

    delta 按**行分段**估计: 布料明暗沿行渐变(同一环带内 gen 胸口
    191 ↔ 环下部 120), 单一环中位对准的是"平均布料"而非贴回行局部
    —— 衣领区(场顶, 最亮)欠锚最重("衣领断层"残因); 每行取 ±24 行
    窗口内的环净布料中位, 与袖口修复(同行布料取样)同一原理。
    加性平移保留参考内部明暗起伏; 印花笔触不受影响 —— blend_core_ref_tone
    的笔触保持取自原始参考, 本函数只动布料背景。净布料 = 环带内排除
    亮中性(笔画/高光白)与 exclude_masks(核心场/鬼场等) 的像素;
    成对样本 <48px 或 |全局 delta|>160(环采样不可信; 袖面高光 vs
    阴影实测 135 合法)时原样返回; 行窗口样本 <32px 或 |delta|>160
    时该行回退全局 delta。行 delta 按 1px 限幅 2 灰阶双向多遍收敛,
    消除环带边界/样本切换处中位突变(2048 尺度约 1-2 灰阶/px 的爬坡,
    贴回区横向条带"涂抹感"来源, r13 GPU 视觉驳回定位; 布料渐变斜率
    实测 <0.1 灰阶/行, 限幅远大于渐变不误伤)。

    Returns:
        与 matched_ref 同尺寸同类型的锚定版(gen 侧带 .info 时保留)
    """
    try:
        gen = np.asarray(gen_img.convert("RGB") if isinstance(gen_img, Image.Image) else gen_img, np.float32)
        mat = np.asarray(
            matched_ref.convert("RGB") if isinstance(matched_ref, Image.Image) else matched_ref, np.float32
        )
        if gen.shape != mat.shape:
            logger.warning("anchor_ref_local: gen/matched 形状不一致, 原样返回")
            return matched_ref
        import cv2

        keep = np.asarray(keep_mask)
        keep = keep[..., 0] if keep.ndim == 3 else keep
        ka = cv2.erode((keep > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        reg = np.asarray(region_mask)
        reg = reg[..., 0] if reg.ndim == 3 else reg
        reg_b = reg > 127
        ring = (
            (
                cv2.dilate(
                    reg_b.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)),
                )
                > 0
            )
            & ~reg_b
            & ka
        )
        if exclude_masks:
            for ex in exclude_masks:
                if ex is None:
                    continue
                exa = np.asarray(ex)
                exa = exa[..., 0] if exa.ndim == 3 else exa
                ring &= ~(exa > 127)
        if ring.sum() < 48:
            return matched_ref
        # 净布料过滤(两侧同口径): 排除亮中性(白墨笔画/高光白)与匹配侧
        # 画布白底; 成对过滤 —— 同一像素须两侧都是布料, 中位数对的才
        # 是同一块布

        def _fabric_px(a):
            return ((a.max(-1) - a.min(-1)) > 35) | (a.mean(-1) < 190)

        paired = ring & _fabric_px(gen) & _fabric_px(mat) & (mat.mean(-1) < 245)
        if paired.sum() < 48:
            return matched_ref
        rg = gen[paired]
        rm = mat[paired]
        delta_g = np.median(rg, axis=0) - np.median(rm, axis=0)
        if np.abs(delta_g).max() > 160.0:
            logger.warning("anchor_ref_local: delta %s 过大, 环采样不可信, 原样返回", delta_g.round(1).tolist())
            return matched_ref
        # 行分段 delta: 每行 ±24 行窗口内的成对环布料中位
        h = mat.shape[0]
        band = 24
        delta_row = np.repeat(delta_g[None, :], h, axis=0)
        ring_rows = np.flatnonzero(paired.any(axis=1))
        if ring_rows.size:
            y0, y1 = int(ring_rows[0]), int(ring_rows[-1])
            for y in range(max(0, y0 - band), min(h, y1 + band + 1)):
                lo, hi = max(0, y - band), min(h, y + band + 1)
                pw = paired[lo:hi]
                if pw.sum() < 32:
                    continue
                d_y = np.median(gen[lo:hi][pw], axis=0) - np.median(mat[lo:hi][pw], axis=0)
                if np.abs(d_y).max() <= 160.0:
                    delta_row[y] = d_y
            # 带边界跳变平滑: ±24 行窗在环样本稀疏/切换处中位 delta 突变
            # (r13 GPU 实测: 贴回区行带边界处相邻行 delta 跳 40+ 灰阶,
            # 贴回区可见横向条带"涂抹感")。行带内 delta 按 1px 限幅 2 灰阶
            # (布料明暗渐变斜率 r13 实测 <0.1 灰阶/行, 限幅远大于渐变
            # 不误伤): 行带内 delta 视为若干水平段(中位不突变) + 跳变
            # 边界, 跳变段(两个平台间的行, 贴回区外无成对样本)填线性爬坡
            # (每行 ≤2 灰阶), 跳变前后平台行不变 —— 贴回区(平台内)局部
            # 锚定不变, 爬坡端点处 delta 与平台值连续; 逐元素标量赋值,
            limit = 2.0
            y0s, y1s = max(0, int(ring_rows[0]) - band), min(h, int(ring_rows[-1]) + band + 1)
            dr = delta_row[y0s:y1s].copy()
            N = dr.shape[0]
            for _pass in range(8):
                step_mag = np.abs(np.diff(dr, axis=0)).max(axis=1)
                bound = np.flatnonzero(step_mag > limit).tolist()
                if not bound:
                    break
                progressed = False
                for b in bound:
                    b = int(b)
                    b2 = b
                    while b2 + 1 < step_mag.shape[0] and step_mag[b2] > limit:
                        b2 += 1
                    left_v = dr[b - 1].copy()
                    right_v = dr[b2 + 1].copy() if b2 + 1 < N else dr[b2].copy()
                    jump_mag = float(np.abs(left_v - right_v).max())
                    if jump_mag <= limit:
                        continue
                    # 爬坡段 [lo, hi] 首尾平台值不变, 中间行(lo+1..hi)
                    # 摊成线性爬坡; 容量 = (行数) * limit, 不足则向两侧
                    # 扩到行带边缘
                    lo, hi = b, b2
                    while lo > 0 and hi < N - 1:
                        span = hi - lo + 1
                        left_v = dr[lo - 1].copy()
                        right_v = dr[hi].copy()
                        cap = float(span * limit)
                        jump_mag = float(np.abs(left_v - right_v).max())
                        if cap >= jump_mag:
                            break
                        lo -= 1
                        hi = min(hi + 1, N - 1)
                    # 爬坡段 [lo, hi] 首尾平台值不变, 中间行(lo+1..hi)
                    # 摊成线性爬坡(含 hi-lo 个间隔, 斜率 ≤limit)
                    left_v = dr[lo].copy()
                    right_v = dr[hi].copy()
                    n = hi - lo
                    if n <= 0:
                        continue
                    wts = np.linspace(0.0, 1.0, n + 1)
                    for i in range(1, n + 1):
                        w0 = float(1.0 - wts[i])
                        w1 = float(wts[i])
                        row_idx = lo + i
                        for c in range(3):
                            dr[row_idx, c] = float(left_v[c] * w0 + right_v[c] * w1)
                    progressed = True
                if not progressed:
                    break
            delta_row[y0s:y1s] = dr
        out = np.clip(mat + delta_row[:, None, :], 0, 255).astype(np.uint8)
        if isinstance(matched_ref, Image.Image):
            im = Image.fromarray(out)
            im.info = dict(matched_ref.info)
            return im
        return out
    except Exception:
        logger.exception("anchor_ref_local failed; returning matched as-is")
        return matched_ref


def ghost_inpaint_pass(
    gen_img: Image.Image,
    stroke_mask: np.ndarray | Image.Image,
    grow_px: int = 5,
    inpaint_radius: int = 5,
    exclude_mask: np.ndarray | Image.Image | None = None,
) -> Image.Image:
    """鬼影笔画局部修补(Telea inpaint): 只填笔画线, 袖褶光影全保。

    恒等贴参考会把整个掩膜区拍成平铺衣料(r11 GPU 视觉验收"衣袖失真",
    袖区 26.5k 自然袖料被凸包吞进恒等贴平区); inpaint 从笔画周边取色,
    零笔画残留时修补结果与周围布料无缝 —— 印花网密的块 Telea 填不净,
    由 ghost_strict_residual 检出后调用方小范围贴参考兜底。

    exclude_mask(核心场): 涂改范围不越核心场本体(core+0) —— 核心场由后续
    核心恒等贴印整体重贴(alpha=1 + feather 1px), 底层涂抹不透出; 但
    [core, core+4) 环内的笔画必须允许修补, 裁掉该环会残留鬼影
    (r12 重放实测擦除带 +17 灰阶)。
    """
    try:
        import cv2

        arr = np.asarray(gen_img.convert("RGB"))
        m = _coerce_mask_to_img(stroke_mask, arr.shape[:2])
        # >0(非 >127): 任意非零=待修补(cv2 掩膜语义), 0/1 掩膜不得静默失效
        m = cv2.dilate(
            (m > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1))
        )
        if exclude_mask is not None:
            ex = _coerce_mask_to_img(exclude_mask, arr.shape[:2])
            m[ex > 127] = 0
        if m.max() == 0:
            return gen_img.convert("RGB")
        out = cv2.inpaint(arr, m, inpaint_radius, cv2.INPAINT_TELEA)
        # 色锚: Telea 在大掩膜区域上把填充白化(袖区实测 填充(240,234,219)
        # vs 净袖料(203,181,116) —— 亮度近似但色度丢失, 填充自身成浅白
        # 斑) → 填充向最近净布料色平移**局部**色偏(填充局部均值差), 保留
        # 填充内部明暗起伏; 掩膜边界 8px 羽化(边缘填充本就贴合周边,
        # 全量平移会产生接缝)。衣身填充本就贴布料 → delta≈0 不变。
        mb = m > 0
        clean = ~mb
        if exclude_mask is not None:
            clean = clean & (ex <= 127)
        if mb.sum() >= 32 and clean.sum() >= 24:
            _outf = out.astype(np.float32)
            near = _nearest_fabric_color(_outf, clean)
            _w = mb.astype(np.float32)
            _fsum = cv2.blur(_outf * _w[..., None], (41, 41))
            _fw = cv2.blur(_w, (41, 41))
            _floc = _fsum / np.maximum(_fw, 1e-6)[..., None]
            _dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            _feather = np.clip(_dt / 8.0, 0.0, 1.0)[..., None]
            _anchored = np.clip(_outf + (near - _floc) * _feather, 0, 255)
            _anchored[~mb] = _outf[~mb]
            out = _anchored.astype(np.uint8)
        return Image.fromarray(out)
    except Exception:
        logger.exception("ghost_inpaint_pass failed; returning input")
        return gen_img.convert("RGB")


def ghost_strict_residual(
    img: Image.Image | np.ndarray,
    base_ref: Image.Image,
    keep_mask: np.ndarray | Image.Image,
    core_mask: np.ndarray | Image.Image | None,
    grow_px: int = 11,
    stroke_mask: np.ndarray | Image.Image | None = None,
) -> np.ndarray | None:
    """擦除后仍残留的鬼影亮块检测(密集笔画网 inpaint 填不净的兜底定位)。

    判据与 ghost_print_mask 同色轴(亮中性+大色距), 但基色取 base_ref
    (色调匹配后的参考) —— Telea 密网填充的中亮垃圾(150-190 vs 生成图
    衣身 94)距原始参考基色(220)不足 60 灰阶会逃过检测(r12 重放实测
    擦除带 +31.6)。检测区在 keep 内缩、核心场本体(core+0)之外;
    返回已外扩 grow_px 的 uint8 掩膜(255=兜底贴参考区), 外扩不越
    核心场本体(越入部分由核心贴印重贴)。

    全局基色只作初筛; 逐像素再用最近净布料色复核 —— 全局基色(生成图衣身
    ~96)会把亮袖料(丝绸反光 183-216)上与袖料同色的合法 inpaint 填充
    判成垃圾 → 贴暗参考 → 亮袖上压深色斑块(r12 重放实测: 袖区笔画带
    gen 224 → inpaint 213 → 兜底后 123, r11 "衣袖失真"的变体回归)。
    检测区限笔画领地(ghost 掩膜外扩 5px = inpaint 足迹)内 —— Telea
    垃圾只可能出现在修补过的区域, 领地外的亮袖料是净布料; 每个像素
    的复核基色取距它**最近**的领地外净布料像素颜色(_nearest_fabric_color
    的 Voronoi 近邻色 —— 高光袖料上鬼影印在高光带(局部 ~220,205,160),
    成分级单一基色无法同时服务跨袖/身成分两侧; 袖面像素对袖料高光 →
    色锚后的填充距 ~20-40 灰阶不判垃圾, 身面像素对衣身暗料 94 →
    中亮垃圾 150-190 距 >60 照判)。净布料 <24px 时回退全局判据。
    stroke_mask 传擦除用的鬼影笔画掩膜。
    """
    try:
        import cv2

        arr = img if isinstance(img, np.ndarray) else np.asarray(img.convert("RGB"))
        arr = arr.astype(np.float32)
        ref = np.asarray(base_ref.convert("RGB"), np.float32)
        ka_img = (_coerce_mask_to_img(keep_mask, arr.shape[:2]) > 127).astype(np.uint8)
        ka_inner = cv2.erode(ka_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        if ka_inner.sum() < 64:
            return None
        zone = ka_inner
        ex_b = None
        if core_mask is not None:
            ex_b = _coerce_mask_to_img(core_mask, arr.shape[:2]) > 127
            zone = zone & ~ex_b
        # 笔画领地 = ghost 掩膜(笔画+9px margin)再外扩 5px = inpaint 足迹:
        # Telea 垃圾只可能出现在修补过的区域; 领地外的亮袖料(丝绸反光
        # 183-216)是净布料, 全局亮中性判据会把整片袖料判成残余 → 贴暗
        # 参考 → 亮袖压深色斑块(r12 重放实测袖带 inpaint 213 → 兜底 119,
        # r11 "衣袖失真"变体回归)。检测区限领地内, 局部基色环从领地外
        # 净布料采样(亮袖料底色 ~211 → 填充 216 距 5 灰阶不判垃圾;
        # 衣身底色 94 → 中亮填充 150-190 照判)。
        territory = None
        if stroke_mask is not None:
            territory = (
                cv2.dilate(
                    (_coerce_mask_to_img(stroke_mask, arr.shape[:2]) > 0).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                )
                > 0
            )
        detect_zone = zone & territory if territory is not None else zone
        base = np.median(ref.reshape(-1, 3)[ka_inner.reshape(-1)], axis=0)
        px = arr.reshape(-1, 3)
        dist = np.linalg.norm(px - base, axis=-1)
        neutral = (px.max(-1) - px.min(-1)) < 40.0
        bright = px.mean(-1) >= float(base.mean()) + 10.0
        cand = (detect_zone.reshape(-1) & (dist > 60.0) & (neutral & bright)).reshape(arr.shape[:2])
        if cand.sum() < 32:
            return None
        # 净布料 = 领地外(有笔画掩膜时); 无 stroke_mask 时退化为候选外
        # (亮块自身不得当自己的基色 —— 最近邻=自己 → 色距恒 0 永不检出)
        clean_zone = zone & ~territory if territory is not None else zone & ~cand
        # 逐像素最近净布料色复核(领地内填充 vs 领地外净布料):
        # 高光袖料上鬼影印在高光带(局部 ~220,205,160), 单一基色(全局 94
        # 或成分环中位数)都会把与高光同色的合法填充(231,218,179)误判
        # (dist 75) → 贴暗成深斑; 跨袖/身成分两侧布料一亮一暗, 成分级
        # 基色无法同时服务两侧 → 每像素取距它**最近**的净布料像素颜色
        # (_nearest_fabric_color, 见其 docstring) —— 袖面像素自动对
        # 袖料高光(色锚后填充距 ~20-40 灰阶不判垃圾), 身面像素自动对
        # 衣身暗料(94, 中亮垃圾 150-190 距 >60 照判)。
        resid = np.zeros(arr.shape[:2], bool)
        if int(clean_zone.sum()) >= 24:
            _near = _nearest_fabric_color(arr, clean_zone)
            dist_v = np.linalg.norm(arr - _near, axis=-1)
            raw = cand & (dist_v > 60.0)
            _n2, _l2, _s2, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8))
            for j in range(1, _n2):
                if _s2[j, cv2.CC_STAT_AREA] >= 32:
                    resid |= _l2 == j
        else:
            # 净布料不可用(整件衣被鬼影盖满等极端几何): 全局判据已成立,
            # 保留(宁贴勿漏 —— 领地内垃圾漏贴是可见缺陷)
            resid |= cand
        if resid.sum() < 32:
            return None
        roi = cv2.dilate(
            resid.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)),
        )
        # 只裁核心场本体(core+0): 兜底越进核心场的部分由核心贴印重贴;
        # [core, core+4) 环内的残留必须兜底, 否则该环鬼影残留
        if ex_b is not None:
            roi[ex_b] = 0
        return np.minimum(roi, ka_img * 255)
    except Exception:
        logger.exception("ghost_strict_residual failed")
        return None


def match_ref_tone(
    gen_img: Image.Image,
    cloth_ref: Image.Image,
    keep_mask: np.ndarray | Image.Image,
    avoid_mask: np.ndarray | Image.Image | list[np.ndarray | Image.Image] | None = None,
    sample_radius: int = 45,
    min_sample_px: int = 3000,
) -> Image.Image:
    """把平铺参考衣料逐通道色调平移到生成图衣身色调(鬼影擦除贴回前的预处理)。

    FASHN 对同件衣服的明暗诠释与平铺照片差异巨大(r10 GPU 实测: gen 衣身
    [117,103,69] vs ref [220,206,168], 相差百余灰阶), 直接把参考贴回生成
    图, 擦除带比周围衣身亮 +100 灰阶 → 视觉验收"印花周围一圈发白发光
    轮廓带"。修法: 在远离核心场/鬼场的干净衣身采样区, 取两图逐通道中位
    数做差, 平移参考整体 —— ref 是 TPS 落位后的同构图, 空间逐像素对应;
    folds/hf 光影由调用方贴回时叠加在匹配后的底色上。

    Args:
        avoid_mask: 采样需避开的场(核心场/鬼场), 单掩膜或掩膜列表
        sample_radius: avoid 外扩像素(避开场边过渡带), 小画布按边长自适应收窄
        min_sample_px: 采样区低于该像素数时不做匹配, 原样返回(小画布按
            keep 像素 2% 自适应放宽, 下限 120px)

    Returns:
        色调匹配后的 RGB PIL 图; 采样不足/失败返回原 cloth_ref
    """
    try:
        import cv2

        gen = np.asarray(gen_img.convert("RGB"), np.float32)
        ref = np.asarray(cloth_ref.convert("RGB"), np.float32)
        if gen.shape[:2] != ref.shape[:2]:
            logger.warning("match_ref_tone: gen/ref 尺寸不一致, 跳过色调匹配")
            return cloth_ref
        keep = np.asarray(keep_mask)
        if keep.ndim == 3:
            keep = keep[..., 0]
        zone = keep > 127
        keep_px = int(zone.sum())
        hh, ww = gen.shape[:2]
        radius = min(sample_radius, max(4, min(hh, ww) // 24))
        min_px = min(min_sample_px, max(120, keep_px // 50))
        if avoid_mask is not None:
            avoid_list = avoid_mask if isinstance(avoid_mask, (list, tuple)) else [avoid_mask]
            combined = np.zeros(gen.shape[:2], dtype=np.uint8)
            for m in avoid_list:
                a = np.asarray(m)
                if a.ndim == 3:
                    a = a[..., 0]
                combined = np.maximum(combined, (a > 0).astype(np.uint8))
            if combined.max() > 0:
                dil = cv2.dilate(
                    combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
                )
                zone = zone & (dil == 0)
        if int(zone.sum()) < min_px:
            logger.info("match_ref_tone: 干净采样区 %dpx 不足 %dpx, 跳过", int(zone.sum()), min_px)
            return cloth_ref
        gen_med = np.median(gen[zone], axis=0)
        ref_med = np.median(ref[zone], axis=0)
        delta = gen_med - ref_med
        # 小差值门: 生成衣身与参考色差 <8 灰阶时, 整体平移只会把 SKU 真色
        # 染偏(视觉验收"图案色彩保真"诉求); r10 鬼影发白场景(差百余灰阶)
        # 仍正常平移, 擦除带语义不变。
        if float(np.abs(delta).max()) < 8.0:
            logger.info(
                "match_ref_tone: 色差 %s < 8 灰阶, 保持参考原色(SKU 色彩保真)",
                np.round(delta, 1),
            )
            return cloth_ref
        matched = np.clip(ref + delta, 0, 255).astype(np.uint8)
        logger.info(
            "match_ref_tone: gen %s / ref %s -> delta %s",
            np.round(gen_med, 1),
            np.round(ref_med, 1),
            np.round(delta, 1),
        )
        return Image.fromarray(matched)
    except Exception:
        logger.exception("match_ref_tone failed")
        return cloth_ref


def blend_core_ref_tone(
    cloth_ref: Image.Image,
    core_mask: np.ndarray | Image.Image,
    matched_ref: Image.Image,
    bright_thresh: int = 225,
    chroma_thresh: int = 35,
) -> Image.Image:
    """核心场贴回参考混合: 白墨笔触保持原样, 场内衣料像素换成色调匹配版。

    核心场 = 逐成分凸包(笔触+笔触间/周边衣料)。整场恒等贴回会把平铺
    照片的亮衣料(r10 实测 198+) 拍上生成图的暗衣身(96), 印花外围一圈
    发光 —— 视觉验收"印花周围发白轮廓带"的主源。白墨笔触本身是客观
    亮色(印刷品不受衣料明暗影响), 必须保持原样, 否则印花发灰。

    笔触判定(参考侧): 亮(bright_thresh)且低彩度(chroma_thresh) ——
    真丝底色 (215,197,149) min=149 不入, 高光 (240,226,180) min=180
    不入, 白墨 (255,255,255)/(247,246,240) 入。中性像素按亮度在
    衣料中位(非笔画像素中位)与 bright_thresh 之间 smoothstep 渐变,
    TPS warp 拉糊的笔触 AA 边不再有 225 处 130 灰阶悬崖(r11 GPU
    视觉验收"印花被暗色图层盖住"); 非笔画像素占比 <20% 的实心印花场
    回退硬门语义(笔画原样, 其余匹配)。

    Returns:
        混合后的 RGB PIL 图; matched_ref 与 cloth_ref 一致时直接返回后者
    """
    try:
        ref = np.asarray(cloth_ref.convert("RGB"))
        mat = np.asarray(matched_ref.convert("RGB"))
        if ref.shape != mat.shape:
            logger.warning("blend_core_ref_tone: ref/matched 形状不一致, 用原始参考")
            return cloth_ref
        if np.array_equal(ref, mat):
            return cloth_ref
        core = np.asarray(core_mask)
        if core.ndim == 3:
            core = core[..., 0]
        zone = core > 127
        if not zone.any():
            return cloth_ref
        strokes = zone & (ref.min(axis=-1) >= bright_thresh) & ((ref.max(axis=-1) - ref.min(axis=-1)) <= chroma_thresh)
        # 软边双轴: ①亮度 smoothstep —— 硬门在 225 处的 130 灰阶悬崖会把
        # TPS warp 拉糊的笔触 AA 边(220-250, 中性)整圈打暗(r11 GPU 视觉
        # 验收"印花被一层暗色图层盖着"); ②笔画距离渐变 —— 亮丝绸 SKU 的
        # 衣料中位(218)与笔触档(225)贴脸, 亮度轴分不出 AA fringe 与衣料
        # (r12 重放实测 AA 环仍 99 vs 参考 218), 距离轴按"距最近笔触 ≤3px"
        # 渐变补位。非中性像素(高光/彩色)维持匹配语义不变。
        # 锚点取非笔画像素中位而非整场中位: 密笔画凸包(实心logo)整场
        # 中位≈笔画亮度, 会把 ramp 锚到笔触档 → 整场全被匹配打暗;
        # 非笔画像素占比 <20% 或锚点够不到门限以下时, 判定该场为实心
        # 印花, 回退硬门语义(笔画原样, 其余匹配)。
        lum = ref.mean(axis=-1)
        non_stroke = zone & ~strokes
        cloth_med = -1.0
        if non_stroke.sum() >= 0.2 * max(int(zone.sum()), 1):
            cloth_med = float(np.median(lum[non_stroke]))
        if 0.0 <= cloth_med <= float(bright_thresh) - 4.0:
            _t = np.clip((lum - (cloth_med + 6.0)) / max(float(bright_thresh) - cloth_med, 1.0), 0.0, 1.0)
            _w = _t * _t * (3.0 - 2.0 * _t)
            if strokes.any():
                try:
                    import cv2  # (距离轴可选: 无 cv2 时仅亮度轴)
                except ImportError:
                    pass
                else:
                    # 距最近笔画 ≤3px 渐变到原样(笔画自身 d=0 → w=1 自洽)
                    _sd = cv2.distanceTransform((~strokes).astype(np.uint8), cv2.DIST_L2, 5)
                    _wd = np.clip(1.0 - _sd / 3.0, 0.0, 1.0)
                    _wd = _wd * _wd * (3.0 - 2.0 * _wd)
                    _w = np.maximum(_w, _wd)
            # 中性门最后统一收口: 高光/彩色贴着笔画也不得借距离轴漏进原样
            # (r10 发白轮廓带回归风险)
            neutral = (ref.max(axis=-1) - ref.min(axis=-1)) <= chroma_thresh
            _w = np.where(neutral, _w, 0.0)
        else:
            _w = strokes.astype(np.float32)
        _w3 = _w[..., None]
        blended = np.clip(ref * _w3 + mat * (1.0 - _w3), 0, 255).astype(np.uint8)
        blended[~zone] = ref[~zone]
        logger.info(
            "blend_core_ref_tone: core %dpx 中笔触保持 %dpx (%.0f%%), 其余用色调匹配版",
            int(zone.sum()),
            int(strokes.sum()),
            strokes.sum() / max(zone.sum(), 1) * 100,
        )
        return Image.fromarray(blended)
    except Exception:
        logger.exception("blend_core_ref_tone failed")
        return cloth_ref


def clip_mask_to_garment_alpha(
    apply_mask: np.ndarray | Image.Image | None,
    ref_alpha: Image.Image | None,
    grow_px: int = 6,
) -> np.ndarray | Image.Image | None:
    """mask ∩ 衣料 alpha(膨胀余量): 纯裁剪, 不并集不外扩。

    与 intersect_mask_with_garment_alpha 相反方向的守卫: 那个把 mask
    并上实心 alpha 再裁(为 SDXL 模板路径兜住宽松款悬垂面料), 本函数
    只做 min(mask, dilate(alpha)) —— FASHN 路径衣服由生成器画定,
    检测区就是衣服边界, 再向模板 alpha 形并集会把贴印推进空气里
    (GPU 实测 apply 0.161→0.3685, 印花重影)。
    """
    if apply_mask is None or ref_alpha is None:
        return apply_mask
    try:
        import cv2
    except ImportError:
        return apply_mask
    m = np.asarray(apply_mask)
    if m.ndim == 3:
        m = m[..., 0]
    h, w = m.shape
    ra_src = ref_alpha.getchannel("A") if ref_alpha.mode == "RGBA" else ref_alpha.convert("L")
    ra = np.asarray(ra_src, dtype=np.uint8)
    if ra.shape != (h, w):
        ra = np.asarray(ra_src.resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, grow_px) * 2 + 1, max(1, grow_px) * 2 + 1))
    ra_dil = cv2.dilate(ra, ke)
    return np.minimum(m, ra_dil)


def refit_cloth_to_drawn_garment(
    garment: Image.Image,
    canvas_size: tuple[int, int],
    drawn_mask: np.ndarray | Image.Image,
) -> tuple[Image.Image, Image.Image] | None:
    """把平铺衣重投到生成器实际画的衣服几何(bbox 八锚点 TPS)。

    身体模板先验(torso_box)与 FASHN 实际画的衣服存在落位/比例差,
    模板 warp 的衣料形直接当贴回区会把印花贴到空气里(重影/气泡感,
    GPU 实测 fidelity 0.847)。这里以生成图检测出的衣服区(drawn_mask)
    bbox 为目标, 四角+四边中点共 8 锚点 TPS 把平铺衣投到同一几何,
    贴印/保真分全部以新几何为准; 生成图检测区即衣服边界, 后续用
    clip_mask_to_garment_alpha 纯裁剪(不并集)。

    Args:
        garment: 平铺衣 RGB (白底商品图)
        canvas_size: 生成图尺寸 (w, h)
        drawn_mask: 生成图检测出的衣服区 (L/ndarray, 255=衣服)

    Returns:
        (RGB 白底参考, RGBA alpha 参考) 同 align_garment_to_pose 口径;
        drawn_mask 无效/退化/TPS 失败时 None, 调用方回退模板路径。
    """
    try:
        from engine import garment_alpha
    except ImportError:
        logger.debug("refit_cloth_to_drawn_garment: engine.garment_alpha unavailable")
        return None
    m = np.asarray(drawn_mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != (canvas_size[1], canvas_size[0]):
        logger.warning("refit_cloth_to_drawn_garment: mask shape %s != canvas %s", m.shape, canvas_size)
        return None
    ys, xs = np.where(m > 127)
    if len(ys) < 16:
        return None
    dx0, dx1 = float(xs.min()), float(xs.max())
    dy0, dy1 = float(ys.min()), float(ys.max())
    cw, ch = float(canvas_size[0]), float(canvas_size[1])
    dw, dh = dx1 - dx0, dy1 - dy0
    if dw < 8 or dh < 8:
        return None
    if not (0.04 <= dw / cw <= 0.95 and 0.05 <= dh / ch <= 0.98):
        logger.info("refit_cloth_to_drawn_garment: drawn bbox degenerate (%.2f, %.2f); fallback", dw / cw, dh / ch)
        return None

    g = garment.convert("RGBA")
    if int(g.split()[-1].getextrema()[0]) >= 250:
        alpha_ch = garment_alpha(garment)
        # 纯色无白底: 色距 alpha 可能全零, 回退实心衣料(整图即内容)
        if np.asarray(alpha_ch).max() <= 10:
            alpha_ch = Image.new("L", g.size, 255)
        else:
            # 白墨印花洞修复: 四角背景色估计把与画布同色的印花笔触判成
            # 透明(真实 SKU alpha 实测 min 0), 贴回 content 门把笔触整
            # 个跳过 → 只贴笔触间基色, 白印花全程留 FASHN 鬼影(r9 GPU
            # 实测 alpha<0.5 区 NCC 0.05, |diff| 102)。
            # 连通性(任何 alpha 阈值)在此失败: 满印 SKU 笔触网触到衣料
            # 轮廓, 画布/羽化/笔触连成一个边框连通块(GPU 实测 holes=0)。
            # 改用 solid 质量(alpha>=150)洪泛: 从边框外洪泛 alpha<150 的
            # 外部区域, 被芥末黄质量包围而未触达的 = 内部洞(印花/白色
            # 部件), 补实心并连带 3px 笔触羽化。
            try:
                import cv2

                a_np = np.asarray(alpha_ch).copy()
                h_, w_ = a_np.shape
                edge = np.zeros((h_, w_), dtype=bool)
                edge[0, :] = a_np[0, :] < 150
                edge[-1, :] = a_np[-1, :] < 150
                edge[:, 0] = a_np[:, 0] < 150
                edge[:, -1] = a_np[:, -1] < 150
                if edge.any():
                    _ys, _xs = np.where(edge)
                    seed_x, seed_y = int(_xs[0]), int(_ys[0])
                    ff = (a_np >= 150).astype(np.uint8)
                    ffmask = np.zeros((h_ + 2, w_ + 2), np.uint8)
                    cv2.floodFill(ff, ffmask, (seed_x, seed_y), 2)
                    holes = (ff == 0) & (a_np < 150)
                    if holes.any():
                        _k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        raise_px = cv2.dilate(holes.astype(np.uint8), _k3) > 0
                        a_np[raise_px & (a_np < 150)] = 255
                        alpha_ch = Image.fromarray(a_np)
            except Exception:  # noqa: BLE001
                logger.debug("refit alpha hole repair skipped")
        g.putalpha(alpha_ch)
    a = np.asarray(g.split()[-1])
    ays, axs = np.where(a > 10)
    if len(ays) == 0:
        axs = np.array([0, g.size[0] - 1])
        ays = np.array([0, g.size[1] - 1])
    sx0, sx1 = float(axs.min()), float(axs.max())
    sy0, sy1 = float(ays.min()), float(ays.max())
    sw, sh = max(1.0, sx1 - sx0), max(1.0, sy1 - sy0)

    fracs = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (1.0, 0.5), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    src_pts = [(sx0 + fx * sw, sy0 + fy * sh) for fx, fy in fracs]
    dst_pts = [(dx0 + fx * dw, dy0 + fy * dh) for fx, fy in fracs]
    try:
        warped = tps_warp(g, src_pts, dst_pts, canvas_size)
    except Exception:
        logger.exception("refit_cloth_to_drawn_garment: tps failed; fallback to template path")
        return None
    if warped.mode != "RGBA":
        warped = warped.convert("RGBA")
    # 覆盖率守卫: warp 塌缩(数值病态/锚点退化)时 alpha 只剩零散斜条,
    # 贴印/保真分会拿到稀疏 sliver(GPU r5 实测 frac 0.011 vs 预期 0.14,
    # fidelity 0.498); 低于 dst bbox 面积 x 源内容填充率的一半即判失败,
    # 回退模板路径。
    wa = np.asarray(warped.split()[-1])
    dst_frac = (dw * dh) / (cw * ch)
    src_fill = len(axs) / max(1.0, sw * sh)
    if float((wa > 10).mean()) < 0.5 * dst_frac * src_fill:
        logger.warning("refit_cloth_to_drawn_garment: warped coverage collapsed; fallback to template path")
        return None
    out = Image.new("RGB", canvas_size, (255, 255, 255))
    out.paste(warped, (0, 0), warped)
    return out, warped


def optical_flow_refit(
    refit_img: Image.Image,
    gen_img: Image.Image,
    recast_mask: Image.Image,
) -> Image.Image:
    """Farneback 光流微移: 把贴印像素向生成图褶皱方向微移, 局部对齐。

    在重绘区域内计算光流, 微移贴印像素使褶皱与生成图一致, 不引入新像素。

    Args:
        refit_img: TPS 贴印结果 (RGBA)
        gen_img: 扩散生成图 (RGB), 与 refit_img 同尺寸
        recast_mask: 重绘区域掩膜 (L, 255=重绘)

    Returns:
        光流微移后的 RGBA 图
    """
    try:
        import cv2

        ref = np.array(refit_img, dtype=np.uint8)
        gen = np.array(gen_img, dtype=np.uint8)
        if gen.ndim == 2:
            gen = np.stack([gen] * 3, axis=2)
        if ref.ndim == 2:
            ref = np.stack([ref] * 3, axis=2)

        mask = np.asarray(recast_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] != ref.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR)
            )

        if not (mask > 20).any():
            return refit_img

        ref_gray = (
            cv2.cvtColor(ref[..., :3], cv2.COLOR_RGB2GRAY) if ref.shape[2] > 3 else ref.mean(axis=2).astype(np.uint8)
        )
        gen_gray = cv2.cvtColor(gen[..., :3], cv2.COLOR_RGB2GRAY) if gen.ndim == 3 else gen

        flow = cv2.calcOpticalFlowFarneback(
            ref_gray,
            gen_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        # 微移: 光流 * 0.5 (保守, 避免吃进非褶皱区)
        h, w = ref.shape[:2]
        xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = np.clip(xg + flow[..., 0] * 0.5, 0, w - 1).astype(np.float32)
        map_y = np.clip(yg + flow[..., 1] * 0.5, 0, h - 1).astype(np.float32)

        warped = cv2.remap(ref, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # 仅重绘区域内应用光流结果, 保护区域保持原贴印
        out = ref.copy()
        region = mask > 20
        out[region] = warped[region]
        return Image.fromarray(out, "RGBA")
    except Exception:
        logger.exception("optical_flow_refit failed; returning unmodified refit")
        return refit_img


def run_multiseed(
    task_fn,
    seeds: list[int],
    scorer,
) -> tuple[Image.Image, list[tuple[int, float]], int]:
    """多 seed 出图按质检分排序, 返回最高分结果。

    Args:
        task_fn: 可调 task_fn(seed) -> Image.Image
        seeds: seed 列表
        scorer: 可调 scorer(image, seed) -> float (质检分, 越高越好)

    Returns:
        (best_result, all_scores, chosen_seed)
    """
    results = []
    for seed in seeds:
        img = task_fn(seed)
        score = float(scorer(img, seed))
        results.append((seed, score, img))
    results.sort(key=lambda x: x[1], reverse=True)
    best_seed, best_score, best_img = results[0]
    all_scores = [(s, sc) for s, sc, _ in results]
    logger.info("multiseed: %d seeds, chosen=%d score=%.4f", len(seeds), best_seed, best_score)
    return best_img, all_scores, best_seed


def refit_seeds_default() -> list[int]:
    """默认多 seed 列表: VTON_MULTI_SEED 控制数量, 纯色 =1."""
    n = int(os.environ.get("VTON_MULTI_SEED", "3"))
    n = max(1, min(n, 8))
    return [42 + i * 1000 for i in range(n)]
