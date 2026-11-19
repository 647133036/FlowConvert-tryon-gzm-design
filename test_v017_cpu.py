"""v0.1.7 CPU-side component validation with real images.

Tests without GPU/rembg:
  1. align_garment_to_pose  — warp + vertical alignment
  2. _detect_skin           — YCbCr + morphological cleaning
  3. _estimate_waist_ratio  — waist heuristic on real silhouette
  4. clothing_mask          — mask with waist-based cut
  5. _correct_skin_tone     — skin tone correction (face exclusion)
"""

import sys, os, numpy as np, cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
os.environ["VTON_WARP"] = "1"
os.environ["VTON_WARP_VERTICAL"] = "1.0"

from platform_presets import (
    _detect_skin,
    _correct_skin_tone,
    _estimate_waist_ratio,
    clothing_mask,
    detect_garment_extent,
)
from engine_sdxl import align_garment_to_pose

OUT = "test_images/out"
os.makedirs(OUT, exist_ok=True)

# ---------- helpers ----------

def simple_subject_mask(img: Image.Image) -> np.ndarray:
    """White-background threshold fallback (no rembg)."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # non-white = foreground
    fg = (gray < 230).astype(np.uint8)
    # morphological close to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # largest connected component
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if nlabels < 2:
        return fg * 255
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def overlay_mask(img: Image.Image, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    m = (mask > 0).astype(np.uint8)
    overlay = arr.copy()
    overlay[m > 0] = color
    return Image.fromarray(cv2.addWeighted(overlay, alpha, arr, 1 - alpha, 0))


# ---------- load images ----------

model = Image.open("test_images/model_01.jpg")
garment = Image.open("test_images/garment_01.jpg")
print(f"model: {model.size}, garment: {garment.size}")

# ---------- 1. align_garment_to_pose ----------

print("\n[1] align_garment_to_pose (warp + vertical alignment)")
for gtype in ["upper", "dress", "lower"]:
    warped = align_garment_to_pose(garment, "general", "asian", gtype, canvas=1024)
    warped.save(f"{OUT}/warp_{gtype}.png")
    print(f"  {gtype}: saved warp_{gtype}.png ({warped.size})")

# also test VTON_WARP_VERTICAL=0 (horizontal only, v0.1.6 behavior)
os.environ["VTON_WARP_VERTICAL"] = "0"
warped_h_only = align_garment_to_pose(garment, "general", "asian", "upper", canvas=1024)
warped_h_only.save(f"{OUT}/warp_upper_h_only.png")
os.environ["VTON_WARP_VERTICAL"] = "1.0"
print("  upper (h-only): saved warp_upper_h_only.png")

# ---------- 2. _detect_skin ----------

print("\n[2] _detect_skin (YCbCr + morphology)")
model_rgb = np.array(model.convert("RGB"))
skin = _detect_skin(model_rgb)
skin_pct = (skin > 0).mean() * 100
print(f"  skin coverage: {skin_pct:.1f}%")
Image.fromarray(skin).save(f"{OUT}/skin_mask.png")
overlay_mask(model, skin, color=(255, 100, 100)).save(f"{OUT}/skin_overlay.png")
print("  saved skin_mask.png, skin_overlay.png")

# ---------- 3. _estimate_waist_ratio ----------

print("\n[3] _estimate_waist_ratio (waist heuristic)")
subj = simple_subject_mask(model)
ys = np.where(subj > 0)[0]
y0, y1 = int(ys.min()), int(ys.max())
span = max(1, y1 - y0)
waist_r = _estimate_waist_ratio(subj, y0, span)
print(f"  subject bbox: y0={y0}, y1={y1}, span={span}")
print(f"  waist_ratio: {waist_r:.4f} (default fallback=0.62)")
print(f"  waist y-pixel: {y0 + int(span * waist_r)}")

# visualize waist line
vis = np.array(model.convert("RGB")).copy()
wy = y0 + int(span * waist_r)
cv2.line(vis, (0, wy), (vis.shape[1], wy), (0, 255, 255), 2)
cv2.line(vis, (0, y0), (vis.shape[1], y0), (255, 0, 0), 1)
cv2.line(vis, (0, y1), (vis.shape[1], y1), (255, 0, 0), 1)
Image.fromarray(vis).save(f"{OUT}/waist_line.png")
print("  saved waist_line.png (yellow=waist, blue=subject top/bottom)")

# ---------- 4. clothing_mask ----------

print("\n[4] clothing_mask (waist-based cut)")
extent = detect_garment_extent(garment)
print(f"  garment_extent: {extent}")
for gtype in ["upper", "lower", "dress"]:
    cmask = clothing_mask(model, garment_type=gtype, subject_mask=subj, garment_extent=extent)
    if cmask is not None:
        overlay_mask(model, cmask, color=(0, 255, 0)).save(f"{OUT}/cloth_mask_{gtype}.png")
        pct = (cmask > 0).mean() * 100
        print(f"  {gtype}: coverage={pct:.1f}%, saved cloth_mask_{gtype}.png")
    else:
        print(f"  {gtype}: None (mask construction failed)")

# ---------- 5. _correct_skin_tone ----------

print("\n[5] _correct_skin_tone (face exclusion)")
# simulate a "result" where skin is shifted (whitened)
res_arr = model_rgb.copy()
skin_for_shift = _detect_skin(res_arr)
# whiten skin region to simulate SDXL output
res_arr[skin_for_shift > 0] = np.clip(
    res_arr[skin_for_shift > 0].astype(np.int16) + 40, 0, 255
).astype(np.uint8)
res_img = Image.fromarray(res_arr)
res_img.save(f"{OUT}/skin_whitened.png")

corrected = _correct_skin_tone(model, res_img)
corrected.save(f"{OUT}/skin_corrected.png")
print("  saved skin_whitened.png (input), skin_corrected.png (output)")

# verify face area was excluded: compare face region pixel diff
fg_bbox = subj
ys2 = np.where(fg_bbox > 0)[0]
top_y = int(ys2.min())
face_h = int((ys2.max() - top_y) * 0.18)
face_region = slice(top_y, top_y + face_h)
face_diff_before = np.abs(
    res_arr[face_region].astype(np.int16) - model_rgb[face_region].astype(np.int16)
).mean()
corr_arr = np.array(corrected.convert("RGB"))
face_diff_after = np.abs(
    corr_arr[face_region].astype(np.int16) - model_rgb[face_region].astype(np.int16)
).mean()
print(f"  face region diff: before={face_diff_before:.1f}, after={face_diff_after:.1f}")
print(f"  (face should stay similar — excluded from correction)")

# arm region: should be corrected back toward original
arm_region = slice(top_y + face_h, int(top_y + (ys2.max() - top_y) * 0.5))
arm_diff_before = np.abs(
    res_arr[arm_region].astype(np.int16) - model_rgb[arm_region].astype(np.int16)
).mean()
arm_diff_after = np.abs(
    corr_arr[arm_region].astype(np.int16) - model_rgb[arm_region].astype(np.int16)
).mean()
print(f"  arm region diff: before={arm_diff_before:.1f}, after={arm_diff_after:.1f}")
print(f"  (arm should decrease — corrected toward original skin tone)")

print("\n=== Done. Results in test_images/out/ ===")
