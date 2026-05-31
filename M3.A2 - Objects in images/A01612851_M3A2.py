import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

np.random.seed(42)

IMG_PATH = "imagenpildoras.jpg"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PARAMS = dict(
    binary_thresh    = 80,
    bh_thresh        = 0.15,
    ratio_min_thresh = 0.10,
    std_thresh       = 6.0,
    fill_thresh      = 0.93,
    lcs_thresh       = 0.30,
    hough_param2     = 22,
    hough_min_dist   = 130,
)

FIXED     = 64
BW_FRAC   = 0.10
WING_FRAC = 0.30

_candidates = [
    IMG_PATH,
    os.path.join(SCRIPT_DIR, IMG_PATH),
]
full_img = None
for _p in _candidates:
    full_img = cv2.imread(_p)
    if full_img is not None:
        break

if full_img is None:
    sys.exit(f"[ERROR] No se encontró la imagen.")

img_h, img_w = full_img.shape[:2]

img_bgr = full_img
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def run_pipeline(params):
    th    = int(params["binary_thresh"])
    p2    = int(params["hough_param2"])
    mdist = int(params["hough_min_dist"])

    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
    _, binary = cv2.threshold(blurred, th, 255, cv2.THRESH_BINARY)

    k_ell  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_ell, iterations=2)
    opened = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  k_ell, iterations=1)

    sobel_h = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_v = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    abs_h   = cv2.convertScaleAbs(sobel_h)
    abs_v   = cv2.convertScaleAbs(sobel_v)

    raw = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=mdist,
        param1=60, param2=p2,
        minRadius=65, maxRadius=170,
    )

    circles = np.empty((0, 3), int)
    if raw is not None:
        candidates = np.round(raw[0]).astype(int)
        valid = []
        for cx, cy, r in candidates:
            m = np.zeros(gray.shape, np.uint8)
            cv2.circle(m, (cx, cy), r, 255, -1)
            if cv2.mean(opened, mask=m)[0] / 255.0 >= 0.40:
                valid.append((cx, cy, r))
        circles = np.array(valid, int) if valid else np.empty((0, 3), int)

    cross_mask = np.zeros(len(circles), bool)
    for i, (cx, cy, r) in enumerate(circles):
        cross_mask[i] = _has_cross(cx, cy, r, opened, params)

    debug = dict(binary=binary, cleaned=opened, sobel_h=abs_h, sobel_v=abs_v)
    return circles, cross_mask, debug

def _has_cross(cx, cy, r, cleaned, params):
    bh_t   = float(params["bh_thresh"])
    rm_t   = float(params["ratio_min_thresh"])
    std_t  = float(params["std_thresh"])
    fill_t = float(params["fill_thresh"])
    lcs_t  = float(params["lcs_thresh"])

    m60 = np.zeros(gray.shape, np.uint8)
    cv2.circle(m60, (cx, cy), int(r * 0.60), 255, -1)
    if cv2.mean(cleaned, mask=m60)[0] / 255.0 < fill_t:
        return False

    m70 = np.zeros(gray.shape, np.uint8)
    cv2.circle(m70, (cx, cy), int(r * 0.70), 255, -1)
    if float(gray[m70 == 255].std()) < std_t:
        return False

    pr  = int(r * 0.78)
    x0  = max(0, cx - pr);  x1 = min(gray.shape[1], cx + pr + 1)
    y0  = max(0, cy - pr);  y1 = min(gray.shape[0], cy + pr + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < 6 or roi.shape[1] < 6:
        return False

    s   = cv2.resize(roi, (FIXED, FIXED))
    mid = FIXED // 2
    bw  = max(1, int(FIXED * BW_FRAC))
    wn  = int(FIXED * WING_FRAC)

    k_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (FIXED // 4, FIXED // 4))
    bh   = cv2.morphologyEx(s, cv2.MORPH_BLACKHAT, k_sq)
    h_band   = bh[mid - bw : mid + bw + 1, :]
    v_band   = bh[:, mid - bw : mid + bw + 1]
    bh_score = (h_band.sum() + v_band.sum()) / (bh.sum() + 1)
    if bh_score < bh_t:
        return False

    h_prof = bh[mid - bw : mid + bw + 1, :].mean(axis=0)
    v_prof = bh[:, mid - bw : mid + bw + 1].mean(axis=1)
    hc = h_prof[mid - bw : mid + bw + 1].mean()
    hw = (h_prof[:wn].mean() + h_prof[-wn:].mean()) / 2.0
    hr = hc / (hw + 1e-5)
    vc = v_prof[mid - bw : mid + bw + 1].mean()
    vw = (v_prof[:wn].mean() + v_prof[-wn:].mean()) / 2.0
    vr = vc / (vw + 1e-5)
    if min(hr, vr) < rm_t:
        return False

    pr_local = int(r * 0.70)
    x0l = max(0, cx - pr_local);  x1l = min(gray.shape[1], cx + pr_local + 1)
    y0l = max(0, cy - pr_local);  y1l = min(gray.shape[0], cy + pr_local + 1)
    roi_l = gray[y0l:y1l, x0l:x1l]
    if roi_l.shape[0] < 6 or roi_l.shape[1] < 6:
        return False
    sl = cv2.resize(roi_l, (FIXED, FIXED))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    sl = clahe.apply(sl)
    adapt = cv2.adaptiveThreshold(
        sl, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )
    k3 = np.ones((3, 3), np.uint8)
    adapt_clean = cv2.erode(adapt, k3, iterations=1)
    h_b = adapt_clean[mid - bw : mid + bw + 1, :]
    v_b = adapt_clean[:, mid - bw : mid + bw + 1]
    h_fill = float((h_b > 0).any(axis=1).mean())
    v_fill = float((v_b > 0).any(axis=0).mean())
    
    center = adapt_clean[
        mid - bw : mid + bw + 1,
        mid - bw : mid + bw + 1
    ]
    center_fill = np.mean(center > 0)

    if center_fill < 0.05:
        return False

    ys, xs = np.where(adapt_clean > 0)
    if len(xs) == 0:
        return False

    cx_feat = np.mean(xs)
    cy_feat = np.mean(ys)
    dist = np.sqrt((cx_feat - mid)**2 + (cy_feat - mid)**2)
    if dist > 5.25:
        return False
    return min(h_fill, v_fill) > lcs_t

BG, FG, ACC = "#12121f", "#e8e8f0", "#ff3c5a"

def _annotate(circles, cross_mask):
    out = img_rgb.copy()
    for i, (cx, cy, r) in enumerate(circles):
        if cross_mask[i]:
            cv2.circle(out, (cx, cy), 19, (255, 50, 50), -1)
    return out

def static_figure():
    circles, cross_mask, dbg = run_pipeline(DEFAULT_PARAMS)
    n_cross = int(cross_mask.sum())
    result_img = _annotate(circles, cross_mask)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor("#1c1c30")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a50")
        ax.tick_params(colors=FG)

    panels = [
        ("1. Original",                       img_rgb,        None),
        ("2. Binarización global",             dbg["binary"],  "gray"),
        ("3. Morfología (cierre + apertura)",  dbg["cleaned"], "gray"),
        ("4. Sobel Horizonral",               dbg["sobel_h"], "gray"),
        ("5. Sobel Vertical",                 dbg["sobel_v"], "gray"),
        (f"6. Resultado — Cruz: {n_cross}", result_img, None),
    ]
    for ax, (title, im, cmap) in zip(axes.flat, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title,
                     color=FG if "Resultado" not in title else ACC,
                     fontsize=10, fontfamily="monospace", fontweight="bold", pad=5)
        ax.axis("off")

    fig.suptitle(
        f"Detección de Píldoras con Cruz (+)  ·  Con cruz: {n_cross}",
        color=ACC, fontsize=14, fontweight="bold", fontfamily="monospace", y=0.99,
    )
    p1 = mpatches.Patch(color=(1, .2, .22), label=f"Con cruz  ({n_cross})")
    fig.legend(handles=[p1], loc="lower center", ncol=1,
               facecolor="#1c1c30", labelcolor=FG, fontsize=11,
               framealpha=0.95, edgecolor=ACC)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    out_path = os.path.join(SCRIPT_DIR, "resultado_deteccion.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Hay {n_cross} píldoras con cruz en la imagen.")
    print()
    print(f"Resultado en {out_path}")
    plt.show()

if __name__ == "__main__":
    static_figure()