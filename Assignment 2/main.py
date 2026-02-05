"""
Repetitive K-Means Color Clustering + Skin Region Detection (Hand Photos)

What this script does (per image):
- For k in {2, 3, 5}:
  - Run k-means 100 times with RANDOM pixel-centroid initialization each time
  - Align cluster labels across repetitions by sorting centroids (lexicographic R,G,B)
  - Build k probability maps: P(pixel belongs to cluster i)
  - Auto-pick likely "skin" cluster(s) using a simple RGB heuristic (overrideable)
  - Threshold to create a binary skin mask

REQUIREMENTS:
pip install numpy opencv-python scikit-learn matplotlib

USAGE:
python rep_kmeans_skin.py --img1 path/to/hand1.jpg --img2 path/to/hand2.jpg --reps 100 --out outputs

TIP:
If your photos are huge, set --max_side (e.g., 800) to speed things up.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ----------------------------
# Utilities
# ----------------------------
def read_bgr_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_keep_aspect(bgr: np.ndarray, max_side: int | None):
    if not max_side:
        return bgr
    h, w = bgr.shape[:2]
    scale = max(h, w) / float(max_side)
    if scale <= 1.0:
        return bgr
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_rgb_float(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Core: one kmeans run with custom init
# ----------------------------
def kmeans_single_run_rgb(X_flat: np.ndarray, k: int, rng: np.random.Generator):
    """
    X_flat: (N,3) float RGB
    returns: labels (N,), centroids (k,3)
    """
    N = X_flat.shape[0]
    idx = rng.integers(0, N, size=k)
    init_centroids = X_flat[idx].copy()

    km = KMeans(
        n_clusters=k,
        init=init_centroids,
        n_init=1,
        max_iter=300,
        random_state=None,  # randomness handled by init_centroids
        algorithm="lloyd",
    )
    km.fit(X_flat)
    return km.labels_.astype(np.int32), km.cluster_centers_.astype(np.float32)


def align_labels_by_centroid_sort(labels: np.ndarray, centroids: np.ndarray):
    """
    Step 3.5: align cluster numbering for consistency across repetitions.
    Fixed rule: sort clusters by centroid (R, then G, then B) ascending.
    Returns aligned_labels (N,), aligned_centroids (k,3), and mapping old->new.
    """
    # centroids shape: (k,3) with columns [R,G,B]
    sort_idx = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))  # B,G,R keys -> lexsort
    # sort_idx: new_order in terms of old cluster ids
    old_to_new = np.empty(len(sort_idx), dtype=np.int32)
    for new_id, old_id in enumerate(sort_idx):
        old_to_new[old_id] = new_id

    aligned_labels = old_to_new[labels]
    aligned_centroids = centroids[sort_idx]
    return aligned_labels, aligned_centroids, old_to_new


# ----------------------------
# Repetitive K-Means
# ----------------------------
def repetitive_kmeans_probability_maps(rgb: np.ndarray, k: int, reps: int, seed: int = 0):
    """
    rgb: (H,W,3) float RGB
    Returns:
      prob_maps: (H,W,k) probabilities
      mean_centroids: (k,3) average centroid values across reps (after alignment)
    """
    H, W = rgb.shape[:2]
    X_flat = rgb.reshape(-1, 3)  # (N,3)
    N = X_flat.shape[0]

    rng = np.random.default_rng(seed)

    # counts per pixel per cluster
    counts = np.zeros((N, k), dtype=np.int32)
    centroids_accum = np.zeros((k, 3), dtype=np.float64)

    for r in range(reps):
        labels, cents = kmeans_single_run_rgb(X_flat, k, rng)
        labels_aligned, cents_aligned, _ = align_labels_by_centroid_sort(labels, cents)

        # accumulate counts
        counts[np.arange(N), labels_aligned] += 1
        centroids_accum += cents_aligned

    prob_flat = counts.astype(np.float32) / float(reps)  # (N,k)
    prob_maps = prob_flat.reshape(H, W, k)
    mean_centroids = (centroids_accum / float(reps)).astype(np.float32)  # (k,3)

    return prob_maps, mean_centroids


# ----------------------------
# Skin cluster selection + thresholding
# ----------------------------
def auto_pick_skin_clusters(mean_centroids_rgb: np.ndarray, top_m: int = 1):
    """
    Heuristic: skin in RGB usually has:
      - R relatively high
      - R > G > B often
      - moderate chroma (not extremely saturated like pure red)
    We'll score each centroid and pick top_m indices.
    """
    R = mean_centroids_rgb[:, 0]
    G = mean_centroids_rgb[:, 1]
    B = mean_centroids_rgb[:, 2]

    # avoid divide-by-zero
    eps = 1e-6

    # Encourage R dominance and moderate brightness
    r_ratio = R / (G + B + eps)
    rg_gap = (R - G)
    gb_gap = (G - B)

    brightness = (R + G + B) / 3.0
    # penalize extremely dark or extremely bright
    bright_penalty = np.abs(brightness - 160.0) / 160.0  # centered roughly mid-high for skin under indoor light

    # A simple combined score (tweakable)
    score = (
        2.0 * r_ratio
        + 0.01 * rg_gap
        + 0.005 * gb_gap
        - 1.0 * bright_penalty
    )

    pick = np.argsort(score)[::-1][:top_m]
    return pick.tolist(), score


def skin_mask_from_probmaps(prob_maps: np.ndarray, skin_cluster_ids: list[int], prob_thresh: float):
    """
    For each pixel:
      - find argmax cluster
      - pixel is skin if argmax in skin_cluster_ids AND maxprob >= prob_thresh
    """
    max_ids = np.argmax(prob_maps, axis=2)
    max_p = np.max(prob_maps, axis=2)
    is_skin_cluster = np.isin(max_ids, np.array(skin_cluster_ids, dtype=np.int32))
    mask = (is_skin_cluster & (max_p >= prob_thresh)).astype(np.uint8)
    return mask


# ----------------------------
# Visualization / Saving
# ----------------------------
def save_probability_maps_and_mask(out_dir: str, base_name: str, rgb: np.ndarray, k: int,
                                  prob_maps: np.ndarray, mean_centroids: np.ndarray,
                                  skin_clusters: list[int], skin_mask: np.ndarray):
    ensure_dir(out_dir)

    H, W = rgb.shape[:2]

    # Save original
    orig_path = os.path.join(out_dir, f"{base_name}_orig.png")
    cv2.imwrite(orig_path, cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Save probability maps as figures + raw arrays
    npy_path = os.path.join(out_dir, f"{base_name}_k{k}_probmaps.npy")
    np.save(npy_path, prob_maps)

    for i in range(k):
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(prob_maps[:, :, i])  # default colormap
        plt.title(f"{base_name}: k={k}  P(cluster {i})")
        plt.axis("off")
        fig_path = os.path.join(out_dir, f"{base_name}_k{k}_P{i}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close(fig)

    # Save centroid swatches figure
    fig = plt.figure(figsize=(6, 2))
    swatch = np.zeros((50, 50 * k, 3), dtype=np.uint8)
    for i in range(k):
        c = np.clip(mean_centroids[i], 0, 255).astype(np.uint8)
        swatch[:, 50 * i:50 * (i + 1), :] = c
    plt.imshow(swatch)
    plt.title(f"{base_name}: k={k} mean centroids (aligned). Skin clusters: {skin_clusters}")
    plt.axis("off")
    fig_path = os.path.join(out_dir, f"{base_name}_k{k}_centroids.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Save skin mask
    mask_path = os.path.join(out_dir, f"{base_name}_k{k}_skinmask.png")
    cv2.imwrite(mask_path, (skin_mask * 255).astype(np.uint8))

    # Save overlay
    overlay = rgb.copy().astype(np.uint8)
    # green-ish tint without explicitly setting matplotlib colors:
    # We'll do it in RGB directly (this is fine for saving images).
    overlay[skin_mask == 1] = np.clip(
        0.6 * overlay[skin_mask == 1] + 0.4 * np.array([0, 255, 0], dtype=np.float32),
        0, 255
    ).astype(np.uint8)

    overlay_path = os.path.join(out_dir, f"{base_name}_k{k}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save a quick summary text
    txt_path = os.path.join(out_dir, f"{base_name}_k{k}_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"Image: {base_name}\n")
        f.write(f"k: {k}\n")
        f.write(f"mean centroids (R,G,B):\n{mean_centroids}\n")
        f.write(f"skin clusters (auto/override): {skin_clusters}\n")
        f.write(f"skin prob threshold: saved in run log\n")


# ----------------------------
# Main pipeline per image
# ----------------------------
def process_one_image(img_path: str, out_root: str, reps: int, seed: int,
                      prob_thresh: float, max_side: int | None,
                      skin_top_m: int, skin_override: dict[int, list[int]] | None):
    bgr = read_bgr_image(img_path)
    bgr = resize_keep_aspect(bgr, max_side=max_side)
    rgb = bgr_to_rgb_float(bgr)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(out_root, base_name)
    ensure_dir(out_dir)

    # Log run settings
    with open(os.path.join(out_dir, "run_settings.txt"), "w") as f:
        f.write(f"img_path={img_path}\n")
        f.write(f"reps={reps}\n")
        f.write(f"seed={seed}\n")
        f.write(f"prob_thresh={prob_thresh}\n")
        f.write(f"max_side={max_side}\n")
        f.write(f"skin_top_m={skin_top_m}\n")
        f.write(f"skin_override={skin_override}\n")

    for k in [2, 3, 5]:
        prob_maps, mean_centroids = repetitive_kmeans_probability_maps(
            rgb_u8.astype(np.float32), k=k, reps=reps, seed=seed + 10 * k
        )

        # choose skin clusters
        if skin_override and k in skin_override:
            skin_clusters = skin_override[k]
            _scores = None
        else:
            skin_clusters, _scores = auto_pick_skin_clusters(mean_centroids, top_m=skin_top_m)

        mask = skin_mask_from_probmaps(prob_maps, skin_clusters, prob_thresh=prob_thresh)

        save_probability_maps_and_mask(
            out_dir=out_dir,
            base_name=base_name,
            rgb=rgb_u8,
            k=k,
            prob_maps=prob_maps,
            mean_centroids=mean_centroids,
            skin_clusters=skin_clusters,
            skin_mask=mask
        )


# ----------------------------
# CLI
# ----------------------------
def parse_skin_override(s: str | None):
    """
    Optional override format example:
      "2:0;3:1,2;5:2"
    meaning:
      k=2 -> [0]
      k=3 -> [1,2]
      k=5 -> [2]
    """
    if not s:
        return None
    out = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for p in parts:
        k_str, ids_str = p.split(":")
        k = int(k_str.strip())
        ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
        out[k] = ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", required=True, help="Path to first hand image")
    ap.add_argument("--img2", required=True, help="Path to second hand image (different background)")
    ap.add_argument("--reps", type=int, default=100, help="Number of k-means repetitions (default 100)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed base (default 0)")
    ap.add_argument("--prob_thresh", type=float, default=0.60,
                    help="Threshold on max probability to mark skin (default 0.60)")
    ap.add_argument("--max_side", type=int, default=900,
                    help="Resize so max(H,W)=max_side for speed; set 0 to disable (default 900)")
    ap.add_argument("--skin_top_m", type=int, default=1,
                    help="Auto-pick top_m clusters as 'skin' (default 1)")
    ap.add_argument("--skin_override", type=str, default=None,
                    help='Override skin clusters per k. Format: "2:0;3:1,2;5:2"')
    ap.add_argument("--out", type=str, default="outputs", help="Output directory (default outputs)")
    args = ap.parse_args()

    max_side = None if args.max_side == 0 else args.max_side
    skin_override = parse_skin_override(args.skin_override)

    ensure_dir(args.out)

    process_one_image(
        img_path=args.img1,
        out_root=args.out,
        reps=args.reps,
        seed=args.seed,
        prob_thresh=args.prob_thresh,
        max_side=max_side,
        skin_top_m=args.skin_top_m,
        skin_override=skin_override
    )

    process_one_image(
        img_path=args.img2,
        out_root=args.out,
        reps=args.reps,
        seed=args.seed + 1,  # small change so two images don't reuse identical RNG streams
        prob_thresh=args.prob_thresh,
        max_side=max_side,
        skin_top_m=args.skin_top_m,
        skin_override=skin_override
    )

    print(f"Done. Results saved under: {args.out}/")


if __name__ == "__main__":
    main()
