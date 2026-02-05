# ================================
# Section 1 — Load & resize images
# ================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- Paths (edit if needed) ----
img1_path = "/Users/tanaypatel/Desktop/Purdue/TheFinalRun-2/CSCI448-Biometrics/Assignment 2/hand1.jpg"
img2_path = "/Users/tanaypatel/Desktop/Purdue/TheFinalRun-2/CSCI448-Biometrics/Assignment 2/hand2.jpg"

# ---- Target resize (KEEP SAME for both images) ----
# Recommend: 512 x 912 (height x width)
TARGET_H, TARGET_W = 912, 512

def load_rgb(path: str) -> np.ndarray:
    """Load an image from disk and return RGB uint8 array of shape (H, W, 3)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def resize_rgb(img_rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize RGB image to (target_h, target_w) using good downscale interpolation."""
    return cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

# ---- Load originals ----
img1_orig = load_rgb(img1_path)
img2_orig = load_rgb(img2_path)

# ---- Resize ----
img1 = resize_rgb(img1_orig, TARGET_H, TARGET_W)
img2 = resize_rgb(img2_orig, TARGET_H, TARGET_W)

print("Original shapes:", img1_orig.shape, img2_orig.shape)
print("Resized shapes: ", img1.shape, img2.shape)

# ---- Quick visualization (original vs resized) ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(img1_orig); axes[0, 0].set_title("Image 1 (original)"); axes[0, 0].axis("off")
axes[0, 1].imshow(img1);      axes[0, 1].set_title("Image 1 (resized)");  axes[0, 1].axis("off")
axes[1, 0].imshow(img2_orig); axes[1, 0].set_title("Image 2 (original)"); axes[1, 0].axis("off")
axes[1, 1].imshow(img2);      axes[1, 1].set_title("Image 2 (resized)");  axes[1, 1].axis("off")
plt.tight_layout()
plt.show()


# ================================
# Section 2 — Single k-means run
# ================================


def random_init_centroids(img_rgb: np.ndarray, k: int) -> np.ndarray:
    """
    Pick k initial centroids by sampling k random pixels from the image.
    Returns array of shape (k, 3), dtype float32.
    """
    H, W, C = img_rgb.shape
    X = img_rgb.reshape(-1, 3).astype(np.float32)
    N = X.shape[0]
    indices = np.random.choice(N, size=min(k, N), replace=False)
    return X[indices].copy()

def kmeans_single_run(img_rgb: np.ndarray,
                      init_centroids: np.ndarray,
                      max_iters: int = 30,
                      tol: float = 1e-3,
                      seed: int | None = None):
    """
    Run one instance of k-means on an RGB image.

    Parameters
    ----------
    img_rgb : np.ndarray
        Image array of shape (H, W, 3), dtype uint8 or float.
    init_centroids : np.ndarray
        Initial centroids of shape (k, 3) in RGB space.
    max_iters : int
        Maximum number of k-means iterations (within a single run).
    tol : float
        Convergence threshold on centroid movement (L2 norm).
    seed : int or None
        Optional seed (usually not needed here).

    Returns
    -------
    labels_img : np.ndarray
        Cluster labels of shape (H, W), values in {0..k-1}.
    centroids : np.ndarray
        Final centroids of shape (k, 3), float32.
    """
    if seed is not None:
        np.random.seed(seed)

    H, W, C = img_rgb.shape
    assert C == 3, "Expected RGB image with 3 channels."

    # Flatten pixels to (N, 3) and convert to float for math
    X = img_rgb.reshape(-1, 3).astype(np.float32)
    centroids = init_centroids.astype(np.float32).copy()
    k = centroids.shape[0]

    labels = np.zeros(X.shape[0], dtype=np.int32)

    for it in range(max_iters):
        # --- Assign step: nearest centroid for each pixel ---
        # distances shape: (N, k)
        # Using (x-c)^2 = x^2 + c^2 - 2xc trick for speed
        x2 = np.sum(X * X, axis=1, keepdims=True)          # (N,1)
        c2 = np.sum(centroids * centroids, axis=1)         # (k,)
        distances = x2 + c2 - 2 * (X @ centroids.T)        # (N,k)

        new_labels = np.argmin(distances, axis=1).astype(np.int32)

        # --- Update step: recompute centroids as mean of assigned pixels ---
        new_centroids = centroids.copy()
        for ci in range(k):
            mask = (new_labels == ci)
            if np.any(mask):
                new_centroids[ci] = X[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize to a random pixel (prevents crash)
                rand_idx = np.random.randint(0, X.shape[0])
                new_centroids[ci] = X[rand_idx]

        # --- Convergence check ---
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels

        if shift < tol:
            break

    labels_img = labels.reshape(H, W)
    return labels_img, centroids



# ================================
# Section 3 — Cluster label alignment
# ================================

def align_clusters_by_red(labels_img: np.ndarray,
                          centroids: np.ndarray):
    """
    Reassign cluster labels so that cluster indices are ordered
    by increasing R-value of their centroids.

    Parameters
    ----------
    labels_img : np.ndarray
        Cluster labels of shape (H, W), values in {0..k-1}.
    centroids : np.ndarray
        Centroids of shape (k, 3) in RGB.

    Returns
    -------
    labels_aligned : np.ndarray
        Relabeled cluster map of shape (H, W).
    centroids_aligned : np.ndarray
        Centroids reordered by increasing R-value.
    """
    # Sort centroid indices by R channel (index 0)
    order = np.argsort(centroids[:, 0])  # ascending R

    # Create mapping old_label -> new_label
    label_map = np.zeros(len(order), dtype=np.int32)
    for new_label, old_label in enumerate(order):
        label_map[old_label] = new_label

    # Apply mapping to labels
    labels_aligned = label_map[labels_img]

    # Reorder centroids
    centroids_aligned = centroids[order]

    return labels_aligned, centroids_aligned

# =========================
# Section 4 — Repetitive k-means (store assignments in M)
# =========================

def my_rep_kmeans(img_rgb: np.ndarray,
                 k: int,
                 num_reps: int = 100,
                 max_iters: int = 30,
                 tol: float = 1e-3,
                 seed: int | None = None):
    """
    Repetitive k-means:
    Runs k-means num_reps times with random initialization,
    aligns cluster labels after each run, and stores labels in M.

    Returns
    -------
    M : np.ndarray
        Assignment history of shape (H, W, num_reps), dtype uint8 or int16.
        M[:,:,r] is the aligned label map from repetition r.
    centroids_last : np.ndarray
        Centroids from the last repetition (aligned order).
        (Useful for debugging / reference.)
    """
    if seed is not None:
        np.random.seed(seed)

    H, W, _ = img_rgb.shape
    M = np.zeros((H, W, num_reps), dtype=np.uint8)

    centroids_last = None

    for rep in range(num_reps):
        # 1) random init using existing pixels
        init_centroids = random_init_centroids(img_rgb, k)

        # 2) run one k-means instance to convergence
        labels_img, centroids = kmeans_single_run(
            img_rgb,
            init_centroids,
            max_iters=max_iters,
            tol=tol
        )

        # 3) align labels for consistency across repetitions
        labels_aligned, centroids_aligned = align_clusters_by_red(labels_img, centroids)

        # 4) store aligned labels
        M[:, :, rep] = labels_aligned.astype(np.uint8)
        centroids_last = centroids_aligned

        # (optional) progress print every 10 runs
        if (rep + 1) % 10 == 0:
            print(f"k={k}: finished {rep+1}/{num_reps} repetitions")

    return M, centroids_last



# Section 4 test (must run so M3_test is defined for Section 5)
'''
M3_test, cent_last = my_rep_kmeans(img1, k=3, num_reps=10, seed=0)
print("M shape:", M3_test.shape)
print("Last centroids:\n", cent_last)

# visualize the labels from repetition 0
plt.figure(figsize=(6,4))
plt.imshow(M3_test[:, :, 0], cmap="tab20")
plt.title("Example: labels from repetition 1 (k=3)")
plt.axis("off")
plt.show()
'''


# ================================
# Section 5 — Probability maps from M
# ================================


def compute_probability_maps(M: np.ndarray, k: int) -> np.ndarray:
    """
    Convert assignment history M (H, W, num_reps) into probability maps P (k, H, W),
    where P[i, y, x] = fraction of repetitions pixel (y,x) was assigned to cluster i.

    Parameters
    ----------
    M : np.ndarray
        Cluster labels over repetitions, shape (H, W, num_reps), values in {0..k-1}
    k : int
        Number of clusters

    Returns
    -------
    P : np.ndarray
        Probability maps, shape (k, H, W), dtype float32
    """
    H, W, R = M.shape
    P = np.zeros((k, H, W), dtype=np.float32)

    # Count frequency for each cluster label
    for i in range(k):
        P[i] = np.mean(M == i, axis=2)  # average over repetitions = probability

    return P

def sanity_check_prob_maps(P: np.ndarray, atol: float = 1e-5):
    """
    Check that probabilities sum to ~1 at each pixel.
    """
    s = P.sum(axis=0)  # (H, W)
    print("Sum(P) stats: min=", float(s.min()), "max=", float(s.max()), "mean=", float(s.mean()))
    if not np.allclose(s, 1.0, atol=atol):
        print("WARNING: Probabilities do not sum to 1 everywhere (check labels / k).")

def plot_probability_maps(P: np.ndarray, k: int, title_prefix: str = ""):
    """
    Plot the k probability maps as heatmaps with colorbars.
    """
    # P shape: (k, H, W)
    fig, axes = plt.subplots(1, k, figsize=(4*k, 4))
    if k == 1:
        axes = [axes]

    for i in range(k):
        im = axes[i].imshow(P[i], cmap="hot", vmin=0, vmax=1)
        axes[i].set_title(f"{title_prefix}P(cluster {i})")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()



# Example: if you already have M3_test from Section 4
'''
k = 3
P3 = compute_probability_maps(M3_test, k)
sanity_check_prob_maps(P3)

plot_probability_maps(P3, k, title_prefix="k=3 ")
'''

# =========================
# Section 6 — Visualization (probability maps + helpers)
# =========================

import numpy as np
import matplotlib.pyplot as plt

def plot_prob_maps(P: np.ndarray, title_prefix: str = "", show_colorbar: bool = True):
    """
    Plot k probability maps from P of shape (k, H, W).
    """
    k, H, W = P.shape
    fig, axes = plt.subplots(1, k, figsize=(4*k, 4))
    if k == 1:
        axes = [axes]

    for i in range(k):
        im = axes[i].imshow(P[i], cmap="hot", vmin=0, vmax=1)
        axes[i].set_title(f"{title_prefix}P(cluster {i})")
        axes[i].axis("off")
        if show_colorbar:
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot_skin_overlay(img_rgb: np.ndarray, skin_mask: np.ndarray, title: str = "Skin mask overlay"):
    """
    Show original image, skin mask, and overlay.
    """
    skin_mask_bool = skin_mask.astype(bool)

    # Overlay: keep skin pixels, darken background
    overlay = img_rgb.copy().astype(np.float32)
    overlay[~skin_mask_bool] *= 0.25
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(skin_mask, cmap="gray"); axes[1].set_title("Skin mask"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title(title); axes[2].axis("off")
    plt.tight_layout()
    plt.show()

# =========================
# Section 7 — Skin detection from probability maps
# =========================

def pick_skin_cluster_by_red(centroids_aligned: np.ndarray) -> int:
    """
    If you aligned clusters by increasing R, then the skin-like cluster
    is often the one with the highest R centroid => last index.
    """
    # centroids_aligned already sorted by R ascending
    return int(np.argmax(centroids_aligned[:, 0]))

def detect_skin_from_prob_maps(P: np.ndarray,
                               skin_clusters: list[int],
                               threshold: float = 0.6):
    """
    Skin detection rule:
    - For each pixel, compute max probability and argmax cluster
    - Skin if:
        argmax in skin_clusters AND max_prob >= threshold

    Parameters
    ----------
    P : np.ndarray
        Probability maps of shape (k, H, W)
    skin_clusters : list[int]
        Which cluster indices correspond to skin
    threshold : float
        Probability threshold

    Returns
    -------
    skin_mask : np.ndarray
        Binary mask of shape (H, W), dtype uint8 (0 or 1)
    argmax_map : np.ndarray
        Argmax cluster index per pixel, shape (H, W)
    maxprob_map : np.ndarray
        Max probability per pixel, shape (H, W)
    """
    argmax_map = np.argmax(P, axis=0)       # (H, W)
    maxprob_map = np.max(P, axis=0)         # (H, W)

    skin_cluster_mask = np.isin(argmax_map, skin_clusters)
    skin_mask = (skin_cluster_mask & (maxprob_map >= threshold)).astype(np.uint8)

    return skin_mask, argmax_map, maxprob_map


# ================================
# FINAL PIPELINE (RUN EXPERIMENT)
# ================================

K_VALUES = [2, 3, 5]
THRESH = 0.6

for idx, img in enumerate([img1, img2], start=1):
    for k in K_VALUES:
        print(f"\n=== Image {idx}, k={k} ===============")
        M, cent_last = my_rep_kmeans(img, k=k, num_reps=100, seed=0)
        P = compute_probability_maps(M, k=k)
        sanity_check_prob_maps(P)

        # Section 6
        plot_prob_maps(P, title_prefix=f"img{idx} k={k} ")

        # Section 7
        skin_cluster = pick_skin_cluster_by_red(cent_last)
        skin_mask, _, _ = detect_skin_from_prob_maps(P, [skin_cluster], threshold=THRESH)
        plot_skin_overlay(img, skin_mask, title=f"img{idx} k={k} thr={THRESH} skin={skin_cluster}")

