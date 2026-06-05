"""
figure_1_method_comparison.py — Fixed: reads image blobs from DB, not file paths
"""
import os, sys, sqlite3, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
import arhayas_math as am

DB_PATH = "arhayas_library.db"; TARGET_SIZE = 256; SEED = 42
ROW_CATEGORIES = [
    ("real_faces_1k",    "Portrait face"),
    ("coral (healthy)",  "Healthy coral"),
    ("quasicrystal",     "Quasicrystal"),
]

def get_image_from_db(db_path, category):
    """Pull one raw image blob from the DB and decode it."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT raw_image FROM samples WHERE category=? ORDER BY RANDOM() LIMIT 1", (category,))
    row = cur.fetchone()
    if row is None:
        cur.execute("SELECT raw_image FROM samples WHERE category LIKE ? ORDER BY RANDOM() LIMIT 1", (f"%{category}%",))
        row = cur.fetchone()
    con.close()
    if row is None:
        return None
    blob = row[0]
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img

def centre_crop(img, size):
    h, w = img.shape[:2]; m = min(h, w)
    top = (h - m) // 2; left = (w - m) // 2
    return cv2.resize(img[top:top+m, left:left+m], (size, size), interpolation=cv2.INTER_AREA)

def phase_surrogate(img_gray):
    return am.phase_randomize(img_gray, rng=np.random.default_rng(SEED))

def iaaft_surrogate(img_gray):
    rng = np.random.default_rng(SEED)
    if hasattr(am, 'iaaft_surrogate'):
        return am.iaaft_surrogate(img_gray, rng=rng)
    elif hasattr(am, 'amplitude_adjusted_surrogate'):
        return am.amplitude_adjusted_surrogate(img_gray, rng=rng)
    else:
        raise AttributeError("IAAFT function not found in arhayas_math.")

def main():
    os.makedirs("exports", exist_ok=True); np.random.seed(SEED)

    print("Loading images from database...")
    rows = []
    for cat, label in ROW_CATEGORIES:
        img = get_image_from_db(DB_PATH, cat)
        if img is None:
            print(f"  WARNING: no image found for '{cat}' -- skipping"); continue
        img_norm = am.normalize_image(centre_crop(img, TARGET_SIZE))
        print(f"  {label}: loaded ({img.shape})")
        rows.append((label, img_norm))

    if not rows:
        print("ERROR: no images loaded."); return

    print("Generating surrogates...")
    panels = []
    for label, img in rows:
        ps = phase_surrogate(img)
        ia = iaaft_surrogate(img)
        panels.append((label, img, ps, ia))
        print(f"  {label}: done")

    n_rows = len(panels)
    col_labels = ["Original", "Phase surrogate\n(Gaussian washout)", "IAAFT surrogate\n(distribution preserved)"]

    fig = plt.figure(figsize=(10, 3.5 * n_rows), dpi=300)
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.05, wspace=0.03,
                           top=0.93, bottom=0.03, left=0.12, right=0.98)

    for i, (label, orig, phase_s, iaaft_s) in enumerate(panels):
        for j, img_arr in enumerate([orig, phase_s, iaaft_s]):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img_arr, cmap='gray', vmin=0, vmax=255, aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(col_labels[j], fontsize=9, fontweight='bold', pad=6)
            if j == 0: ax.set_ylabel(label, fontsize=9, fontweight='bold', rotation=90, labelpad=8, va='center')

    plt.suptitle("Figure 1. Original images alongside phase-randomized and IAAFT surrogates.", fontsize=9, y=0.98)

    for ext in ["png", "pdf"]:
        out = os.path.join("exports", f"figure_1_method_comparison.{ext}")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()

if __name__ == "__main__":
    main()
