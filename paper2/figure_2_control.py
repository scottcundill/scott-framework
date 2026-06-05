"""
figure_2_control.py — Fixed: reads image blobs from DB, not file paths
"""
import os, sys, sqlite3, numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
import arhayas_math as am

DB_PATH = "arhayas_library.db"; TARGET_SIZE = 256; SEED = 42

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

def iaaft_surrogate(img_gray, seed=SEED):
    rng = np.random.default_rng(seed)
    if hasattr(am, 'iaaft_surrogate'):
        return am.iaaft_surrogate(img_gray, rng=rng)
    elif hasattr(am, 'amplitude_adjusted_surrogate'):
        return am.amplitude_adjusted_surrogate(img_gray, rng=rng)
    else:
        raise AttributeError("IAAFT function not found in arhayas_math.")

def make_gamma_noise(size, seed=SEED):
    rng = np.random.default_rng(seed)
    noise = rng.gamma(shape=2.0, scale=30.0, size=(size, size))
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    return noise.astype(np.uint8)

def diff_map(orig, surro):
    d = np.abs(orig.astype(float) - surro.astype(float))
    if d.max() > 0: d = d / d.max() * 255
    return d.astype(np.uint8)

def main():
    os.makedirs("exports", exist_ok=True); np.random.seed(SEED)

    print("Building control image...")
    noise_norm = am.normalize_image(make_gamma_noise(TARGET_SIZE))

    print("Loading face image from database...")
    face_raw = get_image_from_db(DB_PATH, "real_faces_1k")
    if face_raw is None:
        face_raw = get_image_from_db(DB_PATH, "ai_faces_1k")
    if face_raw is None:
        print("ERROR: no face images in database."); return
    face_norm = am.normalize_image(centre_crop(face_raw, TARGET_SIZE))
    print(f"  Face loaded ({face_raw.shape})")

    print("Generating IAAFT surrogates...")
    noise_surro = iaaft_surrogate(noise_norm)
    face_surro  = iaaft_surrogate(face_norm)
    noise_diff  = diff_map(noise_norm, noise_surro)
    face_diff   = diff_map(face_norm, face_surro)

    print("Plotting...")
    row_labels = ["Original", "IAAFT surrogate", "Pixel difference\n(enhanced 4x)"]
    col_labels = ["Non-Gaussian noise\n(no spatial structure)", "Real face\n(genuine spatial structure)"]

    fig = plt.figure(figsize=(6, 9.5), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.06, wspace=0.04,
                           top=0.92, bottom=0.03, left=0.18, right=0.98)
    cells = [[noise_norm, face_norm], [noise_surro, face_surro], [noise_diff, face_diff]]

    for i in range(3):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            img_to_show = cells[i][j]
            if i == 2:
                img_to_show = np.clip(img_to_show.astype(float) * 4, 0, 255).astype(np.uint8)
            ax.imshow(img_to_show, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(col_labels[j], fontsize=8.5, fontweight='bold', pad=5)
            if j == 0: ax.set_ylabel(row_labels[i], fontsize=8.5, rotation=90, labelpad=8, va='center')

    plt.suptitle(
        "Figure 2. The non-Gaussian unstructured control.\n"
        "IAAFT surrogates preserve distribution; pixel differences reveal\n"
        "that structure is destroyed in the face but not in the noise.",
        fontsize=8, y=0.98)

    for ext in ["png", "pdf"]:
        out = os.path.join("exports", f"figure_2_control.{ext}")
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()

if __name__ == "__main__":
    main()
