"""
surrogate_by_category.py
=========================
Multi-category IAAFT surrogate experiment for Paper 2.

For every (category, state) group in the database, draws N_IMAGES_PER_CAT
images, generates N_SURROGATES IAAFT surrogates per image, and reports
what fraction of images separate from their surrogates on each metric.

This is the core data for Paper 2's main results section. Run AFTER
defense1_experiment.py confirms the method is valid.

Outputs (in exports/):
  surrogate_by_category_raw.csv     — one row per (image × metric),
                                      z-scores and verdicts. Goes to git.
  surrogate_by_category_summary.csv — per-(category,state,metric)
                                      separation rates. Goes to paper table.
  surrogate_by_category_summary.txt — human-readable. Goes to handover.

TIMING:  ~5-7 min per image (IAAFT + full TDA pipeline × N_SURROGATES).
         With defaults (5 images × 20 surrogates) expect ~30 min per
         category group. Run overnight for full dataset.

         Quick test: set N_IMAGES_PER_CAT=2, N_SURROGATES=10 at top of file.

Results are written after EVERY image so a crash loses nothing.

Usage:   python scripts/surrogate_by_category.py
"""

import os
import sys
import time
import csv
import json
import sqlite3
import random
import numpy as np
import cv2

# --- Path setup ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import arhayas_math as am
import arhayas_library as al

# =============================================================================
# CONFIGURATION  — edit these before running
# =============================================================================
N_IMAGES_PER_CAT = 30    # images sampled per (category, state) pair
N_SURROGATES     = 20   # IAAFT surrogates per image
SEED             = 42   # random seed for reproducible sampling
DB_PATH          = "arhayas_library.db"
EXPORTS_DIR      = "exports"
# Categories to SKIP (already covered by defense1 as calibration anchors,
# or not meaningful for the paper claim)
SKIP_CATEGORIES  = {"quarantined", "rejected"}
# =============================================================================

# Correct metric keys as returned by analyze_gray
METRICS = [
    # TDA
    "betti_0", "betti_1",
    "persistence_entropy_h1", "total_persistence_h1",
    # Topological — Euler
    "euler_mean", "euler_std", "euler_range",
    # Structural
    "bilateral_symmetry",
    "f_dim",                   # NOTE: was miskeyed as "fractal_dim" in defense1
    "lacunarity_mean",
    "multifractal_width",
    "orientation_coherence",
    "multiscale_entropy_slope",
    # Graph
    "junctions", "clustering", "plateau",
    # Basic
    "entropy",
    # Forbidden symmetry
    "forbidden_coverage",
    "tau_ratio_score",
]

SEP_THRESHOLD     = 3.0   # |z| > 3.0 = SEPARATED
MARGINAL_THRESHOLD = 2.0  # 2 < |z| < 3 = MARGINAL


def extract_metrics(result_dict):
    out = {}
    for k in METRICS:
        val = result_dict.get(k, float("nan"))
        try:
            out[k] = float(val) if val is not None else float("nan")
        except (TypeError, ValueError):
            out[k] = float("nan")
    return out


def compute_z(orig_val, surro_vals):
    """Return (z, verdict). Handles NaN and zero-variance gracefully."""
    arr = np.array([v for v in surro_vals if np.isfinite(v)], dtype=float)
    if not np.isfinite(orig_val) or len(arr) < 3:
        return float("nan"), "skip"
    s_mean = arr.mean()
    s_std  = arr.std()
    # Zero-variance surrogates: note as artifact rather than giving huge z
    if s_std < 1e-10:
        if abs(orig_val - s_mean) < 1e-10:
            return 0.0, "no sep"
        return float("nan"), "zero-var"  # artifact — report separately
    z = (orig_val - s_mean) / s_std
    if abs(z) >= SEP_THRESHOLD:
        verdict = "SEPARATED"
    elif abs(z) >= MARGINAL_THRESHOLD:
        verdict = "MARGINAL"
    else:
        verdict = "no sep"
    return z, verdict


def get_category_groups(db_path):
    """
    Return dict of {group_label: [(image_hash, source_path, raw_image), ...]}.

    Groups are collapsed for Paper 2 relevance:
      - 'faces' (FG-NET): all age_X states merged into one pool
      - 'quasicrystal': all states merged
      - 'coral': SPLIT by state (H=healthy, B=bleached) because the
        difference is scientifically meaningful
      - everything else: grouped by category, state ignored
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT category, state, image_hash, source_path, raw_image
        FROM samples
        WHERE filter_type = 'raw'
          AND raw_image IS NOT NULL
          AND category NOT IN ({})
        GROUP BY category, state, image_hash
        ORDER BY category, state, image_hash
    """.format(",".join(f"'{c}'" for c in SKIP_CATEGORIES))).fetchall()
    conn.close()

    # Exclude 001Q004 — unverified provenance, dropped from Paper 2
    rows = [r for r in rows if '001Q004' not in (r[3] or '')]

    from collections import defaultdict
    groups = defaultdict(list)
    for cat, state, img_hash, src_path, raw_img in rows:
        # Collapse face ages into one group
        if cat == "faces":
            label = "faces (FG-NET)"
        # Collapse quasicrystal states into one group
        elif cat == "quasicrystal":
            label = "quasicrystal"
        # Keep coral split by health state
        elif cat == "coral":
            label = f"coral ({'healthy' if state == 'H' else 'bleached'})"
        # Everything else: category name is the label
        else:
            label = cat
        groups[label].append((img_hash, src_path, raw_img))
    return dict(groups)


def process_image(raw_image_bytes, source_path, n_surrogates):
    """
    Run full SCOTT + IAAFT surrogate test on one image.
    Returns (orig_metrics_dict, list_of_surro_metrics_dicts).
    """
    # Decode raw image
    img = cv2.imdecode(np.frombuffer(raw_image_bytes, np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {source_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = am.normalize_image(gray)

    # Original metrics
    orig_result  = al.analyze_gray(gray)
    orig_metrics = extract_metrics(orig_result)

    # IAAFT surrogate metrics
    surro_metrics_list = []
    rng = np.random.default_rng(SEED)
    for i in range(n_surrogates):
        surro     = am.iaaft_surrogate(gray, max_iter=100, rng=rng)
        s_result  = al.analyze_gray(surro)
        surro_metrics_list.append(extract_metrics(s_result))

    return orig_metrics, surro_metrics_list


def main():
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    raw_csv_path     = os.path.join(EXPORTS_DIR, "surrogate_by_category_raw.csv")
    summary_csv_path = os.path.join(EXPORTS_DIR, "surrogate_by_category_summary.csv")
    summary_txt_path = os.path.join(EXPORTS_DIR, "surrogate_by_category_summary.txt")

    # Raw CSV fieldnames
    raw_fields = [
        "category", "state", "image_name", "image_hash",
        "metric", "orig_val", "surro_mean", "surro_std",
        "z_score", "verdict", "n_surrogates",
    ]
    # Write fresh (overwrite any previous partial run)
    raw_f  = open(raw_csv_path, "w", newline="")
    raw_w  = csv.DictWriter(raw_f, fieldnames=raw_fields)
    raw_w.writeheader()

    # Load DB groups and sample
    print("Querying database...")
    groups = get_category_groups(DB_PATH)
    if not groups:
        print("No images found. Check DB_PATH and that images have filter_type='raw'.")
        raw_f.close()
        return

    rng_sample = random.Random(SEED)
    print(f"\nFound {len(groups)} groups:")
    sampled = {}
    for label, images in sorted(groups.items()):
        n_avail = len(images)
        n_take  = min(n_avail, N_IMAGES_PER_CAT)
        selected = rng_sample.sample(images, n_take)
        sampled[label] = selected
        print(f"  {label:<35} "
              f"{n_avail:>5} available -> {n_take} sampled")

    total_images = sum(len(v) for v in sampled.values())
    print(f"\nTotal images to process: {total_images}")
    print(f"Surrogates per image:     {N_SURROGATES}")
    est_min = total_images * N_SURROGATES * 6 / 60
    print(f"Estimated time:           ~{est_min:.0f} minutes "
          f"(~6s per surrogate+analysis)\n")

    # Accumulate for summary
    # results[(cat, state)][metric] = list of (z, verdict)
    from collections import defaultdict
    all_results = defaultdict(lambda: defaultdict(list))

    t_global = time.time()
    img_idx  = 0

    for label, images in sorted(sampled.items()):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        for img_hash, src_path, raw_img in images:
            img_idx += 1
            name = os.path.basename(src_path) if src_path else img_hash[:16]
            print(f"\n  [{img_idx}/{total_images}] {name}")

            try:
                t0 = time.time()
                orig_m, surro_list = process_image(raw_img, src_path, N_SURROGATES)
                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.1f}s")
            except Exception as e:
                print(f"  ERROR: {e} — skipping")
                continue

            # Compute z-scores and write immediately
            for metric in METRICS:
                orig_val   = orig_m[metric]
                surro_vals = [s[metric] for s in surro_list]
                surro_arr  = [v for v in surro_vals if np.isfinite(v)]
                s_mean = float(np.mean(surro_arr)) if surro_arr else float("nan")
                s_std  = float(np.std(surro_arr))  if surro_arr else float("nan")
                z, verdict = compute_z(orig_val, surro_vals)

                raw_w.writerow({
                    "category":    label,
                    "state":       "",
                    "image_name":  name,
                    "image_hash":  img_hash,
                    "metric":      metric,
                    "orig_val":    round(orig_val, 6) if np.isfinite(orig_val) else "nan",
                    "surro_mean":  round(s_mean, 6)   if np.isfinite(s_mean)  else "nan",
                    "surro_std":   round(s_std, 6)    if np.isfinite(s_std)   else "nan",
                    "z_score":     round(z, 4)         if np.isfinite(z)       else "nan",
                    "verdict":     verdict,
                    "n_surrogates": N_SURROGATES,
                })
                all_results[label][metric].append((z, verdict))

            raw_f.flush()

        elapsed_global = time.time() - t_global
        remaining = total_images - img_idx
        eta = (elapsed_global / max(img_idx, 1)) * remaining
        print(f"\n  Category done. "
              f"Total elapsed: {elapsed_global/60:.1f}min, "
              f"ETA remaining: {eta/60:.1f}min")

    raw_f.close()
    print(f"\nRaw results saved: {raw_csv_path}")

    # === Build summary ===
    summary_rows = []
    summary_lines = [
        "=" * 72,
        "  MULTI-CATEGORY IAAFT SURROGATE EXPERIMENT — SUMMARY",
        "=" * 72,
        f"  N_IMAGES_PER_CAT = {N_IMAGES_PER_CAT}",
        f"  N_SURROGATES     = {N_SURROGATES}",
        f"  SEED             = {SEED}",
        "",
    ]

    for label in sorted(all_results.keys()):
        metric_results = all_results[label]
        summary_lines.append(f"--- {label} ---")
        summary_lines.append(
            f"  {'Metric':<28} {'n_images':>8} {'n_sep':>6} "
            f"{'sep_rate':>9} {'mean_|z|':>9}")
        summary_lines.append("  " + "-" * 60)

        for metric in METRICS:
            pairs = metric_results.get(metric, [])
            valid = [(z, v) for z, v in pairs if np.isfinite(z)]
            n_img = len(valid)
            n_sep = sum(1 for _, v in valid if v == "SEPARATED")
            n_mar = sum(1 for _, v in valid if v == "MARGINAL")
            sep_rate = n_sep / n_img if n_img else float("nan")
            mean_absz = np.mean([abs(z) for z, _ in valid]) if valid else float("nan")

            flag = ""
            if sep_rate >= 0.8:  flag = "  *** STRONG"
            elif sep_rate >= 0.4: flag = "  ** MODERATE"
            elif n_mar > 0:       flag = "  * marginal"

            summary_lines.append(
                f"  {metric:<28} {n_img:>8} {n_sep:>6} "
                f"{sep_rate:>9.1%} {mean_absz:>9.2f}{flag}")

            summary_rows.append({
                "category":    label,
                "state":       "",
                "metric":      metric,
                "n_images":    n_img,
                "n_separated": n_sep,
                "n_marginal":  n_mar,
                "sep_rate":    round(sep_rate, 4) if np.isfinite(sep_rate) else "nan",
                "mean_abs_z":  round(mean_absz, 4) if np.isfinite(mean_absz) else "nan",
            })

        summary_lines.append("")

    # Overall verdict
    summary_lines.append("=" * 72)
    summary_lines.append("  METRICS WITH >=80% SEPARATION ACROSS ALL CATEGORIES")
    summary_lines.append("=" * 72)
    for metric in METRICS:
        all_rates = []
        for label in all_results:
            pairs = all_results[label].get(metric, [])
            valid = [(z, v) for z, v in pairs if np.isfinite(z)]
            if valid:
                rate = sum(1 for _, v in valid if v == "SEPARATED") / len(valid)
                all_rates.append(rate)
        if all_rates:
            min_rate = min(all_rates)
            if min_rate >= 0.8:
                summary_lines.append(
                    f"  {metric:<28}  min sep rate across categories: {min_rate:.0%}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(summary_txt_path, "w") as f:
        f.write(summary_text)

    summary_fields = ["category", "state", "metric", "n_images",
                      "n_separated", "n_marginal", "sep_rate", "mean_abs_z"]
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSummary CSV saved: {summary_csv_path}")
    print(f"Summary TXT saved: {summary_txt_path}")
    total_elapsed = time.time() - t_global
    print(f"Total runtime: {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
