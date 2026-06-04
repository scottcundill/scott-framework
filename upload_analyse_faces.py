# analyse_faces.py
# =============================================================================
# SCOTT — Structural and Coherent Order in Topological Transforms
# Example script: measure bilateral symmetry on a folder of images.
#
# Usage:
#   python analyse_faces.py --folder path/to/images --output results.csv
#
# Output CSV columns:
#   filename, filter, bilateral_symmetry, lr_imbalance
# =============================================================================

import os
import csv
import argparse
import cv2
from bilateral_symmetry import normalize_image, bilateral_symmetry
from image_filters import raw_filter, topographic_filter, edge_filter


FILTERS = {
    'raw':         raw_filter,
    'topographic': topographic_filter,
    'edge':        edge_filter,
}


def analyse_folder(folder, output_csv):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in extensions]
    files.sort()

    if not files:
        print(f"No image files found in {folder}")
        return

    print(f"Found {len(files)} images. Processing...")

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'filter', 'bilateral_symmetry',
                         'lr_imbalance'])

        for i, fname in enumerate(files):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Skipping {fname} — could not read.")
                continue

            img_norm = normalize_image(img)

            for filter_name, filter_fn in FILTERS.items():
                filtered = filter_fn(img_norm)
                # convert back to greyscale for metric computation
                if filtered.ndim == 3:
                    filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
                else:
                    filtered_gray = filtered

                bs, imbalance = bilateral_symmetry(filtered_gray)
                writer.writerow([fname, filter_name, bs, imbalance])

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(files)}")

    print(f"\nDone. Results saved to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Measure bilateral symmetry on a folder of images.')
    parser.add_argument('--folder', required=True,
                        help='Path to folder containing images')
    parser.add_argument('--output', default='results.csv',
                        help='Output CSV filename (default: results.csv)')
    args = parser.parse_args()
    analyse_folder(args.folder, args.output)
