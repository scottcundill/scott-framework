# SCOTT — Structural and Coherent Order in Topological Transforms

An open-source Python framework for measuring geometric and topological order in 2D images.

## What this is

SCOTT provides a set of image analysis metrics that measure structural properties of 2D images — symmetry, topological complexity, edge structure, and multi-scale organisation. The framework is domain-agnostic: it operates on any greyscale image, whether a face photograph, a microscopy image, or a synthetic pattern.

This repository contains the core metric and filter implementations used in:

> Cundill, S. (2026). *Bilateral Symmetry as a Filter-Robust Discriminator Between Real and GAN-Generated Face Images.* [preprint]

## What this repository contains

| File | Description |
|------|-------------|
| `bilateral_symmetry.py` | Global bilateral symmetry metric (BS) and image normalisation |
| `image_filters.py` | Three image preprocessing filters: raw, topographic isoline, edge |
| `analyse_faces.py` | Example script: run bilateral symmetry on a folder of images |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
pip install -r requirements.txt
python analyse_faces.py --folder path/to/images --output results.csv
```

This produces a CSV with bilateral symmetry scores for each image under all three filter representations.

## The bilateral symmetry metric

```python
from bilateral_symmetry import normalize_image, bilateral_symmetry

import cv2
img = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)
img_norm = normalize_image(img)           # resize to 512x512
bs, imbalance = bilateral_symmetry(img_norm)
print(f"Bilateral symmetry: {bs:.4f}")   # 0.0 = no symmetry, 1.0 = perfect
```

The metric is defined as:

```
BS = 1 - MAD(I, F)

where I  = normalised image (512x512, float [0,1])
      F  = horizontal flip of I
      MAD = mean absolute difference, pixel-wise
```

## The three filters

Each filter suppresses a different class of image information:

- **Raw** — control, no processing, retains all spatial information
- **Topographic** — CLAHE + bilateral smoothing + intensity isolines; suppresses texture, retains large-scale geometric structure
- **Edge** — Gaussian blur + Canny + morphological closing; responds to local gradient magnitude, largely insensitive to global lighting and pose

Running a metric across all three filters provides three partially independent tests of the same hypothesis.

## Reproducing the paper results

The full measurement CSV from the paper is available at https://github.com/scottcundill/scott-framework.

To reproduce the statistical tables:

```bash
# install dependencies
pip install -r requirements.txt pandas scipy

# run analysis on your own copy of the datasets
python analyse_faces.py --folder path/to/ffhq_1k --output real_faces.csv
python analyse_faces.py --folder path/to/stylegan_1k --output ai_faces.csv
```

Datasets used in the paper:
- Real faces: [FFHQ](https://github.com/NVlabs/ffhq-dataset) (Karras et al., 2019)
- GAN faces: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (xhlulu, 2020)

## Citation

If you use this code, please cite:

```
Cundill, S. (2026). Bilateral Symmetry as a Filter-Robust Discriminator
Between Real and GAN-Generated Face Images. [preprint — citation to update on publication]
```

## License

MIT License. See LICENSE file.

## Author

Scott Cundill — Independent Researcher, Okinawa, Japan
