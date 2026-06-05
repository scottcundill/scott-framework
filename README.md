# SCOTT — Structural and Coherent Order in Topological Transforms
An open-source Python framework for measuring geometric and topological order in 2D images.

## About
SCOTT provides image analysis metrics that measure structural properties of 2D images — symmetry, topological complexity, edge structure, and multi-scale organisation. The framework is domain-agnostic: it operates on any greyscale image.

## Papers

**Paper 1** (root directory)
> Cundill, S. (2026). *Bilateral Symmetry as a Filter-Robust Discriminator Between Real and GAN-Generated Face Images.* [preprint]

**Paper 2** (`paper2/` folder)
> Cundill, S. (2026). *Spatial Structure Beyond the Power Spectrum: Amplitude-Adjusted Surrogate Testing in Natural and Synthetic Images.* [preprint]

Scripts and data for Paper 2 are in the `paper2/` folder.

---

## Paper 1 — Repository contents

| File | Description |
|------|-------------|
| `bilateral_symmetry.py` | Global bilateral symmetry metric (BS) and image normalisation |
| `image_filters.py` | Three preprocessing filters: raw, topographic isoline, edge |
| `analyse_faces.py` | Example: compute bilateral symmetry on a folder of images |
| `paper1_reproduce_results.py` | Reproduce all tables and figures from the paper |
| `paper1_measurements_raw.csv` | Main analysis data — raw filter (n=2,000) |
| `paper1_measurements_edge.csv` | Main analysis data — edge filter (n=2,000) |
| `paper1_measurements_topographic.csv` | Main analysis data — topographic filter (n=2,000) |
| `paper1_validation_results.csv` | Independent validation sample (n=2,000, seed 42) |
| `paper1_diffusion_results.csv` | Diffusion model analysis — SFHQ-T2I (n=1,000, seed 42) |
| `paper1.tex` | LaTeX source for the paper |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
# Clone the repository
git clone https://github.com/scottcundill/scott-framework.git
cd scott-framework

# Install dependencies
pip install -r requirements.txt
pip install pandas scipy matplotlib

# Reproduce all paper results (tables printed to console, figures saved as PNG)
python paper1_reproduce_results.py
```

## Using the bilateral symmetry metric

```python
from bilateral_symmetry import normalize_image, bilateral_symmetry
import cv2

img = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)
img_norm = normalize_image(img)           # resize to 512x512
bs, imbalance = bilateral_symmetry(img_norm)
print(f"Bilateral symmetry: {bs:.4f}")   # 0.0 = no symmetry, 1.0 = perfect
```

## The metric

Bilateral symmetry is defined as:

```
BS = 1 - MAD(I, F)

where I  = normalised greyscale image (512x512, float [0,1])
      F  = horizontal flip of I
      MAD = (1/HW) * sum |I(h,w) - F(h,w)|
```

Result ranges from 0 (no mirror correspondence) to 1.0 (perfect bilateral symmetry).

## The three filters

Each filter suppresses a different class of image information:

- **Raw** — greyscale, no processing. Retains all spatial information.
- **Topographic** — CLAHE (clip limit 4) + bilateral smoothing + intensity isolines at 15-level intervals. Suppresses texture, retains geometric skeleton.
- **Edge** — Gaussian blur + Canny (30/120) + morphological closing. Responds to local gradient magnitude, largely insensitive to global lighting and pose.

Running a metric across all three filters provides three partially independent tests of the same hypothesis.

## Datasets

- **Real faces:** [FFHQ](https://github.com/NVlabs/ffhq-dataset) (Karras et al., 2019)
- **GAN faces:** [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (xhlulu, 2020)
- **Diffusion faces:** [SFHQ-T2I](https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models) (Beniaguev, 2024)

## Citation

```
Cundill, S. (2026). Bilateral Symmetry as a Filter-Robust Discriminator
Between Real and GAN-Generated Face Images.
[citation to update on publication]
```

## License

MIT License. See LICENSE file.

## Author

Scott Cundill — Independent Researcher, Okinawa, Japan
