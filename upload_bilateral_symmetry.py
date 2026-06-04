# bilateral_symmetry.py
# =============================================================================
# SCOTT — Structural and Coherent Order in Topological Transforms
# https://github.com/scottcundill/scott-framework
#
# Bilateral symmetry metric.
# Measures global left-right mirror correspondence in a greyscale image.
# Used in: Cundill, S. (2026). "Bilateral Symmetry as a Filter-Robust
# Discriminator Between Real and GAN-Generated Face Images."
# =============================================================================

import numpy as np
import cv2

NORMALIZED_SIZE = 512


def normalize_image(img_gray, size=NORMALIZED_SIZE):
    """
    Resize any greyscale image to size x size pixels.

    Uses INTER_AREA when shrinking (best quality), INTER_CUBIC when
    upscaling. All metrics in SCOTT operate on normalized images so that
    results are comparable across images of different original resolutions.

    Parameters
    ----------
    img_gray : np.ndarray
        Greyscale image (2D) or BGR image (3D, will be converted).
    size : int
        Target side length in pixels. Default 512.

    Returns
    -------
    np.ndarray
        Greyscale image of shape (size, size), dtype uint8.
    """
    if img_gray is None:
        raise ValueError("normalize_image received None")
    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_CUBIC
    return cv2.resize(img_gray, (size, size), interpolation=interp)


def bilateral_symmetry(img_gray):
    """
    Compute global bilateral (left-right mirror) symmetry of a greyscale image.

    The image is normalised to [0, 1], flipped horizontally, and the mean
    absolute difference (MAD) between original and mirror is computed
    pixel-wise. Bilateral symmetry is defined as:

        BS = 1 - MAD(I, F)

    where I is the normalised image, F is its horizontal flip, and:

        MAD = (1 / H*W) * sum( |I(h,w) - F(h,w)| )

    Result ranges from 0.0 (no mirror correspondence) to 1.0 (perfect
    bilateral symmetry). This is a direct pixel-wise deviation from mirror
    symmetry, not a normalised cross-correlation.

    A secondary measure, left-right brightness imbalance, is also returned.
    This is the absolute difference in mean pixel intensity between the left
    and right halves of the image, capturing gross brightness asymmetry
    independently of fine spatial structure.

    Parameters
    ----------
    img_gray : np.ndarray
        Greyscale image, dtype uint8. Should be normalized to 512x512
        before calling (use normalize_image).

    Returns
    -------
    symmetry : float
        Global bilateral symmetry score, 0.0 to 1.0.
    imbalance : float
        Left-right brightness imbalance, 0.0 to 1.0.
    """
    g = img_gray.astype(np.float32) / 255.0
    flipped = np.fliplr(g)
    mad = float(np.mean(np.abs(g - flipped)))
    symmetry = float(max(0.0, 1.0 - mad))

    w = g.shape[1]
    left = g[:, :w // 2]
    right = np.fliplr(g[:, w - w // 2:])
    imbalance = float(abs(np.mean(left) - np.mean(right)))

    return round(symmetry, 4), round(imbalance, 4)
