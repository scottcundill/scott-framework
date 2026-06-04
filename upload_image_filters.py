# image_filters.py
# =============================================================================
# SCOTT — Structural and Coherent Order in Topological Transforms
# https://github.com/[USERNAME]/scott-framework
#
# Image preprocessing filters.
# Each filter produces a different representation of the input image,
# suppressing a different class of information. Running a metric across
# all three filters provides partially independent tests of the same
# hypothesis.
# =============================================================================

import cv2
import numpy as np


def raw_filter(img_gray):
    """
    Return the greyscale image unchanged.

    This is the control representation, retaining all spatial information
    including lighting gradients, texture, and fine detail.

    Parameters
    ----------
    img_gray : np.ndarray
        Greyscale image, uint8, 512x512.

    Returns
    -------
    np.ndarray
        3-channel (H, W, 3) uint8 image (greyscale duplicated to RGB).
    """
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


def topographic_filter(img_gray, step_interval=15, blur_strength=9,
                       clahe_strength=4):
    """
    Topographic isoline representation.

    Applies CLAHE contrast normalisation, then bilateral smoothing, then
    extracts intensity isolines at uniform grey-level intervals. The result
    is a topographic map of the intensity surface: texture and fine detail
    are suppressed, large-scale geometric structure is retained.

    Parameters used in the paper (Cundill 2026):
        step_interval = 15
        blur_strength = 9
        clahe_strength = 4

    Parameters
    ----------
    img_gray : np.ndarray
        Greyscale image, uint8, 512x512.
    step_interval : int
        Grey-level spacing between isoline levels. Default 15 gives 16
        contour levels across the 0-255 range.
    blur_strength : int
        Bilateral filter diameter. Controls smoothing before contouring.
    clahe_strength : float
        CLAHE clip limit. 0 disables CLAHE. Default 4.

    Returns
    -------
    np.ndarray
        3-channel (H, W, 3) uint8 image — white isolines on black canvas.
    """
    if clahe_strength > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_strength),
                                tileGridSize=(8, 8))
        processing_img = clahe.apply(img_gray)
    else:
        processing_img = img_gray.copy()

    h, w = processing_img.shape
    smoothed = cv2.bilateralFilter(processing_img, blur_strength, 75, 75)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for level in range(step_interval, 255, step_interval):
        _, thresh = cv2.threshold(smoothed, level, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1)

    return canvas


def edge_filter(img_gray, low_gate=30, high_gate=120):
    """
    Canny edge representation.

    Applies Gaussian smoothing, Canny edge detection, and a morphological
    closing operation to produce a binary edge map. Responds primarily to
    local gradient magnitude; largely insensitive to global brightness
    gradients and low-frequency composition.

    Thresholds low=30, high=120 are set to retain fine facial detail
    including shallow surface gradients.

    Parameters
    ----------
    img_gray : np.ndarray
        Greyscale image, uint8, 512x512.
    low_gate : int
        Canny lower threshold. Default 30.
    high_gate : int
        Canny upper threshold. Default 120.

    Returns
    -------
    np.ndarray
        3-channel (H, W, 3) uint8 image — white edges on black canvas.
    """
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_gate, high_gate)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
