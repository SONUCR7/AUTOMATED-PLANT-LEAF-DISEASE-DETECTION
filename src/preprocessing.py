"""
Module 1 & 2: Image Acquisition, Filtering, Enhancement, and Transformations
=============================================================================
This module provides standard and advanced Computer Vision algorithms for leaf
image preprocessing, including color-space conversions, noise removal filters,
and Contrast-Limited Adaptive Histogram Equalization (CLAHE).
"""

import cv2
import numpy as np


def load_image(image_path: str, target_size: tuple = None) -> np.ndarray:
    """
    Loads an image from disk in BGR format and optionally resizes it.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Converts BGR image (OpenCV default) to RGB format."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image to BGR format."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image to HSV (Hue, Saturation, Value) color space."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image to CIE-L*a*b* color space."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image to single-channel Grayscale."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def apply_gaussian_blur(image_rgb: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 1.0) -> np.ndarray:
    """
    Applies Gaussian Filtering for high-frequency noise removal.
    G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2)/(2*sigma^2))
    """
    return cv2.GaussianBlur(image_rgb, kernel_size, sigmaX=sigma, sigmaY=sigma)


def apply_median_blur(image_rgb: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Applies Median Filtering to eliminate salt-and-pepper / impulse noise.
    """
    return cv2.medianBlur(image_rgb, ksize)


def apply_bilateral_filter(image_rgb: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Applies Bilateral Filtering for edge-preserving denoising.
    Combines spatial closeness and photometric similarity.
    """
    return cv2.bilateralFilter(image_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_clahe(image_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the
    Luminance (L*) channel in LAB color space to boost lesion contrast without
    introducing color distortion or amplifying background noise.
    """
    lab = rgb_to_lab(image_rgb)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_channel = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def apply_global_histogram_equalization(image_rgb: np.ndarray) -> np.ndarray:
    """
    Standard Global Histogram Equalization on Y channel of YCrCb space.
    Included for academic comparative analysis against CLAHE.
    """
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    eq_ycrcb = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(eq_ycrcb, cv2.COLOR_YCrCb2RGB)


def compute_canny_edges(image_rgb: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Canny Edge Detection:
    1. Gaussian smoothing
    2. Gradient magnitude & direction (Sobel)
    3. Non-maximum suppression
    4. Hysteresis thresholding
    """
    gray = rgb_to_gray(image_rgb)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def compute_sobel_gradients(image_rgb: np.ndarray) -> dict:
    """
    Computes first-order spatial derivatives using Sobel operator.
    """
    gray = rgb_to_gray(image_rgb)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return {
        "grad_x": grad_x,
        "grad_y": grad_y,
        "magnitude": magnitude_norm
    }


def preprocess_pipeline(image_rgb: np.ndarray, clip_limit: float = 2.5) -> dict:
    """
    Runs the complete Module 1 & 2 preprocessing pipeline:
    1. Gaussian Smoothing (noise reduction)
    2. LAB-based CLAHE (contrast enhancement for disease spots)
    3. Grayscale conversion
    4. Canny edge mapping
    
    Returns a dictionary of all intermediate processed representations.
    """
    smoothed = apply_gaussian_blur(image_rgb, kernel_size=(5, 5), sigma=1.0)
    enhanced = apply_clahe(smoothed, clip_limit=clip_limit, tile_grid_size=(8, 8))
    gray = rgb_to_gray(enhanced)
    edges = compute_canny_edges(enhanced)
    
    return {
        "original_rgb": image_rgb,
        "smoothed_rgb": smoothed,
        "enhanced_rgb": enhanced,
        "gray": gray,
        "edges": edges
    }
