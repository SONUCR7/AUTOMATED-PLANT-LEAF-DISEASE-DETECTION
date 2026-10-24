"""
Module 3: Image Segmentation & Lesion Extraction
==================================================
Implements foreground leaf segmentation (Otsu + Morphological Filtering)
and multi-method disease spot isolation (HSV Thresholding & K-Means Color Clustering).
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def segment_leaf_otsu(image_rgb: np.ndarray) -> np.ndarray:
    """
    Segments leaf from background using Otsu's automatic global thresholding
    applied to grayscale and saturation channels.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up small holes using morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    return mask_opened


def segment_leaf_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """
    Segments the entire leaf body from neutral/dark/gray background using
    broad HSV range covering green, yellow, brown, and pale leaf tissue.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Broad range capturing any plant tissue (green, yellow, brown, dark spots)
    lower_plant = np.array([10, 25, 25], dtype=np.uint8)
    upper_plant = np.array([100, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_plant, upper_plant)
    
    # Combine with intensity threshold to catch darker lesions inside leaf
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, dark_thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    combined = cv2.bitwise_and(mask_hsv, dark_thresh)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Keep largest connected contour (the leaf)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(combined)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
        # Fill internal holes in the contour
        cv2.drawContours(final_mask, [largest_cnt], -1, 255, thickness=-1)
    else:
        final_mask = combined

    return final_mask


def segment_disease_hsv(image_rgb: np.ndarray, leaf_mask: np.ndarray) -> tuple:
    """
    Segments diseased tissue (chlorosis/yellowing and necrosis/browning/black spots)
    from healthy green chlorophyll within the segmented leaf mask.
    
    Returns:
        disease_mask (np.ndarray): Binary mask of disease lesions.
        healthy_mask (np.ndarray): Binary mask of healthy green leaf tissue.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Healthy Green Chlorophyll range in OpenCV HSV (H: 0-180, S: 0-255, V: 0-255)
    lower_healthy = np.array([35, 38, 38], dtype=np.uint8)
    upper_healthy = np.array([86, 255, 255], dtype=np.uint8)
    healthy_green_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
    
    # Restrict healthy mask to leaf area
    healthy_mask = cv2.bitwise_and(healthy_green_mask, healthy_green_mask, mask=leaf_mask)
    
    # Diseased tissue is leaf area that is NOT healthy green
    raw_disease = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(healthy_mask))
    
    # Filter out leaf veins and natural minor specular highlights:
    # 1. Necrotic / Brown / Black spots (low H or low V)
    lower_brown = np.array([0, 30, 20], dtype=np.uint8)
    upper_brown = np.array([32, 255, 240], dtype=np.uint8)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # 2. Chlorotic / Yellow halo spots (H between 20 and 34)
    lower_yellow = np.array([18, 40, 60], dtype=np.uint8)
    upper_yellow = np.array([34, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 3. Dark necrotic lesions (Value < 55 inside leaf)
    dark_mask = np.zeros_like(leaf_mask)
    dark_mask[(hsv[:, :, 2] < 55) & (leaf_mask > 0)] = 255
    
    disease_candidates = cv2.bitwise_or(brown_mask, yellow_mask)
    disease_candidates = cv2.bitwise_or(disease_candidates, dark_mask)
    disease_mask = cv2.bitwise_and(disease_candidates, raw_disease)
    
    # Morphological noise removal (remove speckle noise < 3x3)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_clean)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_clean)
    
    return disease_mask, healthy_mask


def segment_disease_kmeans(image_rgb: np.ndarray, leaf_mask: np.ndarray, k: int = 3) -> tuple:
    """
    Module 3 Unsupervised Clustering:
    Segments leaf into K clusters using K-Means in L*a*b* color space:
    - Cluster 1: Healthy green chlorophyll tissue
    - Cluster 2: Necrotic / Chlorotic disease lesions
    - Cluster 3: Transitional / Petiole / Stem regions
    
    Returns:
        disease_mask (np.ndarray): Binary mask of disease lesions.
        cluster_vis (np.ndarray): Color-coded visualization of the K clusters.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    # Extract only leaf pixels
    leaf_indices = np.where(leaf_mask > 0)
    if len(leaf_indices[0]) == 0:
        return np.zeros_like(leaf_mask), np.zeros_like(image_rgb)
    
    pixel_features = lab[leaf_indices].astype(np.float32)
    
    # Run K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixel_features)
    centers = kmeans.cluster_centers_
    
    # In LAB:
    # a* channel represents Green (negative) to Red/Magenta (positive).
    # b* channel represents Blue (negative) to Yellow (positive).
    # Diseased spots (brown/yellow) have higher a* and higher b* (less negative green).
    # Healthy green tissue has most negative a* values.
    a_values = centers[:, 1]
    diseased_cluster_idx = int(np.argmax(a_values))
    
    disease_mask = np.zeros_like(leaf_mask)
    disease_pixels_x = leaf_indices[0][labels == diseased_cluster_idx]
    disease_pixels_y = leaf_indices[1][labels == diseased_cluster_idx]
    disease_mask[disease_pixels_x, disease_pixels_y] = 255
    
    # Color-coded cluster visualization
    palette = [
        [34, 139, 34],    # Forest Green (Healthy)
        [220, 20, 60],    # Crimson Red (Disease)
        [218, 165, 32],   # Goldenrod (Halo / Boundary)
        [70, 130, 180]    # Steel Blue (Extra)
    ]
    cluster_vis = np.zeros_like(image_rgb)
    for c_idx in range(k):
        c_pixels_x = leaf_indices[0][labels == c_idx]
        c_pixels_y = leaf_indices[1][labels == c_idx]
        color = palette[c_idx % len(palette)]
        cluster_vis[c_pixels_x, c_pixels_y] = color
        
    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    
    return disease_mask, cluster_vis


def extract_lesion_contours(disease_mask: np.ndarray, min_area: int = 15) -> list:
    """
    Finds and filters individual disease lesion contours.
    Returns bounding boxes and area statistics for each lesion spot.
    """
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            lesions.append({
                "contour": cnt,
                "area": float(area),
                "bbox": (x, y, w, h)
            })
            
    # Sort largest lesions first
    lesions.sort(key=lambda item: item["area"], reverse=True)
    return lesions
