"""
Module 3 (Twist): Quantitative Severity Estimation & Visual Heatmapping
========================================================================
Calculates the exact ratio of diseased pixel area to total leaf area,
assigns agricultural severity grades, and produces visual overlay heatmaps.
"""

import cv2
import numpy as np


def compute_severity_metrics(leaf_mask: np.ndarray, disease_mask: np.ndarray) -> dict:
    """
    Computes quantitative disease severity metrics:
    - Total Leaf Pixels
    - Diseased Pixels
    - Healthy Pixels
    - Disease Severity Percentage = (Diseased Pixels / Total Leaf Pixels) * 100
    - Severity Grade and Clinical Stage
    """
    total_leaf_pixels = int(np.count_nonzero(leaf_mask > 0))
    diseased_pixels = int(np.count_nonzero(disease_mask > 0))
    
    if total_leaf_pixels == 0:
        severity_percentage = 0.0
        healthy_pixels = 0
    else:
        # Constrain diseased pixels not to exceed total leaf pixels
        diseased_pixels = min(diseased_pixels, total_leaf_pixels)
        severity_percentage = (diseased_pixels / float(total_leaf_pixels)) * 100.0
        healthy_pixels = total_leaf_pixels - diseased_pixels
        
    severity_percentage = round(severity_percentage, 2)
    
    # Standard Agricultural Severity Grading (FAO / Horsfall-Barratt scale)
    if severity_percentage < 0.5:
        stage = "Healthy / Trace"
        grade = "Grade 0"
        urgency = "Normal"
        color_code = (46, 204, 113) # Emerald Green
    elif severity_percentage <= 10.0:
        stage = "Mild / Early Infection"
        grade = "Grade 1"
        urgency = "Monitor & Preventive Action"
        color_code = (241, 196, 15) # Sunflower Yellow
    elif severity_percentage <= 25.0:
        stage = "Moderate Infection"
        grade = "Grade 2"
        urgency = "Immediate Treatment Required"
        color_code = (230, 126, 34) # Carrot Orange
    elif severity_percentage <= 50.0:
        stage = "Severe Outbreak"
        grade = "Grade 3"
        urgency = "High Threat - Intensive Treatment"
        color_code = (231, 76, 60)  # Alizarin Red
    else:
        stage = "Critical / Destructive"
        grade = "Grade 4"
        urgency = "Crop Destruction Risk - Quarantine Plant"
        color_code = (142, 68, 173) # Amethyst / Dark Purple

    return {
        "total_leaf_pixels": total_leaf_pixels,
        "diseased_pixels": diseased_pixels,
        "healthy_pixels": healthy_pixels,
        "severity_percentage": severity_percentage,
        "grade": grade,
        "stage": stage,
        "urgency": urgency,
        "color_code": color_code
    }


def create_severity_overlay(
    image_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    disease_mask: np.ndarray,
    severity_metrics: dict,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Creates an enhanced color-coded diagnostic visual:
    - Base leaf image
    - Healthy areas subtly masked
    - Diseased spots highlighted in vivid Crimson/Amber with contour borders
    - Diagnostic summary badge with % severity, grade, and urgency.
    """
    overlay = image_rgb.copy()
    
    # Highlight diseased pixels with crimson red (220, 20, 60)
    red_highlight = np.array([220, 20, 60], dtype=np.uint8)
    lesion_pixels = disease_mask > 0
    
    # Blend semi-transparent highlight
    overlay[lesion_pixels] = (
        (1.0 - alpha) * overlay[lesion_pixels] + alpha * red_highlight
    ).astype(np.uint8)
    
    # Draw sharp borders around disease lesions
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1) # Bright yellow contour
    
    # Draw leaf perimeter in neon green
    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_contours, -1, (0, 255, 0), 2)
    
    # Add annotated badge on top of image
    h, w, _ = overlay.shape
    badge_height = max(42, int(h * 0.14))
    badge_bg = np.zeros((badge_height, w, 3), dtype=np.uint8)
    
    # Blend black background bar
    overlay[0:badge_height, 0:w] = cv2.addWeighted(
        overlay[0:badge_height, 0:w], 0.35, badge_bg, 0.65, 0
    )
    
    # Format text
    text_sev = f"Severity: {severity_metrics['severity_percentage']}% ({severity_metrics['grade']})"
    text_stage = f"{severity_metrics['stage']} | {severity_metrics['urgency']}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(0.48, w / 600.0)
    cv2.putText(overlay, text_sev, (12, int(badge_height * 0.45)), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, text_stage, (12, int(badge_height * 0.85)), font, font_scale * 0.85, (0, 255, 255), 1, cv2.LINE_AA)
    
    return overlay


def generate_disease_heatmap(image_rgb: np.ndarray, disease_mask: np.ndarray) -> np.ndarray:
    """
    Generates a continuous thermal density heatmap of disease lesion concentration.
    Uses Gaussian smoothing on the binary disease mask followed by COLORMAP_JET.
    """
    # Smooth binary mask to generate density gradient
    density = cv2.GaussianBlur(disease_mask.astype(np.float32), (31, 31), 10.0)
    density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply JET colormap (blue = healthy/none, red = intense lesion concentration)
    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    blended = cv2.addWeighted(image_rgb, 0.6, heatmap_rgb, 0.4, 0)
    return blended
