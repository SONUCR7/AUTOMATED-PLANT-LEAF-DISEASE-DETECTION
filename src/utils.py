"""
Diagnostic Visualization and Pipeline Utilities
================================================
Generates high-resolution 5-stage pipeline figures, CSV lab reports,
and color-coded diagnostic summaries for reports and presentations.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_5stage_figure(
    original_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    disease_overlay: np.ndarray,
    classification_result: dict,
    severity_metrics: dict,
    save_path: str = None
) -> plt.Figure:
    """
    Creates the signature 5-stage visual pipeline figure for reports and viva:
    Stage 1: Original Leaf Image
    Stage 2: Gaussian Denoised & CLAHE Enhanced
    Stage 3: Segmented Leaf Mask (Otsu + Morphology)
    Stage 4: Isolated Lesions with % Severity Calculation
    Stage 5: MobileNetV2 Disease Prediction & Diagnostic Summary
    """
    fig = plt.figure(figsize=(20, 4.5), dpi=150)
    
    # 1. Original
    ax1 = plt.subplot(1, 5, 1)
    ax1.imshow(original_rgb)
    ax1.set_title("Stage 1: Original Leaf\n(Input Acquisition)", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")
    
    # 2. Preprocessed (CLAHE)
    ax2 = plt.subplot(1, 5, 2)
    ax2.imshow(enhanced_rgb)
    ax2.set_title("Stage 2: Gaussian + CLAHE\n(Contrast Enhanced)", fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")
    
    # 3. Leaf Mask
    ax3 = plt.subplot(1, 5, 3)
    ax3.imshow(leaf_mask, cmap="gray")
    ax3.set_title("Stage 3: Segmented Leaf Mask\n(Otsu + Morphological Clean)", fontsize=11, fontweight="bold", pad=8)
    ax3.axis("off")
    
    # 4. Disease Spots & Severity
    ax4 = plt.subplot(1, 5, 4)
    ax4.imshow(disease_overlay)
    sev_title = f"Stage 4: Disease Lesions\nSeverity: {severity_metrics['severity_percentage']}% ({severity_metrics['grade']})"
    ax4.set_title(sev_title, fontsize=11, fontweight="bold", pad=8, color="#B22222")
    ax4.axis("off")
    
    # 5. Diagnostic Summary Card
    ax5 = plt.subplot(1, 5, 5)
    ax5.axis("off")
    
    pred_name = classification_result.get("formatted_name", "Unknown")
    confidence = classification_result.get("confidence", 0.0)
    urgency = severity_metrics.get("urgency", "Monitor")
    stage_name = severity_metrics.get("stage", "Unknown")
    
    summary_text = (
        f"Stage 5: MobileNetV2 Diagnosis\n"
        f"------------------------------------\n"
        f"• Disease: {pred_name}\n"
        f"• Confidence: {confidence}%\n"
        f"• Severity: {severity_metrics['severity_percentage']}%\n"
        f"• Stage: {stage_name}\n"
        f"• Urgency: {urgency}\n\n"
        f"CVIP Modules Used:\n"
        f"M1-2: Color/CLAHE/Gauss\n"
        f"M3: Otsu + K-Means Lesion Mask\n"
        f"M4-5: Transfer Learning CNN"
    )
    
    ax5.text(
        0.05, 0.5, summary_text,
        fontsize=10, family="monospace", verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#F0F8FF", edgecolor="#4682B4", linewidth=1.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        
    return fig


def cv2_to_pil(image_rgb: np.ndarray) -> Image.Image:
    """Converts a numpy RGB image array into a PIL Image."""
    return Image.fromarray(image_rgb.astype(np.uint8))
