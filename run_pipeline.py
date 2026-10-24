"""
Automated Plant Leaf Disease Detection & Severity Estimation
Complete End-to-End Diagnostic Pipeline (CLI & Module Runner)
=============================================================
Usage:
    python run_pipeline.py --image path/to/leaf.jpg --output results/
    python run_pipeline.py --all-samples
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import preprocess_pipeline, load_image, bgr_to_rgb
from src.segmentation import segment_leaf_hsv, segment_disease_hsv, segment_disease_kmeans
from src.severity import compute_severity_metrics, create_severity_overlay, generate_disease_heatmap
from src.classifier import PlantDiseaseClassifier
from src.remedies import get_disease_remedy
from src.utils import generate_5stage_figure


def run_single_image_pipeline(
    image_path: str,
    output_dir: str = "results",
    segmentation_method: str = "hsv",
    save_visuals: bool = True
) -> dict:
    """
    Executes the full 5-stage diagnostic pipeline on an individual leaf image.
    """
    print(f"\n{'='*70}")
    print(f"[PIPELINE] Processing: {os.path.basename(image_path)}")
    print(f"{'='*70}")

    # Ensure safe console printing on Windows
    sys.stdout.reconfigure(encoding='utf-8')

    # Stage 1: Acquisition & Load
    image_bgr = load_image(image_path, target_size=(512, 512))
    image_rgb = bgr_to_rgb(image_bgr)
    print("[+] Stage 1: Image acquired & normalized to 512x512 RGB.")

    # Stage 2: Filtering & CLAHE Enhancement (Modules 1 & 2)
    prep_res = preprocess_pipeline(image_rgb, clip_limit=2.5)
    enhanced_rgb = prep_res["enhanced_rgb"]
    print("[+] Stage 2: Gaussian smoothing applied & CLAHE contrast enhancement completed.")

    # Stage 3: Leaf Foreground Segmentation (Module 3)
    leaf_mask = segment_leaf_hsv(enhanced_rgb)
    print("[+] Stage 3: Foreground leaf isolated using Otsu & Morphological Closing.")

    # Stage 4: Disease Lesion Segmentation & Severity Calculation (Module 3 Twist)
    if segmentation_method.lower() == "kmeans":
        disease_mask, _ = segment_disease_kmeans(enhanced_rgb, leaf_mask, k=3)
        method_used = "K-Means L*a*b* Clustering"
    else:
        disease_mask, _ = segment_disease_hsv(enhanced_rgb, leaf_mask)
        method_used = "HSV Color Space Thresholding"

    severity_metrics = compute_severity_metrics(leaf_mask, disease_mask)
    disease_overlay = create_severity_overlay(image_rgb, leaf_mask, disease_mask, severity_metrics)
    disease_heatmap = generate_disease_heatmap(image_rgb, disease_mask)

    print(f"[+] Stage 4: Lesions isolated via {method_used}.")
    print(f"   -> Total Leaf Pixels : {severity_metrics['total_leaf_pixels']:,}")
    print(f"   -> Diseased Pixels   : {severity_metrics['diseased_pixels']:,}")
    print(f"   -> Severity Score    : {severity_metrics['severity_percentage']}%")
    print(f"   -> Infection Grade   : {severity_metrics['grade']} ({severity_metrics['stage']})")

    # Stage 5: MobileNetV2 Classification & Treatment Advisory (Modules 4 & 5)
    classifier = PlantDiseaseClassifier()
    classification_result = classifier.predict(image_rgb, severity_metrics)
    remedy_result = get_disease_remedy(classification_result["predicted_class"], severity_metrics)

    print("[+] Stage 5: Deep Learning Classification complete.")
    print(f"   -> Predicted Class   : {classification_result['formatted_name']}")
    print(f"   -> Model Confidence  : {classification_result['confidence']}%")
    print(f"   -> Action Required   : {remedy_result['action_plan']}")

    # Save visual outputs
    if save_visuals:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 5-Stage signature figure
        fig_path = os.path.join(output_dir, f"{base_name}_5stage_pipeline.png")
        generate_5stage_figure(
            image_rgb, enhanced_rgb, leaf_mask, disease_overlay,
            classification_result, severity_metrics, save_path=fig_path
        )
        
        # Overlay & Heatmap standalone
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_severity_overlay.png"), cv2.cvtColor(disease_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_heatmap.png"), cv2.cvtColor(disease_heatmap, cv2.COLOR_RGB2BGR))
        
        # Save JSON diagnostic report
        report_data = {
            "image_filename": os.path.basename(image_path),
            "severity_analysis": severity_metrics,
            "classification": classification_result,
            "agronomic_advisory": remedy_result
        }
        with open(os.path.join(output_dir, f"{base_name}_report.json"), "w") as f:
            json.dump(report_data, f, indent=4)

        print(f"[SUCCESS] Visual report saved to: {fig_path}")

    return {
        "image_path": image_path,
        "severity": severity_metrics,
        "classification": classification_result,
        "remedy": remedy_result
    }


def main():
    parser = argparse.ArgumentParser(description="Plant Leaf Disease Detection & Severity Estimation")
    parser.add_argument("--image", type=str, help="Path to input plant leaf image")
    parser.add_argument("--output", type=str, default="results", help="Directory to save visual diagnostic reports")
    parser.add_argument("--method", type=str, default="hsv", choices=["hsv", "kmeans"], help="Lesion segmentation method")
    parser.add_argument("--all-samples", action="store_true", help="Process all images in dataset/samples directory")

    args = parser.parse_args()

    if args.all_samples or args.image is None:
        samples_dir = os.path.join("dataset", "samples")
        if not os.path.exists(samples_dir):
            print(f"Error: {samples_dir} not found.")
            return

        sample_files = [
            os.path.join(samples_dir, f)
            for f in os.listdir(samples_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not sample_files:
            print(f"No sample images found in {samples_dir}. Run sample generator first.")
            return

        print(f"Found {len(sample_files)} sample leaves. Running batch diagnostic pipeline...")
        for img_path in sample_files:
            run_single_image_pipeline(img_path, output_dir=args.output, segmentation_method=args.method)
    else:
        run_single_image_pipeline(args.image, output_dir=args.output, segmentation_method=args.method)


if __name__ == "__main__":
    main()
