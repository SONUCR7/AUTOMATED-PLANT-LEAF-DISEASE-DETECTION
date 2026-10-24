"""
Automated Plant Leaf Disease Detection & Severity Estimation
Interactive Streamlit Dashboard
=============================================================
A comprehensive Computer Vision & Deep Learning web application providing:
- 5-Stage Visual Diagnostic Pipeline
- Quantitative Disease Severity Estimation (% Calculation & FAO Grading)
- MobileNetV2 Deep Learning Crop Disease Classification
- Agronomic Remedy & Treatment Recommendation Engine
- CVIP Algorithm Inspector (Color Spaces, Histograms, Edge Gradients)
- Batch Diagnostic Evaluation with CSV Export
- Viva Voce & Theory Cheat Sheet for Academic Evaluation
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src modules are reachable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    load_image, bgr_to_rgb, apply_gaussian_blur, apply_clahe,
    rgb_to_hsv, rgb_to_lab, rgb_to_gray, compute_canny_edges, compute_sobel_gradients
)
from src.segmentation import (
    segment_leaf_hsv, segment_leaf_otsu, segment_disease_hsv,
    segment_disease_kmeans, extract_lesion_contours
)
from src.severity import (
    compute_severity_metrics, create_severity_overlay, generate_disease_heatmap
)
from src.classifier import PlantDiseaseClassifier, CLASS_NAMES
from src.remedies import get_disease_remedy
from src.utils import generate_5stage_figure

# Set Page Config
st.set_page_config(
    page_title="Plant Leaf Disease Detection & Severity Estimator",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* 1. Hide Streamlit Branding & Deploy Button */
    .stAppDeployButton, 
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    #MainMenu, 
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }

    /* 2. Global Typography & Spacing */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* 3. Hero Header Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.04) 100%);
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-radius: 16px;
        padding: 22px 26px 18px 26px;
        margin-bottom: 22px;
        backdrop-filter: blur(12px);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 12px;
        font-weight: 400;
    }
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 6px;
    }
    .badge-pill {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* 4. Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #10b981 !important;
        color: #34d399 !important;
    }

    /* 5. Stage Visual Columns */
    div[data-testid="column"] {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="column"]:hover {
        border-color: rgba(16, 185, 129, 0.35);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    }

    /* 6. Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* 7. Action Buttons */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 8px 20px !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35) !important;
    }
    .stDownloadButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 8px 20px !important;
    }
    .stDownloadButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    return PlantDiseaseClassifier()


def colorize_channel(channel_2d: np.ndarray, colormap_name: str = "viridis") -> np.ndarray:
    """Normalizes a 2D single-channel matrix and applies a Matplotlib colormap returning RGB uint8 array."""
    norm = cv2.normalize(channel_2d.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    cmap = plt.get_cmap(colormap_name)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def main():
    # Hero Header Banner
    st.markdown("""
    <div class="hero-header">
        <div class="main-title">🌿 Automated Plant Leaf Disease Detection & Severity Estimation</div>
        <div class="hero-subtitle">
            Diagnostic Computer Vision (Modules 1–3) & Deep Learning Transfer Pipeline (Modules 4–5) with Agronomic Advisory
        </div>
        <div class="badge-container">
            <span class="badge-pill">⚙️ M1-2: Gaussian & LAB-CLAHE</span>
            <span class="badge-pill">✂️ M3: Otsu & Morphological Masking</span>
            <span class="badge-pill">🎯 M3 Twist: Quantitative Severity % Calculation</span>
            <span class="badge-pill">🧠 M4-5: MobileNetV2 Classification</span>
            <span class="badge-pill">💊 Agronomic Cure Advisory</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    classifier = get_classifier()

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration & Input")
    
    input_source = st.sidebar.radio(
        "Select Leaf Image Source:",
        ["Curated Benchmark Samples", "Upload Custom Image"]
    )

    samples_dir = os.path.join("dataset", "samples")
    sample_files = []
    if os.path.exists(samples_dir):
        sample_files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    image_rgb = None
    image_name = "Leaf Image"

    if input_source == "Curated Benchmark Samples":
        if sample_files:
            sample_labels = {
                f: f.replace("_", " ").replace(".jpg", "").replace(".png", "").title()
                for f in sample_files
            }
            selected_file = st.sidebar.selectbox(
                "Choose Benchmark Leaf:",
                sample_files,
                format_func=lambda x: sample_labels.get(x, x)
            )
            image_path = os.path.join(samples_dir, selected_file)
            image_bgr = load_image(image_path, target_size=(512, 512))
            image_rgb = bgr_to_rgb(image_bgr)
            image_name = selected_file
        else:
            st.sidebar.warning("No sample files found. Please generate samples.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_bgr = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA)
            image_rgb = bgr_to_rgb(image_bgr)
            image_name = uploaded_file.name

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Algorithm Parameters")
    seg_method = st.sidebar.selectbox(
        "Lesion Segmentation Method:",
        ["HSV Color Space Thresholding", "K-Means L*a*b* Clustering (K=3)"]
    )
    clahe_clip = st.sidebar.slider("CLAHE Contrast Clip Limit:", 1.0, 5.0, 2.5, step=0.5)

    if image_rgb is None:
        st.info("👈 Please select or upload a plant leaf image in the sidebar to begin automated diagnosis.")
        return

    # Process Core Pipeline
    with st.spinner("Executing 5-Stage Diagnostic Pipeline..."):
        # 1. Preprocessing & CLAHE (Modules 1-2)
        smoothed = apply_gaussian_blur(image_rgb, kernel_size=(5, 5), sigma=1.0)
        enhanced = apply_clahe(smoothed, clip_limit=clahe_clip)

        # 2. Leaf Foreground Segmentation (Module 3)
        leaf_mask = segment_leaf_hsv(enhanced)

        # 3. Lesion Extraction (Module 3)
        if "K-Means" in seg_method:
            disease_mask, cluster_vis = segment_disease_kmeans(enhanced, leaf_mask, k=3)
        else:
            disease_mask, healthy_mask = segment_disease_hsv(enhanced, leaf_mask)
            cluster_vis = None

        # 4. Severity Metrics & Overlay (Module 3 Twist)
        severity_metrics = compute_severity_metrics(leaf_mask, disease_mask)
        disease_overlay = create_severity_overlay(image_rgb, leaf_mask, disease_mask, severity_metrics)
        disease_heatmap = generate_disease_heatmap(image_rgb, disease_mask)

        # 5. MobileNetV2 Deep Learning Classification & Remedies (Modules 4-5)
        classification = classifier.predict(image_rgb, severity_metrics)
        remedy = get_disease_remedy(classification["predicted_class"], severity_metrics)

    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 5-Stage Diagnostic Pipeline",
        "🔬 CVIP Algorithm Inspector",
        "📁 Batch Evaluation & CSV Export"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: 5-STAGE PIPELINE DASHBOARD
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("🎯 Automated 5-Stage Diagnostic Pipeline")
        
        # Row of 5 Visual Stages
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.image(image_rgb, caption="Stage 1: Original Leaf", use_container_width=True)
            st.caption("Raw leaf acquisition on neutral background.")
        with c2:
            st.image(enhanced, caption="Stage 2: Gaussian + CLAHE", use_container_width=True)
            st.caption("Denoised & contrast-enhanced lesions.")
        with c3:
            st.image(leaf_mask, caption="Stage 3: Segmented Leaf", use_container_width=True, clamp=True)
            st.caption("Otsu + Morphological background removal.")
        with c4:
            st.image(disease_overlay, caption=f"Stage 4: Lesions ({severity_metrics['severity_percentage']}%)", use_container_width=True)
            st.caption("Isolated necrotic spots with yellow boundary.")
        with c5:
            st.image(disease_heatmap, caption="Stage 5: Thermal Density", use_container_width=True)
            st.caption("JET colormap lesion intensity gradient.")

        st.markdown("---")

        # Summary Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Predicted Disease", classification["formatted_name"])
            st.caption(f"Confidence: **{classification['confidence']}%**")
        with col_m2:
            st.metric("Disease Severity", f"{severity_metrics['severity_percentage']}%", delta=f"{severity_metrics['grade']}")
            st.caption(f"Stage: **{severity_metrics['stage']}**")
        with col_m3:
            st.metric("Diseased Pixel Count", f"{severity_metrics['diseased_pixels']:,} px")
            st.caption(f"Total Leaf: **{severity_metrics['total_leaf_pixels']:,} px**")
        with col_m4:
            st.metric("Urgency Level", severity_metrics["urgency"].split(" - ")[0])
            st.caption(severity_metrics["urgency"])

        st.markdown("---")

        # Clinical Diagnosis & Treatment Plan
        st.subheader("💊 Agronomic Diagnosis & Cure Advisory")
        col_diag, col_rem = st.columns([1, 1.2])

        with col_diag:
            st.markdown(f"**Target Crop:** `{remedy['crop']}`")
            st.markdown(f"**Identified Pathogen:** *{remedy['pathogen']}*")
            st.markdown(f"**Pathological Symptoms:** {remedy['symptoms']}")
            
            # Severity Alert Callout
            sev = severity_metrics["severity_percentage"]
            if sev < 0.5:
                st.success(f"✅ **Healthy Foliage:** {remedy['action_plan']}")
            elif sev <= 10.0:
                st.warning(f"⚠️ **Mild Infection (<10%):** {remedy['action_plan']}")
            elif sev <= 25.0:
                st.error(f"🚨 **Moderate Infection (10-25%):** {remedy['action_plan']}")
            else:
                st.error(f"🛑 **Critical Outbreak (>25%):** {remedy['action_plan']}")

            # Top-3 Model Confidences
            st.markdown("##### 📈 Top Model Probability Distribution")
            top_df = pd.DataFrame(classification["top_predictions"])
            top_df = top_df.rename(columns={"formatted_name": "Class", "confidence": "Confidence (%)"})
            st.dataframe(top_df[["Class", "Confidence (%)"]], hide_index=True, use_container_width=True)

        with col_rem:
            st.markdown("##### 🌿 Recommended Organic & Biological Controls")
            for bio in remedy["organic_remedies"]:
                st.markdown(f"- 🍃 {bio}")

            st.markdown("##### 🧪 Recommended Chemical Fungicides / Bactericides")
            for chem in remedy["chemical_remedies"]:
                st.markdown(f"- 💉 `{chem}`")

            st.markdown("##### 🚜 Agronomic Cultural Sanitation Practices")
            for cult in remedy["cultural_practices"]:
                st.markdown(f"- 🔄 {cult}")

        # Download Report Section
        st.markdown("---")
        st.subheader("📥 Export Diagnostic Report (PNG)")
        
        fig = generate_5stage_figure(
            image_rgb, enhanced, leaf_mask, disease_overlay,
            classification, severity_metrics
        )
        report_img_path = os.path.join("results", f"{os.path.splitext(image_name)[0]}_report.png")
        os.makedirs("results", exist_ok=True)
        fig.savefig(report_img_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        with open(report_img_path, "rb") as file:
            st.download_button(
                label="📥 Download High-Resolution 5-Stage Figure (PNG)",
                data=file,
                file_name=f"{os.path.splitext(image_name)[0]}_5stage_diagnostic.png",
                mime="image/png"
            )

    # --------------------------------------------------------------------------
    # TAB 2: CVIP ALGORITHM INSPECTOR
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🔬 Module-by-Module Computer Vision Inspector")
        st.markdown("Analyze how the underlying mathematical transformations extract meaningful diagnostic features.")

        # Sub-tab: Color Spaces
        st.markdown("#### 1. Color Space Decomposition (RGB vs HSV vs L*a*b*)")
        hsv_img = rgb_to_hsv(image_rgb)
        lab_img = rgb_to_lab(image_rgb)
        gray_img = rgb_to_gray(image_rgb)

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.image(colorize_channel(hsv_img[:, :, 0], "hsv"), caption="HSV: Hue Channel (Color Identity)", use_container_width=True)
            st.caption("Hue isolates chlorophyll (green: ~40-80) from necrotic browning (~10-25).")
        with col_c2:
            st.image(colorize_channel(hsv_img[:, :, 1], "viridis"), caption="HSV: Saturation Channel (Purity)", use_container_width=True)
            st.caption("High saturation highlights vibrant leaf area; background is near zero.")
        with col_c3:
            st.image(colorize_channel(lab_img[:, :, 1], "coolwarm"), caption="LAB: a* Channel (Green to Red)", use_container_width=True)
            st.caption("Negative values = healthy green; positive values = diseased brown/red lesions.")

        st.markdown("---")

        # Sub-tab: Histogram Equalization vs CLAHE
        st.markdown("#### 2. Contrast Enhancement: Standard Histogram Eq vs Adaptive CLAHE")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.image(image_rgb, caption="Original RGB Image", use_container_width=True)
        with col_h2:
            st.image(enhanced, caption=f"CLAHE Enhanced (Clip Limit={clahe_clip})", use_container_width=True)

        # Plot Histograms
        fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 3))
        ax_hist[0].hist(gray_img.ravel(), 256, [0, 256], color="gray")
        ax_hist[0].set_title("Original Grayscale Histogram")
        ax_hist[0].set_xlim([0, 256])

        enhanced_gray = rgb_to_gray(enhanced)
        ax_hist[1].hist(enhanced_gray.ravel(), 256, [0, 256], color="teal")
        ax_hist[1].set_title(f"CLAHE Enhanced Histogram (Tiles: 8x8)")
        ax_hist[1].set_xlim([0, 256])
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        st.markdown("---")

        # Sub-tab: Edge Gradients
        st.markdown("#### 3. Structural Gradient & Edge Detection (Sobel vs Canny)")
        canny_edges = compute_canny_edges(enhanced)
        sobel_res = compute_sobel_gradients(enhanced)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.image(sobel_res["magnitude"], caption="Sobel Gradient Magnitude sqrt(Gx^2 + Gy^2)", use_container_width=True)
            st.caption("First-order spatial derivatives highlighting leaf boundary and lesion contours.")
        with col_e2:
            st.image(canny_edges, caption="Canny Multi-Stage Hysteresis Edges", use_container_width=True)
            st.caption("Non-maximum suppression and hysteresis thresholding (50, 150).")

    # --------------------------------------------------------------------------
    # TAB 3: BATCH PROCESSING & CSV EXPORT
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("📁 Batch Diagnostic Evaluation Across All Benchmark Leaves")
        st.markdown("Process all test leaves in the dataset to produce a comprehensive academic evaluation table.")

        if st.button("🚀 Run Batch Diagnostic on All Sample Leaves"):
            results_records = []
            progress_bar = st.progress(0)

            for i, fname in enumerate(sample_files):
                fpath = os.path.join(samples_dir, fname)
                bgr = load_image(fpath, target_size=(512, 512))
                rgb = bgr_to_rgb(bgr)

                # Run pipeline
                prep = apply_clahe(apply_gaussian_blur(rgb, (5, 5), 1.0))
                l_mask = segment_leaf_hsv(prep)
                d_mask, _ = segment_disease_hsv(prep, l_mask)
                sev = compute_severity_metrics(l_mask, d_mask)
                clf = classifier.predict(rgb, sev)
                med = get_disease_remedy(clf["predicted_class"], sev)

                results_records.append({
                    "Image File": fname,
                    "Predicted Crop & Disease": clf["formatted_name"],
                    "Confidence (%)": clf["confidence"],
                    "Total Leaf Pixels": sev["total_leaf_pixels"],
                    "Diseased Pixels": sev["diseased_pixels"],
                    "Severity (%)": sev["severity_percentage"],
                    "Grade": sev["grade"],
                    "Stage": sev["stage"],
                    "Primary Remedy": med["organic_remedies"][0] if med["organic_remedies"] else "None"
                })
                progress_bar.progress((i + 1) / len(sample_files))

            df_batch = pd.DataFrame(results_records)
            st.success(f"✅ Successfully processed {len(results_records)} leaves!")
            st.dataframe(df_batch, use_container_width=True)

            # CSV Download
            csv_data = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Batch Results (CSV)",
                data=csv_data,
                file_name="plant_leaf_disease_batch_evaluation.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
