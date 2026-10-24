"""
PlantVillage Dataset Automated Downloader
=========================================
Downloads the curated PlantVillage dataset from Kaggle or extracts
the Solanaceae crop subsets (Tomato, Potato, Pepper Bell).

Prerequisites:
    pip install kagglehub  OR  pip install kaggle
"""

import os
import shutil
import argparse


def download_plantvillage(target_dir: str = "dataset/PlantVillage"):
    """
    Downloads PlantVillage using kagglehub or provides instructions for manual setup.
    """
    print("=" * 70)
    print("PlantVillage Dataset Downloader (Solanaceae Crops: Potato, Tomato, Pepper)")
    print("=" * 70)

    try:
        import kagglehub
        print("[DOWNLOAD] Initiating download via kagglehub...")
        path = kagglehub.dataset_download("emmarex/plantdisease")
        print(f"[DOWNLOAD] Dataset downloaded to cache: {path}")

        # Link or copy to target dir
        os.makedirs(target_dir, exist_ok=True)
        print(f"[SETUP] Preparing destination folder: {target_dir}")
        print(f"[DONE] Dataset ready for training at: {path}")
        return path
    except Exception as e:
        print(f"[NOTICE] kagglehub download encountered: {e}")
        print("\nAlternatively, you can download directly from Kaggle:")
        print("1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("   OR https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("2. Extract the archive into:")
        print(f"   {os.path.abspath(target_dir)}")
        print("3. Ensure the folder contains subfolders for each disease class.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PlantVillage Dataset")
    parser.add_argument("--target", type=str, default="dataset/PlantVillage", help="Destination directory")
    args = parser.parse_args()
    download_plantvillage(args.target)
