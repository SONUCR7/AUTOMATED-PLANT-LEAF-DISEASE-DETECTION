"""
MobileNetV2 Transfer Learning Model Training Script
===================================================
Trains MobileNetV2 on the PlantVillage dataset (Potato, Tomato, and Pepper subsets).
Designed to run locally or in Google Colab with GPU acceleration in ~15 minutes.

Usage:
    python train.py --data-dir path/to/PlantVillage --epochs 15 --batch-size 32
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.classifier import CLASS_NAMES, build_mobilenetv2_model


def create_data_generators(data_dir: str, target_size=(224, 224), batch_size=32):
    """
    Constructs training and validation data generators with data augmentation:
    - Random rotation (+/- 20 deg)
    - Random zoom (0.2)
    - Random horizontal/vertical flip
    - Width/Height shift (0.15)
    """
    datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2
    )

    train_gen = datagen_train.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        classes=CLASS_NAMES,
        class_mode="sparse",
        subset="training",
        shuffle=True
    )

    val_gen = datagen_val.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        classes=CLASS_NAMES,
        class_mode="sparse",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen


def train_model(data_dir: str, epochs: int = 15, batch_size: int = 32, output_model_path: str = "models/plant_disease_mobilenetv2.keras"):
    """
    Two-Phase Transfer Learning Pipeline:
    Phase 1: Feature Extraction (Train top classification layers with frozen base)
    Phase 2: Fine-Tuning (Unfreeze top 30 layers of MobileNetV2 with low learning rate)
    """
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"\n[TRAIN] Loading dataset from: {data_dir}")
    train_gen, val_gen = create_data_generators(data_dir, batch_size=batch_size)

    print(f"[TRAIN] Total Training Samples   : {train_gen.samples}")
    print(f"[TRAIN] Total Validation Samples : {val_gen.samples}")
    print(f"[TRAIN] Total Target Classes     : {len(CLASS_NAMES)}")

    # Build model
    model = build_mobilenetv2_model(num_classes=len(CLASS_NAMES))

    # Phase 1: Train classification head
    print("\n--- Phase 1: Training Classification Head (Backbone Frozen) ---")
    cb_phase1 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
        callbacks.ModelCheckpoint(output_model_path, monitor="val_accuracy", save_best_only=True)
    ]

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cb_phase1
    )

    # Phase 2: Fine-tuning top layers of MobileNetV2
    print("\n--- Phase 2: Fine-Tuning Top 30 Layers of MobileNetV2 ---")
    base_model = model.get_layer("mobilenetv2_1.00_224")
    base_model.trainable = True
    # Freeze all layers before layer 120
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=min(5, epochs // 2),
        callbacks=cb_phase1
    )

    # Save final model
    model.save(output_model_path)
    print(f"[TRAIN] Model saved successfully to: {output_model_path}")

    # Evaluate & Plot
    print("\n[EVAL] Generating Evaluation Report and Confusion Matrix...")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    # Classification Report
    cls_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("\nClassification Report:\n", cls_report)
    with open("results/classification_report.txt", "w") as f:
        f.write(cls_report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=[c.split("___")[-1] for c in CLASS_NAMES],
                yticklabels=[c.split("___")[-1] for c in CLASS_NAMES])
    plt.title("Plant Leaf Disease Classification - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=200)
    plt.close()

    print("[SUCCESS] All training artifacts saved in 'models/' and 'results/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on PlantVillage")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to PlantVillage dataset root directory")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default="models/plant_disease_mobilenetv2.keras", help="Save path")

    args = parser.parse_args()
    train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, output_model_path=args.output)
