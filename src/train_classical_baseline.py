import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "IMAGES"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGE_SIZE = (128, 128)
RANDOM_SEED = 42


def get_class_name(sample_name: str) -> str:
    """Extract the class name from a sample name."""
    return sample_name.rsplit("_", maxsplit=1)[0]


def extract_hog_feature(image_path: Path) -> np.ndarray:
    """Convert one image into a HOG feature vector."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.uint8)

    feature_vector = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return feature_vector.astype(np.float32)


def load_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split and extract HOG features."""
    split_path = SPLIT_DIR / f"{split_name}.txt"

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    sample_names = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    features = []
    labels = []

    print(f"Loading {split_name} split: {len(sample_names)} images")

    for index, sample_name in enumerate(sample_names, start=1):
        image_path = IMAGE_DIR / f"{sample_name}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        features.append(extract_hog_feature(image_path))
        labels.append(get_class_name(sample_name))

        if index % 100 == 0 or index == len(sample_names):
            print(f"  Processed {index}/{len(sample_names)}")

    return np.asarray(features), np.asarray(labels)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NEU-DET Classical Classification Baseline")
    print("Model: HOG + Linear SVM")
    print("=" * 60)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    print()
    print(f"Training feature matrix: {x_train.shape}")
    print(f"Validation feature matrix: {x_val.shape}")
    print(f"Test feature matrix: {x_test.shape}")

    model = LinearSVC(
        C=1.0,
        random_state=RANDOM_SEED,
        max_iter=10000,
    )

    print()
    print("Training the Linear SVM...")
    model.fit(x_train, y_train)
    print("Training completed.")

    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print()
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print()
    print("Test classification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    test_report = classification_report(
        y_test,
        test_predictions,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    test_confusion_matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=CLASS_NAMES,
    )

    metrics = {
        "model": "HOG + LinearSVC",
        "random_seed": RANDOM_SEED,
        "image_size": list(IMAGE_SIZE),
        "hog_parameters": {
            "orientations": 9,
            "pixels_per_cell": [8, 8],
            "cells_per_block": [2, 2],
            "block_norm": "L2-Hys",
        },
        "svm_parameters": {
            "C": 1.0,
            "max_iter": 10000,
        },
        "feature_dimensions": int(x_train.shape[1]),
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_confusion_matrix.tolist(),
    }

    metrics_path = METRICS_DIR / "classical_baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=test_confusion_matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        f"HOG + Linear SVM Confusion Matrix\n"
        f"Test Accuracy: {test_accuracy:.2%}"
    )
    plt.xticks(rotation=30, ha="right")
    figure.tight_layout()

    figure_path = FIGURE_DIR / "classical_baseline_confusion_matrix.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {figure_path}")


if __name__ == "__main__":
    main()