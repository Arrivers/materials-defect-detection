import csv
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from train_cnn_classifier import (
    BATCH_SIZE,
    CLASS_NAMES,
    FIGURE_DIR,
    IMAGE_SIZE,
    METRICS_DIR,
    MODEL_DIR,
    NEUClassificationDataset,
    SmallCNN,
)


HOG_SVM_TEST_ACCURACY = 0.685185


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = MODEL_DIR / "best_small_cnn.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    test_dataset = NEUClassificationDataset(
        "test",
        test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model = SmallCNN(
        number_of_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loss_function = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    all_confidences = []

    total_loss = 0.0
    total_samples = 0

    print("=" * 60)
    print("NEU-DET CNN Final Test Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Checkpoint best epoch: {checkpoint['best_epoch']}")
    print(
        "Checkpoint validation accuracy: "
        f"{checkpoint['best_validation_accuracy']:.4f}"
    )
    print(f"Test samples: {len(test_dataset)}")
    print()
    print("Evaluating the untouched test split...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_function(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(
                confidences.cpu().tolist()
            )

    test_loss = total_loss / total_samples
    test_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    improvement = (
        test_accuracy - HOG_SVM_TEST_ACCURACY
    )

    print()
    print(f"Test loss: {test_loss:.4f}")
    print(f"CNN test accuracy: {test_accuracy:.4f}")
    print(
        "HOG + SVM test accuracy: "
        f"{HOG_SVM_TEST_ACCURACY:.4f}"
    )
    print(
        "Absolute accuracy improvement: "
        f"{improvement:.4f}"
    )

    print()
    print("CNN test classification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    metrics = {
        "model": "SmallCNN",
        "checkpoint_best_epoch": int(
            checkpoint["best_epoch"]
        ),
        "checkpoint_validation_accuracy": float(
            checkpoint["best_validation_accuracy"]
        ),
        "test_samples": len(test_dataset),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "hog_svm_test_accuracy": HOG_SVM_TEST_ACCURACY,
        "absolute_accuracy_improvement": float(
            improvement
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    metrics_path = METRICS_DIR / "cnn_test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    predictions_path = (
        METRICS_DIR / "cnn_test_predictions.csv"
    )

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_name",
                "true_class",
                "predicted_class",
                "correct",
                "confidence",
            ]
        )

        for sample_name, true_index, predicted_index, confidence in zip(
            test_dataset.sample_names,
            all_labels,
            all_predictions,
            all_confidences,
        ):
            writer.writerow(
                [
                    sample_name,
                    CLASS_NAMES[true_index],
                    CLASS_NAMES[predicted_index],
                    true_index == predicted_index,
                    f"{confidence:.6f}",
                ]
            )

    figure, axis = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        "Small CNN Test Confusion Matrix\n"
        f"Accuracy: {test_accuracy:.2%}"
    )
    plt.xticks(rotation=30, ha="right")
    figure.tight_layout()

    figure_path = (
        FIGURE_DIR / "cnn_test_confusion_matrix.png"
    )
    figure.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Confusion matrix saved to: {figure_path}")
    print()
    print(
        "The test result is now recorded and must not "
        "be used for further hyperparameter tuning."
    )


if __name__ == "__main__":
    main()