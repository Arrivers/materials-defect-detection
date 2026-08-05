import csv
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.svm import LinearSVC

from train_classical_baseline import (
    FIGURE_DIR,
    IMAGE_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    SPLIT_DIR,
    load_split,
)


MAX_EXAMPLES = 12
MAX_PER_TRUE_CLASS = 2


def read_sample_names(split_name: str) -> list[str]:
    split_path = SPLIT_DIR / f"{split_name}.txt"

    return [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_balanced_errors(
    error_indices: np.ndarray,
    true_labels: np.ndarray,
    margins: np.ndarray,
) -> list[int]:
    """Select confident errors while limiting each true class."""
    sorted_indices = sorted(
        error_indices,
        key=lambda index: margins[index],
        reverse=True,
    )

    selected = []
    class_counts = Counter()

    for index in sorted_indices:
        true_class = true_labels[index]

        if class_counts[true_class] < MAX_PER_TRUE_CLASS:
            selected.append(index)
            class_counts[true_class] += 1

        if len(selected) == MAX_EXAMPLES:
            break

    if len(selected) < MAX_EXAMPLES:
        for index in sorted_indices:
            if index not in selected:
                selected.append(index)

            if len(selected) == MAX_EXAMPLES:
                break

    return selected


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Classical Baseline Validation Error Analysis")
    print("=" * 60)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    val_sample_names = read_sample_names("val")

    if len(val_sample_names) != len(y_val):
        raise ValueError("Validation sample names and labels do not match.")

    model = LinearSVC(
        C=1.0,
        random_state=RANDOM_SEED,
        max_iter=10000,
    )

    print()
    print("Training the baseline model...")
    model.fit(x_train, y_train)

    predictions = model.predict(x_val)
    decision_scores = model.decision_function(x_val)

    sorted_scores = np.sort(decision_scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]

    error_indices = np.flatnonzero(predictions != y_val)
    correct_count = len(y_val) - len(error_indices)
    accuracy = correct_count / len(y_val)

    confusion_counts = Counter(
        (y_val[index], predictions[index])
        for index in error_indices
    )

    print()
    print(f"Validation samples: {len(y_val)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {len(error_indices)}")
    print(f"Validation accuracy: {accuracy:.4f}")

    print()
    print("Most common confusion directions:")

    for (true_class, predicted_class), count in confusion_counts.most_common(10):
        print(
            f"  {true_class} -> {predicted_class}: "
            f"{count} image(s)"
        )

    csv_path = METRICS_DIR / "classical_baseline_validation_predictions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_name",
                "true_class",
                "predicted_class",
                "correct",
                "decision_margin",
            ]
        )

        for index, sample_name in enumerate(val_sample_names):
            writer.writerow(
                [
                    sample_name,
                    y_val[index],
                    predictions[index],
                    bool(predictions[index] == y_val[index]),
                    f"{margins[index]:.6f}",
                ]
            )

    selected_indices = select_balanced_errors(
        error_indices,
        y_val,
        margins,
    )

    figure, axes = plt.subplots(3, 4, figsize=(14, 11))
    axes = axes.ravel()

    for axis, index in zip(axes, selected_indices):
        image_path = IMAGE_DIR / f"{val_sample_names[index]}.jpg"

        with Image.open(image_path) as image:
            axis.imshow(image.convert("L"), cmap="gray")

        true_class = y_val[index].replace("_", " ")
        predicted_class = predictions[index].replace("_", " ")

        axis.set_title(
            f"True: {true_class}\n"
            f"Predicted: {predicted_class}\n"
            f"Margin: {margins[index]:.3f}",
            color="darkred",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes[len(selected_indices):]:
        axis.axis("off")

    figure.suptitle(
        "Representative HOG + Linear SVM Validation Errors",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    figure_path = (
        FIGURE_DIR
        / "classical_baseline_validation_errors.png"
    )
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print()
    print(f"Prediction table saved to: {csv_path}")
    print(f"Error figure saved to: {figure_path}")
    print()
    print(
        "Note: the decision margin measures model confidence, "
        "but it is not a probability."
    )


if __name__ == "__main__":
    main()