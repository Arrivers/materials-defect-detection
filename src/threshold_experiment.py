import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


project_root = Path(__file__).resolve().parents[1]

image_path = (
    project_root
    / "data"
    / "sample"
    / "synthetic_scratch.png"
)

ground_truth_path = (
    project_root
    / "data"
    / "sample"
    / "synthetic_scratch_ground_truth.png"
)

csv_path = (
    project_root
    / "outputs"
    / "metrics"
    / "threshold_results.csv"
)

figure_path = (
    project_root
    / "outputs"
    / "figures"
    / "threshold_comparison.png"
)

# 读取合成图像
image = Image.open(image_path).convert("L")
pixels = np.asarray(image)

# 根据划痕生成时的已知位置，建立独立真实标注
ground_truth_image = Image.new(
    mode="L",
    size=image.size,
    color=0,
)

ground_truth_drawer = ImageDraw.Draw(ground_truth_image)

ground_truth_drawer.line(
    xy=(30, 220, 220, 40),
    fill=255,
    width=3,
)

ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
ground_truth_image.save(ground_truth_path)

ground_truth_mask = np.asarray(ground_truth_image) > 0


def calculate_metrics(prediction, ground_truth):
    true_positive = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    false_positive = np.logical_and(
        prediction,
        ~ground_truth,
    ).sum()

    false_negative = np.logical_and(
        ~prediction,
        ground_truth,
    ).sum()

    precision = true_positive / (
        true_positive + false_positive
    )

    recall = true_positive / (
        true_positive + false_negative
    )

    intersection_over_union = true_positive / (
        true_positive
        + false_positive
        + false_negative
    )

    return precision, recall, intersection_over_union


thresholds = [40, 60, 80, 100, 120, 140]
results = []

print(
    "Threshold | Precision | Recall | IoU | "
    "Predicted Pixels"
)

for threshold in thresholds:
    prediction_mask = pixels < threshold

    precision, recall, iou = calculate_metrics(
        prediction_mask,
        ground_truth_mask,
    )

    predicted_pixel_count = np.count_nonzero(
        prediction_mask
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "predicted_pixels": predicted_pixel_count,
        }
    )

    print(
        f"{threshold:9d} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{iou:6.4f} | "
        f"{predicted_pixel_count:16d}"
    )

# 保存实验数据
csv_path.parent.mkdir(parents=True, exist_ok=True)

with csv_path.open(
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=results[0].keys(),
    )
    writer.writeheader()
    writer.writerows(results)

# 绘制评价指标和预测像素数量
precision_values = [
    result["precision"] for result in results
]
recall_values = [
    result["recall"] for result in results
]
iou_values = [
    result["iou"] for result in results
]
predicted_pixel_values = [
    result["predicted_pixels"] for result in results
]

figure, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(
    thresholds,
    precision_values,
    marker="o",
    label="Precision",
)
axes[0].plot(
    thresholds,
    recall_values,
    marker="s",
    label="Recall",
)
axes[0].plot(
    thresholds,
    iou_values,
    marker="^",
    label="IoU",
)

axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Metric Value")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Segmentation Metrics")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(
    thresholds,
    predicted_pixel_values,
    marker="o",
    color="darkred",
    label="Predicted Pixels",
)

axes[1].axhline(
    y=np.count_nonzero(ground_truth_mask),
    color="black",
    linestyle="--",
    label="Ground Truth Pixels",
)

axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Pixel Count")
axes[1].set_title("Predicted Defect Area")
axes[1].grid(alpha=0.3)
axes[1].legend()

figure.tight_layout()

figure_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(figure_path, dpi=200)
plt.close(figure)

print(f"Results saved to: {csv_path}")
print(f"Figure saved to: {figure_path}")