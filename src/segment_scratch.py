from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


project_root = Path(__file__).resolve().parents[1]

image_path = (
    project_root
    / "data"
    / "sample"
    / "synthetic_scratch.png"
)

figure_path = (
    project_root
    / "outputs"
    / "figures"
    / "scratch_segmentation.png"
)

mask_path = (
    project_root
    / "outputs"
    / "masks"
    / "synthetic_scratch_mask.png"
)

# 读取灰度图，并转换成二维矩阵
image = Image.open(image_path).convert("L")
pixels = np.asarray(image)

# 阈值分割：像素值小于 80 时判为划痕
threshold = 80
scratch_mask = pixels < threshold

# 统计划痕像素
scratch_pixel_count = np.count_nonzero(scratch_mask)
total_pixel_count = scratch_mask.size
scratch_ratio = scratch_pixel_count / total_pixel_count * 100

# 获取划痕区域的外接矩形
y_coordinates, x_coordinates = np.where(scratch_mask)

x_min = x_coordinates.min()
x_max = x_coordinates.max()
y_min = y_coordinates.min()
y_max = y_coordinates.max()

print(f"Threshold: {threshold}")
print(f"Scratch pixels: {scratch_pixel_count}")
print(f"Total pixels: {total_pixel_count}")
print(f"Scratch area ratio: {scratch_ratio:.2f}%")
print(
    "Bounding box: "
    f"({x_min}, {y_min}) to ({x_max}, {y_max})"
)

# 保存黑白掩膜：白色表示划痕
mask_image = (scratch_mask * 255).astype(np.uint8)

mask_path.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(mask_image).save(mask_path)

# 可视化原图、掩膜和叠加结果
figure, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].imshow(pixels, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original Image")

bounding_box = Rectangle(
    (x_min, y_min),
    x_max - x_min + 1,
    y_max - y_min + 1,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
)

axes[0].add_patch(bounding_box)
axes[0].axis("off")

axes[1].imshow(scratch_mask, cmap="gray")
axes[1].set_title("Binary Mask")
axes[1].axis("off")

axes[2].imshow(pixels, cmap="gray", vmin=0, vmax=255)

masked_scratch = np.ma.masked_where(
    ~scratch_mask,
    scratch_mask,
)

axes[2].imshow(
    masked_scratch,
    cmap="Reds",
    alpha=0.65,
    vmin=0,
    vmax=1,
)

axes[2].set_title("Segmentation Overlay")
axes[2].axis("off")

figure.tight_layout()

figure_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(figure_path, dpi=200)
plt.close(figure)

print(f"Mask saved to: {mask_path}")
print(f"Figure saved to: {figure_path}")