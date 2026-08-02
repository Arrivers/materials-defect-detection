from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


project_root = Path(__file__).resolve().parents[1]

image_path = (
    project_root
    / "data"
    / "sample"
    / "synthetic_scratch.png"
)

output_path = (
    project_root
    / "outputs"
    / "figures"
    / "image_analysis.png"
)

# 读取图像，并确保它是灰度图
image = Image.open(image_path).convert("L")

# 将图像转换为 NumPy 矩阵
pixels = np.asarray(image)

print(f"Image shape: {pixels.shape}")
print(f"Data type: {pixels.dtype}")
print(f"Minimum pixel: {pixels.min()}")
print(f"Maximum pixel: {pixels.max()}")
print(f"Mean pixel: {pixels.mean():.2f}")

# 创建“原始图像 + 灰度直方图”
figure, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(
    pixels,
    cmap="gray",
    vmin=0,
    vmax=255,
)
axes[0].set_title("Synthetic Surface Scratch")
axes[0].axis("off")

axes[1].hist(
    pixels.ravel(),
    bins=50,
    range=(0, 255),
    color="steelblue",
)
axes[1].set_title("Grayscale Histogram")
axes[1].set_xlabel("Pixel Value")
axes[1].set_ylabel("Pixel Count")

output_path.parent.mkdir(parents=True, exist_ok=True)

figure.tight_layout()
figure.savefig(output_path, dpi=200)
plt.close(figure)

print(f"Analysis figure saved to: {output_path}")