from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


project_root = Path(__file__).resolve().parents[1]
output_path = project_root / "data" / "sample" / "synthetic_scratch.png"

# 固定随机种子，使每次运行生成完全相同的结果
random_generator = np.random.default_rng(seed=42)

# 生成 256×256 的灰度材料表面
surface_pixels = random_generator.normal(
    loc=150,
    scale=15,
    size=(256, 256),
)

# 将像素限制在 0～255，并转换成 8 位无符号整数
surface_pixels = np.clip(surface_pixels, 0, 255).astype(np.uint8)

image = Image.fromarray(surface_pixels)

# 绘制一条深色斜线，模拟表面划痕
drawer = ImageDraw.Draw(image)
drawer.line(
    xy=(30, 220, 220, 40),
    fill=30,
    width=3,
)

output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(output_path)

print(f"Image saved to: {output_path}")