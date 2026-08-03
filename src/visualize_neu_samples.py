import xml.etree.ElementTree as element_tree
from pathlib import Path

import matplotlib

# 使用无窗口绘图后端，确保脚本稳定运行
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


project_root = Path(__file__).resolve().parents[1]

dataset_root = (
    project_root
    / "data"
    / "raw"
    / "NEU-DET"
)

images_directory = dataset_root / "IMAGES"
annotations_directory = dataset_root / "ANNOTATIONS"

output_path = (
    project_root
    / "outputs"
    / "figures"
    / "neu_det_samples.png"
)

sample_stems = [
    "crazing_172",
    "inclusion_19",
    "patches_232",
    "pitted_surface_181",
    "rolled-in_scale_235",
    "scratches_104",
]


def read_voc_annotation(xml_path):
    xml_root = element_tree.parse(
        xml_path
    ).getroot()

    annotations = []

    for object_element in xml_root.findall("object"):
        class_name = object_element.findtext("name")
        box = object_element.find("bndbox")

        x_min = int(float(box.findtext("xmin")))
        y_min = int(float(box.findtext("ymin")))
        x_max = int(float(box.findtext("xmax")))
        y_max = int(float(box.findtext("ymax")))

        annotations.append(
            {
                "class_name": class_name,
                "box": (
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ),
            }
        )

    return annotations


figure, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8),
)

for axis, sample_stem in zip(
    axes.flat,
    sample_stems,
):
    image_path = (
        images_directory
        / f"{sample_stem}.jpg"
    )

    xml_path = (
        annotations_directory
        / f"{sample_stem}.xml"
    )

    image = Image.open(image_path)
    annotations = read_voc_annotation(
        xml_path
    )

    axis.imshow(image, cmap="gray")

    for annotation in annotations:
        class_name = annotation["class_name"]

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = annotation["box"]

        rectangle = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )

        axis.add_patch(rectangle)

        label_y = max(y_min - 3, 0)

        axis.text(
            x_min,
            label_y,
            class_name,
            color="white",
            fontsize=7,
            bbox={
                "facecolor": "red",
                "alpha": 0.75,
                "pad": 1,
            },
        )

    display_class = sample_stem.rsplit(
        "_",
        maxsplit=1,
    )[0]

    axis.set_title(
        f"{display_class}\n"
        f"{len(annotations)} object(s)"
    )

    axis.axis("off")

    print(
        f"{sample_stem}: "
        f"{len(annotations)} bounding box(es)"
    )

figure.suptitle(
    "NEU-DET Surface Defect Samples",
    fontsize=16,
)

figure.tight_layout(
    rect=(0, 0, 1, 0.96)
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

figure.savefig(
    output_path,
    dpi=200,
)

plt.close(figure)

print(f"Figure saved to: {output_path}")