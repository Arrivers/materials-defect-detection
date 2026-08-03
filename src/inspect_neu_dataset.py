import json
import re
import xml.etree.ElementTree as element_tree
from collections import Counter
from pathlib import Path

import numpy as np
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

report_path = (
    project_root
    / "outputs"
    / "metrics"
    / "neu_dataset_audit.json"
)

expected_classes = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
}


def get_class_from_filename(file_stem):
    match = re.match(r"(.+)_\d+$", file_stem)

    if match is None:
        return "unknown"

    return match.group(1)


image_files = sorted(images_directory.glob("*.jpg"))
annotation_files = sorted(
    annotations_directory.glob("*.xml")
)

image_map = {
    file_path.stem: file_path
    for file_path in image_files
}

annotation_map = {
    file_path.stem: file_path
    for file_path in annotation_files
}

image_stems = set(image_map)
annotation_stems = set(annotation_map)

paired_stems = sorted(
    image_stems & annotation_stems
)

missing_annotations = sorted(
    image_stems - annotation_stems
)

missing_images = sorted(
    annotation_stems - image_stems
)

image_class_counts = Counter()
object_class_counts = Counter()
image_size_counts = Counter()
image_mode_counts = Counter()

problems = []

for file_stem in paired_stems:
    image_path = image_map[file_stem]
    annotation_path = annotation_map[file_stem]

    filename_class = get_class_from_filename(
        file_stem
    )
    image_class_counts[filename_class] += 1

    # 检查图片
    try:
        with Image.open(image_path) as image:
            image.load()

            image_size_counts[str(image.size)] += 1
            image_mode_counts[image.mode] += 1

            pixels = np.asarray(image)

            if image.size != (200, 200):
                problems.append(
                    f"Unexpected size: {image_path.name}"
                )

            if pixels.ndim != 3 or pixels.shape[2] != 3:
                problems.append(
                    f"Unexpected channels: {image_path.name}"
                )
            else:
                channels_identical = (
                    np.array_equal(
                        pixels[:, :, 0],
                        pixels[:, :, 1],
                    )
                    and np.array_equal(
                        pixels[:, :, 1],
                        pixels[:, :, 2],
                    )
                )

                if not channels_identical:
                    problems.append(
                        f"Non-grayscale RGB image: "
                        f"{image_path.name}"
                    )

    except Exception as error:
        problems.append(
            f"Unreadable image: "
            f"{image_path.name}: {error}"
        )
        continue

    # 检查 Pascal VOC XML
    try:
        xml_root = element_tree.parse(
            annotation_path
        ).getroot()

        xml_filename = xml_root.findtext("filename")

        if xml_filename is not None:
            if Path(xml_filename).stem != file_stem:
                problems.append(
                    f"Filename mismatch: "
                    f"{annotation_path.name}"
                )

        size_element = xml_root.find("size")

        width = int(
            size_element.findtext("width")
        )
        height = int(
            size_element.findtext("height")
        )

        if (width, height) != (200, 200):
            problems.append(
                f"XML size error: "
                f"{annotation_path.name}"
            )

        objects = xml_root.findall("object")

        if not objects:
            problems.append(
                f"No objects: {annotation_path.name}"
            )

        for object_element in objects:
            class_name = object_element.findtext(
                "name"
            )

            object_class_counts[class_name] += 1

            if class_name not in expected_classes:
                problems.append(
                    f"Unknown class {class_name}: "
                    f"{annotation_path.name}"
                )

            box = object_element.find("bndbox")

            x_min = int(float(box.findtext("xmin")))
            y_min = int(float(box.findtext("ymin")))
            x_max = int(float(box.findtext("xmax")))
            y_max = int(float(box.findtext("ymax")))

            valid_box = (
                0 <= x_min < x_max <= width
                and 0 <= y_min < y_max <= height
            )

            if not valid_box:
                problems.append(
                    f"Invalid box: "
                    f"{annotation_path.name}"
                )

    except Exception as error:
        problems.append(
            f"Invalid XML: "
            f"{annotation_path.name}: {error}"
        )

report = {
    "dataset": "NEU-DET",
    "image_count": len(image_files),
    "annotation_count": len(annotation_files),
    "paired_count": len(paired_stems),
    "missing_annotation_count": len(
        missing_annotations
    ),
    "missing_image_count": len(missing_images),
    "image_class_counts": dict(
        sorted(image_class_counts.items())
    ),
    "object_class_counts": dict(
        sorted(object_class_counts.items())
    ),
    "image_size_counts": dict(
        image_size_counts
    ),
    "image_mode_counts": dict(
        image_mode_counts
    ),
    "problem_count": len(problems),
    "problems": problems,
}

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with report_path.open(
    mode="w",
    encoding="utf-8",
) as report_file:
    json.dump(
        report,
        report_file,
        ensure_ascii=False,
        indent=2,
    )

print("=" * 50)
print("NEU-DET Dataset Audit")
print("=" * 50)

print(f"Images: {len(image_files)}")
print(f"Annotations: {len(annotation_files)}")
print(f"Paired samples: {len(paired_stems)}")
print(f"Missing annotations: {len(missing_annotations)}")
print(f"Missing images: {len(missing_images)}")

print("\nImages per class:")

for class_name, count in sorted(
    image_class_counts.items()
):
    print(f"  {class_name}: {count}")

print("\nObjects per class:")

for class_name, count in sorted(
    object_class_counts.items()
):
    print(f"  {class_name}: {count}")

print(f"\nProblems: {len(problems)}")

if problems:
    print("Dataset audit failed.")
else:
    print("Dataset audit passed.")

print(f"Report saved to: {report_path}")