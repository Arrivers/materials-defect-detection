import hashlib
import json
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "IMAGES"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def numeric_suffix(path: Path) -> int:
    """Extract the number at the end of a filename."""
    match = re.search(r"_(\d+)$", path.stem)

    if match is None:
        raise ValueError(f"Invalid filename: {path.name}")

    return int(match.group(1))


def get_class_name(stem: str) -> str:
    """Extract the defect class from a sample name."""
    match = re.match(r"(.+)_\d+$", stem)

    if match is None:
        raise ValueError(f"Invalid sample name: {stem}")

    return match.group(1)


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for class_name in CLASS_NAMES:
        image_paths = sorted(
            IMAGE_DIR.glob(f"{class_name}_*.jpg"),
            key=numeric_suffix,
        )

        if len(image_paths) != 300:
            raise ValueError(
                f"{class_name}: expected 300 images, found {len(image_paths)}"
            )

        sample_names = [path.stem for path in image_paths]

        seed_text = f"{RANDOM_SEED}:{class_name}"
        class_seed = int(
            hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16],
            16,
        )
        random.Random(class_seed).shuffle(sample_names)

        train_end = int(len(sample_names) * TRAIN_RATIO)
        val_end = train_end + int(len(sample_names) * VAL_RATIO)

        splits["train"].extend(sample_names[:train_end])
        splits["val"].extend(sample_names[train_end:val_end])
        splits["test"].extend(sample_names[val_end:])

    for split_name in splits:
        splits[split_name].sort()

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    all_samples = train_set | val_set | test_set

    overlap_count = (
        len(train_set & val_set)
        + len(train_set & test_set)
        + len(val_set & test_set)
    )

    expected_samples = {path.stem for path in IMAGE_DIR.glob("*.jpg")}
    missing_count = len(expected_samples - all_samples)

    if overlap_count != 0:
        raise ValueError("Data leakage detected: split overlap is not zero.")

    if all_samples != expected_samples:
        raise ValueError("Split samples do not match the complete dataset.")

    per_class = {}

    for class_name in CLASS_NAMES:
        per_class[class_name] = {}

        for split_name, sample_names in splits.items():
            per_class[class_name][split_name] = sum(
                get_class_name(name) == class_name
                for name in sample_names
            )

    fingerprint_text = "\n".join(
        f"{split_name}:{sample_name}"
        for split_name in ("train", "val", "test")
        for sample_name in splits[split_name]
    )
    fingerprint = hashlib.sha256(
        fingerprint_text.encode("utf-8")
    ).hexdigest()

    for split_name, sample_names in splits.items():
        output_path = OUTPUT_DIR / f"{split_name}.txt"
        output_path.write_text(
            "\n".join(sample_names) + "\n",
            encoding="utf-8",
        )

    summary = {
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        },
        "total_samples": len(all_samples),
        "split_counts": {
            name: len(samples)
            for name, samples in splits.items()
        },
        "per_class_counts": per_class,
        "overlap_count": overlap_count,
        "missing_count": missing_count,
        "split_fingerprint_sha256": fingerprint,
    }

    summary_path = OUTPUT_DIR / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 50)
    print("NEU-DET Stratified Dataset Split")
    print("=" * 50)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Validation samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"Overlap: {overlap_count}")
    print(f"Missing: {missing_count}")
    print()

    print("Samples per class:")
    for class_name, counts in per_class.items():
        print(
            f"  {class_name}: "
            f"train={counts['train']}, "
            f"val={counts['val']}, "
            f"test={counts['test']}"
        )

    print()
    print(f"Fingerprint: {fingerprint}")
    print(f"Split files saved to: {OUTPUT_DIR}")
    print("Dataset split passed.")


if __name__ == "__main__":
    main()