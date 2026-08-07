import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "IMAGES"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
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

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

RANDOM_SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 7


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NEUClassificationDataset(Dataset):
    def __init__(
        self,
        split_name: str,
        transform,
    ) -> None:
        split_path = SPLIT_DIR / f"{split_name}.txt"

        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )

        self.sample_names = [
            line.strip()
            for line in split_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, index: int):
        sample_name = self.sample_names[index]
        class_name = sample_name.rsplit("_", maxsplit=1)[0]

        if class_name not in CLASS_TO_INDEX:
            raise ValueError(f"Unknown class: {class_name}")

        image_path = IMAGE_DIR / f"{sample_name}.jpg"

        with Image.open(image_path) as image:
            image = image.convert("L")
            image = self.transform(image)

        label = CLASS_TO_INDEX[class_name]
        return image, label


class SmallCNN(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def run_epoch(
    model,
    data_loader,
    loss_function,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            predictions == labels
        ).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def main() -> None:
    set_random_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("NEU-DET CNN Classification Training")
    print("=" * 60)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    train_dataset = NEUClassificationDataset(
        "train",
        train_transform,
    )
    validation_dataset = NEUClassificationDataset(
        "val",
        validation_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    model = SmallCNN(
        number_of_classes=len(CLASS_NAMES)
    ).to(device)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {number_of_parameters:,}")

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )

    history = []
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0

    model_path = MODEL_DIR / "best_small_cnn.pt"

    print()
    print("Starting training...")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "learning_rate": current_learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        improved = validation_accuracy > best_validation_accuracy

        if improved:
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "best_epoch": epoch,
                    "best_validation_accuracy": (
                        best_validation_accuracy
                    ),
                    "random_seed": RANDOM_SEED,
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1

        marker = "  <-- best" if improved else ""

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"LR {current_learning_rate:.6f} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_accuracy:.4f} | "
            f"Val loss {validation_loss:.4f} | "
            f"Val acc {validation_accuracy:.4f}"
            f"{marker}"
        )

        scheduler.step()

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping: validation accuracy "
                "did not improve."
            )
            break

    history_path = METRICS_DIR / "cnn_training_history.csv"

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    validation_losses = [
        row["validation_loss"] for row in history
    ]
    train_accuracies = [
        row["train_accuracy"] for row in history
    ]
    validation_accuracies = [
        row["validation_accuracy"] for row in history
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epochs,
        train_losses,
        marker="o",
        label="Training",
    )
    axes[0].plot(
        epochs,
        validation_losses,
        marker="o",
        label="Validation",
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("CNN Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_accuracies,
        marker="o",
        label="Training",
    )
    axes[1].plot(
        epochs,
        validation_accuracies,
        marker="o",
        label="Validation",
    )
    axes[1].axhline(
        y=0.662963,
        color="gray",
        linestyle="--",
        label="HOG + SVM baseline",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CNN Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()

    figure_path = FIGURE_DIR / "cnn_training_curves.png"
    figure.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    best_row = max(
        history,
        key=lambda row: row["validation_accuracy"],
    )

    print()
    print("=" * 60)
    print("Training completed")
    print(f"Best epoch: {best_row['epoch']}")
    print(
        "Best validation accuracy: "
        f"{best_row['validation_accuracy']:.4f}"
    )
    print(f"Best model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Training curves saved to: {figure_path}")
    print("The test split has not been evaluated.")


if __name__ == "__main__":
    main()