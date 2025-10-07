import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple
from scipy import interpolate

from ..constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEQUENCE_LENGTH,
    GESTURE_MODEL_METADATA_PATH,
    GESTURE_MODEL_PATH,
    LANDMARKS_DIR,
    MODELS_TRAINED_DIR,
)


def get_model_path(skip_pattern: int):
    """Get model file path for given skip pattern"""
    skip_names = {0: "0_skip", 1: "1_skip", 2: "2_skip"}
    if skip_pattern not in skip_names:
        return GESTURE_MODEL_PATH
    suffix = skip_names[skip_pattern]
    # Use Path operations instead of string replace
    return (
        GESTURE_MODEL_PATH.parent
        / f"{GESTURE_MODEL_PATH.stem}_{suffix}{GESTURE_MODEL_PATH.suffix}"
    )


def get_metadata_path(skip_pattern: int):
    """Get metadata file path for given skip pattern"""
    skip_names = {0: "no_skip", 1: "1_skip", 2: "2_skip"}
    if skip_pattern not in skip_names:
        return GESTURE_MODEL_METADATA_PATH
    suffix = skip_names[skip_pattern]
    return (
        GESTURE_MODEL_METADATA_PATH.parent
        / f"{GESTURE_MODEL_METADATA_PATH.stem}_{suffix}{GESTURE_MODEL_METADATA_PATH.suffix}"
    )


torch.manual_seed(69)
np.random.seed(69)


class SignLanguageDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LandmarkDataLoader:
    def __init__(self, landmarks_dir: str):
        self.landmarks_dir = landmarks_dir
        self.feature_info = None

    def extract_features_from_frame(
        self, frame_data: Dict, feature_info: Dict
    ) -> np.ndarray:
        features = []

        # Hand landmarks
        if feature_info["hand_landmarks"] > 0:
            max_hands = feature_info["max_hands"]
            hand_dim_per_hand = feature_info["hand_landmarks_per_hand"]
            hand_features = np.zeros(feature_info["hand_landmarks"])

            if frame_data.get("hands"):
                for i, hand_data in enumerate(frame_data["hands"][:max_hands]):
                    if hand_data and "landmarks" in hand_data:
                        landmarks = np.array(hand_data["landmarks"][:21])[
                            :, :3
                        ].flatten()
                        start_idx = i * hand_dim_per_hand
                        end_idx = start_idx + len(landmarks)
                        hand_features[start_idx:end_idx] = landmarks
            features.extend(hand_features)

        # Pose landmarks
        if feature_info["pose_landmarks"] > 0:
            pose_features = np.zeros(feature_info["pose_landmarks"])
            if frame_data.get("pose") and frame_data["pose"]:
                landmarks = np.array(frame_data["pose"]["landmarks"]).flatten()
                pose_features[: min(len(landmarks), len(pose_features))] = landmarks[
                    : len(pose_features)
                ]
            features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def interpolate_missing_frames(
        self, sequence: List[np.ndarray], skip_indices: List[int]
    ) -> List[np.ndarray]:
        """
        Interpolate missing frames using cubic spline interpolation
        """
        if not skip_indices or len(sequence) < 3:
            return sequence

        sequence_array = np.array(sequence)
        interpolated_sequence = sequence_array.copy()

        # Get available frame indices (non-skipped frames)
        all_indices = list(range(len(sequence)))
        available_indices = [i for i in all_indices if i not in skip_indices]

        if len(available_indices) < 2:
            # Not enough frames for interpolation, return original
            return sequence

        # Interpolate each feature dimension
        for feature_dim in range(sequence_array.shape[1]):
            available_values = sequence_array[available_indices, feature_dim]

            # Handle edge cases where all available values are the same
            if np.all(available_values == available_values[0]):
                interpolated_sequence[skip_indices, feature_dim] = available_values[0]
                continue

            try:
                # Use cubic spline interpolation
                if len(available_indices) >= 4:
                    interp_func = interpolate.interp1d(
                        available_indices,
                        available_values,
                        kind="cubic",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                else:
                    # Use linear interpolation for fewer points
                    interp_func = interpolate.interp1d(
                        available_indices,
                        available_values,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )

                # Interpolate missing values
                interpolated_values = interp_func(skip_indices)
                interpolated_sequence[skip_indices, feature_dim] = interpolated_values

            except Exception:
                # Fallback to nearest neighbor if interpolation fails
                for skip_idx in skip_indices:
                    nearest_available = min(
                        available_indices, key=lambda x: abs(x - skip_idx)
                    )
                    interpolated_sequence[skip_idx, feature_dim] = sequence_array[
                        nearest_available, feature_dim
                    ]

        return [interpolated_sequence[i] for i in range(len(sequence))]

    def apply_frame_skipping(
        self, sequence: List[np.ndarray], skip_pattern: int
    ) -> List[np.ndarray]:
        """
        Apply frame skipping pattern and interpolate missing frames
        skip_pattern: 0 = no skip, 1 = skip every other frame, 2 = skip 2 out of every 3 frames
        """
        if skip_pattern == 0 or len(sequence) < 3:
            return sequence

        skip_indices = []

        if skip_pattern == 1:
            # Skip every other frame (1, 3, 5, ...)
            skip_indices = list(range(1, len(sequence), 2))
        elif skip_pattern == 2:
            # Skip 2 out of every 3 frames (1, 2, 4, 5, 7, 8, ...)
            for i in range(len(sequence)):
                if i % 3 != 0:  # Keep frames at positions 0, 3, 6, 9, ...
                    skip_indices.append(i)

        # Create sequence with missing frames (set to zeros)
        modified_sequence = []
        for i, frame in enumerate(sequence):
            if i in skip_indices:
                modified_sequence.append(np.zeros_like(frame))
            else:
                modified_sequence.append(frame)

        # Interpolate missing frames
        return self.interpolate_missing_frames(modified_sequence, skip_indices)

    def pad_or_truncate_sequence(
        self, sequence: List, target_length: int, feature_size: int
    ) -> np.ndarray:
        if len(sequence) > target_length:
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            padded = list(sequence)
            while len(padded) < target_length:
                padded.append(np.zeros(feature_size))
            return np.array(padded)

    def load_data(
        self, sequence_length: int = 30, skip_pattern: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        all_sequences = []
        all_labels = []
        class_names = []

        print(
            f"Loading data from {self.landmarks_dir} with skip pattern {skip_pattern}"
        )

        for class_folder in sorted(os.listdir(self.landmarks_dir)):
            class_path = os.path.join(self.landmarks_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            class_names.append(class_folder)
            pickle_files = [f for f in os.listdir(class_path) if f.endswith(".pkl")]

            for pickle_file in tqdm(pickle_files, desc=f"Loading {class_folder}"):
                pickle_path = os.path.join(class_path, pickle_file)

                try:
                    with open(pickle_path, "rb") as f:
                        landmark_data = pickle.load(f)

                    # Get feature info from first file
                    if self.feature_info is None:
                        self.feature_info = landmark_data.get("feature_info", {})
                        print(f"Feature info: {self.feature_info}")

                    frame_features = []
                    for frame in landmark_data.get("frames", []):
                        features = self.extract_features_from_frame(
                            frame, self.feature_info
                        )
                        frame_features.append(features)

                    # Apply frame skipping and interpolation
                    if skip_pattern > 0:
                        frame_features = self.apply_frame_skipping(
                            frame_features, skip_pattern
                        )

                    padded_sequence = self.pad_or_truncate_sequence(
                        frame_features,
                        sequence_length,
                        self.feature_info["total_features"],
                    )
                    all_sequences.append(padded_sequence)
                    all_labels.append(len(class_names) - 1)

                except Exception as e:
                    print(f"Error loading {pickle_path}: {e}")

        print(f"Loaded {len(all_sequences)} sequences from {len(class_names)} classes")
        return (
            np.array(all_sequences),
            np.array(all_labels),
            class_names,
            self.feature_info,
        )


class GestureRecognizerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(
        self, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_features, batch_labels in tqdm(dataloader, desc="Training"):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def validate(
        self, dataloader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_features, batch_labels in tqdm(dataloader, desc="Validating"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 0.001,
        model_path = GESTURE_MODEL_PATH,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5
        )

        best_val_acc = 0
        patience_counter = 0
        early_stop_patience = 20

        print(f"Starting training on {self.device}")

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)

            scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "train_accuracy": train_acc,
                    },
                    str(model_path),
                )
                print(
                    f"New best model saved with validation accuracy: {best_val_acc:.4f}"
                )
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    def plot_training_history(self, skip_pattern: int = 0):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.train_losses, label="Training Loss")
        ax1.plot(self.val_losses, label="Validation Loss")
        ax1.set_title(f"Model Loss (Skip Pattern {skip_pattern})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accuracies, label="Training Accuracy")
        ax2.plot(self.val_accuracies, label="Validation Accuracy")
        ax2.set_title(f"Model Accuracy (Skip Pattern {skip_pattern})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                MODELS_TRAINED_DIR, f"training_history_skip_{skip_pattern}.png"
            )
        )
        plt.show()


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str,
    skip_pattern: int = 0,
):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in tqdm(test_loader, desc="Evaluating"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy (Skip Pattern {skip_pattern}): {accuracy:.4f}")

    report = classification_report(
        all_labels, all_predictions, target_names=class_names, zero_division=0
    )
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix (Skip Pattern {skip_pattern})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODELS_TRAINED_DIR, f"confusion_matrix_skip_{skip_pattern}.png")
    )
    plt.show()


def train_single_model(args, skip_pattern: int):
    """Train a single model with specified skip pattern"""
    print(f"\n{'='*60}")
    print(f"Training model with skip pattern {skip_pattern}")
    print(f"{'='*60}")

    # Define paths
    model_path = get_model_path(skip_pattern)
    metadata_path = get_metadata_path(skip_pattern)

    print("Loading landmark data...")
    data_loader = LandmarkDataLoader(args.landmarks_dir)
    sequences, labels, class_names, feature_info = data_loader.load_data(
        args.sequence_length, skip_pattern
    )

    if len(sequences) == 0:
        print("No data loaded. Please check the landmarks directory.")
        return

    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of classes: {len(class_names)}")
    print(f"Feature size: {feature_info['total_features']}")

    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    test_dataset = SignLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = GestureRecognizerModel(
        input_size=feature_info["total_features"],
        num_classes=len(class_names),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )

    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader, args.epochs, args.learning_rate, model_path)
    trainer.plot_training_history(skip_pattern)

    # Load best model for evaluation
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(model, test_loader, class_names, trainer.device, skip_pattern)

    # Save model metadata
    model_info = {
        "class_names": class_names,
        "input_size": feature_info["total_features"],
        "feature_info": feature_info,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "skip_pattern": skip_pattern,
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(model_info, f)

    print(f"Model with skip pattern {skip_pattern} completed successfully!")
    return model_info


def main():
    parser = argparse.ArgumentParser(
        description="Train sign language recognition models with frame skipping"
    )
    parser.add_argument(
        "--landmarks_dir",
        default=LANDMARKS_DIR,
        help="Directory containing landmark data",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Sequence length for model input",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Hidden size for model",
    )
    parser.add_argument(
        "--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate"
    )
    parser.add_argument(
        "--skip_patterns",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Skip patterns to train (0=no skip, 1=skip every other, 2=skip 2 of 3)",
    )

    args = parser.parse_args()

    os.makedirs(MODELS_TRAINED_DIR, exist_ok=True)

    model_infos = {}

    # Train models for each skip pattern
    for skip_pattern in args.skip_patterns:
        if skip_pattern not in [0, 1, 2]:
            print(f"Warning: Skip pattern {skip_pattern} not supported. Skipping.")
            continue

        model_info = train_single_model(args, skip_pattern)
        if model_info:
            model_infos[skip_pattern] = model_info

    print("\n" + "=" * 60)
    print("All models training completed!")
    print("=" * 60)

    # Print summary
    for skip_pattern, info in model_infos.items():
        skip_names = {0: "0_skip", 1: "1_skip", 2: "2_skip"}
        print(f"✓ Model {skip_names[skip_pattern]}: {len(info['class_names'])} classes")


if __name__ == "__main__":
    main()
