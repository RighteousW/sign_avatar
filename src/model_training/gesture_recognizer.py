import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    GESTURE_MODEL_HANDS_ONLY_DIR,
    GESTURE_MODEL_HANDS_POSE_DIR,
    LANDMARKS_DIR_HANDS_ONLY,
    LANDMARKS_DIR_HANDS_POSE,
    get_gesture_metadata_path,
    get_gesture_model_path,
)
from ..utils.interpolation import apply_frame_skipping

torch.manual_seed(42)
np.random.seed(42)


def get_path(
    skip_pattern: int, use_pose: bool, is_metadata: bool, model_type: str = "lstm"
):
    """Get model or metadata file path for given skip pattern"""
    base_path = (
        get_gesture_metadata_path(use_pose, skip_pattern, model_type)
        if is_metadata
        else get_gesture_model_path(use_pose, skip_pattern, model_type)
    )
    return base_path


def save_training_metrics(
    metrics: Dict, output_dir: str, skip_pattern: int, model_type: str = "cnn"
):
    """Save training metrics to JSON file"""
    suffix = f"_{model_type}" if model_type != "cnn" else ""
    path = os.path.join(
        output_dir, f"training_metrics_skip_{skip_pattern}{suffix}.json"
    )
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {path}")


def save_evaluation_metrics(
    metrics: Dict, output_dir: str, skip_pattern: int, model_type: str = "cnn"
):
    """Save evaluation metrics to JSON and confusion matrix to text"""
    suffix = f"_{model_type}" if model_type != "cnn" else ""
    json_path = os.path.join(
        output_dir, f"evaluation_metrics_skip_{skip_pattern}{suffix}.json"
    )
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]
    txt_path = os.path.join(
        output_dir, f"confusion_matrix_skip_{skip_pattern}{suffix}.txt"
    )

    with open(txt_path, "w") as f:
        f.write(
            f"Confusion Matrix ({model_type.upper()} - Skip Pattern {skip_pattern})\n{'='*80}\n\n"
        )
        f.write(
            "True\\Pred".ljust(15)
            + "".join(n[:10].ljust(12) for n in class_names)
            + "\n"
            + "-" * 80
            + "\n"
        )
        for i, name in enumerate(class_names):
            f.write(
                name[:10].ljust(15) + "".join(str(v).ljust(12) for v in cm[i]) + "\n"
            )

    print(f"Evaluation metrics saved to {json_path} and {txt_path}")


class SignLanguageDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LandmarkDataLoader:
    def __init__(self, use_pose: bool):
        self.landmarks_dir = (
            LANDMARKS_DIR_HANDS_POSE if use_pose else LANDMARKS_DIR_HANDS_ONLY
        )
        self.feature_info = None

    def extract_features_from_frame(
        self, frame_data: Dict, feature_info: Dict
    ) -> np.ndarray:
        features = []

        # Hand landmarks
        if feature_info["hand_landmarks"] > 0:
            hand_features = np.zeros(feature_info["hand_landmarks"])
            if frame_data.get("hands"):
                for i, hand_data in enumerate(
                    frame_data["hands"][: feature_info["max_hands"]]
                ):
                    if hand_data and "landmarks" in hand_data:
                        landmarks = np.array(hand_data["landmarks"][:21])[
                            :, :3
                        ].flatten()
                        start_idx = i * feature_info["hand_landmarks_per_hand"]
                        hand_features[start_idx : start_idx + len(landmarks)] = (
                            landmarks
                        )
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

    def pad_or_truncate_sequence(
        self, sequence: List, target_length: int, feature_size: int
    ) -> np.ndarray:
        if len(sequence) > target_length:
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        return np.array(
            sequence + [np.zeros(feature_size)] * (target_length - len(sequence))
        )

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
                try:
                    with open(os.path.join(class_path, pickle_file), "rb") as f:
                        landmark_data = pickle.load(f)

                    if self.feature_info is None:
                        self.feature_info = landmark_data.get("feature_info", {})
                        print(f"Feature info: {self.feature_info}")

                    frame_features = [
                        self.extract_features_from_frame(frame, self.feature_info)
                        for frame in landmark_data.get("frames", [])
                    ]

                    if skip_pattern > 0:
                        frame_features = apply_frame_skipping(
                            frame_features, skip_pattern
                        )

                    padded = self.pad_or_truncate_sequence(
                        frame_features,
                        sequence_length,
                        self.feature_info["total_features"],
                    )
                    all_sequences.append(padded)
                    all_labels.append(len(class_names) - 1)
                except Exception as e:
                    print(f"Error loading {os.path.join(class_path, pickle_file)}: {e}")

        print(f"Loaded {len(all_sequences)} sequences from {len(class_names)} classes")
        return (
            np.array(all_sequences),
            np.array(all_labels),
            class_names,
            self.feature_info,
        )


class GestureRecognizerLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from both directions
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        last_hidden = torch.cat(
            [h_n[-2], h_n[-1]], dim=1
        )  # Concatenate forward and backward
        return self.classifier(last_hidden)


class GestureRecognizerCNN(nn.Module):
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
        return self.classifier(x)


class GestureRecognizerScaleCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 5 different temporal scales for capturing variations
        # Scale 1: Single frame - instantaneous pose
        self.conv_tiny = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        # Scale 2: 3 frames - very fast micro-movements
        self.conv_small = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        # Scale 3: 5 frames - fast hand transitions
        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        # Scale 4: 7 frames - medium gesture phases
        self.conv_large = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        # Scale 5: 11 frames - slow, full gesture context
        self.conv_xlarge = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

        # Combine all 5 scales (5 * hidden_size channels)
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_size * 5, hidden_size * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 4, hidden_size, kernel_size=1),
            nn.Sigmoid(),
        )

        # Dual pooling for robustness
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        x = x.transpose(1, 2)  # -> (batch, input_size, sequence_length)

        # Extract features at all 5 temporal scales
        feat_tiny = self.conv_tiny(x)  # Instantaneous pose
        feat_small = self.conv_small(x)  # Very fast movements
        feat_medium = self.conv_medium(x)  # Fast transitions
        feat_large = self.conv_large(x)  # Medium phases
        feat_xlarge = self.conv_xlarge(x)  # Full gesture context

        # Concatenate all scales along channel dimension
        x = torch.cat(
            [feat_tiny, feat_small, feat_medium, feat_large, feat_xlarge], dim=1
        )

        # Fuse multi-scale features
        x = self.fusion(x)

        # Apply attention to focus on important temporal regions
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Dual pooling (captures both average and peak activations)
        avg_pool = self.global_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Classification
        return self.classifier(x)


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.history = {
            "epochs": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def run_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        is_training: bool,
    ) -> Tuple[float, float]:
        """Run single epoch for training or validation"""
        self.model.train() if is_training else self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for features, labels in tqdm(
                dataloader, desc="Training" if is_training else "Validating"
            ):
                features, labels = features.to(self.device), labels.to(self.device)

                if is_training:
                    optimizer.zero_grad()

                outputs = self.model(features)
                loss = criterion(outputs, labels)

                if is_training:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        model_path,
        output_dir: str,
        skip_pattern: int,
        model_type: str = "lstm",
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

        print(f"Starting training on {self.device}")

        for epoch in range(num_epochs):
            train_loss, train_acc = self.run_epoch(
                train_loader, optimizer, criterion, True
            )
            val_loss, val_acc = self.run_epoch(val_loader, optimizer, criterion, False)
            scheduler.step(val_loss)

            self.history["epochs"].append(epoch + 1)
            self.history["train_loss"].append(float(train_loss))
            self.history["train_acc"].append(float(train_acc))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_acc"].append(float(val_acc))

            print(
                f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

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
                print(f"✓ Best model saved: val_acc={best_val_acc:.4f}\n")
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        save_training_metrics(self.history, output_dir, skip_pattern, model_type)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str,
    output_dir: str,
    skip_pattern: int,
    model_type: str = "cnn",  # ADD THIS
):
    """Evaluate model and save metrics"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(
        f"Test Accuracy ({model_type.upper()} - Skip Pattern {skip_pattern}): {accuracy:.4f}"
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_predictions)

    metrics = {
        "model_type": model_type,
        "skip_pattern": skip_pattern,
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    save_evaluation_metrics(
        metrics, output_dir, skip_pattern, model_type
    )


def train_single_model(args, skip_pattern: int, model_type: str = "lstm"):
    """Train a single model with specified skip pattern"""
    print(f"\n{'='*60}\nTraining model with skip pattern {skip_pattern}\n{'='*60}")

    model_path = get_path(
        skip_pattern, args.use_pose, is_metadata=False, model_type=model_type
    )
    metadata_path = get_path(
        skip_pattern, args.use_pose, is_metadata=True, model_type=model_type
    )
    model_dir = (
        GESTURE_MODEL_HANDS_POSE_DIR if args.use_pose else GESTURE_MODEL_HANDS_ONLY_DIR
    )

    data_loader = LandmarkDataLoader(args.use_pose)
    sequences, labels, class_names, feature_info = data_loader.load_data(
        args.sequence_length, skip_pattern
    )

    if len(sequences) == 0:
        print("No data loaded. Check landmarks directory.")
        return None

    # Split data: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(
        f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, "
        f"Classes: {len(class_names)}, Features: {feature_info['total_features']}"
    )

    loaders = {
        "train": DataLoader(
            SignLanguageDataset(X_train, y_train),
            batch_size=args.batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            SignLanguageDataset(X_val, y_val), batch_size=args.batch_size
        ),
        "test": DataLoader(
            SignLanguageDataset(X_test, y_test), batch_size=args.batch_size
        ),
    }

    if model_type == "lstm":
        model = GestureRecognizerLSTM(
            feature_info["total_features"],
            len(class_names),
            args.hidden_size,
            args.dropout,
        )
    else:
        model = GestureRecognizerCNN(
            feature_info["total_features"],
            len(class_names),
            args.hidden_size,
            args.dropout,
        )

    trainer = ModelTrainer(model)
    trainer.train(
        loaders["train"],
        loaders["val"],
        args.epochs,
        args.learning_rate,
        model_path,
        model_dir,
        skip_pattern,
        model_type,
    )

    # Load best model and evaluate
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate_model(
        model,
        loaders["test"],
        class_names,
        trainer.device,
        model_dir,
        skip_pattern,
        model_type,
    )

    # Save metadata (format preserved for compatibility)
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

    print(f"✓ Model with skip pattern {skip_pattern} completed!")
    return model_info


def main():
    parser = argparse.ArgumentParser(
        description="Train sign language recognition models"
    )
    parser.add_argument("--use_pose", default=False, help="Use pose landmarks")
    parser.add_argument("--sequence_length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--skip_patterns", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--model_type", type=str, default="scale_cnn", choices=["cnn", "scale_cnn", "lstm"]
    )
    args = parser.parse_args()

    model_infos = {}
    for skip_pattern in args.skip_patterns:
        if skip_pattern in [0, 1, 2]:
            model_info = train_single_model(args, skip_pattern, args.model_type)
            if model_info:
                model_infos[skip_pattern] = model_info

    print(f"\n{'='*60}\nTraining completed!\n{'='*60}")
    for skip_pattern, info in model_infos.items():
        print(f"✓ Skip {skip_pattern}: {len(info['class_names'])} classes")


if __name__ == "__main__":
    main()
