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

from constants import (
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
        self, sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        all_sequences = []
        all_labels = []
        class_names = []

        print(f"Loading data from {self.landmarks_dir}")

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

                    if len(frame_features) >= 5:
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


class SignLanguageModel(nn.Module):
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
                    GESTURE_MODEL_PATH,
                )
                print(
                    f"New best model saved with validation accuracy: {best_val_acc:.4f}"
                )
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.train_losses, label="Training Loss")
        ax1.plot(self.val_losses, label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accuracies, label="Training Accuracy")
        ax2.plot(self.val_accuracies, label="Validation Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_TRAINED_DIR, "training_history.png"))
        plt.show()


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, class_names: List[str], device: str
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
    print(f"Test Accuracy: {accuracy:.4f}")

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
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_TRAINED_DIR, "confusion_matrix.png"))
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Train sign language recognition model"
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

    args = parser.parse_args()

    os.makedirs(MODELS_TRAINED_DIR, exist_ok=True)

    print("Loading landmark data...")
    data_loader = LandmarkDataLoader(args.landmarks_dir)
    sequences, labels, class_names, feature_info = data_loader.load_data(
        args.sequence_length
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

    model = SignLanguageModel(
        input_size=feature_info["total_features"],
        num_classes=len(class_names),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )

    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader, args.epochs, args.learning_rate)
    trainer.plot_training_history()

    checkpoint = torch.load(GESTURE_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(model, test_loader, class_names, trainer.device)

    model_info = {
        "class_names": class_names,
        "input_size": feature_info["total_features"],
        "feature_info": feature_info,
        "sequence_length": args.sequence_length,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
    }

    with open(GESTURE_MODEL_METADATA_PATH, "wb") as f:
        pickle.dump(model_info, f)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
