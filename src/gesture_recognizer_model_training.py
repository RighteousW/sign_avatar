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
import logging

from constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEQUENCE_LENGTH,
    LANDMARKS_DIR,
    MODELS_TRAINED_DIR,
)
import mediapipe as mp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureConfig:
    """Configuration class to control which features to extract and define their dimensions.
    Updated to use only hand and pose landmarks."""

    def __init__(
        self,
        use_hand_landmarks: bool = True,
        use_pose_landmarks: bool = True,
        max_hands: int = 2,
    ):
        self.use_hand_landmarks = use_hand_landmarks
        # Set hand connections to False as they are no longer used
        self.use_hand_connections = False
        self.use_pose_landmarks = use_pose_landmarks
        # Set pose connections to False as they are no longer used
        self.use_pose_connections = False
        self.max_hands = max_hands

        # Define dimensions for a single hand
        # 21 landmarks * 3 coords (x, y, z) + 2 new distance features
        self.hand_landmarks_dim_per_hand = (21 * 3) + 2
        # Hand connections are removed
        self.hand_connections_dim_per_hand = 0

        # Calculate total hand feature dimensions based on max_hands
        self.hand_landmarks_dim = (
            self.hand_landmarks_dim_per_hand * self.max_hands
            if use_hand_landmarks
            else 0
        )
        self.hand_connections_dim = 0

        # Pose dimensions
        # 16 specific landmarks * 4 coordinates (x, y, z, visibility)
        self.pose_landmarks_dim = 16 * 4 if use_pose_landmarks else 0
        # Pose connections are removed
        self.pose_connections_dim = 0

        # Calculate the total feature size
        self.feature_size = self.hand_landmarks_dim + self.pose_landmarks_dim

    def __str__(self):
        features = []
        if self.use_hand_landmarks:
            features.append(f"hand_landmarks({self.hand_landmarks_dim})")
        if self.use_pose_landmarks:
            features.append(f"pose_landmarks({self.pose_landmarks_dim})")
        return f"FeatureConfig(total={self.feature_size}): {', '.join(features)}"


class SignLanguageDataset(Dataset):
    """Dataset class for sign language landmarks"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LandmarkDataLoader:
    """Enhanced data loader with configurable feature extraction.
    Updated to remove connection feature logic."""

    def __init__(self, landmarks_dir: str, feature_config: FeatureConfig):
        self.landmarks_dir = landmarks_dir
        self.feature_config = feature_config
        self.pose_landmark_indices = [
            0,
            2,
            5,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ]
        self.pose_landmark_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ]
        self.pose_landmark_indices = [
            0,
            2,
            5,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ]
        # These are the specific indices requested by the user
        self.pose_landmark_indices = [
            0,
            2,
            5,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ]

    def _calculate_distance(self, p1, p2):
        """Calculates Euclidean distance between two 3D points."""
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    def pad_or_truncate_sequence(
        self, sequence: List, target_length: int
    ) -> np.ndarray:
        """Pad or truncate sequence to target length"""
        if len(sequence) > target_length:
            # Uniformly sample frames
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            # Pad with zeros
            padded = list(sequence)
            last_frame = np.zeros(self.feature_config.feature_size)
            while len(padded) < target_length:
                padded.append(
                    last_frame.copy()
                    if hasattr(last_frame, "copy")
                    else np.copy(last_frame)
                )
            return np.array(padded)

    def extract_features_from_frame(self, frame_data: Dict) -> np.ndarray:
        """Extract features from a single frame based on configuration"""
        features = []

        # Hand landmarks
        for hand_idx in range(self.feature_config.max_hands):
            hand_features = np.zeros(self.feature_config.hand_landmarks_dim_per_hand)
            if (
                frame_data.get("hands")
                and hand_idx < len(frame_data["hands"])
                and frame_data["hands"][hand_idx]
            ):
                hand_data = frame_data["hands"][hand_idx]
                landmarks = hand_data.get("landmarks", [])

                if (
                    self.feature_config.use_hand_landmarks
                    and isinstance(landmarks, list)
                    and len(landmarks) >= 21
                ):
                    # Flatten the coordinates of all 21 landmarks
                    flat_landmarks = np.array(landmarks[:21])[:, :3].flatten()
                    hand_features[: len(flat_landmarks)] = flat_landmarks

                    # Calculate new distance features and append
                    thumb_tip = landmarks[4]
                    ring_joint = landmarks[15]
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]

                    dist_thumb_ring = self._calculate_distance(thumb_tip, ring_joint)
                    dist_index_middle = self._calculate_distance(index_tip, middle_tip)

                    hand_features[len(flat_landmarks)] = dist_thumb_ring
                    hand_features[len(flat_landmarks) + 1] = dist_index_middle
            features.extend(hand_features)

        # Pose landmarks
        if self.feature_config.use_pose_landmarks:
            pose_features = np.zeros(self.feature_config.pose_landmarks_dim)
            if frame_data.get("pose") and "landmarks" in frame_data["pose"]:
                all_pose_landmarks = frame_data["pose"]["landmarks"]

                # Filter for the specific pose landmarks
                selected_landmarks = []
                for idx in self.pose_landmark_indices:
                    if idx < len(all_pose_landmarks):
                        selected_landmarks.append(all_pose_landmarks[idx])

                flat_pose_landmarks = np.array(selected_landmarks).flatten()

                pose_features[: min(len(flat_pose_landmarks), len(pose_features))] = (
                    flat_pose_landmarks[: len(pose_features)]
                )
            features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def load_data(
        self, sequence_length: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load all landmark data and prepare for training"""
        all_sequences = []
        all_labels = []
        class_names = []

        logger.info(f"Loading data from {self.landmarks_dir}")
        logger.info(f"Feature configuration: {self.feature_config}")

        for class_folder in sorted(os.listdir(self.landmarks_dir)):
            class_path = os.path.join(self.landmarks_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            class_names.append(class_folder)
            logger.info(f"Processing class: {class_folder}")

            pickle_files = [
                f
                for f in os.listdir(class_path)
                if f.endswith(".pkl") or f.endswith("_landmarks.pkl")
            ]

            for pickle_file in tqdm(pickle_files, desc=f"Loading {class_folder}"):
                pickle_path = os.path.join(class_path, pickle_file)

                try:
                    with open(pickle_path, "rb") as f:
                        landmark_data = pickle.load(f)

                    frame_features = []
                    for frame in landmark_data.get("frames", []):
                        features = self.extract_features_from_frame(frame)
                        frame_features.append(features)

                    if len(frame_features) >= 5:
                        padded_sequence = self.pad_or_truncate_sequence(
                            frame_features, sequence_length
                        )
                        all_sequences.append(padded_sequence)
                        all_labels.append(len(class_names) - 1)

                except Exception as e:
                    logger.warning(f"Error loading {pickle_path}: {e}")

        logger.info(
            f"Loaded {len(all_sequences)} sequences from {len(class_names)} classes"
        )
        logger.info(f"Feature size: {self.feature_config.feature_size}")
        logger.info(f"Sequence length: {sequence_length}")

        return np.array(all_sequences), np.array(all_labels), class_names


class SignLanguageModel(nn.Module):
    """Sign language model architecture using a Conv1D model."""

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

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Temporal convolution
        x = self.temporal_conv(x)

        # Global pooling
        x = self.global_pool(x)  # (batch, hidden_size, 1)
        x = x.squeeze(-1)  # (batch, hidden_size)

        # Classification
        x = self.classifier(x)
        return x


class ModelTrainer:
    """Training pipeline for sign language models"""

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
        """Train for one epoch"""
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
        """Validate the model"""
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
        weight_decay: float = 1e-5,
    ) -> None:
        """Full training loop"""

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5
        )

        best_val_acc = 0
        patience_counter = 0
        early_stop_patience = 20

        logger.info(f"Starting training on {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)

            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_acc,
                        "train_accuracy": train_acc,
                    },
                    os.path.join(MODELS_TRAINED_DIR, "best_model.pth"),
                )
                logger.info(
                    f"New best model saved with validation accuracy: {best_val_acc:.4f}"
                )
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.train_losses, label="Training Loss")
        ax1.plot(self.val_losses, label="Validation Loss")
        ax1.set_title("Model Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
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
) -> None:
    """Evaluate model and generate classification report"""
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

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(
        all_labels, all_predictions, target_names=class_names, zero_division=0
    )
    logger.info("Classification Report:")
    logger.info(report)

    # Confusion matrix
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

    # Feature selection arguments - simplified
    parser.add_argument(
        "--max_hands", type=int, default=2, help="Maximum number of hands to track"
    )

    args = parser.parse_args()

    # Create feature configuration - only hand and pose landmarks
    feature_config = FeatureConfig(
        use_hand_landmarks=True,
        use_pose_landmarks=True,
        max_hands=args.max_hands,
    )

    # Create model save directory
    os.makedirs(MODELS_TRAINED_DIR, exist_ok=True)

    # Load data
    logger.info("Loading landmark data...")
    data_loader = LandmarkDataLoader(args.landmarks_dir, feature_config)
    sequences, labels, class_names = data_loader.load_data(args.sequence_length)

    if len(sequences) == 0:
        logger.error("No data loaded. Please check the landmarks directory.")
        return

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Number of classes: {len(class_names)}")
    logger.info(f"Classes: {sorted(class_names)}")

    # Create datasets and dataloaders
    train_dataset = SignLanguageDataset(X_train, y_train)
    val_dataset = SignLanguageDataset(X_val, y_val)
    test_dataset = SignLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    input_size = feature_config.feature_size
    num_classes = len(class_names)

    model = SignLanguageModel(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )

    # Train model
    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader, args.epochs, args.learning_rate)

    # Plot training history
    trainer.plot_training_history()

    # Load best model for evaluation
    checkpoint = torch.load(os.path.join(MODELS_TRAINED_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set
    evaluate_model(model, test_loader, class_names, trainer.device)

    # Save model info with feature configuration
    model_info = {
        "class_names": class_names,
        "input_size": input_size,
        "model_type": "conv1d",
        "sequence_length": args.sequence_length,
        "feature_size": feature_config.feature_size,
        "feature_config": {
            "use_hand_landmarks": feature_config.use_hand_landmarks,
            "use_hand_connections": feature_config.use_hand_connections,
            "use_pose_landmarks": feature_config.use_pose_landmarks,
            "use_pose_connections": feature_config.use_pose_connections,
            "max_hands": feature_config.max_hands,
        },
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
    }

    with open(os.path.join(MODELS_TRAINED_DIR, "model_info.pkl"), "wb") as f:
        pickle.dump(model_info, f)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
