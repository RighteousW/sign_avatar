import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import random

from constants import LANDMARKS_DIR, MODELS_TRAINED_DIR, SEQ2SEQ_CONFIG

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class LandmarkProcessor:
    """Processes MediaPipe landmark data into standardized feature vectors"""

    def __init__(self):
        self.scaler = None
        self.is_fitted = False

        # Feature dimensions based on your landmark extractor
        self.hand_landmarks_dim = 21 * 3 * 2  # 126 (max 2 hands)
        self.hand_connections_dim = 20 * 2  # 40 (20 connections per hand, max 2 hands)
        self.pose_landmarks_dim = 33 * 4  # 132 (33 landmarks with x,y,z,visibility)
        self.pose_connections_dim = 35  # 35 pose connections

        self.total_features = (
            self.hand_landmarks_dim
            + self.hand_connections_dim
            + self.pose_landmarks_dim
            + self.pose_connections_dim
        )

        print(f"Total feature dimensions: {self.total_features}")

    def extract_frame_features(self, frame_data):
        """Extract features from a single frame"""
        features = np.zeros(self.total_features, dtype=np.float32)

        # Process hand landmarks and connections
        hand_features = np.zeros(
            self.hand_landmarks_dim + self.hand_connections_dim, dtype=np.float32
        )

        for i, hand_data in enumerate(frame_data.get("hands", [])[:2]):  # Max 2 hands
            landmarks = hand_data.get("landmarks", [])
            connections = hand_data.get("connection_features", [])

            # Hand landmarks (21 * 3 = 63 features per hand)
            start_idx = i * 63
            for j, landmark in enumerate(landmarks[:21]):
                if j < 21:  # Ensure we don't exceed bounds
                    hand_features[start_idx + j * 3 : start_idx + (j + 1) * 3] = (
                        landmark[:3]
                    )

            # Hand connections (20 features per hand)
            conn_start_idx = self.hand_landmarks_dim + i * 20
            for j, conn in enumerate(connections[:20]):
                if j < 20:
                    hand_features[conn_start_idx + j] = conn

        features[: self.hand_landmarks_dim + self.hand_connections_dim] = hand_features

        # Process pose landmarks and connections
        pose_data = frame_data.get("pose")
        if pose_data:
            landmarks = pose_data.get("landmarks", [])
            connections = pose_data.get("connection_features", [])

            # Pose landmarks (33 * 4 = 132 features)
            pose_start = self.hand_landmarks_dim + self.hand_connections_dim
            for i, landmark in enumerate(landmarks[:33]):
                if i < 33:
                    features[pose_start + i * 4 : pose_start + (i + 1) * 4] = landmark[
                        :4
                    ]

            # Pose connections (35 features)
            conn_start = pose_start + self.pose_landmarks_dim
            for i, conn in enumerate(connections[:35]):
                if i < 35:
                    features[conn_start + i] = conn

        return features

    def get_all_landmark_files(self, landmarks_dir):
        """Get all landmark pickle files without loading them"""
        landmark_files = []
        landmarks_path = Path(landmarks_dir)

        if not landmarks_path.exists():
            raise ValueError(f"Landmarks directory not found: {landmarks_dir}")

        for folder in landmarks_path.iterdir():
            if folder.is_dir():
                pkl_files = list(folder.glob("*_landmarks.pkl"))
                landmark_files.extend(pkl_files)

        print(f"Found {len(landmark_files)} landmark files")
        return landmark_files

    def fit_scaler_on_sample(self, landmarks_dir, sample_size=1000):
        """Fit scaler on a sample of data to avoid loading everything"""
        landmark_files = self.get_all_landmark_files(landmarks_dir)

        # Sample files for fitting scaler
        sample_files = random.sample(
            landmark_files, min(len(landmark_files), sample_size // 50)
        )

        sample_features = []
        for pkl_file in sample_files:
            try:
                with open(pkl_file, "rb") as f:
                    landmark_data = pickle.load(f)

                # Extract features from first few frames
                for frame_data in landmark_data.get("frames", [])[
                    :20
                ]:  # Sample first 20 frames
                    frame_features = self.extract_frame_features(frame_data)
                    sample_features.append(frame_features)

                    if len(sample_features) >= sample_size:
                        break

                if len(sample_features) >= sample_size:
                    break

            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")

        if len(sample_features) > 0:
            sample_features = np.array(sample_features)
            self.scaler = StandardScaler()
            self.scaler.fit(sample_features)
            self.is_fitted = True
            print(f"Fitted scaler on {len(sample_features)} sample frames")
        else:
            raise ValueError("No valid frames found for fitting scaler")

    def normalize_features(self, features):
        """Normalize features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before normalization")

        if features.ndim == 1:
            features = features.reshape(1, -1)
            return self.scaler.transform(features).squeeze()
        else:
            return self.scaler.transform(features)


class GestureDataset(Dataset):
    """Optimized dataset for gesture gap filling"""

    def __init__(self, landmark_files, processor, config):
        self.landmark_files = landmark_files
        self.processor = processor
        self.gap_size = config["gap_size"]
        self.max_samples = config["max_samples"]
        self.total_sequence_length = self.gap_size + 2

        # Pre-load and process all data into memory for faster training
        self.samples = []
        self._load_all_samples()

        print(f"Dataset ready with {len(self.samples)} samples")

    def _load_all_samples(self):
        """Pre-load all samples for faster training"""
        print("Pre-loading dataset samples...")

        # Sample files if we have too many
        files_to_process = self.landmark_files[:2000]  # Limit files

        for file_idx, pkl_file in enumerate(files_to_process):
            if file_idx % 200 == 0:
                print(f"  Loading from file {file_idx}/{len(files_to_process)}")

            try:
                with open(pkl_file, "rb") as f:
                    landmark_data = pickle.load(f)

                frames = landmark_data.get("frames", [])

                if len(frames) < self.total_sequence_length:
                    continue

                # Convert all frames to features first
                frame_features = []
                for frame_data in frames:
                    features = self.processor.extract_frame_features(frame_data)
                    normalized = self.processor.normalize_features(features)
                    frame_features.append(normalized)

                frame_features = np.array(frame_features)

                # Create gap pairs from this file
                max_start = len(frame_features) - self.total_sequence_length

                # Sample pairs from this file
                num_pairs = min(max_start + 1, 10)  # Max 10 pairs per file
                if max_start > 0:
                    start_indices = np.random.choice(
                        max_start + 1, size=num_pairs, replace=False
                    )
                else:
                    start_indices = [0]

                for start_idx in start_indices:
                    end_idx = start_idx + self.gap_size + 1
                    sequence = frame_features[
                        start_idx : start_idx + self.total_sequence_length
                    ]

                    self.samples.append(
                        {
                            "start_frame": sequence[0],
                            "end_frame": sequence[-1],
                            "target_sequence": sequence,
                        }
                    )

                    if len(self.samples) >= self.max_samples:
                        print(f"  Reached max samples ({self.max_samples})")
                        return

            except Exception as e:
                continue

        print(f"Loaded {len(self.samples)} samples into memory")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "start_frame": torch.FloatTensor(sample["start_frame"]),
            "end_frame": torch.FloatTensor(sample["end_frame"]),
            "target_sequence": torch.FloatTensor(sample["target_sequence"]),
        }


class LSTMGenerator(nn.Module):
    """LSTM-based Generator following proven GAN architectures"""

    def __init__(self, feature_dim=333, noise_dim=100, hidden_dim=128, num_layers=2):
        super(LSTMGenerator, self).__init__()
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Conditioning network (start + end frames -> context)
        self.condition_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
        )

        # Noise + context -> initial hidden state
        self.noise_to_hidden = nn.Sequential(
            nn.Linear(noise_dim + hidden_dim, hidden_dim * num_layers),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * num_layers),
        )

        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh(),  # Bounded output
        )

    def forward(self, start_frame, end_frame, noise=None):
        batch_size = start_frame.size(0)
        device = start_frame.device

        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim).to(device)

        # Encode conditioning information
        condition = torch.cat([start_frame, end_frame], dim=1)
        context = self.condition_encoder(condition)

        # Combine noise and context
        noise_context = torch.cat([noise, context], dim=1)
        hidden_init = self.noise_to_hidden(noise_context)

        # Initialize LSTM hidden state
        h_0 = (
            hidden_init.view(batch_size, self.num_layers, self.hidden_dim)
            .transpose(0, 1)
            .contiguous()
        )
        c_0 = torch.zeros_like(h_0)

        # Generate sequence
        sequence_length = 7  # start + 5 middle + end
        lstm_input = context.unsqueeze(1).repeat(1, sequence_length, 1)

        lstm_out, _ = self.lstm(lstm_input, (h_0, c_0))

        # Project to feature space
        lstm_out_flat = lstm_out.contiguous().view(-1, self.hidden_dim)
        features = self.output_projection(lstm_out_flat)

        # Reshape back to sequence
        generated_sequence = features.view(
            batch_size, sequence_length, self.feature_dim
        )

        # Create output tensor without in-place operations
        output = torch.zeros_like(generated_sequence)

        # Set start frame (first position)
        output[:, 0, :] = start_frame

        # Set middle frames (generated content)
        output[:, 1:-1, :] = generated_sequence[:, 1:-1, :]

        # Set end frame (last position)
        output[:, -1, :] = end_frame

        return output


class LSTMDiscriminator(nn.Module):
    """LSTM-based Discriminator following proven GAN architectures"""

    def __init__(self, feature_dim=333, hidden_dim=128, num_layers=2):
        super(LSTMDiscriminator, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Feature processing
        self.feature_embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )

        # LSTM for sequence analysis
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 * 7, hidden_dim
            ),  # *2 for bidirectional, *7 for sequence length
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, sequences):
        batch_size, seq_len, feat_dim = sequences.shape

        # Embed features
        sequences_flat = sequences.view(-1, feat_dim)
        embedded = self.feature_embedding(sequences_flat)
        embedded = embedded.view(batch_size, seq_len, -1)

        # LSTM processing
        lstm_out, _ = self.lstm(embedded)

        # Flatten for classification
        lstm_out_flat = lstm_out.contiguous().view(batch_size, -1)

        # Classify
        output = self.classifier(lstm_out_flat)

        return output


class GestureGAN:
    """Improved GAN with proven training techniques"""

    def __init__(self, feature_dim=333, config=SEQ2SEQ_CONFIG):
        self.config = config
        self.feature_dim = feature_dim

        # Initialize networks with proven architectures
        self.generator = LSTMGenerator(
            feature_dim=feature_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )

        self.discriminator = LSTMDiscriminator(
            feature_dim=feature_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )

        # Separate optimizers with rebalanced learning rates
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.get("generator_lr", 0.0002),  # Increased generator LR
            betas=(config.get("beta1", 0.5), config.get("beta2", 0.999)),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get("discriminator_lr", 0.0001),  # Decreased discriminator LR
            betas=(config.get("beta1", 0.5), config.get("beta2", 0.999)),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Loss function with label smoothing
        self.criterion = nn.BCELoss()

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        print(f"Using device: {self.device}")

        # Training history
        self.history = {
            "g_losses": [],
            "d_losses": [],
            "d_real_acc": [],
            "d_fake_acc": [],
        }

    def get_labels(self, batch_size, real=True):
        """Get labels with stronger label smoothing"""
        if real:
            # More aggressive label smoothing for real samples
            labels = (
                torch.ones(batch_size, 1) - torch.rand(batch_size, 1) * 0.3
            )  # Increased from 0.1
        else:
            # Add some noise to fake labels too
            labels = (
                torch.zeros(batch_size, 1) + torch.rand(batch_size, 1) * 0.3
            )  # Increased from 0.1

        return labels.to(self.device)

    def train_step(self, batch):
        start_frames = batch["start_frame"].to(self.device)
        end_frames = batch["end_frame"].to(self.device)
        real_sequences = batch["target_sequence"].to(self.device)
        batch_size = start_frames.size(0)

        # Train Discriminator
        self.d_optimizer.zero_grad()

        # Real sequences
        real_labels = self.get_labels(batch_size, real=True)
        d_real_output = self.discriminator(real_sequences)
        d_real_loss = self.criterion(d_real_output, real_labels)

        # Fake sequences - use detach() to avoid gradients
        fake_labels = self.get_labels(batch_size, real=False)
        fake_sequences = self.generator(start_frames, end_frames)
        d_fake_output = self.discriminator(
            fake_sequences.detach()
        )  # Detach to avoid generator gradients
        d_fake_loss = self.criterion(d_fake_output, fake_labels)

        # Combined discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()

        # Generate fresh sequences for generator training
        fake_sequences_for_g = self.generator(start_frames, end_frames)
        real_labels_for_g = self.get_labels(batch_size, real=True)
        d_fake_output_for_g = self.discriminator(fake_sequences_for_g)
        g_loss = self.criterion(d_fake_output_for_g, real_labels_for_g)

        g_loss.backward()
        self.g_optimizer.step()

        # Calculate accuracies using detached tensors
        with torch.no_grad():
            d_real_acc = ((d_real_output > 0.5).float().mean()).item()
            d_fake_acc = ((d_fake_output <= 0.5).float().mean()).item()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "d_real_acc": d_real_acc,
            "d_fake_acc": d_fake_acc,
        }

    def train(self, dataloader, epochs=None):
        if epochs is None:
            epochs = self.config["epochs"]

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_metrics = {"g_loss": 0, "d_loss": 0, "d_real_acc": 0, "d_fake_acc": 0}
            num_batches = 0

            for batch in dataloader:
                metrics = self.train_step(batch)
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1

            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
                self.history[
                    key.replace("_loss", "_losses").replace("_acc", "_acc")
                ].append(epoch_metrics[key])

            # Logging
            if (epoch + 1) % 5 == 0 or epoch < 5:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  G Loss: {epoch_metrics['g_loss']:.4f}")
                print(f"  D Loss: {epoch_metrics['d_loss']:.4f}")
                print(f"  D Real Acc: {epoch_metrics['d_real_acc']:.3f}")
                print(f"  D Fake Acc: {epoch_metrics['d_fake_acc']:.3f}")

                # Training status with better thresholds
                d_avg_acc = (
                    epoch_metrics["d_real_acc"] + epoch_metrics["d_fake_acc"]
                ) / 2
                if (
                    0.5 <= epoch_metrics["g_loss"] <= 4.0
                    and 0.2 <= epoch_metrics["d_loss"] <= 1.5
                    and 0.6 <= d_avg_acc <= 0.85
                ):  # More realistic accuracy range
                    print(f"  Status: GOOD TRAINING")
                elif d_avg_acc > 0.9:
                    print(f"  Status: DISCRIMINATOR TOO STRONG - Need rebalancing")
                elif epoch_metrics["g_loss"] > 4.0:
                    print(f"  Status: GENERATOR STRUGGLING")
                else:
                    print(f"  Status: TRAINING IN PROGRESS")
                print()

    def generate_sequence(self, start_frame, end_frame):
        """Generate sequence between start and end frames"""
        self.generator.eval()
        with torch.no_grad():
            if isinstance(start_frame, np.ndarray):
                start_frame = (
                    torch.FloatTensor(start_frame).unsqueeze(0).to(self.device)
                )
            if isinstance(end_frame, np.ndarray):
                end_frame = torch.FloatTensor(end_frame).unsqueeze(0).to(self.device)

            generated = self.generator(start_frame, end_frame)

        self.generator.train()
        return generated.cpu().numpy()

    def save_models(self, save_dir=None):
        if save_dir is None:
            save_dir = MODELS_TRAINED_DIR

        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.generator.state_dict(), os.path.join(save_dir, "lstm_generator.pth")
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(save_dir, "lstm_discriminator.pth"),
        )

        with open(os.path.join(save_dir, "gan_config.pkl"), "wb") as f:
            pickle.dump(self.config, f)

        print(f"Models saved to {save_dir}")


def plot_training_curves(gan):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Losses
    axes[0, 0].plot(gan.history["g_losses"], label="Generator", alpha=0.8)
    axes[0, 0].plot(gan.history["d_losses"], label="Discriminator", alpha=0.8)
    axes[0, 0].set_title("Training Losses")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Discriminator accuracies
    axes[0, 1].plot(gan.history["d_real_acc"], label="Real Accuracy", alpha=0.8)
    axes[0, 1].plot(gan.history["d_fake_acc"], label="Fake Accuracy", alpha=0.8)
    axes[0, 1].set_title("Discriminator Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training balance
    combined_acc = [
        (r + f) / 2
        for r, f in zip(gan.history["d_real_acc"], gan.history["d_fake_acc"])
    ]
    axes[1, 0].plot(combined_acc, alpha=0.8, color="purple")
    axes[1, 0].axhline(
        y=0.5, color="red", linestyle="--", alpha=0.7, label="Ideal Balance"
    )
    axes[1, 0].set_title("Training Balance")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Combined D Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss ratio
    loss_ratio = [
        g / max(d, 0.001)
        for g, d in zip(gan.history["g_losses"], gan.history["d_losses"])
    ]
    axes[1, 1].plot(loss_ratio, alpha=0.8, color="orange")
    axes[1, 1].axhline(
        y=1.0, color="red", linestyle="--", alpha=0.7, label="Equal Losses"
    )
    axes[1, 1].set_title("G/D Loss Ratio")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main training pipeline with proven GAN architecture"""
    print("=== LSTM-GAN for Gesture Sequence Filling ===")
    print(f"Config: {SEQ2SEQ_CONFIG}")

    # Initialize processor
    processor = LandmarkProcessor()

    # Get landmark files
    try:
        landmark_files = processor.get_all_landmark_files(LANDMARKS_DIR)
    except Exception as e:
        print(f"Error: {e}")
        return

    if not landmark_files:
        print("No landmark files found!")
        return

    # Fit scaler
    processor.fit_scaler_on_sample(LANDMARKS_DIR)

    # Create dataset (pre-loads for faster training)
    dataset = GestureDataset(landmark_files, processor, max_samples=5000)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=SEQ2SEQ_CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Create and train GAN
    gan = GestureGAN(feature_dim=processor.total_features)
    gan.train(dataloader)

    # Plot results
    plot_training_curves(gan)

    # Save models
    gan.save_models()

    # Save processor
    with open(os.path.join(MODELS_TRAINED_DIR, "processor.pkl"), "wb") as f:
        pickle.dump(processor, f)

    print("Training complete!")


if __name__ == "__main__":
    main()
