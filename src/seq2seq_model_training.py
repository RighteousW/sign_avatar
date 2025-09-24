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
import json

from constants import (
    LANDMARKS_DIR,
    LANDMARKS_DIR_METADATA_JSON,
    LANDMARKS_DIR_METADATA_PKL,
    MODELS_TRAINED_DIR,
    SEQ2SEQ_CONFIG,
)
from similarity_metrics import GestureSequenceSimilarityMetrics

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_extraction_metadata():
    """Load metadata from landmark extraction to configure model"""
    metadata = None

    # Try loading from pickle first (more complete), then JSON
    if os.path.exists(LANDMARKS_DIR_METADATA_PKL):
        try:
            with open(LANDMARKS_DIR_METADATA_PKL, "rb") as f:
                metadata = pickle.load(f)
            print(f"Loaded metadata from: {LANDMARKS_DIR_METADATA_PKL}")
        except Exception as e:
            print(f"Error loading pickle metadata: {e}")

    if metadata is None and os.path.exists(LANDMARKS_DIR_METADATA_JSON):
        try:
            with open(LANDMARKS_DIR_METADATA_JSON, "r") as f:
                metadata = json.load(f)
            print(f"Loaded metadata from: {LANDMARKS_DIR_METADATA_JSON}")
        except Exception as e:
            print(f"Error loading JSON metadata: {e}")

    if metadata is None:
        raise FileNotFoundError(
            f"No metadata found. Please run landmark extraction first to generate "
            f"{LANDMARKS_DIR_METADATA_PKL} or {LANDMARKS_DIR_METADATA_JSON}"
        )

    return metadata


def create_config_from_metadata(metadata, base_config=None):
    """Create training configuration based on extraction metadata"""
    if base_config is None:
        base_config = SEQ2SEQ_CONFIG.copy()

    feature_info = metadata.get("feature_info", {})
    landmark_types = metadata.get("landmark_types", [])

    # Configure feature dimensions
    config = base_config.copy()
    config["feature_dim"] = feature_info.get("total_features", 333)
    config["landmark_types"] = landmark_types
    config["feature_breakdown"] = {
        "hand_landmarks_dim": feature_info.get("hand_landmarks", 0),
        "pose_landmarks_dim": feature_info.get("pose_landmarks", 0),
        "total_features": feature_info.get("total_features", 333),
    }

    # Adjust model capacity based on feature complexity
    total_features = config["feature_dim"]
    if total_features > 300:
        config["hidden_dim"] = max(config.get("hidden_dim", 128), 256)
        config["num_layers"] = max(config.get("num_layers", 2), 3)
    elif total_features > 150:
        config["hidden_dim"] = max(config.get("hidden_dim", 128), 192)
        config["num_layers"] = max(config.get("num_layers", 2), 3)

    # Adjust training parameters based on dataset size
    total_videos = metadata.get("total_videos", 0)
    if total_videos < 100:
        config["epochs"] = max(
            config.get("epochs", 50), 100
        )  # More epochs for small datasets
        config["batch_size"] = min(config.get("batch_size", 32), 16)  # Smaller batches
    elif total_videos > 1000:
        config["epochs"] = min(
            config.get("epochs", 50), 30
        )  # Fewer epochs for large datasets
        config["batch_size"] = max(config.get("batch_size", 32), 64)  # Larger batches

    print(f"Configuration created from metadata:")
    print(f"  Feature dimensions: {config['feature_dim']}")
    print(f"  Landmark types: {config['landmark_types']}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Total videos in dataset: {total_videos}")
    print(f"  Adjusted epochs: {config['epochs']}")
    print(f"  Adjusted batch size: {config['batch_size']}")

    return config


class MetadataAwareLandmarkProcessor:
    """Enhanced processor that uses extraction metadata for configuration"""

    def __init__(self, metadata=None):
        if metadata is None:
            metadata = load_extraction_metadata()

        self.metadata = metadata
        self.feature_info = metadata.get("feature_info", {})
        self.landmark_types = metadata.get("landmark_types", [])

        # Set feature dimensions from metadata
        self.hand_landmarks_dim = self.feature_info.get("hand_landmarks", 0)
        self.pose_landmarks_dim = self.feature_info.get("pose_landmarks", 0)
        self.total_features = self.feature_info.get("total_features", 0)

        # Initialize derived dimensions (if using connections - adjust as needed)
        self.hand_connections_dim = (
            20 * 2 if "hand_landmarks" in self.landmark_types else 0
        )
        self.pose_connections_dim = 35 if "pose_landmarks" in self.landmark_types else 0

        # Update total if using connections
        if hasattr(self, "use_connections") and self.use_connections:
            self.total_features += self.hand_connections_dim + self.pose_connections_dim

        self.scaler = None
        self.is_fitted = False

        print(f"MetadataAwareLandmarkProcessor initialized:")
        print(f"  Hand landmarks dim: {self.hand_landmarks_dim}")
        print(f"  Pose landmarks dim: {self.pose_landmarks_dim}")
        print(f"  Total feature dimensions: {self.total_features}")
        print(f"  Landmark types: {self.landmark_types}")

    def extract_frame_features(self, frame_data):
        """Extract features from a single frame based on metadata configuration"""
        features = np.zeros(self.total_features, dtype=np.float32)
        current_idx = 0

        # Process hand landmarks if configured
        if "hand_landmarks" in self.landmark_types and self.hand_landmarks_dim > 0:
            hand_features = np.zeros(self.hand_landmarks_dim, dtype=np.float32)

            # Extract up to 2 hands (based on metadata configuration)
            max_hands = self.feature_info.get("max_hands", 2)
            landmarks_per_hand = self.feature_info.get(
                "hand_landmarks_per_hand", 63
            )  # 21*3

            for i, hand_data in enumerate(frame_data.get("hands", [])[:max_hands]):
                landmarks = hand_data.get("landmarks", [])
                start_idx = i * landmarks_per_hand

                for j, landmark in enumerate(landmarks[:21]):  # 21 landmarks per hand
                    if start_idx + j * 3 + 2 < len(hand_features):
                        hand_features[start_idx + j * 3 : start_idx + (j + 1) * 3] = (
                            landmark[:3]
                        )

            features[current_idx : current_idx + self.hand_landmarks_dim] = (
                hand_features
            )
            current_idx += self.hand_landmarks_dim

        # Process pose landmarks if configured
        if "pose_landmarks" in self.landmark_types and self.pose_landmarks_dim > 0:
            pose_data = frame_data.get("pose")
            if pose_data:
                landmarks = pose_data.get("landmarks", [])
                pose_features = np.zeros(self.pose_landmarks_dim, dtype=np.float32)

                coords_per_landmark = self.feature_info.get(
                    "pose_coords_per_landmark", 4
                )
                max_landmarks = self.feature_info.get("pose_total_landmarks", 33)

                for i, landmark in enumerate(landmarks[:max_landmarks]):
                    if i * coords_per_landmark + coords_per_landmark <= len(
                        pose_features
                    ):
                        pose_features[
                            i * coords_per_landmark : (i + 1) * coords_per_landmark
                        ] = landmark[:coords_per_landmark]

                features[current_idx : current_idx + self.pose_landmarks_dim] = (
                    pose_features
                )
                current_idx += self.pose_landmarks_dim

        return features

    def get_all_landmark_files(self, landmarks_dir):
        """Get all landmark files, using metadata if available"""
        landmark_files = []
        landmarks_path = Path(landmarks_dir)

        if not landmarks_path.exists():
            raise ValueError(f"Landmarks directory not found: {landmarks_dir}")

        # Use metadata info if available
        processed_videos = self.metadata.get("processed_videos", [])
        if processed_videos:
            print(f"Using metadata to locate {len(processed_videos)} processed videos")

            for video_info in processed_videos:
                video_path = video_info.get("video_path", "")
                if video_path:
                    # Convert video path to landmark path
                    video_file = os.path.basename(video_path)
                    video_folder = os.path.dirname(video_path)
                    folder_name = os.path.basename(video_folder)

                    landmark_filename = video_file.replace(".avi", "_landmarks.pkl")
                    landmark_path = landmarks_path / folder_name / landmark_filename

                    if landmark_path.exists():
                        landmark_files.append(landmark_path)

        # Fallback to directory scan if metadata doesn't help
        if not landmark_files:
            print("Metadata didn't help locate files, scanning directories...")
            for folder in landmarks_path.iterdir():
                if folder.is_dir():
                    pkl_files = list(folder.glob("*_landmarks.pkl"))
                    landmark_files.extend(pkl_files)

        print(f"Found {len(landmark_files)} landmark files")
        return landmark_files

    def fit_scaler_on_sample(self, landmarks_dir, sample_size=1000):
        """Fit scaler on a sample of data"""
        landmark_files = self.get_all_landmark_files(landmarks_dir)

        # Use metadata to inform sampling strategy
        total_frames = self.metadata.get("total_frames", 0)
        total_videos = self.metadata.get("total_videos", len(landmark_files))

        if total_videos > 0:
            frames_per_video = total_frames / total_videos
            files_needed = min(
                len(landmark_files), max(10, sample_size // int(frames_per_video))
            )
        else:
            files_needed = min(len(landmark_files), sample_size // 50)

        sample_files = random.sample(landmark_files, files_needed)

        sample_features = []
        for pkl_file in sample_files:
            try:
                with open(pkl_file, "rb") as f:
                    landmark_data = pickle.load(f)

                # Extract features from frames
                for frame_data in landmark_data.get("frames", [])[:20]:
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


class LSTMGenerator(nn.Module):
    """Enhanced LSTM-based Generator with improved architecture"""

    def __init__(
        self,
        feature_dim=333,
        noise_dim=100,
        hidden_dim=128,
        num_layers=2,
        sequence_length=3,
    ):
        super(LSTMGenerator, self).__init__()
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim * 2
        self.num_layers = num_layers + 1

        # Multi-scale conditioning network (stronger feature extraction)
        self.condition_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim),
        )

        # Enhanced noise processing
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim // 2),
        )

        # Combined noise + context -> initial hidden state
        self.noise_to_hidden = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.hidden_dim // 2,
                self.hidden_dim
                * self.num_layers
                * 2
                * 2,  # *2 for bidirectional, *2 for h_0 and c_0
            ),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim * self.num_layers * 2 * 2),
            nn.Dropout(0.1),
        )

        # Project combined features to LSTM input dimension
        self.feature_to_lstm = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim // 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim),
        )

        # Bidirectional LSTM for richer representation
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 if self.num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism for better sequence modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # Multi-layer output projection with residual connections
        self.pre_output = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, feature_dim),
            nn.Tanh(),
        )
        self.sequence_length = sequence_length

        # Learnable interpolation weights for start/end frame blending
        self.interpolation_weights = nn.Parameter(torch.ones(sequence_length, 1))

    def forward(self, start_frame, end_frame, noise=None):
        batch_size = start_frame.size(0)
        device = start_frame.device

        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim).to(device)

        # Enhanced conditioning
        condition = torch.cat([start_frame, end_frame], dim=1)
        context = self.condition_encoder(condition)

        # Process noise separately
        noise_features = self.noise_encoder(noise)

        # Combine noise and context
        combined_features = torch.cat([context, noise_features], dim=1)

        # Initialize hidden states from combined noise+context features
        hidden_init = self.noise_to_hidden(combined_features)

        # Calculate the size needed for each hidden state
        single_hidden_size = (
            self.hidden_dim * self.num_layers * 2
        )  # *2 for bidirectional

        # Split the tensor correctly
        h_0 = (
            hidden_init[:, :single_hidden_size]
            .contiguous()
            .view(self.num_layers * 2, batch_size, self.hidden_dim)
        )
        c_0 = (
            hidden_init[:, single_hidden_size : single_hidden_size * 2]
            .contiguous()
            .view(self.num_layers * 2, batch_size, self.hidden_dim)
        )

        # Project combined features to LSTM input space
        lstm_input_base = self.feature_to_lstm(combined_features)

        # Generate sequence using noise+context information
        positions = torch.linspace(0, 1, self.sequence_length).to(device)

        # Create input sequence that incorporates both noise and positional info
        lstm_inputs = []
        for t in range(self.sequence_length):
            pos_weight = positions[t]
            pos_encoding = torch.full_like(lstm_input_base, pos_weight * 0.1)
            timestep_input = lstm_input_base + pos_encoding
            lstm_inputs.append(timestep_input)

        lstm_input = torch.stack(lstm_inputs, dim=1)

        # LSTM processing with noise-initialized hidden states
        lstm_out, _ = self.lstm(lstm_input, (h_0, c_0))

        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Add residual connection
        lstm_out = lstm_out + attended_out

        # Project to output space
        lstm_out_flat = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        pre_features = self.pre_output(lstm_out_flat)
        features = self.output_projection(pre_features)

        # Reshape back to sequence
        generated_sequence = features.view(
            batch_size, self.sequence_length, self.feature_dim
        )

        # Smart blending with learnable weights
        output = torch.zeros_like(generated_sequence)

        # Start frame (always fixed)
        output[:, 0, :] = start_frame

        # Middle frames with learned blending
        for i in range(1, self.sequence_length - 1):
            # Linear interpolation baseline
            alpha = i / (self.sequence_length - 1)
            interpolated = (1 - alpha) * start_frame + alpha * end_frame

            # Blend with generated content using learnable weights
            weight = torch.sigmoid(self.interpolation_weights[i])
            output[:, i, :] = (
                weight * generated_sequence[:, i, :] + (1 - weight) * interpolated
            )

        # End frame (always fixed)
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

        # Use adaptive pooling instead of fixed dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
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
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]

        # Use adaptive pooling to handle any sequence length
        lstm_out_permuted = lstm_out.permute(
            0, 2, 1
        )  # [batch_size, hidden_dim * 2, seq_len]
        pooled = self.adaptive_pool(
            lstm_out_permuted
        )  # [batch_size, hidden_dim * 2, 1]
        pooled = pooled.squeeze(-1)  # [batch_size, hidden_dim * 2]

        # Classify
        output = self.classifier(pooled)

        return output


class ConfigurableGestureDataset(Dataset):
    """Dataset that uses metadata-driven configuration"""

    def __init__(self, landmark_files, processor, config):
        self.landmark_files = landmark_files
        self.processor = processor
        self.config = config
        self.gap_size = config["gap_size"]
        self.max_samples = config["max_samples"]
        self.total_sequence_length = self.gap_size + 2

        # Pre-load and process all data into memory for faster training
        self.samples = []
        self._load_all_samples()

        print(f"ConfigurableGestureDataset ready with {len(self.samples)} samples")
        print(f"  Feature dimension: {processor.total_features}")
        print(f"  Gap size: {self.gap_size}")
        print(f"  Sequence length: {self.total_sequence_length}")

    def _load_all_samples(self):
        """Pre-load all samples for faster training"""
        print("Pre-loading dataset samples...")

        # Use metadata to inform file processing limits
        if hasattr(self.processor, "metadata"):
            total_videos = self.processor.metadata.get(
                "total_videos", len(self.landmark_files)
            )
            files_to_process = self.landmark_files[: min(2000, total_videos)]
        else:
            files_to_process = self.landmark_files[:2000]

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


class ConfigurableEnhancedGestureGAN:
    """Enhanced GAN that configures itself based on metadata"""

    def __init__(self, config):
        self.config = config
        self.feature_dim = config["feature_dim"]
        self.sequence_length = config["gap_size"] + 2

        print(f"Initializing ConfigurableEnhancedGestureGAN:")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Hidden dimension: {config['hidden_dim']}")
        print(f"  Number of layers: {config['num_layers']}")
        print(f"  Sequence length: {self.sequence_length}")

        # Initialize networks with metadata-driven configuration
        self.generator = LSTMGenerator(
            feature_dim=self.feature_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            sequence_length=self.sequence_length,
        )

        self.discriminator = LSTMDiscriminator(
            feature_dim=self.feature_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
        )

        # Configure optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.get("generator_lr", 0.0003),
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )

        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get("discriminator_lr", 0.0001),
            betas=(0.5, 0.999),
            weight_decay=1e-5,
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Initialize similarity metrics tracker
        self.similarity_tracker = GestureSequenceSimilarityMetrics(self.feature_dim)

        # Enhanced training history
        self.history = {
            "g_losses": [],
            "d_losses": [],
            "d_real_acc": [],
            "d_fake_acc": [],
            "overall_quality": [],
            "temporal_correlation": [],
            "hand_landmarks_quality": [],
            "pose_landmarks_quality": [],
            "overall_mse": [],
            "g_adversarial_loss": [],
            "g_quality_loss": [],
            "g_reconstruction_loss": [],
        }

        # Feature dimension splits for component losses (from config)
        feature_breakdown = config.get("feature_breakdown", {})
        self.hand_landmarks_dim = feature_breakdown.get("hand_landmarks_dim", 126)
        self.pose_landmarks_dim = feature_breakdown.get("pose_landmarks_dim", 132)

        # Derived dimensions (adjust if using connections)
        self.hand_connections_dim = 40  # 20 * 2
        self.pose_connections_dim = 35

        print(f"ConfigurableEnhancedGestureGAN initialized on {self.device}")
        print(
            f"Feature breakdown - Hand: {self.hand_landmarks_dim}, Pose: {self.pose_landmarks_dim}"
        )

    def compute_temporal_consistency_loss(self, sequences):
        """Compute loss based on temporal smoothness and consistency"""
        if sequences.size(1) < 3:
            return torch.tensor(0.0, device=sequences.device)

        # First-order differences (velocity)
        vel = sequences[:, 1:] - sequences[:, :-1]

        # Second-order differences (acceleration)
        acc = vel[:, 1:] - vel[:, :-1]

        # Temporal consistency: penalize large accelerations
        temporal_loss = torch.mean(torch.norm(acc, dim=2))

        # Velocity smoothness: encourage smooth motion
        velocity_loss = torch.mean(torch.norm(vel, dim=2))

        return 0.1 * temporal_loss + 0.05 * velocity_loss

    def compute_feature_specific_losses(self, fake_sequences, real_sequences):
        """Compute losses for different body parts separately"""
        # Ensure sequences have the same length
        min_seq_len = min(fake_sequences.size(1), real_sequences.size(1))
        fake_sequences = fake_sequences[:, :min_seq_len, :]
        real_sequences = real_sequences[:, :min_seq_len, :]

        losses = {}

        # Hand landmarks loss (if hand landmarks are present)
        if self.hand_landmarks_dim > 0:
            hand_start = 0
            hand_end = self.hand_landmarks_dim
            fake_hands = fake_sequences[:, :, hand_start:hand_end]
            real_hands = real_sequences[:, :, hand_start:hand_end]
            losses["hand_landmarks"] = nn.functional.mse_loss(fake_hands, real_hands)
            current_idx = hand_end
        else:
            losses["hand_landmarks"] = torch.tensor(0.0, device=fake_sequences.device)
            current_idx = 0

        # Hand connections loss (if using connections)
        if self.hand_connections_dim > 0:
            conn_start = current_idx
            conn_end = conn_start + self.hand_connections_dim
            if conn_end <= fake_sequences.size(2):
                fake_hand_conn = fake_sequences[:, :, conn_start:conn_end]
                real_hand_conn = real_sequences[:, :, conn_start:conn_end]
                losses["hand_connections"] = nn.functional.mse_loss(
                    fake_hand_conn, real_hand_conn
                )
                current_idx = conn_end
            else:
                losses["hand_connections"] = torch.tensor(
                    0.0, device=fake_sequences.device
                )
        else:
            losses["hand_connections"] = torch.tensor(0.0, device=fake_sequences.device)

        # Pose landmarks loss (if pose landmarks are present)
        if self.pose_landmarks_dim > 0:
            pose_start = current_idx
            pose_end = pose_start + self.pose_landmarks_dim
            if pose_end <= fake_sequences.size(2):
                fake_pose = fake_sequences[:, :, pose_start:pose_end]
                real_pose = real_sequences[:, :, pose_start:pose_end]
                losses["pose_landmarks"] = nn.functional.mse_loss(fake_pose, real_pose)
                current_idx = pose_end
            else:
                losses["pose_landmarks"] = torch.tensor(
                    0.0, device=fake_sequences.device
                )
        else:
            losses["pose_landmarks"] = torch.tensor(0.0, device=fake_sequences.device)

        # Pose connections loss (if using connections)
        if self.pose_connections_dim > 0 and current_idx < fake_sequences.size(2):
            fake_pose_conn = fake_sequences[:, :, current_idx:]
            real_pose_conn = real_sequences[:, :, current_idx:]
            losses["pose_connections"] = nn.functional.mse_loss(
                fake_pose_conn, real_pose_conn
            )
        else:
            losses["pose_connections"] = torch.tensor(0.0, device=fake_sequences.device)

        return losses

    def compute_quality_aware_loss(self, fake_sequences, real_sequences):
        """Compute loss based on quality metrics"""
        try:
            # Convert to numpy for similarity computation
            fake_np = fake_sequences.detach().cpu().numpy()
            real_np = real_sequences.detach().cpu().numpy()

            # Compute similarity metrics
            similarity_report = (
                self.similarity_tracker.compute_comprehensive_similarity(
                    real_np, fake_np
                )
            )
            summary = self.similarity_tracker.summarize_similarity_scores(
                similarity_report
            )

            # Convert quality metrics to losses (lower quality = higher loss)
            quality_loss = 1.0 - summary.get("overall_quality", 0.0)
            temporal_loss = 1.0 - summary.get("temporal_correlation", 0.0)
            mse_loss = summary.get("overall_mse", 0.0)

            # Combine into single quality loss
            total_quality_loss = (
                0.4 * quality_loss + 0.3 * temporal_loss + 0.3 * mse_loss
            )

            return (
                torch.tensor(total_quality_loss, device=fake_sequences.device),
                summary,
            )

        except Exception as e:
            print(f"Error computing quality loss: {e}")
            return torch.tensor(0.0, device=fake_sequences.device), {}

    def enhanced_generator_loss(
        self,
        fake_sequences,
        real_sequences,
        d_fake_output,
        start_frames,
        end_frames,
        epoch,
    ):
        """Enhanced generator loss with multiple components"""
        batch_size = fake_sequences.size(0)

        # 1. Adversarial loss
        real_labels = torch.ones(batch_size, 1).to(self.device)
        adversarial_loss = self.criterion(d_fake_output, real_labels)

        # 2. Reconstruction loss (start/end frame consistency)
        reconstruction_loss = nn.functional.mse_loss(
            fake_sequences[:, 0], start_frames
        ) + nn.functional.mse_loss(fake_sequences[:, -1], end_frames)

        # 3. Temporal consistency loss
        temporal_loss = self.compute_temporal_consistency_loss(fake_sequences)

        # 4. Feature-specific losses
        feature_losses = self.compute_feature_specific_losses(
            fake_sequences, real_sequences
        )
        feature_loss = (
            2.0 * feature_losses["hand_landmarks"]
            + 1.0 * feature_losses["hand_connections"]
            + 2.0 * feature_losses["pose_landmarks"]
            + 1.0 * feature_losses["pose_connections"]
        )

        # 5. Quality-aware loss
        quality_loss, quality_metrics = self.compute_quality_aware_loss(
            fake_sequences, real_sequences
        )

        # 6. Progressive difficulty: weight losses based on training progress
        epoch_progress = epoch / self.config.get("epochs", 50)

        reconstruction_weight = max(0.5, 2.0 - 2.0 * epoch_progress)  # 2.0 -> 0.5
        adversarial_weight = min(2.0, 0.5 + 1.5 * epoch_progress)  # 0.5 -> 2.0
        quality_weight = min(1.0, 2.0 * epoch_progress)  # 0.0 -> 1.0

        # Combine all losses
        total_loss = (
            adversarial_weight * adversarial_loss
            + reconstruction_weight * reconstruction_loss
            + 0.2 * temporal_loss
            + 0.3 * feature_loss
            + quality_weight * quality_loss
        )

        return {
            "total_loss": total_loss,
            "adversarial_loss": adversarial_loss,
            "reconstruction_loss": reconstruction_loss,
            "temporal_loss": temporal_loss,
            "feature_loss": feature_loss,
            "quality_loss": quality_loss,
            "quality_metrics": quality_metrics,
        }

    def should_train_discriminator(self, d_real_acc, d_fake_acc):
        """Determine if discriminator should be trained based on performance"""
        return d_real_acc < 0.8 and d_fake_acc < 0.6

    def train_step(self, batch, epoch):
        start_frames = batch["start_frame"].to(self.device)
        end_frames = batch["end_frame"].to(self.device)
        real_sequences = batch["target_sequence"].to(self.device)
        batch_size = start_frames.size(0)

        # Calculate current discriminator performance
        with torch.no_grad():
            fake_test = self.generator(start_frames, end_frames)
            d_real_test = self.discriminator(real_sequences)
            d_fake_test = self.discriminator(fake_test)

            d_real_acc = ((d_real_test > 0.5).float().mean()).item()
            d_fake_acc = ((d_fake_test <= 0.5).float().mean()).item()

        # Determine training strategy
        train_discriminator = self.should_train_discriminator(d_real_acc, d_fake_acc)

        # Multiple generator steps when discriminator is strong
        if d_real_acc > 0.9 or d_fake_acc > 0.9:
            generator_steps = 3
        elif d_real_acc > 0.8 or d_fake_acc > 0.8:
            generator_steps = 2
        else:
            generator_steps = 1

        # Train Discriminator (only when needed)
        d_loss = torch.tensor(0.0)
        if train_discriminator:
            self.d_optimizer.zero_grad()

            # Real samples
            real_labels = torch.ones(batch_size, 1).to(self.device)
            d_real_output = self.discriminator(real_sequences)
            d_real_loss = self.criterion(d_real_output, real_labels)

            # Fake samples
            fake_sequences = self.generator(start_frames, end_frames)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            d_fake_output = self.discriminator(fake_sequences.detach())
            d_fake_loss = self.criterion(d_fake_output, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            self.d_optimizer.step()

        # Train Generator (multiple steps)
        g_loss_components = None
        for step in range(generator_steps):
            self.g_optimizer.zero_grad()

            # Generate fake sequences
            fake_sequences = self.generator(start_frames, end_frames)

            # Get discriminator output for generator loss
            d_fake_output = self.discriminator(fake_sequences)

            # Enhanced generator loss
            g_loss_components = self.enhanced_generator_loss(
                fake_sequences,
                real_sequences,
                d_fake_output,
                start_frames,
                end_frames,
                epoch,
            )

            total_g_loss = g_loss_components["total_loss"]
            total_g_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
            self.g_optimizer.step()

        # Calculate final accuracies
        with torch.no_grad():
            final_fake = self.generator(start_frames, end_frames)
            d_real_final = self.discriminator(real_sequences)
            d_fake_final = self.discriminator(final_fake)

            final_d_real_acc = ((d_real_final > 0.5).float().mean()).item()
            final_d_fake_acc = ((d_fake_final <= 0.5).float().mean()).item()

        return {
            "g_loss": (
                g_loss_components["total_loss"].item() if g_loss_components else 0.0
            ),
            "d_loss": d_loss.item(),
            "d_real_acc": final_d_real_acc,
            "d_fake_acc": final_d_fake_acc,
            "trained_discriminator": train_discriminator,
            "generator_steps": generator_steps,
            "g_adversarial_loss": (
                g_loss_components["adversarial_loss"].item()
                if g_loss_components
                else 0.0
            ),
            "g_quality_loss": (
                g_loss_components["quality_loss"].item() if g_loss_components else 0.0
            ),
            "g_reconstruction_loss": (
                g_loss_components["reconstruction_loss"].item()
                if g_loss_components
                else 0.0
            ),
            "quality_metrics": (
                g_loss_components["quality_metrics"] if g_loss_components else {}
            ),
        }

    def train(self, dataloader, epochs=None):
        if epochs is None:
            epochs = self.config["epochs"]

        print(f"Starting enhanced training for {epochs} epochs...")
        print("Training strategy:")
        print("- Discriminator trains only when Real Acc < 0.8 AND Fake Acc < 0.6")
        print("- Generator gets multiple steps when discriminator is strong")
        print("- Progressive loss weighting: reconstruction -> adversarial over time")

        for epoch in range(epochs):
            epoch_metrics = {
                "g_loss": 0,
                "d_loss": 0,
                "d_real_acc": 0,
                "d_fake_acc": 0,
                "g_adversarial_loss": 0,
                "g_quality_loss": 0,
                "g_reconstruction_loss": 0,
            }
            epoch_similarity = {
                "overall_quality": [],
                "temporal_correlation": [],
                "hand_landmarks_quality": [],
                "pose_landmarks_quality": [],
                "overall_mse": [],
            }

            discriminator_trained_batches = 0
            total_generator_steps = 0
            num_batches = 0

            for batch in dataloader:
                metrics = self.train_step(batch, epoch + 1)

                # Accumulate metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]

                # Track training strategy stats
                if metrics["trained_discriminator"]:
                    discriminator_trained_batches += 1
                total_generator_steps += metrics["generator_steps"]

                # Similarity metrics
                if metrics.get("quality_metrics"):
                    quality_metrics = metrics["quality_metrics"]
                    for key in epoch_similarity:
                        if key in quality_metrics and not np.isnan(
                            quality_metrics[key]
                        ):
                            epoch_similarity[key].append(quality_metrics[key])

                num_batches += 1

            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            # Update history
            self.history["g_losses"].append(epoch_metrics["g_loss"])
            self.history["d_losses"].append(epoch_metrics["d_loss"])
            self.history["d_real_acc"].append(epoch_metrics["d_real_acc"])
            self.history["d_fake_acc"].append(epoch_metrics["d_fake_acc"])

            # Store component losses
            self.history["g_adversarial_loss"].append(
                epoch_metrics["g_adversarial_loss"]
            )
            self.history["g_quality_loss"].append(epoch_metrics["g_quality_loss"])
            self.history["g_reconstruction_loss"].append(
                epoch_metrics["g_reconstruction_loss"]
            )

            # Average similarity metrics
            for key in epoch_similarity:
                if epoch_similarity[key]:
                    self.history[key].append(np.mean(epoch_similarity[key]))
                else:
                    self.history[key].append(0.0)

            # Enhanced logging
            if (epoch + 1) % self.config.get("log_every", 5) == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}]")
                print(
                    f"  G Loss: {epoch_metrics['g_loss']:.4f} (Adv: {epoch_metrics['g_adversarial_loss']:.4f}, Quality: {epoch_metrics['g_quality_loss']:.4f}, Recon: {epoch_metrics['g_reconstruction_loss']:.4f})"
                )
                print(f"  D Loss: {epoch_metrics['d_loss']:.4f}")
                print(f"  D Real Acc: {epoch_metrics['d_real_acc']:.3f}")
                print(f"  D Fake Acc: {epoch_metrics['d_fake_acc']:.3f}")

                # Training strategy info
                d_train_pct = (discriminator_trained_batches / num_batches) * 100
                avg_g_steps = total_generator_steps / num_batches
                print(
                    f"  Strategy: D trained {d_train_pct:.1f}% of batches, Avg G steps: {avg_g_steps:.1f}"
                )

                # Quality metrics
                if len(self.history["overall_quality"]) > 0:
                    print(
                        f"  Overall Quality: {self.history['overall_quality'][-1]:.3f}"
                    )
                    print(
                        f"  Temporal Corr: {self.history['temporal_correlation'][-1]:.3f}"
                    )
                    print(f"  Overall MSE: {self.history['overall_mse'][-1]:.4f}")

                # Status
                d_total_acc = (
                    epoch_metrics["d_real_acc"] + epoch_metrics["d_fake_acc"]
                ) / 2
                if d_total_acc < 0.7:
                    status = "GENERATOR IMPROVING"
                elif d_total_acc > 0.85:
                    status = "DISCRIMINATOR STRONG - FOCUSING ON GENERATOR"
                else:
                    status = "BALANCED TRAINING"
                print(f"  Status: {status}")

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
        """Save models along with configuration"""
        if save_dir is None:
            save_dir = MODELS_TRAINED_DIR

        os.makedirs(save_dir, exist_ok=True)

        # Save model weights
        torch.save(
            self.generator.state_dict(),
            os.path.join(save_dir, "configurable_lstm_generator.pth"),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(save_dir, "configurable_lstm_discriminator.pth"),
        )

        # Save complete configuration including metadata-derived settings
        with open(os.path.join(save_dir, "configurable_gan_config.pkl"), "wb") as f:
            pickle.dump(self.config, f)

        # Save feature dimension mapping for inference
        feature_config = {
            "feature_dim": self.feature_dim,
            "hand_landmarks_dim": self.hand_landmarks_dim,
            "pose_landmarks_dim": self.pose_landmarks_dim,
            "landmark_types": self.config.get("landmark_types", []),
            "sequence_length": self.sequence_length,
        }

        with open(os.path.join(save_dir, "feature_config.json"), "w") as f:
            json.dump(feature_config, f, indent=2)

        print(f"Configurable models saved to {save_dir}")
        print(f"Feature configuration saved for inference")


def plot_enhanced_training_curves(gan):
    """Plot enhanced training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Generator loss components
    axes[0, 0].plot(
        gan.history["g_losses"], label="Total G Loss", alpha=0.8, linewidth=2
    )
    axes[0, 0].plot(gan.history["g_adversarial_loss"], label="Adversarial", alpha=0.7)
    axes[0, 0].plot(gan.history["g_quality_loss"], label="Quality", alpha=0.7)
    axes[0, 0].plot(
        gan.history["g_reconstruction_loss"], label="Reconstruction", alpha=0.7
    )
    axes[0, 0].set_title("Generator Loss Components")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Discriminator metrics
    axes[0, 1].plot(gan.history["d_losses"], label="D Loss", alpha=0.8, color="red")
    axes[0, 1].set_title("Discriminator Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(gan.history["d_real_acc"], label="Real Accuracy", alpha=0.8)
    axes[0, 2].plot(gan.history["d_fake_acc"], label="Fake Accuracy", alpha=0.8)
    axes[0, 2].axhline(
        y=0.8, color="red", linestyle="--", alpha=0.7, label="Training Thresholds"
    )
    axes[0, 2].axhline(y=0.6, color="red", linestyle="--", alpha=0.7)
    axes[0, 2].set_title("Discriminator Accuracy")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Accuracy")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Quality metrics
    axes[1, 0].plot(gan.history["overall_quality"], alpha=0.8, color="purple")
    axes[1, 0].set_title("Overall Generation Quality")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Quality Score")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(gan.history["temporal_correlation"], alpha=0.8, color="orange")
    axes[1, 1].set_title("Temporal Correlation")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Correlation")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(gan.history["overall_mse"], alpha=0.8, color="red")
    axes[1, 2].set_title("Overall MSE")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("MSE")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Main training pipeline with metadata-driven configuration"""
    print("=== Metadata-Driven LSTM-GAN for Gesture Sequence Filling ===")

    try:
        # Load extraction metadata
        metadata = load_extraction_metadata()

        # Create configuration from metadata
        config = create_config_from_metadata(metadata)
        print(f"Final config: {config}")

        # Initialize metadata-aware processor
        processor = MetadataAwareLandmarkProcessor(metadata)

        # Get landmark files
        landmark_files = processor.get_all_landmark_files(LANDMARKS_DIR)

        if not landmark_files:
            print("No landmark files found!")
            return

        # Fit scaler
        processor.fit_scaler_on_sample(LANDMARKS_DIR)

        # Create configurable dataset
        dataset = ConfigurableGestureDataset(landmark_files, processor, config)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        # Create and train Configurable GAN
        gan = ConfigurableEnhancedGestureGAN(config)
        gan.train(dataloader, epochs=config["epochs"])

        # Plot enhanced results
        plot_enhanced_training_curves(gan)

        # Save models and processor
        gan.save_models()

        with open(
            os.path.join(MODELS_TRAINED_DIR, "metadata_aware_processor.pkl"), "wb"
        ) as f:
            pickle.dump(processor, f)

        print("Metadata-driven training complete!")
        print(
            "Models are configured for the exact landmark types and dimensions from your extraction process."
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the landmark extraction script first to generate metadata.")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
