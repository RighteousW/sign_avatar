import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

from constants import LANDMARKS_DIR, PROCESSED_GESTURE_DATA_PATH


class GestureDataProcessor:
    def __init__(self, landmarks_dir: str):
        self.landmarks_dir = landmarks_dir
        self.scaler = StandardScaler()
        self.feature_dim = None

    def find_landmark_files(self) -> List[Dict]:
        """Find and return paths to all landmark pickle files"""
        all_file_paths = []
        for folder_name in os.listdir(self.landmarks_dir):
            folder_path = os.path.join(self.landmarks_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                if file_name.endswith("_landmarks.pkl"):
                    file_path = os.path.join(folder_path, file_name)
                    all_file_paths.append(
                        {
                            "path": file_path,
                            "folder": folder_name,
                            "filename": file_name,
                        }
                    )
        print(f"Found {len(all_file_paths)} landmark files")
        return all_file_paths

    def extract_features_from_frame(
        self, frame_data: Dict, feature_info: Dict
    ) -> np.ndarray:
        """
        Extract feature vector from a single frame using feature dimensions
        provided by the LandmarkExtractor metadata.
        """
        features = []

        # Hand landmarks
        hand_landmarks_per_hand = feature_info["breakdown"]["hand_landmarks_per_hand"]
        hand_landmarks_dim = feature_info["hand_landmarks"]
        hand_features = np.zeros(hand_landmarks_dim)
        for i, hand_data in enumerate(frame_data.get("hands", [])):
            if i < 2:
                landmarks = np.array(hand_data["landmarks"]).flatten()
                start_idx = i * hand_landmarks_per_hand
                end_idx = start_idx + len(landmarks)
                hand_features[start_idx:end_idx] = landmarks
        features.extend(hand_features)

        # Hand connection features
        hand_connections_per_hand = feature_info["breakdown"][
            "hand_connections_per_hand"
        ]
        hand_connections_dim = feature_info["hand_connections"]
        hand_conn_features = np.zeros(hand_connections_dim)
        for i, hand_connections in enumerate(
            frame_data["connection_features"].get("hands", [])
        ):
            if i < 2:
                start_idx = i * hand_connections_per_hand
                end_idx = start_idx + len(hand_connections)
                hand_conn_features[start_idx:end_idx] = hand_connections
        features.extend(hand_conn_features)

        # Pose landmarks
        pose_landmarks_dim = feature_info["pose_landmarks"]
        pose_features = np.zeros(pose_landmarks_dim)
        if frame_data.get("pose"):
            pose_landmarks = np.array(frame_data["pose"]["landmarks"]).flatten()
            pose_features[: len(pose_landmarks)] = pose_landmarks
        features.extend(pose_features)

        # Pose connection features
        pose_connections_dim = feature_info["pose_connections"]
        pose_conn_features = np.zeros(pose_connections_dim)
        if frame_data["connection_features"].get("pose"):
            pose_connections = frame_data["connection_features"]["pose"]
            pose_conn_features[: len(pose_connections)] = pose_connections
        features.extend(pose_conn_features)

        return np.array(features)

    def extract_and_normalize_sequences(
        self, file_paths: List[Dict], min_sequence_length: int = 10
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Extract feature sequences and normalize them.
        Returns a list of tuples: (file_path, normalized_sequence).
        """
        all_features = []
        sequences = []

        # First pass: Extract all features to fit the scaler
        for file_info in file_paths:
            file_path = file_info["path"]
            with open(file_path, "rb") as f:
                landmark_data = pickle.load(f)

            # Use the feature info from the processed data file
            feature_info = landmark_data["feature_info"]

            frames = landmark_data["frames"]
            if len(frames) < min_sequence_length:
                continue

            sequence_features = [
                self.extract_features_from_frame(frame, feature_info)
                for frame in frames
            ]

            if sequence_features:
                sequence_array = np.array(sequence_features)
                all_features.append(sequence_array)

        if not all_features:
            raise ValueError("No sequences extracted from any file!")

        # Flatten all features for scaler fitting
        all_features_flat = np.concatenate(all_features, axis=0)

        # Fit scaler on the entire dataset
        self.scaler.fit(all_features_flat)
        self.feature_dim = all_features_flat.shape[1]
        print(f"Feature dimension: {self.feature_dim}")

        # Second pass: Create normalized sequences
        normalized_sequences = []
        for file_info in file_paths:
            file_path = file_info["path"]
            with open(file_path, "rb") as f:
                landmark_data = pickle.load(f)

            feature_info = landmark_data["feature_info"]
            frames = landmark_data["frames"]
            if len(frames) < min_sequence_length:
                continue

            sequence_features = [
                self.extract_features_from_frame(frame, feature_info)
                for frame in frames
            ]
            sequence_array = np.array(sequence_features)
            normalized_seq = self.scaler.transform(sequence_array)
            normalized_sequences.append((file_path, normalized_seq))

        print(f"Extracted and normalized {len(normalized_sequences)} sequences")
        return normalized_sequences

    def process_data(
        self,
        min_sequence_length: int = 15,
        target_transition_length: int = 20,
        overlap_frames: int = 3,
        test_size: float = 0.2,
    ) -> Dict:
        """Complete data processing pipeline"""
        print("Starting data processing pipeline...")

        file_paths = self.find_landmark_files()

        # The first file is used to get the feature dimensions from the metadata
        with open(file_paths[0]["path"], "rb") as f:
            first_file_data = pickle.load(f)
            feature_info = first_file_data["feature_info"]

        normalized_sequences = self.extract_and_normalize_sequences(
            file_paths, min_sequence_length
        )

        # Split file paths into train/test
        train_file_paths, test_file_paths = train_test_split(
            [fp for fp, _ in normalized_sequences], test_size=test_size, random_state=42
        )

        processed_data = {
            "train_files": train_file_paths,
            "test_files": test_file_paths,
            "feature_dim": self.feature_dim,
            "scaler": self.scaler,
            "target_length": target_transition_length,
            "overlap_frames": overlap_frames,
        }

        print("Processing complete!")
        print(f"Training files: {len(train_file_paths)}")
        print(f"Test files: {len(test_file_paths)}")
        return processed_data


class LazyGestureDataset(Dataset):
    """PyTorch Dataset that loads gesture data files on-demand"""

    def __init__(
        self,
        file_paths: List[str],
        feature_dim: int,
        scaler: StandardScaler,
        target_length: int,
        overlap_frames: int,
        add_augmentation: bool = True,
    ):
        self.file_paths = file_paths
        self.feature_dim = feature_dim
        self.scaler = scaler
        self.target_length = target_length
        self.overlap_frames = overlap_frames
        self.add_augmentation = add_augmentation

        # Load the feature info from the first file to get dimensions
        if self.file_paths:
            with open(self.file_paths[0], "rb") as f:
                landmark_data = pickle.load(f)
                self.feature_info = landmark_data.get("feature_info")
        else:
            self.feature_info = None

        self.all_pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple]:
        """Generate all training pairs by iterating through file paths once"""
        pairs = []
        processor = GestureDataProcessor(landmarks_dir="")

        for file_path in self.file_paths:
            try:
                with open(file_path, "rb") as f:
                    landmark_data = pickle.load(f)

                frames = landmark_data["frames"]
                if len(frames) < self.target_length + 2 * self.overlap_frames:
                    continue

                sequence_features = [
                    processor.extract_features_from_frame(frame, self.feature_info)
                    for frame in frames
                ]
                sequence_array = np.array(sequence_features)
                normalized_seq = self.scaler.transform(sequence_array)

                for start_idx in range(
                    0,
                    len(normalized_seq)
                    - self.target_length
                    - 2 * self.overlap_frames
                    + 1,
                    5,
                ):
                    source_end = start_idx + self.overlap_frames
                    source_gesture = normalized_seq[start_idx:source_end]

                    transition_start = source_end
                    transition_end = transition_start + self.target_length
                    ground_truth = normalized_seq[transition_start:transition_end]

                    target_start = transition_end
                    target_end = target_start + self.overlap_frames
                    if target_end <= len(normalized_seq):
                        target_gesture = normalized_seq[target_start:target_end]
                        pairs.append((source_gesture, target_gesture, ground_truth))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        if self.add_augmentation:
            augmented_pairs = pairs.copy()
            for source, target, gt in pairs:
                noise_std = 0.01
                noise_source = source + np.random.normal(0, noise_std, source.shape)
                noise_target = target + np.random.normal(0, noise_std, target.shape)
                noise_gt = gt + np.random.normal(0, noise_std, gt.shape)
                augmented_pairs.append((noise_source, noise_target, noise_gt))
            pairs = augmented_pairs

        return pairs

    def __len__(self):
        """Return the total number of training pairs."""
        return len(self.all_pairs)

    def __getitem__(self, idx):
        source, target, ground_truth = self.all_pairs[idx]

        return {
            "source": torch.FloatTensor(source),
            "target": torch.FloatTensor(target),
            "ground_truth": torch.FloatTensor(ground_truth),
        }


def create_data_loaders(
    processed_data: Dict, batch_size: int = 32, add_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch data loaders with the lazy-loading dataset"""
    train_dataset = LazyGestureDataset(
        file_paths=processed_data["train_files"],
        feature_dim=processed_data["feature_dim"],
        scaler=processed_data["scaler"],
        target_length=processed_data["target_length"],
        overlap_frames=processed_data["overlap_frames"],
        add_augmentation=add_augmentation,
    )
    test_dataset = LazyGestureDataset(
        file_paths=processed_data["test_files"],
        feature_dim=processed_data["feature_dim"],
        scaler=processed_data["scaler"],
        target_length=processed_data["target_length"],
        overlap_frames=processed_data["overlap_frames"],
        add_augmentation=False,  # No augmentation on test set
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def save_processed_data(processed_data: Dict, save_path: str = PROCESSED_GESTURE_DATA_PATH):
    """Save processed data metadata to disk"""
    with open(save_path, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Saved processed data to {save_path}")


def load_processed_data(save_path: str) -> Dict:
    """Load processed data metadata from disk"""
    with open(save_path, "rb") as f:
        processed_data = pickle.load(f)
    print(f"Loaded processed data from {save_path}")
    return processed_data


# Example usage
if __name__ == "__main__":
    processor = GestureDataProcessor(LANDMARKS_DIR)

    # Process data to get file paths and scaler
    processed_data = processor.process_data(
        min_sequence_length=15,
        target_transition_length=20,
        overlap_frames=3,
        test_size=0.2,
    )

    # Save processed data metadata
    save_processed_data(processed_data)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(processed_data, batch_size=16)

    # Test data loading
    print("\nTesting data loader:")
    for batch in train_loader:
        print(f"Source shape: {batch['source'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Ground truth shape: {batch['ground_truth'].shape}")
        break
