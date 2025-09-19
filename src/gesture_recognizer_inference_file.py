import os
import cv2
import torch
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import argparse
from typing import Dict
import tkinter as tk
from tkinter import filedialog

from constants import (
    MODELS_TRAINED_DIR,
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
)


# Import the model classes and feature config from the original script
class FeatureConfig:
    """Configuration class to control which features to extract and define their dimensions."""

    def __init__(
        self,
        use_hand_landmarks: bool = True,
        use_hand_connections: bool = True,
        use_pose_landmarks: bool = True,
        use_pose_connections: bool = True,
        max_hands: int = 2,
    ):
        self.use_hand_landmarks = use_hand_landmarks
        self.use_hand_connections = use_hand_connections
        self.use_pose_landmarks = use_pose_landmarks
        self.use_pose_connections = use_pose_connections
        self.max_hands = max_hands

        # Define dimensions for a single hand
        self.hand_landmarks_dim_per_hand = 21 * 3
        self.hand_connections_dim_per_hand = 21

        # Calculate total hand feature dimensions based on max_hands
        self.hand_landmarks_dim = (
            self.hand_landmarks_dim_per_hand * self.max_hands
            if use_hand_landmarks
            else 0
        )
        self.hand_connections_dim = (
            self.hand_connections_dim_per_hand * self.max_hands
            if use_hand_connections
            else 0
        )

        # Pose dimensions
        self.pose_landmarks_dim = 132 if use_pose_landmarks else 0
        self.pose_connections_dim = 35 if use_pose_connections else 0

        # Calculate the total feature size
        self.feature_size = (
            self.hand_landmarks_dim
            + self.hand_connections_dim
            + self.pose_landmarks_dim
            + self.pose_connections_dim
        )


import torch.nn as nn


class SignLanguageModel(nn.Module):
    """Model architecture matching the training script"""

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


class FileBasedLandmarkExtractor:
    """MediaPipe landmark extractor for video file processing"""

    def __init__(self, hand_model_path, pose_model_path, feature_config):
        self.feature_config = feature_config

        # Initialize hand landmarker for VIDEO mode
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # Initialize pose landmarker for VIDEO mode
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Store connection information
        self.hand_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
        self.pose_connections = list(mp.solutions.pose.POSE_CONNECTIONS)

    def calculate_connection_features(self, landmarks, connections):
        """Calculate connection features (distances between connected landmarks)"""
        if not landmarks:
            return []

        connection_features = []
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = np.array(landmarks[start_idx][:3])  # x, y, z
                end_point = np.array(landmarks[end_idx][:3])
                distance = np.linalg.norm(end_point - start_point)
                connection_features.append(distance)
            else:
                connection_features.append(0.0)  # Missing landmark

        return connection_features

    def extract_features_from_frame(self, frame_data: Dict) -> np.ndarray:
        """Extract features from a single frame based on configuration"""
        features = []

        # Hand features
        for hand_idx in range(self.feature_config.max_hands):
            # Hand landmarks
            if self.feature_config.use_hand_landmarks:
                hand_features = np.zeros(
                    self.feature_config.hand_landmarks_dim_per_hand
                )
                if (
                    frame_data.get("hands")
                    and hand_idx < len(frame_data["hands"])
                    and frame_data["hands"][hand_idx]
                ):
                    hand_data = frame_data["hands"][hand_idx]
                    landmarks = hand_data.get("landmarks", [])
                    if isinstance(landmarks, list) and len(landmarks) >= 21:
                        flat_landmarks = np.array(landmarks[:21]).flatten()
                        hand_features[
                            : min(len(flat_landmarks), len(hand_features))
                        ] = flat_landmarks[: len(hand_features)]
                features.extend(hand_features)

            # Hand connections
            if self.feature_config.use_hand_connections:
                hand_conn_features = np.zeros(
                    self.feature_config.hand_connections_dim_per_hand
                )
                if (
                    frame_data.get("connection_features", {}).get("hands")
                    and hand_idx < len(frame_data["connection_features"]["hands"])
                    and frame_data["connection_features"]["hands"][hand_idx]
                ):
                    conn_feats = frame_data["connection_features"]["hands"][hand_idx]
                    hand_conn_features[
                        : min(len(conn_feats), len(hand_conn_features))
                    ] = conn_feats[: len(hand_conn_features)]
                features.extend(hand_conn_features)

        # Pose landmarks
        if self.feature_config.use_pose_landmarks:
            pose_features = np.zeros(self.feature_config.pose_landmarks_dim)
            if frame_data.get("pose") and "landmarks" in frame_data["pose"]:
                pose_landmarks = np.array(frame_data["pose"]["landmarks"]).flatten()
                pose_features[: min(len(pose_landmarks), len(pose_features))] = (
                    pose_landmarks[: len(pose_features)]
                )
            features.extend(pose_features)

        # Pose connections
        if self.feature_config.use_pose_connections:
            pose_conn_features = np.zeros(self.feature_config.pose_connections_dim)
            if frame_data.get("connection_features", {}).get("pose"):
                conn_feats = frame_data["connection_features"]["pose"]
                pose_conn_features[: min(len(conn_feats), len(pose_conn_features))] = (
                    conn_feats[: len(pose_conn_features)]
                )
            features.extend(pose_conn_features)

        return np.array(features, dtype=np.float32)

    def process_frame(self, frame, timestamp_ms):
        """Process a single frame and extract landmark features"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        frame_data = {
            "hands": [],
            "pose": None,
            "connection_features": {"hands": [], "pose": []},
        }

        # Process hand landmarks
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        if hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]

                # Calculate hand connection features
                hand_connections = self.calculate_connection_features(
                    landmarks, self.hand_connections
                )

                # Ensure we have exactly 21 connection features for hands
                if len(hand_connections) == 20:  # Standard MediaPipe connections
                    # Add extra connection to match expected dimension
                    extra_connection = np.linalg.norm(
                        np.array(landmarks[0]) - np.array(landmarks[9])
                    )
                    hand_connections.append(extra_connection)

                hand_data = {
                    "hand_index": i,
                    "handedness": (
                        hand_result.handedness[i][0].category_name
                        if hand_result.handedness
                        else "Unknown"
                    ),
                    "landmarks": landmarks,
                    "connection_features": hand_connections,
                }
                frame_data["hands"].append(hand_data)
                frame_data["connection_features"]["hands"].append(hand_connections)

        # Process pose landmarks
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            landmarks = []
            for lm in pose_landmarks:
                landmarks.append([lm.x, lm.y, lm.z, getattr(lm, "visibility", None)])

            # Calculate pose connection features
            pose_connections = self.calculate_connection_features(
                landmarks, self.pose_connections
            )

            frame_data["pose"] = {
                "landmarks": landmarks,
                "connection_features": pose_connections,
            }
            frame_data["connection_features"]["pose"] = pose_connections

        # Extract features
        features = self.extract_features_from_frame(frame_data)
        return features, frame_data


class FileInferenceSystem:
    """File-based sign language inference system"""

    def __init__(
        self, model_dir, hand_model_path, pose_model_path, confidence_threshold=0.8
    ):
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold

        # Load model info
        with open(os.path.join(model_dir, "model_info.pkl"), "rb") as f:
            self.model_info = pickle.load(f)

        # Create feature config
        feature_config_data = self.model_info["feature_config"]
        self.feature_config = FeatureConfig(**feature_config_data)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignLanguageModel(
            input_size=self.model_info["input_size"],
            num_classes=len(self.model_info["class_names"]),
            hidden_size=self.model_info["hidden_size"],
            dropout=self.model_info["dropout"],
        )

        # Load trained weights
        checkpoint = torch.load(
            os.path.join(model_dir, "best_model.pth"), map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize landmark extractor
        self.landmark_extractor = FileBasedLandmarkExtractor(
            hand_model_path, pose_model_path, self.feature_config
        )

        # Sequence management
        self.sequence_length = self.model_info["sequence_length"]

        print(f"Model loaded: {len(self.model_info['class_names'])} classes")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Feature size: {self.feature_config.feature_size}")

    def select_video_file(self):
        """Open file dialog to select AVI file"""
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        file_path = filedialog.askopenfilename(
            title="Select AVI video file",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")],
        )

        root.destroy()
        return file_path

    def extract_features_from_video(self, video_path):
        """Extract features from all frames in video"""
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")

        features_list = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp in milliseconds
            timestamp_ms = int((frame_count / fps) * 1000)

            # Extract features
            features, _ = self.landmark_extractor.process_frame(frame, timestamp_ms)
            features_list.append(features)

            frame_count += 1

            # Progress indicator
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(
                    f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)"
                )

        cap.release()
        print(f"Feature extraction complete: {len(features_list)} frames processed")

        return np.array(features_list), fps

    def create_overlapping_chunks(self, features, overlap_percent=10):
        """Split features into overlapping chunks"""
        if len(features) < self.sequence_length:
            print(
                f"Warning: Video has {len(features)} frames, but sequence length is {self.sequence_length}"
            )
            if len(features) == 0:
                return []

            # For short videos, pad with zeros to reach sequence_length
            padded_features = np.zeros((self.sequence_length, features.shape[1]))
            padded_features[: len(features)] = features
            return [(padded_features, 0)]

        # Calculate step size (how many frames to advance each time)
        step_size = max(1, int(self.sequence_length * (100 - overlap_percent) / 100))

        chunks = []
        start_idx = 0

        while start_idx + self.sequence_length <= len(features):
            chunk = features[start_idx : start_idx + self.sequence_length]
            chunks.append((chunk, start_idx))
            start_idx += step_size

        print(
            f"Created {len(chunks)} chunks with {overlap_percent}% overlap (step size: {step_size})"
        )
        return chunks

    def run_inference_on_chunk(self, chunk):
        """Run inference on a single chunk"""
        if len(chunk) != self.sequence_length:
            print(
                f"Warning: Chunk has {len(chunk)} frames, expected {self.sequence_length}"
            )
            return None, 0.0

        # Convert to tensor
        sequence_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.model_info["class_names"][predicted.item()]
            confidence_value = confidence.item()

        return predicted_class, confidence_value

    def process_video_file(self, video_path, overlap_percent=10):
        """Process entire video file with chunking and inference"""
        print("\n" + "=" * 50)
        print("STARTING VIDEO PROCESSING")
        print("=" * 50)

        # Extract features from video
        start_time = time.time()
        features, fps = self.extract_features_from_video(video_path)
        extraction_time = time.time() - start_time
        print(f"Feature extraction took: {extraction_time:.2f} seconds")

        if len(features) == 0:
            print("No features extracted from video!")
            return

        # Create overlapping chunks
        chunks = self.create_overlapping_chunks(features, overlap_percent)

        if not chunks:
            print("No chunks created!")
            return

        print("\n" + "=" * 50)
        print("RUNNING INFERENCE ON CHUNKS")
        print("=" * 50)

        results = []

        for i, (chunk, start_frame) in enumerate(chunks):
            # Calculate time range for this chunk
            start_time_sec = start_frame / fps
            end_time_sec = (start_frame + self.sequence_length) / fps

            # Run inference
            prediction, confidence = self.run_inference_on_chunk(chunk)

            if prediction is not None:
                results.append(
                    {
                        "chunk_id": i + 1,
                        "start_frame": start_frame,
                        "end_frame": start_frame + self.sequence_length - 1,
                        "start_time": start_time_sec,
                        "end_time": end_time_sec,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                )

                # Print result
                status = "✓" if confidence > self.confidence_threshold else " "
                print(
                    f"Chunk {i+1:3d}: [{start_frame:4d}-{start_frame + self.sequence_length - 1:4d}] "
                    f"({start_time_sec:6.2f}s-{end_time_sec:6.2f}s) -> "
                    f"{prediction:15s} (conf: {confidence:.3f}) {status}"
                )

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)

        if results:
            high_conf_results = [
                r for r in results if r["confidence"] > self.confidence_threshold
            ]
            print(f"Total chunks processed: {len(results)}")
            print(
                f"High confidence predictions (>{self.confidence_threshold}): {len(high_conf_results)}"
            )

            if high_conf_results:
                print("\nHigh confidence predictions:")
                for result in high_conf_results:
                    print(
                        f"  {result['start_time']:6.2f}s-{result['end_time']:6.2f}s: "
                        f"{result['prediction']} (conf: {result['confidence']:.3f})"
                    )
        else:
            print("No results generated!")

        return results


def main():
    parser = argparse.ArgumentParser(description="File-based sign language recognition")
    parser.add_argument(
        "--model_dir",
        default=MODELS_TRAINED_DIR,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--hand_model",
        default=MEDIAPIPE_HAND_LANDMARKER_PATH,
        help="Path to MediaPipe hand landmark .task model",
    )
    parser.add_argument(
        "--pose_model",
        default=MEDIAPIPE_POSE_LANDMARKER_PATH,
        help="Path to MediaPipe pose landmark .task model",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--overlap_percent",
        type=int,
        default=10,
        help="Overlap percentage between chunks (default: 10%)",
    )
    parser.add_argument(
        "--video_file",
        help="Path to AVI video file (if not provided, file dialog will open)",
    )

    args = parser.parse_args()

    # Check if model files exist
    model_info_path = os.path.join(args.model_dir, "model_info.pkl")
    model_weights_path = os.path.join(args.model_dir, "best_model.pth")

    if not os.path.exists(model_info_path):
        print(f"Error: Model info not found: {model_info_path}")
        return

    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights not found: {model_weights_path}")
        return

    if not os.path.exists(args.hand_model):
        print(f"Error: Hand model file not found: {args.hand_model}")
        return

    if not os.path.exists(args.pose_model):
        print(f"Error: Pose model file not found: {args.pose_model}")
        return

    try:
        # Initialize inference system
        print("Initializing file inference system...")
        inference_system = FileInferenceSystem(
            model_dir=args.model_dir,
            hand_model_path=args.hand_model,
            pose_model_path=args.pose_model,
            confidence_threshold=args.confidence_threshold,
        )

        # Get video file
        if args.video_file:
            video_path = args.video_file
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                return
        else:
            print("Opening file dialog to select AVI video...")
            video_path = inference_system.select_video_file()
            if not video_path:
                print("No file selected. Exiting.")
                return

        # Process the video
        print(
            f"Processing video with {args.overlap_percent}% overlap between chunks..."
        )
        results = inference_system.process_video_file(video_path, args.overlap_percent)

    except Exception as e:
        print(f"Error running file inference system: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
