import os
import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque
import argparse
from typing import Dict, Tuple

from constants import (
    GESTURE_MODEL_METADATA_PATH,
    GESTURE_MODEL_PATH,
    MODELS_TRAINED_DIR,
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
    DEFAULT_SEQUENCE_LENGTH,
)


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


class MediaPipeLandmarkExtractor:
    """MediaPipe landmark extractor using training script's feature extraction logic"""

    def __init__(self, hand_model_path, pose_model_path, feature_info):
        self.feature_info = feature_info

        # Setup MediaPipe
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        self.timestamp_ms = 0

    def extract_features_from_frame(self, frame_data: Dict) -> np.ndarray:
        """Extract features using EXACT same logic as training script"""
        features = []

        # Hand landmarks - exact same logic as LandmarkDataLoader
        if self.feature_info["hand_landmarks"] > 0:
            max_hands = self.feature_info["max_hands"]
            hand_dim_per_hand = self.feature_info["hand_landmarks_per_hand"]
            hand_features = np.zeros(self.feature_info["hand_landmarks"])

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

        # Pose landmarks - exact same logic as LandmarkDataLoader
        if self.feature_info["pose_landmarks"] > 0:
            pose_features = np.zeros(self.feature_info["pose_landmarks"])
            if frame_data.get("pose") and frame_data["pose"]:
                # Only use x,y,z coordinates, skip visibility for consistency with training
                landmarks = np.array(frame_data["pose"]["landmarks"])
                if len(landmarks.shape) > 1 and landmarks.shape[1] > 3:
                    landmarks = landmarks[:, :3]  # Take only x,y,z
                landmarks_flat = landmarks.flatten()
                pose_features[: min(len(landmarks_flat), len(pose_features))] = (
                    landmarks_flat[: len(pose_features)]
                )
            features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def process_frame(self, frame) -> Tuple[np.ndarray, Dict]:
        """Process frame and extract features"""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        hand_result = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += 1

        frame_data = {"hands": [], "pose": None}

        # Process hand results
        if hand_result and hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                frame_data["hands"].append({"landmarks": landmarks})

        # Process pose results
        if pose_result and pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            landmarks = [
                [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
                for lm in pose_landmarks
            ]
            frame_data["pose"] = {"landmarks": landmarks}

        # Extract features using training logic
        features = self.extract_features_from_frame(frame_data)
        return features, frame_data


class InferenceSystem:
    """Minimal real-time sign language inference system"""

    def __init__(
        self, model_dir, hand_model_path, pose_model_path, confidence_threshold=0.8
    ):
        self.confidence_threshold = confidence_threshold

        # Load model info (contains feature_info from training)
        with open(GESTURE_MODEL_METADATA_PATH, "rb") as f:
            self.model_info = pickle.load(f)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignLanguageModel(
            input_size=self.model_info["input_size"],
            num_classes=len(self.model_info["class_names"]),
            hidden_size=self.model_info["hidden_size"],
            dropout=self.model_info["dropout"],
        )

        checkpoint = torch.load(GESTURE_MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize landmark extractor with feature_info from training
        self.landmark_extractor = MediaPipeLandmarkExtractor(
            hand_model_path, pose_model_path, self.model_info["feature_info"]
        )

        # Sequence management
        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)

        # State
        self.last_prediction = ""
        self.current_confidence = 0.0

    def pad_or_truncate_sequence(self, sequence, target_length, feature_size):
        """Same padding logic as training script"""
        if len(sequence) > target_length:
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            padded = list(sequence)
            while len(padded) < target_length:
                padded.append(np.zeros(feature_size))
            return np.array(padded)

    def run_inference(self):
        """Run inference on current sequence"""
        if len(self.feature_queue) < self.sequence_length:
            return None, 0.0

        # Use same padding logic as training
        sequence = self.pad_or_truncate_sequence(
            list(self.feature_queue),
            self.sequence_length,
            self.model_info["feature_info"]["total_features"],
        )

        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.model_info["class_names"][predicted.item()]
            confidence_value = confidence.item()

        return predicted_class, confidence_value

    def run(self):
        """Main inference loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print(f"Loaded model with {len(self.model_info['class_names'])} classes")
        print(
            f"Feature dimensions: {self.model_info['feature_info']['total_features']}"
        )
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            features, frame_data = self.landmark_extractor.process_frame(frame)
            self.feature_queue.append(features)

            # Run inference when queue is full
            if len(self.feature_queue) == self.sequence_length:
                prediction, confidence = self.run_inference()
                self.current_confidence = confidence

                if (
                    confidence > self.confidence_threshold
                    and prediction != self.last_prediction
                ):
                    self.last_prediction = prediction
                    print(f"Detected: {prediction} (confidence: {confidence:.3f})")

                    # Clear some old frames after successful prediction
                    for _ in range(int(0.8 * self.sequence_length)):
                        if self.feature_queue:
                            self.feature_queue.popleft()

            # Simple display
            display_text = f"Queue: {len(self.feature_queue)}/{self.sequence_length}"
            if self.last_prediction:
                display_text += (
                    f" | Last: {self.last_prediction} ({self.current_confidence:.2f})"
                )

            cv2.putText(
                frame,
                display_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Sign Language Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time sign language recognition")
    parser.add_argument(
        "--model_dir",
        default=MODELS_TRAINED_DIR,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--hand_model",
        default=MEDIAPIPE_HAND_LANDMARKER_PATH,
        help="MediaPipe hand model",
    )
    parser.add_argument(
        "--pose_model",
        default=MEDIAPIPE_POSE_LANDMARKER_PATH,
        help="MediaPipe pose model",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.8, help="Confidence threshold"
    )

    args = parser.parse_args()

    # Check files exist
    required_files = [
        GESTURE_MODEL_METADATA_PATH,
        GESTURE_MODEL_PATH,
        args.hand_model,
        args.pose_model,
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return

    # Run inference
    inference_system = InferenceSystem(
        model_dir=args.model_dir,
        hand_model_path=args.hand_model,
        pose_model_path=args.pose_model,
        confidence_threshold=args.confidence_threshold,
    )

    inference_system.run()


if __name__ == "__main__":
    main()
