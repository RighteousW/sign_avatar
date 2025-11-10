import os
import cv2
import torch
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque
import argparse

from ..constants import (
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
    get_gesture_metadata_path,
    get_gesture_model_path,
)

from ..model_training import GestureRecognizerCNN


class MediaPipeLandmarkExtractor:
    """MediaPipe landmark extractor matching training preprocessing"""

    def __init__(self, hand_model_path, pose_model_path, feature_info):
        self.feature_info = feature_info
        self.frame_counter = 0
        self.last_processed_features = None

        # Setup MediaPipe (same as before)
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        self.timestamp_ms = 0

    def extract_features_from_frame(self, frame_data: dict) -> np.ndarray:
        """Extract features using same logic as training script"""
        features = []

        # Hand landmarks
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

        # Pose landmarks
        if self.feature_info["pose_landmarks"] > 0:
            pose_features = np.zeros(self.feature_info["pose_landmarks"])
            if frame_data.get("pose") and frame_data["pose"]:
                landmarks = np.array(frame_data["pose"]["landmarks"])
                if len(landmarks.shape) > 1 and landmarks.shape[1] > 3:
                    landmarks = landmarks[:, :3]
                landmarks_flat = landmarks.flatten()
                pose_features[: min(len(landmarks_flat), len(pose_features))] = (
                    landmarks_flat[: len(pose_features)]
                )
            features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def process_frame(self, frame):
        """
        Process frame with 2_skip pattern matching webcam behavior.
        Process every 3rd frame, repeat last features for skipped frames.
        """
        frame_index = self.frame_counter
        self.frame_counter += 1

        should_process = frame_index % 3 == 0  # Process frames 0, 3, 6, 9...

        if should_process:
            # Process with MediaPipe
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            frame_data = {"hands": [], "pose": None}

            # Hand detection
            if self.feature_info["hand_landmarks"] > 0:
                hand_result = self.hand_landmarker.detect_for_video(
                    mp_image, self.timestamp_ms
                )
                if hand_result and hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                        frame_data["hands"].append({"landmarks": landmarks})

            # Pose detection
            if self.feature_info["pose_landmarks"] > 0:
                pose_result = self.pose_landmarker.detect_for_video(
                    mp_image, self.timestamp_ms
                )
                if pose_result and pose_result.pose_landmarks:
                    pose_landmarks = pose_result.pose_landmarks[0]
                    landmarks = [
                        [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
                        for lm in pose_landmarks
                    ]
                    frame_data["pose"] = {"landmarks": landmarks}

            self.timestamp_ms += 1
            current_features = self.extract_features_from_frame(frame_data)
            self.last_processed_features = current_features
            return [current_features]

        else:
            # Skipped frame - return last processed (or zeros if none)
            if self.last_processed_features is not None:
                return [self.last_processed_features.copy()]
            else:
                return [np.zeros(self.feature_info["total_features"], dtype=np.float32)]


class SimpleInferenceSystem:
    """Simple inference system using 2_skip model"""

    def __init__(
        self,
        hand_model_path,
        pose_model_path,
        confidence_threshold=0.7,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading 2_skip model...")

        # Load model metadata
        with open(str(get_gesture_metadata_path(False, 2)), "rb") as f:
            self.model_info = pickle.load(f)

        # Load model
        self.model = GestureRecognizerCNN(
            input_size=self.model_info["input_size"],
            num_classes=len(self.model_info["class_names"]),
            hidden_size=self.model_info["hidden_size"],
            dropout=self.model_info["dropout"],
        )

        checkpoint = torch.load(
            str(get_gesture_model_path(False, 2)), map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")

        # Initialize landmark extractor
        self.landmark_extractor = MediaPipeLandmarkExtractor(
            hand_model_path,
            pose_model_path,
            self.model_info["feature_info"],
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

        # Prepare sequence
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

        frame_times = deque(maxlen=30)

        print("Starting inference with 2_skip model")
        print("Press 'q' to quit")

        while True:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            feature_list = self.landmark_extractor.process_frame(frame)

            # Add features to queue
            for features in feature_list:
                self.feature_queue.append(features)

            # Run inference when queue is full
            if len(self.feature_queue) == self.sequence_length:
                prediction, confidence = self.run_inference()
                self.current_confidence = confidence

                if (
                    confidence > self.confidence_threshold
                    and prediction
                    and prediction != self.last_prediction
                ):
                    self.last_prediction = prediction
                    print(f"Detected: {prediction} (confidence: {confidence:.3f})")

                    # Clear some old frames after successful prediction
                    for _ in range(int(0.8 * self.sequence_length)):
                        if self.feature_queue:
                            self.feature_queue.popleft()

            # Calculate FPS
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            # Display information
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Queue: {len(self.feature_queue)}/{self.sequence_length}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if self.last_prediction:
                cv2.putText(
                    frame,
                    f"Last: {self.last_prediction} ({self.current_confidence:.2f})",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Sign Language Recognition (2_skip)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Simple sign language recognition with 2_skip model"
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
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for predictions",
    )

    args = parser.parse_args()

    # Check files exist
    required_files = [
        str(args.hand_model),
        str(args.pose_model),
        str(get_gesture_model_path(False, 2)),
        str(get_gesture_metadata_path(False, 2)),
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return

    # Run inference
    inference_system = SimpleInferenceSystem(
        hand_model_path=args.hand_model,
        pose_model_path=args.pose_model,
        confidence_threshold=args.confidence_threshold,
    )

    inference_system.run()


if __name__ == "__main__":
    main()
