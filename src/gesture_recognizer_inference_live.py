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
    MODELS_TRAINED_DIR,
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
    DEFAULT_SEQUENCE_LENGTH,
)


class FeatureConfig:
    """Configuration class to control which features to extract and define their dimensions.
    Updated for live inference to use only hand and pose landmarks."""

    def __init__(
        self,
        use_hand_landmarks: bool = True,
        use_pose_landmarks: bool = True,
        max_hands: int = 2,
    ):
        self.use_hand_landmarks = use_hand_landmarks
        # Hand connections are no longer used
        self.use_hand_connections = False
        self.use_pose_landmarks = use_pose_landmarks
        # Pose connections are no longer used
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
        self.feature_size = (
            self.hand_landmarks_dim + self.pose_landmarks_dim
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


class MediaPipeLandmarkExtractor:
    """MediaPipe landmark extractor for real-time inference, with an option for async."""

    def __init__(self, hand_model_path, pose_model_path, feature_config, async_mode: bool = False):
        self.feature_config = feature_config
        self.async_mode = async_mode
        self.pose_landmark_indices = [0, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

        # Set up MediaPipe based on the chosen mode
        if self.async_mode:
            self._setup_async_mode(hand_model_path, pose_model_path)
        else:
            self._setup_sync_mode(hand_model_path, pose_model_path)
            self.timestamp_ms = 0 # To simulate a timestamp for `detect_for_video`

    def _setup_async_mode(self, hand_model_path, pose_model_path):
        """Initializes MediaPipe in LIVE_STREAM mode for async processing."""
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._hand_result_callback,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self._pose_result_callback,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Results storage for async mode
        self.latest_hand_result = None
        self.latest_pose_result = None
        self.timestamp_counter = 0

    def _setup_sync_mode(self, hand_model_path, pose_model_path):
        """Initializes MediaPipe in VIDEO mode for sync processing."""
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

    def _hand_result_callback(self, result, output_image, timestamp_ms):
        """Callback for hand detection results in async mode."""
        self.latest_hand_result = result

    def _pose_result_callback(self, result, output_image, timestamp_ms):
        """Callback for pose detection results in async mode."""
        self.latest_pose_result = result

    def _calculate_distance(self, p1, p2):
        """Calculates Euclidean distance between two 3D points."""
        return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))


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

                if self.feature_config.use_hand_landmarks and isinstance(landmarks, list) and len(landmarks) >= 21:
                    # Flatten the coordinates of all 21 landmarks
                    flat_landmarks = np.array(landmarks[:21])[:, :3].flatten()
                    hand_features[:len(flat_landmarks)] = flat_landmarks

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

    def process_frame(self, frame) -> Tuple[np.ndarray, Dict]:
        """Process a single frame and extract landmark features"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        hand_result = None
        pose_result = None

        if self.async_mode:
            timestamp_ms = self.timestamp_counter
            self.timestamp_counter += 1
            
            # Submit to MediaPipe processors
            self.hand_landmarker.detect_async(mp_image, timestamp_ms)
            self.pose_landmarker.detect_async(mp_image, timestamp_ms)

            # Small delay to allow processing
            time.sleep(0.001)

            # Use the latest results from the callbacks
            hand_result = self.latest_hand_result
            pose_result = self.latest_pose_result

        else: # Synchronous mode
            hand_result = self.hand_landmarker.detect_for_video(mp_image, self.timestamp_ms)
            pose_result = self.pose_landmarker.detect_for_video(mp_image, self.timestamp_ms)
            self.timestamp_ms += 1

        frame_data = {
            "hands": [],
            "pose": None,
        }

        # Process hand results
        if hand_result and hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                hand_data = {
                    "hand_index": i,
                    "handedness": (
                        hand_result.handedness[i][0].category_name
                        if hand_result.handedness
                        else "Unknown"
                    ),
                    "landmarks": landmarks,
                }
                frame_data["hands"].append(hand_data)

        # Process pose results
        if pose_result and pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            landmarks = []
            for lm in pose_landmarks:
                landmarks.append([lm.x, lm.y, lm.z, getattr(lm, "visibility", None)])

            frame_data["pose"] = {
                "landmarks": landmarks,
            }

        # Extract features
        features = self.extract_features_from_frame(frame_data)
        return features, frame_data


class InferenceSystem:
    """Real-time sign language inference system"""

    def __init__(
        self, model_dir, hand_model_path, pose_model_path, confidence_threshold=0.8, async_mode: bool = False
    ):
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold

        # Load model info
        with open(os.path.join(model_dir, "model_info.pkl"), "rb") as f:
            self.model_info = pickle.load(f)

        # Create feature config
        feature_config_data = self.model_info["feature_config"]
        # Only use necessary parameters
        self.feature_config = FeatureConfig(
            use_hand_landmarks=feature_config_data.get("use_hand_landmarks", True),
            use_pose_landmarks=feature_config_data.get("use_pose_landmarks", True),
            max_hands=feature_config_data.get("max_hands", 2),
        )

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
        self.landmark_extractor = MediaPipeLandmarkExtractor(
            hand_model_path, pose_model_path, self.feature_config, async_mode
        )

        # Sequence management
        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)

        # State variables
        self.last_unique_result = ""
        self.current_prediction = ""
        self.current_confidence = 0.0

        # UI state
        self.show_landmarks = True
        self.paused = False

        # Performance tracking
        self.capture_times = deque(maxlen=30)
        self.mediapipe_times = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)

    def run_inference(self):
        """Run inference on current sequence"""
        if len(self.feature_queue) < self.sequence_length:
            return None, 0.0

        start_time = time.time()

        # Convert to tensor
        sequence = np.array(list(self.feature_queue))
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.model_info["class_names"][predicted.item()]
            confidence_value = confidence.item()

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return predicted_class, confidence_value

    def draw_landmarks(self, frame, frame_data):
        """Draw landmarks on frame if enabled"""
        if not self.show_landmarks:
            return frame

        annotated_frame = frame.copy()

        # Draw hand landmarks
        if frame_data.get("hands"):
            for hand_data in frame_data["hands"]:
                landmarks = hand_data["landmarks"]
                h, w = frame.shape[:2]

                # Draw landmarks
                for landmark in landmarks:
                    x, y = int(landmark[0] * w), int(landmark[1] * h)
                    cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

                # Draw connections
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = (
                            int(landmarks[start_idx][0] * w),
                            int(landmarks[start_idx][1] * h),
                        )
                        end_point = (
                            int(landmarks[end_idx][0] * w),
                            int(landmarks[end_idx][1] * h),
                        )
                        cv2.line(
                            annotated_frame, start_point, end_point, (255, 0, 0), 2
                        )

        # Draw pose landmarks
        if frame_data.get("pose"):
            landmarks = frame_data["pose"]["landmarks"]
            h, w = frame.shape[:2]
            
            # Use the same pose landmark indices to draw
            pose_indices_to_draw = self.landmark_extractor.pose_landmark_indices

            # Draw landmarks
            for idx in pose_indices_to_draw:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    x, y = int(landmark[0] * w), int(landmark[1] * h)
                    cv2.circle(annotated_frame, (x, y), 2, (0, 0, 255), -1)

            # Draw connections
            for connection in mp.solutions.pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                # Only draw if both points are in the selected subset
                if start_idx in pose_indices_to_draw and end_idx in pose_indices_to_draw:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = (
                            int(landmarks[start_idx][0] * w),
                            int(landmarks[start_idx][1] * h),
                        )
                        end_point = (
                            int(landmarks[end_idx][0] * w),
                            int(landmarks[end_idx][1] * h),
                        )
                        cv2.line(annotated_frame, start_point, end_point, (255, 255, 0), 1)

        return annotated_frame

    def draw_statistics(self, frame, fps):
        """Draw statistics panel"""
        h, w = frame.shape[:2]
        stats_height = int(h * 0.2)
        stats_panel = np.zeros((stats_height, int(w * 0.8), 3), dtype=np.uint8)

        y_offset = 30
        line_height = 25

        # Performance stats
        avg_capture = np.mean(self.capture_times) * 1000 if self.capture_times else 0
        avg_mediapipe = (
            np.mean(self.mediapipe_times) * 1000 if self.mediapipe_times else 0
        )
        avg_inference = (
            np.mean(self.inference_times) * 1000 if self.inference_times else 0
        )

        cv2.putText(
            stats_panel,
            f"FPS: {fps:.1f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y_offset += line_height

        cv2.putText(
            stats_panel,
            f"Capture: {avg_capture:.1f}ms",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y_offset += line_height

        cv2.putText(
            stats_panel,
            f"MediaPipe: {avg_mediapipe:.1f}ms",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y_offset += line_height

        cv2.putText(
            stats_panel,
            f"Inference: {avg_inference:.1f}ms",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y_offset += line_height

        cv2.putText(
            stats_panel,
            f"Queue: {len(self.feature_queue)}/{self.sequence_length}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        y_offset += line_height + 10

        # Controls
        cv2.putText(
            stats_panel,
            "Controls:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )
        y_offset += line_height

        cv2.putText(
            stats_panel,
            "L - Toggle landmarks",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_offset += line_height - 5

        cv2.putText(
            stats_panel,
            "P - Pause/Resume",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        y_offset += line_height - 5

        cv2.putText(
            stats_panel,
            "Q - Quit",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return stats_panel

    def draw_inference_panel(self, frame):
        """Draw inference results panel"""
        h, w = frame.shape[:2]
        panel_width = int(w * 0.2)
        inference_panel = np.zeros((h, panel_width, 3), dtype=np.uint8)

        # Title
        cv2.putText(
            inference_panel,
            "Predictions",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        y_offset = 70

        # Current prediction
        if self.current_prediction:
            # Prediction text
            cv2.putText(
                inference_panel,
                "Current:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 25

            cv2.putText(
                inference_panel,
                self.current_prediction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            y_offset += 35

            # Confidence
            cv2.putText(
                inference_panel,
                f"Conf: {self.current_confidence:.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 35

            # Confidence bar
            bar_width = panel_width - 20
            bar_height = 15
            cv2.rectangle(
                inference_panel,
                (10, y_offset),
                (10 + bar_width, y_offset + bar_height),
                (100, 100, 100),
                -1,
            )

            conf_width = int(self.current_confidence * bar_width)
            color = (
                (0, 255, 0)
                if self.current_confidence > self.confidence_threshold
                else (0, 165, 255)
            )
            cv2.rectangle(
                inference_panel,
                (10, y_offset),
                (10 + conf_width, y_offset + bar_height),
                color,
                -1,
            )

            y_offset += 40

        # Last unique result
        if self.last_unique_result:
            cv2.putText(
                inference_panel,
                "Last unique:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 25

            cv2.putText(
                inference_panel,
                self.last_unique_result,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        # Threshold line
        threshold_y = 200
        cv2.line(
            inference_panel,
            (10, threshold_y),
            (panel_width - 10, threshold_y),
            (255, 0, 0),
            1,
        )
        cv2.putText(
            inference_panel,
            f"Threshold: {self.confidence_threshold}",
            (10, threshold_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )

        return inference_panel

    def run(self):
        """Main inference loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Create window
        cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            "Sign Language Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0

        print("Starting inference system...")
        print(f"Model loaded: {len(self.model_info['class_names'])} classes")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"MediaPipe mode: {'Async' if self.landmark_extractor.async_mode else 'Sync'}")
        print("Press 'q' to quit, 'l' to toggle landmarks, 'p' to pause")

        while True:
            if not self.paused:
                # Capture frame
                capture_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                capture_time = time.time() - capture_start
                self.capture_times.append(capture_time)

                # Process with MediaPipe
                mediapipe_start = time.time()
                features, frame_data = self.landmark_extractor.process_frame(frame)
                mediapipe_time = time.time() - mediapipe_start
                self.mediapipe_times.append(mediapipe_time)

                # Add to queue
                self.feature_queue.append(features)

                # Run inference when queue is full
                if len(self.feature_queue) == self.sequence_length:
                    prediction, confidence = self.run_inference()

                    self.current_prediction = prediction
                    self.current_confidence = confidence

                    # Check if confidence is above threshold
                    if confidence > self.confidence_threshold:
                        if prediction != self.last_unique_result:
                            self.last_unique_result = prediction
                            print(
                                f"New sign detected: {prediction} (confidence: {confidence:.3f})"
                            )

                        # Clear 80% of old frames
                        clear_count = int(0.8 * self.sequence_length)
                        for _ in range(clear_count):
                            if self.feature_queue:
                                self.feature_queue.popleft()
            else:
                # When paused, just read frame but don't process
                ret, frame = cap.read()
                if not ret:
                    break
                frame_data = {"hands": [], "pose": None}

            # Draw frame with landmarks
            main_frame = self.draw_landmarks(frame, frame_data)

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # Create layout
            h, w = main_frame.shape[:2]

            # Main frame (top left 80% x 80%)
            main_height = int(h * 0.8)
            main_width = int(w * 0.8)
            main_resized = cv2.resize(main_frame, (main_width, main_height))

            # Statistics panel (bottom left 80% x 20%)
            stats_panel = self.draw_statistics(main_frame, current_fps)

            # Inference panel (right 20% x 100%)
            inference_panel = self.draw_inference_panel(main_frame)

            # Combine panels
            left_panel = np.vstack([main_resized, stats_panel])
            full_display = np.hstack([left_panel, inference_panel])

            # Add pause indicator
            if self.paused:
                cv2.putText(
                    full_display,
                    "PAUSED",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    3,
                )

            cv2.imshow("Sign Language Recognition", full_display)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks display: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord("p"):
                self.paused = not self.paused
                print(f"Processing: {'PAUSED' if self.paused else 'RESUMED'}")

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
        "--sequence_length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Override sequence length from model",
    )
    parser.add_argument(
        "--async_mediapipe",
        default=False,
        help="Use MediaPipe's asynchronous live stream mode",
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
        print("Initializing inference system...")
        inference_system = InferenceSystem(
            model_dir=args.model_dir,
            hand_model_path=args.hand_model,
            pose_model_path=args.pose_model,
            confidence_threshold=args.confidence_threshold,
            async_mode=args.async_mediapipe,
        )

        # Override sequence length if specified
        if args.sequence_length != DEFAULT_SEQUENCE_LENGTH:
            inference_system.sequence_length = args.sequence_length
            inference_system.feature_queue = deque(maxlen=args.sequence_length)

        # Run the system
        inference_system.run()

    except Exception as e:
        print(f"Error running inference system: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()