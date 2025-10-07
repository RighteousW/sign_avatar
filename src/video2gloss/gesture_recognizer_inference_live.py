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
from typing import Dict, Tuple, List, Optional

from constants import (
    GESTURE_MODEL_METADATA_PATH,
    GESTURE_MODEL_PATH,
    MEDIAPIPE_HAND_LANDMARKER_PATH,
    MEDIAPIPE_POSE_LANDMARKER_PATH,
)


# Helper functions for different model names
def get_model_path(skip_pattern: int):
    """Get model file path for given skip pattern"""
    skip_names = {0: "no_skip", 1: "1_skip", 2: "2_skip"}
    if skip_pattern not in skip_names:
        return GESTURE_MODEL_PATH
    suffix = skip_names[skip_pattern]
    # Use Path operations for pathlib compatibility
    model_path = GESTURE_MODEL_PATH
    if hasattr(model_path, "parent"):  # pathlib.Path
        return model_path.parent / f"{model_path.stem}_{suffix}{model_path.suffix}"
    else:  # string path
        return model_path.replace(".pth", f"_{suffix}.pth")


def get_metadata_path(skip_pattern: int):
    """Get metadata file path for given skip pattern"""
    skip_names = {0: "no_skip", 1: "1_skip", 2: "2_skip"}
    if skip_pattern not in skip_names:
        return GESTURE_MODEL_METADATA_PATH
    suffix = skip_names[skip_pattern]
    # Use Path operations for pathlib compatibility
    metadata_path = GESTURE_MODEL_METADATA_PATH
    if hasattr(metadata_path, "parent"):  # pathlib.Path
        return (
            metadata_path.parent
            / f"{metadata_path.stem}_{suffix}{metadata_path.suffix}"
        )
    else:  # string path
        return metadata_path.replace(".pkl", f"_{suffix}.pkl")


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
    """MediaPipe landmark extractor with frame drop detection and interpolation"""

    def __init__(
        self, hand_model_path, pose_model_path, feature_info, skip_processing=0
    ):
        self.feature_info = feature_info
        self.skip_processing = (
            skip_processing  # 0=process all, 1=skip every other, 2=skip 2/3
        )
        self.frame_counter = 0
        self.recent_features = deque(
            maxlen=10
        )  # Store recent processed features for interpolation

        # Setup MediaPipe
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

        # Frame timing for drop detection
        self.last_frame_time = time.time()
        self.expected_fps = 30
        self.frame_drop_threshold = 1.5 / self.expected_fps  # 1.5x expected frame time
        self.frame_buffer = []  # Buffer for interpolation
        self.last_processed_features = None

    def interpolate_features(self) -> np.ndarray:
        """Interpolate features based on recent processed frames"""
        if len(self.recent_features) < 2:
            # Not enough data for interpolation, return zeros
            return np.zeros(self.feature_info["total_features"], dtype=np.float32)

        # Simple linear interpolation between last two processed frames
        if len(self.recent_features) >= 2:
            return (self.recent_features[-1] + self.recent_features[-2]) / 2
        else:
            return self.recent_features[-1]

    def interpolate_features_advanced(self) -> np.ndarray:
        """Advanced interpolation using multiple recent frames"""
        if len(self.recent_features) < 2:
            return np.zeros(self.feature_info["total_features"], dtype=np.float32)

        features_array = np.array(list(self.recent_features))

        if len(features_array) >= 4:
            # Use weighted average with more weight on recent frames
            weights = np.array([0.1, 0.2, 0.3, 0.4])[: len(features_array)]
            weights = weights[-len(features_array) :]  # Take last N weights
            weights = weights / weights.sum()  # Normalize

            return np.average(features_array, axis=0, weights=weights)
        elif len(features_array) >= 2:
            # Linear interpolation between last two
            return (features_array[-1] + features_array[-2]) / 2
        else:
            return features_array[-1]

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

    def detect_frame_drops(self) -> bool:
        """Detect if frames are being dropped based on timing"""
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Return True if frame time is significantly longer than expected
        return time_diff > self.frame_drop_threshold


    def process_frame(self, frame) -> List[Tuple[np.ndarray, Dict, bool]]:
        """
        Process frame and return list of features (processed + interpolated).
        Returns multiple feature sets when interpolation occurs.
        """
        frame_dropped = self.detect_frame_drops()
        self.frame_counter += 1
        should_process = self._should_process_current_frame()

        if should_process:
            # Process with MediaPipe
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            frame_data = {"hands": [], "pose": None}

            # Only run hand detection if model uses it
            if self.feature_info["hand_landmarks"] > 0:
                hand_result = self.hand_landmarker.detect_for_video(
                    mp_image, self.timestamp_ms
                )
                if hand_result and hand_result.hand_landmarks:
                    for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                        frame_data["hands"].append({"landmarks": landmarks})

            # Only run pose detection if model uses it
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

            # Now interpolate for buffered frames
            results = []

            if self.last_processed_features is not None and len(self.frame_buffer) > 0:
                # Interpolate between last_processed and current
                num_skipped = len(self.frame_buffer)

                for i in range(num_skipped):
                    # Linear interpolation
                    weight = (i + 1) / (num_skipped + 1)
                    interpolated = (
                        1 - weight
                    ) * self.last_processed_features + weight * current_features
                    results.append(
                        (interpolated, {"hands": [], "pose": None}, frame_dropped)
                    )

            # Add current processed frame
            results.append((current_features, frame_data, frame_dropped))

            # Update state
            self.last_processed_features = current_features
            self.recent_features.append(current_features)
            self.frame_buffer = []

            return results

        else:
            # Buffer this frame for later interpolation
            self.frame_buffer.append(frame)
            return []  # No features yet, waiting for next processed frame
    
    def _should_process_current_frame(self) -> bool:
        if self.skip_processing == 0:
            return True
        elif self.skip_processing == 1:
            return self.frame_counter % 2 == 1
        elif self.skip_processing == 2:
            return self.frame_counter % 3 == 1
        return True


class MultiModelInferenceSystem:
    """Enhanced inference system with multiple models for different frame skip scenarios"""

    def __init__(
        self,
        hand_model_path,
        pose_model_path,
        confidence_threshold=0.7,
        ensemble_voting=True,
        selected_models=None,
        mediapipe_skip=0,
    ):
        self.confidence_threshold = confidence_threshold
        self.ensemble_voting = ensemble_voting
        self.smart_selection = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detect all available models
        available_models = {}
        self.skip_patterns = [0, 1, 2]
        skip_names = {0: "no_skip", 1: "1_skip", 2: "2_skip"}

        print("Detecting available models...")
        for skip_pattern in self.skip_patterns:
            model_suffix = skip_names[skip_pattern]
            model_path = get_model_path(skip_pattern)
            metadata_path = get_metadata_path(skip_pattern)

            if os.path.exists(str(model_path)) and os.path.exists(str(metadata_path)):
                available_models[skip_pattern] = {
                    "name": model_suffix,
                    "model_path": model_path,
                    "metadata_path": metadata_path,
                }
                print(f"✓ Found {model_suffix} model")
            else:
                print(f"✗ Missing {model_suffix} model")

        if not available_models:
            raise ValueError("No trained models found! Please train models first.")

        # Let user select models to use
        if selected_models == "auto_all":
            selected_models = list(available_models.keys())
            self._auto_all = True
        elif selected_models is None:
            selected_models = self.select_models_interactive(available_models)

        # Load selected models
        self.models = {}
        self.model_infos = {}

        for skip_pattern in selected_models:
            if skip_pattern in available_models:
                print(f"Loading {available_models[skip_pattern]['name']} model...")

                # Load model info
                with open(
                    str(available_models[skip_pattern]["metadata_path"]), "rb"
                ) as f:
                    model_info = pickle.load(f)
                self.model_infos[skip_pattern] = model_info

                # Load model
                model = SignLanguageModel(
                    input_size=model_info["input_size"],
                    num_classes=len(model_info["class_names"]),
                    hidden_size=model_info["hidden_size"],
                    dropout=model_info["dropout"],
                )

                checkpoint = torch.load(
                    str(available_models[skip_pattern]["model_path"]),
                    map_location=self.device,
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.models[skip_pattern] = model
                print(f"✓ Loaded {available_models[skip_pattern]['name']} model")

        # Use the first available model's info for feature extraction
        primary_model_info = list(self.model_infos.values())[0]

        # Initialize landmark extractor with skip option
        self.landmark_extractor = MediaPipeLandmarkExtractor(
            hand_model_path,
            pose_model_path,
            primary_model_info["feature_info"],
            skip_processing=mediapipe_skip,
        )

        # Sequence management
        self.sequence_length = primary_model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)

        # Frame drop tracking
        self.recent_frame_drops = deque(maxlen=10)
        self.frame_drop_count = 0

        # State
        self.last_prediction = ""
        self.current_confidence = 0.0
        self.prediction_history = deque(maxlen=5)

    def get_recommendation(self, available_models):
        """Get recommendation based on available models"""
        recommendations = {
            "single_best": "Use automatic model selection (fastest, good accuracy)",
            "ensemble_all": "Use all models with ensemble voting (slower, best accuracy)",
            "ensemble_robust": "Use 1_skip + 2_skip models (balanced speed/robustness)",
        }

        if len(available_models) == 1:
            return "single_best"
        elif len(available_models) == 2:
            return "ensemble_robust" if 0 not in available_models else "ensemble_all"
        else:
            return "ensemble_all"

    def select_models_interactive(self, available_models):
        """Interactive model selection"""
        skip_names = {0: "no_skip", 1: "1_skip", 2: "2_skip"}

        # Handle auto_all case
        if hasattr(self, "_auto_all") and self._auto_all:
            return list(available_models.keys())

        print(f"\n{'='*60}")
        print("MODEL SELECTION")
        print(f"{'='*60}")
        print("Available models:")

        description = {
            0: "Best for stable, high FPS conditions",
            1: "Good for moderate frame drops",
            2: "Best for low FPS, high frame drops",
        }

        for skip_pattern, info in available_models.items():
            print(f"  {skip_pattern}: {info['name']} - {description[skip_pattern]}")

        recommendation = self.get_recommendation(available_models)

        print(f"\n📋 RECOMMENDATION: {recommendation}")

        print("\nOptions:")
        print("1. Smart single-model (automatic selection based on conditions)")
        print("2. Use all models with ensemble voting (best accuracy)")
        print("3. Custom selection")
        print("4. Use recommendation")

        while True:
            try:
                choice = input(f"\nSelect option (1-4, default=4): ").strip() or "4"

                if choice == "1":
                    # Smart single model selection
                    selected = list(available_models.keys())  # Load all for switching
                    self.ensemble_voting = False
                    self.smart_selection = True
                    print(
                        "Selected: Smart single-model selection (will auto-switch based on conditions)"
                    )
                    break

                elif choice == "2":
                    # Use all available models
                    selected = list(available_models.keys())
                    self.ensemble_voting = True
                    self.smart_selection = False
                    print(
                        f"Selected: All models with ensemble voting ({len(selected)} models)"
                    )
                    break

                elif choice == "3":
                    # Custom selection
                    print(
                        "\nEnter model numbers separated by commas (e.g., 0,1 or 1,2):"
                    )
                    for skip_pattern, info in available_models.items():
                        print(f"  {skip_pattern}: {info['name']}")

                    custom_input = input("Selection: ").strip()
                    try:
                        selected = [int(x.strip()) for x in custom_input.split(",")]
                        selected = [x for x in selected if x in available_models]

                        if selected:
                            model_names = [
                                available_models[x]["name"] for x in selected
                            ]
                            print(f"Selected: {', '.join(model_names)}")

                            if len(selected) > 1:
                                ensemble_choice = (
                                    input("Use ensemble voting? (y/N): ")
                                    .strip()
                                    .lower()
                                )
                                self.ensemble_voting = ensemble_choice == "y"
                                self.smart_selection = not self.ensemble_voting
                            else:
                                self.ensemble_voting = False
                                self.smart_selection = False
                            break
                        else:
                            print("No valid models selected. Please try again.")
                    except ValueError:
                        print("Invalid input format. Please try again.")

                elif choice == "4":
                    # Use recommendation
                    if recommendation == "single_best":
                        selected = list(available_models.keys())
                        self.ensemble_voting = False
                        self.smart_selection = True
                    elif recommendation == "ensemble_all":
                        selected = list(available_models.keys())
                        self.ensemble_voting = True
                        self.smart_selection = False
                    elif recommendation == "ensemble_robust":
                        selected = [k for k in available_models.keys() if k != 0]
                        self.ensemble_voting = True
                        self.smart_selection = False

                    strategy = (
                        "Smart selection"
                        if self.smart_selection
                        else ("Ensemble" if self.ensemble_voting else "Single")
                    )
                    print(
                        f"Using recommendation: {strategy} with {len(selected)} models"
                    )
                    break
                else:
                    print("Invalid option. Please select 1-4.")

            except KeyboardInterrupt:
                print("\nUsing recommendation")
                selected = list(available_models.keys())
                self.ensemble_voting = True
                self.smart_selection = False
                break

        return selected

    def select_best_model(self) -> int:
        """Select the best model based on recent frame drop pattern"""
        if len(self.recent_frame_drops) < 5:
            return list(self.models.keys())[0]  # Use first available model by default

        # Calculate frame drop rate in recent history
        drop_rate = sum(self.recent_frame_drops) / len(self.recent_frame_drops)

        if drop_rate > 0.6:  # High frame drop rate
            return (
                2
                if 2 in self.models
                else (1 if 1 in self.models else list(self.models.keys())[0])
            )
        elif drop_rate > 0.3:  # Medium frame drop rate
            return 1 if 1 in self.models else list(self.models.keys())[0]
        else:  # Low frame drop rate
            return 0 if 0 in self.models else list(self.models.keys())[0]

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

    def run_inference_single_model(
        self, model_skip_pattern: int
    ) -> Tuple[Optional[str], float]:
        """Run inference using a single model"""
        if len(self.feature_queue) < self.sequence_length:
            return None, 0.0

        if model_skip_pattern not in self.models:
            return None, 0.0

        model = self.models[model_skip_pattern]
        model_info = self.model_infos[model_skip_pattern]

        # Use same padding logic as training
        sequence = self.pad_or_truncate_sequence(
            list(self.feature_queue),
            self.sequence_length,
            model_info["feature_info"]["total_features"],
        )

        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = model_info["class_names"][predicted.item()]
            confidence_value = confidence.item()

        return predicted_class, confidence_value

    def run_ensemble_inference(self) -> Tuple[Optional[str], float]:
        """Run inference using ensemble of available models"""
        if len(self.feature_queue) < self.sequence_length:
            return None, 0.0

        predictions = {}
        confidences = {}

        # Get predictions from all models
        for skip_pattern in self.models.keys():
            pred, conf = self.run_inference_single_model(skip_pattern)
            if pred is not None:
                predictions[skip_pattern] = pred
                confidences[skip_pattern] = conf

        if not predictions:
            return None, 0.0

        # Weighted voting based on confidence and model appropriateness
        class_votes = {}
        total_weight = 0

        for skip_pattern, prediction in predictions.items():
            # Weight based on confidence and how appropriate the model is for current conditions
            drop_rate = sum(self.recent_frame_drops) / max(
                len(self.recent_frame_drops), 1
            )

            # Model appropriateness weights
            if skip_pattern == 0:  # no_skip
                appropriateness = 1.0 - drop_rate
            elif skip_pattern == 1:  # 1_skip
                appropriateness = 1.0 - abs(drop_rate - 0.5)
            else:  # 2_skip
                appropriateness = drop_rate

            weight = confidences[skip_pattern] * (0.5 + 0.5 * appropriateness)

            if prediction not in class_votes:
                class_votes[prediction] = 0
            class_votes[prediction] += weight
            total_weight += weight

        # Get the class with highest weighted vote
        best_class = max(class_votes.keys(), key=lambda k: class_votes[k])
        ensemble_confidence = (
            class_votes[best_class] / total_weight if total_weight > 0 else 0.0
        )

        return best_class, ensemble_confidence

    def run_inference(self) -> Tuple[Optional[str], float]:
        """Run inference using selected strategy"""
        if self.smart_selection:
            # Use single best model based on current conditions
            best_model = self.select_best_model()
            return self.run_inference_single_model(best_model)
        elif self.ensemble_voting and len(self.models) > 1:
            return self.run_ensemble_inference()
        else:
            # Use single model (first available)
            model_key = list(self.models.keys())[0]
            return self.run_inference_single_model(model_key)

    def run(self):
        """Main inference loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Performance monitoring
        frame_times = deque(maxlen=30)
        processing_times = deque(maxlen=30)

        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        print(f"MediaPipe skip pattern: {self.landmark_extractor.skip_processing}")
        print("Press 'q' to quit, 'e' to toggle ensemble, 's' to change skip pattern")

        while True:
            frame_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            process_start = time.time()
            feature_list = self.landmark_extractor.process_frame(frame)
            process_time = time.time() - process_start
            processing_times.append(process_time)

            # Add features to queue
            for features, frame_data, frame_dropped in feature_list:
                self.feature_queue.append(features)
                self.recent_frame_drops.append(frame_dropped)
                if frame_dropped:
                    self.frame_drop_count += 1

            # Run inference when queue is full
            prediction, confidence = None, 0.0
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

            # Calculate performance stats
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # Calculate stats
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
            avg_process_time = (
                sum(processing_times) / len(processing_times) if processing_times else 0
            )
            drop_rate = sum(self.recent_frame_drops) / max(len(self.recent_frame_drops), 1)
            selected_model = self.select_best_model() if self.smart_selection else "N/A"

            # Display information including performance
            display_lines = [
                f"FPS: {avg_fps:.1f} | Process: {avg_process_time*1000:.1f}ms",
                f"MP Skip: {self.landmark_extractor.skip_processing} | Queue: {len(self.feature_queue)}/{self.sequence_length}",
                f"Frame drops: {drop_rate:.1%} | Model: {selected_model}",
                f"Strategy: {'Smart' if self.smart_selection else ('Ensemble' if self.ensemble_voting else 'Single')}",
            ]

            if self.last_prediction:
                display_lines.append(
                    f"Last: {self.last_prediction} ({self.current_confidence:.2f})"
                )

            # Draw information on frame
            y_offset = 25
            for i, line in enumerate(display_lines):
                cv2.putText(
                    frame,
                    line,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Multi-Model Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                self.ensemble_voting = not self.ensemble_voting
                if self.ensemble_voting:
                    self.smart_selection = False
                print(f"Ensemble voting: {'ON' if self.ensemble_voting else 'OFF'}")
            elif key == ord("s"):
                # Cycle through skip patterns
                current_skip = self.landmark_extractor.skip_processing
                new_skip = (current_skip + 1) % 3
                self.landmark_extractor.skip_processing = new_skip
                self.landmark_extractor.frame_counter = 0  # Reset counter
                print(f"MediaPipe skip pattern changed to: {new_skip}")

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model real-time sign language recognition"
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
        "--confidence_threshold", type=float, default=0.7, help="Confidence threshold"
    )
    parser.add_argument(
        "--no_ensemble", action="store_true", help="Disable ensemble voting"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=int,
        choices=[0, 1, 2],
        help="Specify which models to use (0=no_skip, 1=1_skip, 2=2_skip). If not specified, interactive selection will be used.",
    )
    parser.add_argument(
        "--auto_all",
        action="store_true",
        help="Automatically use all available models without interactive selection",
    )
    parser.add_argument(
        "--mediapipe_skip",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Skip MediaPipe processing: 0=process all frames, 1=skip every other frame, 2=skip 2/3 of frames",
    )
    parser.add_argument(
        "--show_processing_stats",
        action="store_true",
        help="Show processing time statistics",
    )

    args = parser.parse_args()

    # Check basic files exist
    required_files = [str(args.hand_model), str(args.pose_model)]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return

    # Determine selected models
    selected_models = None
    ensemble_voting = not args.no_ensemble

    if args.auto_all:
        # Use all available models automatically
        selected_models = "auto_all"
    elif args.models:
        # Use command-line specified models
        selected_models = args.models
        if len(selected_models) == 1:
            ensemble_voting = False
        print(f"Using specified models: {selected_models}")

    # Run inference
    inference_system = MultiModelInferenceSystem(
        hand_model_path=args.hand_model,
        pose_model_path=args.pose_model,
        confidence_threshold=args.confidence_threshold,
        ensemble_voting=ensemble_voting,
        selected_models=selected_models,
        mediapipe_skip=args.mediapipe_skip,
    )

    if args.show_processing_stats:
        inference_system.show_stats = True

    inference_system.run()


if __name__ == "__main__":
    main()
