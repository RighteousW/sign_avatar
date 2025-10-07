import cv2
import torch
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from typing import List, Optional, Tuple

from constants import (
    GESTURE_MODEL_PATH,
    GESTURE_MODEL_METADATA_PATH,
)
from src.landmark_extraction.landmark_extraction import LandmarkExtractor

class GestureRecognizer:
    """Wrapper for the trained sign language model"""

    def __init__(self, model_path=None, metadata_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use default paths if not provided
        if model_path is None:
            model_path = GESTURE_MODEL_PATH
        if metadata_path is None:
            metadata_path = GESTURE_MODEL_METADATA_PATH

        # Load model metadata
        with open(str(metadata_path), "rb") as f:
            self.model_info = pickle.load(f)

        # Initialize model
        self.model = SignLanguageModel(
            input_size=self.model_info["input_size"],
            num_classes=len(self.model_info["class_names"]),
            hidden_size=self.model_info["hidden_size"],
            dropout=self.model_info["dropout"],
        )

        # Load trained weights
        checkpoint = torch.load(str(model_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.sequence_length = self.model_info["sequence_length"]
        self.feature_info = self.model_info["feature_info"]
        self.class_names = self.model_info["class_names"]

    def predict(self, landmarks_sequence: np.ndarray, confidence_threshold=0.5):
        """
        Predict gesture from a sequence of landmarks.

        Args:
            landmarks_sequence: numpy array of shape (sequence_length, feature_dim)
            confidence_threshold: minimum confidence for prediction

        Returns:
            Tuple of (predicted_class, confidence) or (None, 0.0) if below threshold
        """
        # Pad or truncate sequence to required length
        landmarks_sequence = self._pad_or_truncate_sequence(landmarks_sequence)

        # Convert to tensor
        sequence_tensor = (
            torch.FloatTensor(landmarks_sequence).unsqueeze(0).to(self.device)
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.class_names[predicted.item()]
            confidence_value = confidence.item()

        # Return prediction only if confidence is above threshold
        if confidence_value >= confidence_threshold:
            return predicted_class, confidence_value
        else:
            return None, confidence_value

    def _pad_or_truncate_sequence(self, sequence):
        """Pad or truncate sequence to match required sequence length"""
        target_length = self.sequence_length
        feature_size = self.feature_info["total_features"]

        if len(sequence) > target_length:
            # Uniformly sample frames
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            # Pad with zeros
            padded = list(sequence)
            while len(padded) < target_length:
                padded.append(np.zeros(feature_size))
            return np.array(padded)


class Video2Gloss:
    """Convert video chunks to gloss sequences using MediaPipe and gesture recognition"""

    def __init__(
        self,
        gesture_recognizer: GestureRecognizer,
        window_size: int = None,
        stride: int = None,
        confidence_threshold: float = 0.5,
        use_pose: bool = False,
    ):
        self.landmark_extractor = LandmarkExtractor(use_pose=use_pose)
        self.gesture_recognizer = gesture_recognizer
        self.confidence_threshold = confidence_threshold

        # Use model's sequence length if not specified
        self.window_size = window_size or gesture_recognizer.sequence_length
        self.stride = stride or (self.window_size // 2)  # 50% overlap by default

        self.feature_info = gesture_recognizer.feature_info

    def infer(self, video_path: str) -> List[Tuple[str, float, int]]:
        """
        Process video and return detected glosses with confidence and frame positions.

        Args:
            video_path: Path to input video file

        Returns:
            List of tuples: (gloss, confidence, frame_index)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Extract all landmarks from video
        all_landmarks = []
        frame_count = 0

        print(f"Processing video: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = self.landmark_extractor.extract_landmarks_from_frame(frame)
            hand_result = frame_data.get("hand_landmarks")
            pose_result = frame_data.get("pose_landmarks")
            landmarks = self.extract_landmarks(hand_result, pose_result)

            if landmarks is not None:
                all_landmarks.append(landmarks)
            else:
                # Add zero features if detection failed
                all_landmarks.append(
                    np.zeros(self.feature_info["total_features"], dtype=np.float32)
                )

            frame_count += 1

        cap.release()

        print(f"Extracted landmarks from {frame_count} frames")

        # Run sliding window inference
        gloss_sequence = self._sliding_window_inference(all_landmarks)

        return gloss_sequence

    def extract_landmarks(self, hand_result, pose_result) -> Optional[np.ndarray]:
        """
        Extract landmarks from MediaPipe results and format them for the model.

        Args:
            hand_result: MediaPipe hand detection result
            pose_result: MediaPipe pose detection result

        Returns:
            Numpy array of flattened landmarks or None if extraction fails
        """
        features = []

        # Extract hand landmarks
        if self.feature_info["hand_landmarks"] > 0:
            max_hands = self.feature_info["max_hands"]
            hand_dim_per_hand = self.feature_info["hand_landmarks_per_hand"]
            hand_features = np.zeros(self.feature_info["hand_landmarks"])

            if hand_result and hand_result.hand_landmarks:
                for i, hand_landmarks in enumerate(
                    hand_result.hand_landmarks[:max_hands]
                ):
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    landmarks_flat = np.array(landmarks[:21]).flatten()
                    start_idx = i * hand_dim_per_hand
                    end_idx = start_idx + len(landmarks_flat)
                    hand_features[start_idx:end_idx] = landmarks_flat

            features.extend(hand_features)

        # Extract pose landmarks
        if self.feature_info["pose_landmarks"] > 0:
            pose_features = np.zeros(self.feature_info["pose_landmarks"])

            if pose_result and pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks[0]
                # Only use x,y,z coordinates (skip visibility for consistency)
                landmarks = [[lm.x, lm.y, lm.z] for lm in pose_landmarks]
                landmarks_flat = np.array(landmarks).flatten()
                pose_features[: min(len(landmarks_flat), len(pose_features))] = (
                    landmarks_flat[: len(pose_features)]
                )

            features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def _sliding_window_inference(
        self, landmarks: List[np.ndarray]
    ) -> List[Tuple[str, float, int]]:
        """
        Run sliding window inference over landmark sequence.

        Args:
            landmarks: List of landmark arrays for each frame

        Returns:
            List of (gloss, confidence, frame_index) tuples
        """
        gloss_sequence = []
        landmarks_array = np.array(landmarks)

        # Use sliding window
        for start_idx in range(
            0, len(landmarks_array) - self.window_size + 1, self.stride
        ):
            end_idx = start_idx + self.window_size
            window = landmarks_array[start_idx:end_idx]

            # Run inference on window
            predicted_class, confidence = self.gesture_recognizer.predict(
                window, self.confidence_threshold
            )

            # Only add if prediction is confident enough
            if predicted_class is not None:
                center_frame = start_idx + self.window_size // 2
                gloss_sequence.append((predicted_class, confidence, center_frame))

        # Remove duplicate consecutive predictions
        gloss_sequence = self._remove_duplicates(gloss_sequence)

        return gloss_sequence

    def _remove_duplicates(
        self, gloss_sequence: List[Tuple[str, float, int]]
    ) -> List[Tuple[str, float, int]]:
        """Remove consecutive duplicate glosses, keeping the one with highest confidence"""
        if not gloss_sequence:
            return []

        cleaned = [gloss_sequence[0]]

        for gloss, conf, frame in gloss_sequence[1:]:
            # If same as previous gloss and frames are close, keep higher confidence one
            if gloss == cleaned[-1][0] and abs(frame - cleaned[-1][2]) < self.stride:
                if conf > cleaned[-1][1]:
                    cleaned[-1] = (gloss, conf, frame)
            else:
                cleaned.append((gloss, conf, frame))

        return cleaned


class LiveStream2Gloss:
    """Real-time gloss detection from webcam/livestream"""

    def __init__(
        self,
        mediapipe_processor: MediapipeProcessor,
        gesture_recognizer: GestureRecognizer,
        confidence_threshold: float = 0.6,
        display: bool = True,
    ):
        self.mediapipe_processor = mediapipe_processor
        self.gesture_recognizer = gesture_recognizer
        self.confidence_threshold = confidence_threshold
        self.display = display

        self.feature_info = gesture_recognizer.feature_info
        self.sequence_length = gesture_recognizer.sequence_length

        # Rolling buffer for features
        self.feature_buffer = deque(maxlen=self.sequence_length)

        # Prediction state
        self.last_prediction = None
        self.last_confidence = 0.0
        self.prediction_history = deque(maxlen=10)

        # Statistics
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = None

    def infer(self, camera_id: int = 0, callback=None) -> None:
        """
        Run real-time inference on webcam/livestream.

        Args:
            camera_id: Camera device ID (0 for default webcam)
            callback: Optional callback function(gloss, confidence) called on new predictions
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        print(f"Starting livestream inference from camera {camera_id}")
        print("Press 'q' to quit, 'r' to reset buffer, 'c' to clear predictions")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Update FPS
                current_time = cv2.getTickCount()
                if self.last_frame_time is not None:
                    fps = cv2.getTickFrequency() / (current_time - self.last_frame_time)
                    self.fps_history.append(fps)
                self.last_frame_time = current_time

                # Process frame
                hand_result, pose_result = self.mediapipe_processor.process(frame)
                landmarks = self._extract_landmarks(hand_result, pose_result)

                # Add to buffer
                if landmarks is not None:
                    self.feature_buffer.append(landmarks)
                else:
                    # Add zero features if detection failed
                    self.feature_buffer.append(
                        np.zeros(self.feature_info["total_features"], dtype=np.float32)
                    )

                self.frame_count += 1

                # Run inference when buffer is full
                prediction = None
                if len(self.feature_buffer) == self.sequence_length:
                    sequence = np.array(list(self.feature_buffer))
                    predicted_class, confidence = self.gesture_recognizer.predict(
                        sequence, self.confidence_threshold
                    )

                    if predicted_class is not None:
                        # Only update if different from last prediction or confidence is higher
                        if (
                            predicted_class != self.last_prediction
                            or confidence > self.last_confidence + 0.1
                        ):
                            self.last_prediction = predicted_class
                            self.last_confidence = confidence
                            self.prediction_history.append(
                                (predicted_class, confidence, self.frame_count)
                            )
                            prediction = (predicted_class, confidence)

                            print(
                                f"[Frame {self.frame_count:5d}] Detected: {predicted_class:20s} "
                                f"(confidence: {confidence:.3f})"
                            )

                            # Call callback if provided
                            if callback:
                                callback(predicted_class, confidence)

                # Display
                if self.display:
                    display_frame = self._draw_display(frame, prediction)
                    cv2.imshow("Live Sign Language Recognition", display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.feature_buffer.clear()
                    print("Buffer reset")
                elif key == ord("c"):
                    self.prediction_history.clear()
                    self.last_prediction = None
                    print("Predictions cleared")

        finally:
            cap.release()
            if self.display:
                cv2.destroyAllWindows()

            # Print summary
            print(f"\n{'='*60}")
            print("Session Summary")
            print(f"{'='*60}")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total predictions: {len(self.prediction_history)}")
            if self.prediction_history:
                print("\nPrediction history:")
                for gloss, conf, frame in list(self.prediction_history)[-10:]:
                    print(f"  Frame {frame:5d}: {gloss:20s} (conf: {conf:.3f})")

    def _extract_landmarks(self, hand_result, pose_result) -> Optional[np.ndarray]:
        """Extract landmarks from MediaPipe results"""
        features = []

        # Extract hand landmarks
        if self.feature_info["hand_landmarks"] > 0:
            max_hands = self.feature_info["max_hands"]
            hand_dim_per_hand = self.feature_info["hand_landmarks_per_hand"]
            hand_features = np.zeros(self.feature_info["hand_landmarks"])

            if hand_result and hand_result.hand_landmarks:
                for i, hand_landmarks in enumerate(
                    hand_result.hand_landmarks[:max_hands]
                ):
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    landmarks_flat = np.array(landmarks[:21]).flatten()
                    start_idx = i * hand_dim_per_hand
                    end_idx = start_idx + len(landmarks_flat)
                    hand_features[start_idx:end_idx] = landmarks_flat

            features.extend(hand_features)

        # Extract pose landmarks
        if self.feature_info["pose_landmarks"] > 0:
            pose_features = np.zeros(self.feature_info["pose_landmarks"])

            if pose_result and pose_result.pose_landmarks:
                pose_landmarks = pose_result.pose_landmarks[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in pose_landmarks]
                landmarks_flat = np.array(landmarks).flatten()
                pose_features[: min(len(landmarks_flat), len(pose_features))] = (
                    landmarks_flat[: len(pose_features)]
                )

            features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def _draw_display(self, frame, prediction):
        """Draw information overlay on frame"""
        display_frame = frame.copy()

        # Calculate stats
        avg_fps = (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        )

        # Draw information
        info_lines = [
            f"FPS: {avg_fps:.1f}",
            f"Buffer: {len(self.feature_buffer)}/{self.sequence_length}",
            f"Frames: {self.frame_count}",
        ]

        # Add current prediction
        if self.last_prediction:
            info_lines.append("")
            info_lines.append(f"Current: {self.last_prediction}")
            info_lines.append(f"Confidence: {self.last_confidence:.2f}")

            # Draw confidence bar
            bar_width = int(200 * self.last_confidence)
            cv2.rectangle(
                display_frame,
                (10, 150),
                (10 + bar_width, 170),
                (0, 255, 0),
                -1,
            )
            cv2.rectangle(display_frame, (10, 150), (210, 170), (255, 255, 255), 2)

        # Draw text
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                display_frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset += 30

        # Draw recent predictions history
        if self.prediction_history:
            history_text = "Recent: " + " -> ".join(
                [gloss for gloss, _, _ in list(self.prediction_history)[-5:]]
            )
            cv2.putText(
                display_frame,
                history_text,
                (10, display_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # Highlight if new prediction
        if prediction:
            cv2.rectangle(
                display_frame,
                (5, 5),
                (display_frame.shape[1] - 5, display_frame.shape[0] - 5),
                (0, 255, 0),
                3,
            )

        return display_frame


# Example usage
def main():
    """Example usage of the simplified inference pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Sign Language Inference")
    parser.add_argument(
        "--mode",
        choices=["video", "livestream"],
        default="livestream",
        help="Inference mode",
    )
    parser.add_argument("--input", help="Input video path (for video mode)")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera ID (for livestream mode)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.6, help="Confidence threshold"
    )
    parser.add_argument(
        "--no_display", action="store_true", help="Disable video display"
    )

    args = parser.parse_args()

    gesture_recognizer = GestureRecognizer(
        model_path=GESTURE_MODEL_PATH,
        metadata_path=GESTURE_MODEL_METADATA_PATH,
    )

    if args.mode == "video":
        # Video file inference
        if not args.input:
            print("Error: --input required for video mode")
            return

        video2gloss = Video2Gloss(
            gesture_recognizer=gesture_recognizer,
            confidence_threshold=args.confidence,
        )

        try:
            gloss_sequence = video2gloss.infer(args.input)

            print(f"\nDetected {len(gloss_sequence)} glosses:")
            for gloss, confidence, frame in gloss_sequence:
                print(f"  Frame {frame:4d}: {gloss:20s} (conf: {confidence:.3f})")

            # Extract just the gloss labels
            gloss_labels = [gloss for gloss, _, _ in gloss_sequence]
            print(f"\nGloss sequence: {' '.join(gloss_labels)}")

        except Exception as e:
            print(f"Error processing video: {e}")

    else:
        # Livestream inference
        def prediction_callback(gloss, confidence):
            """Optional callback for predictions"""
            pass  # Add custom handling here if needed

        livestream2gloss = LiveStream2Gloss(
            mediapipe_processor=mediapipe_processor,
            gesture_recognizer=gesture_recognizer,
            confidence_threshold=args.confidence,
            display=not args.no_display,
        )

        try:
            livestream2gloss.infer(camera_id=args.camera, callback=prediction_callback)
        except Exception as e:
            print(f"Error in livestream: {e}")


if __name__ == "__main__":
    main()
