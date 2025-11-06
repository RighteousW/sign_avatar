"""
Video File to Gloss GUI
Select a video file and extract glosses from it
"""

import sys
import threading
import pickle
import torch
import cv2
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, QObject

try:
    from ..constants import (
        MEDIAPIPE_HAND_LANDMARKER_PATH,
        MEDIAPIPE_POSE_LANDMARKER_PATH,
        get_gesture_metadata_path,
        get_gesture_model_path,
    )
    from ..model_training import GestureRecognizerModel
    from ..landmark_extraction import LandmarkExtractor
except ImportError:
    print("Import error: Ensure required modules are available")


class VideoFileProcessor(QObject):
    """Processes video file and detects glosses"""

    gloss_detected = pyqtSignal(str, float)
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_processing = False
        self.confidence_threshold = 0.7

        # Load model
        with open(str(get_gesture_metadata_path(False, 2)), "rb") as f:
            self.model_info = pickle.load(f)

        self.model = GestureRecognizerModel(
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

        # Initialize landmark extractor (use_pose=True for hands+pose)
        self.landmark_extractor = LandmarkExtractor(use_pose=True)

        # Store feature info from model
        self.feature_info = self.model_info["feature_info"]

        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)
        self.detected_glosses = []
        self.last_prediction = ""

        # Frame processing state for 2_skip pattern
        self.frame_counter = 0
        self.last_processed_features = None

    def extract_features_from_frame_data(self, frame_data: dict) -> np.ndarray:
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

    def process_frame_with_skip(self, frame):
        """
        Process frame with 2_skip pattern matching webcam behavior.
        Process every 3rd frame, repeat last features for skipped frames.
        Returns list of features (same as webcam inference).
        """
        frame_index = self.frame_counter
        self.frame_counter += 1

        should_process = frame_index % 3 == 0  # Process frames 0, 3, 6, 9...

        if should_process:
            # Process with LandmarkExtractor
            frame_data = self.landmark_extractor.extract_landmarks_from_frame(frame)
            current_features = self.extract_features_from_frame_data(frame_data)
            self.last_processed_features = current_features
            return [current_features]
        else:
            # Skipped frame - return last processed (or zeros if none)
            if self.last_processed_features is not None:
                return [self.last_processed_features.copy()]
            else:
                return [np.zeros(self.feature_info["total_features"], dtype=np.float32)]

    def process_video_file(self, video_path):
        """Process video file"""
        self.is_processing = True
        self.detected_glosses = []
        self.feature_queue.clear()
        self.frame_counter = 0
        self.last_processed_features = None
        self.last_prediction = ""

        thread = threading.Thread(target=self._process_video_file, args=(video_path,))
        thread.daemon = True
        thread.start()

    def _process_video_file(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while self.is_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process frame using 2_skip pattern (same as webcam)
                # Returns list of features: same as webcam
                feature_list = self.process_frame_with_skip(frame)

                # Add all returned features to queue (handles interpolation)
                for features in feature_list:
                    self.feature_queue.append(features)

                    # Run inference when queue is full (for each frame)
                    if len(self.feature_queue) == self.sequence_length:
                        prediction, confidence = self._run_inference()

                        if (
                            confidence > self.confidence_threshold
                            and prediction
                            and prediction != self.last_prediction
                        ):
                            self.last_prediction = prediction
                            self.detected_glosses.append(prediction)
                            self.gloss_detected.emit(prediction, confidence)

                            # Clear some frames after detection
                            for _ in range(int(0.8 * self.sequence_length)):
                                if self.feature_queue:
                                    self.feature_queue.popleft()

            cap.release()
            self.processing_complete.emit(self.detected_glosses)

        except Exception as e:
            self.error_occurred.emit(f"Video processing error: {str(e)}")

    def _run_inference(self):
        """Run model inference on current sequence"""
        if len(self.feature_queue) < self.sequence_length:
            return None, 0.0

        # Prepare sequence
        sequence = self._pad_or_truncate_sequence(
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

    def _pad_or_truncate_sequence(self, sequence, target_length, feature_size):
        """Same padding logic as training script"""
        if len(sequence) > target_length:
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            padded = list(sequence)
            while len(padded) < target_length:
                padded.append(np.zeros(feature_size))
            return np.array(padded)

    def stop_processing(self):
        self.is_processing = False


class VideoFile2GlossWidget(QWidget):
    """Main widget for video file to gloss conversion"""

    update_gloss_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.processor = VideoFileProcessor()
        self.current_video_path = None

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label)

        self.select_btn = QPushButton("Select Video")
        self.select_btn.clicked.connect(self.select_video)
        file_layout.addWidget(self.select_btn)

        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        file_layout.addWidget(self.process_btn)

        layout.addLayout(file_layout)

        # Gloss output
        layout.addWidget(QLabel("Detected Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        layout.addWidget(self.gloss_output)

        self.setLayout(layout)

    def setup_connections(self):
        self.processor.gloss_detected.connect(self.on_gloss_detected)
        self.processor.processing_complete.connect(self.on_processing_complete)
        self.processor.error_occurred.connect(self.on_error)
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def select_video(self):
        """Open file dialog to select video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.current_video_path = file_path
            import os

            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.process_btn.setEnabled(True)
            self.update_status_signal.emit("Ready to process")

    def process_video(self):
        """Start processing the selected video"""
        if not self.current_video_path:
            return

        self.gloss_output.clear()
        self.update_status_signal.emit("Processing...")
        self.select_btn.setEnabled(False)
        self.process_btn.setEnabled(False)

        self.processor.process_video_file(self.current_video_path)

    def on_gloss_detected(self, gloss, confidence):
        """Handle new gloss detection"""
        current_text = self.gloss_output.toPlainText()
        if current_text:
            self.update_gloss_signal.emit(f"{current_text} {gloss}")
        else:
            self.update_gloss_signal.emit(gloss)

        self.update_status_signal.emit(f"Detected: {gloss} ({confidence:.2f})")

    def on_processing_complete(self, glosses):
        """Handle completion of video processing"""
        self.update_status_signal.emit(f"Complete - {len(glosses)} glosses detected")
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"Error: {error_msg}")
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video File to Gloss")
        self.setGeometry(100, 100, 600, 400)

        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = VideoFile2GlossWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
