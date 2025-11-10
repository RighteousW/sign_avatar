"""
Video2Gloss Record Demo
Record video from webcam -> Detect glosses in real-time
"""

import sys
import threading
import pickle
import numpy as np
import torch
import cv2
from collections import deque
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

try:
    from demo_utils import get_dark_stylesheet, VideoDisplayLabel
    from ..constants import (
        FRAME_RATE,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        get_gesture_metadata_path,
        get_gesture_model_path,
    )
    from ..data_creation.video_recording import FrameTimer
    from ..model_training import GestureRecognizerCNN
    from ..landmark_extraction import LandmarkExtractor
    from ..utils.interpolation import apply_frame_skipping
except ImportError:
    try:
        from .demo_utils import get_dark_stylesheet, VideoDisplayLabel
        from ..constants import (
            FRAME_RATE,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            get_gesture_metadata_path,
            get_gesture_model_path,
        )
        from ..data_creation.video_recording import FrameTimer
        from ..model_training import GestureRecognizerCNN
        from ..landmark_extraction import LandmarkExtractor
        from ..utils.interpolation import apply_frame_skipping
    except ImportError:
        print("Import error - ensure modules are available")


class VideoRecordingProcessor(QObject):
    """Handles video recording and real-time gloss detection"""

    frame_ready = pyqtSignal(object)
    gloss_detected = pyqtSignal(str, float)
    recording_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = 0.7

        # Load model
        with open(str(get_gesture_metadata_path(False, 2)), "rb") as f:
            self.model_info = pickle.load(f)

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

        self.landmark_extractor = LandmarkExtractor(use_pose=True)
        self.feature_info = self.model_info["feature_info"]
        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)
        self.last_prediction = ""

    def start_recording(self):
        self.is_recording = True
        self.feature_queue.clear()
        self.last_prediction = ""
        self.frame_buffer = []

        thread = threading.Thread(target=self._record)
        thread.daemon = True
        thread.start()

    def _record(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.error_occurred.emit("Could not open camera")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            frame_timer = FrameTimer(FRAME_RATE)

            while self.is_recording and cap.isOpened():
                frame_timer.wait_for_next_frame()

                ret, frame = cap.read()
                if not ret:
                    break

                # Display frame
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    "REC",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                self.frame_ready.emit(display_frame)

                # Process for gloss detection
                feature_list = self.process_frame_with_skip(frame)

                for features in feature_list:
                    self.feature_queue.append(features)

                    if len(self.feature_queue) == self.sequence_length:
                        prediction, confidence = self._run_inference()

                        if (
                            confidence > self.confidence_threshold
                            and prediction
                            and prediction != self.last_prediction
                        ):
                            self.last_prediction = prediction
                            self.gloss_detected.emit(prediction, confidence)

                            # Clear some frames after detection
                            for _ in range(int(0.8 * self.sequence_length)):
                                if self.feature_queue:
                                    self.feature_queue.popleft()

            cap.release()
            self.recording_stopped.emit()

        except Exception as e:
            self.error_occurred.emit(f"Recording error: {str(e)}")

    def extract_features_from_frame_data(self, frame_data: dict) -> np.ndarray:
        features = []

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
        frame_data = self.landmark_extractor.extract_landmarks_from_frame(frame)
        current_features = self.extract_features_from_frame_data(frame_data)

        self.frame_buffer.append(current_features)

        skip_pattern = 2
        window_size = 6

        if len(self.frame_buffer) >= window_size:
            interpolated_frames = apply_frame_skipping(
                self.frame_buffer[:window_size], skip_pattern
            )
            self.frame_buffer = self.frame_buffer[3:]
            return interpolated_frames

        return []

    def _run_inference(self):
        if len(self.feature_queue) < self.sequence_length:
            return None, 0.0

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
        if len(sequence) > target_length:
            indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
            return np.array([sequence[i] for i in indices])
        else:
            padded = list(sequence)
            while len(padded) < target_length:
                padded.append(np.zeros(feature_size))
            return np.array(padded)

    def stop_recording(self):
        self.is_recording = False


class Video2GlossRecordWidget(QWidget):
    """Main widget for recording and detecting glosses"""

    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.processor = VideoRecordingProcessor()
        self.detected_sequences = []  # List of (gloss, confidence) tuples
        self.sequence_offset = 0  # For navigation
        self.is_recording = False

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Video to Gloss - Recording Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("✓ Ready - Click Start Recording")
        self.status_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Video display
        self.video_display = VideoDisplayLabel("Camera preview will appear here")
        layout.addWidget(self.video_display, 50)

        # Current detection display
        detection_layout = QHBoxLayout()
        detection_layout.addWidget(QLabel("Current Detection:"))
        self.current_gloss_label = QLabel("---")
        self.current_gloss_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #14a085; padding: 5px;"
        )
        detection_layout.addWidget(self.current_gloss_label)

        self.current_confidence_label = QLabel("")
        self.current_confidence_label.setStyleSheet("font-size: 14px; padding: 5px;")
        detection_layout.addWidget(self.current_confidence_label)
        detection_layout.addStretch()
        layout.addLayout(detection_layout)

        # Recording controls
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.start_recording)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Recording")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Sequence history display
        layout.addWidget(QLabel("Detected Sequence History:"))

        history_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        self.prev_btn.setEnabled(False)
        history_layout.addWidget(self.prev_btn)

        self.sequence_display = QLabel("No sequences detected yet")
        self.sequence_display.setStyleSheet(
            "background-color: #2b2b2b; padding: 5px; border-radius: 4px; font-size: 13px;"
        )
        self.sequence_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        history_layout.addWidget(self.sequence_display, 80)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.show_next)
        self.next_btn.setEnabled(False)
        history_layout.addWidget(self.next_btn)

        layout.addLayout(history_layout)

        self.setLayout(layout)

    def setup_connections(self):
        self.processor.frame_ready.connect(self.update_frame)
        self.processor.gloss_detected.connect(self.on_gloss_detected)
        self.processor.recording_stopped.connect(self.on_recording_stopped)
        self.processor.error_occurred.connect(self.on_error)
        self.update_status_signal.connect(self.status_label.setText)

    def start_recording(self):
        self.is_recording = True
        self.detected_sequences = []
        self.sequence_offset = 0
        self.current_gloss_label.setText("---")
        self.current_confidence_label.setText("")
        self.sequence_display.setText("Recording... detections will appear here")

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        self.update_status_signal.emit(
            "🔴 Recording... perform signs in front of camera"
        )
        self.processor.start_recording()

    def stop_recording(self):
        self.processor.stop_recording()
        self.update_status_signal.emit("Stopping recording...")
        self.stop_btn.setEnabled(False)

    def update_frame(self, frame):
        self.video_display.display_frame(frame)

    def on_gloss_detected(self, gloss, confidence):
        self.detected_sequences.append((gloss, confidence))
        self.current_gloss_label.setText(gloss.upper())
        self.current_confidence_label.setText(f"({confidence:.2%})")
        self.update_sequence_display()

        self.update_status_signal.emit(
            f"🔴 Recording... detected {len(self.detected_sequences)} signs"
        )

    def on_recording_stopped(self):
        self.is_recording = False
        self.start_btn.setEnabled(True)

        if self.detected_sequences:
            self.update_status_signal.emit(
                f"✓ Recording complete - {len(self.detected_sequences)} signs detected"
            )
            self.prev_btn.setEnabled(len(self.detected_sequences) > 5)
            self.next_btn.setEnabled(False)
        else:
            self.update_status_signal.emit("✓ Recording complete - No signs detected")
            self.sequence_display.setText("No sequences detected")

    def update_sequence_display(self):
        if not self.detected_sequences:
            self.sequence_display.setText("No sequences detected yet")
            return

        # Show 5 sequences at a time
        visible_count = 5
        start_idx = self.sequence_offset
        end_idx = min(start_idx + visible_count, len(self.detected_sequences))

        visible_sequences = self.detected_sequences[start_idx:end_idx]

        display_text = ""
        for i, (gloss, conf) in enumerate(visible_sequences):
            seq_num = start_idx + i + 1
            display_text += f"{seq_num}. {gloss.upper()} ({conf:.2%})\n"

        if len(self.detected_sequences) > visible_count:
            display_text += f"\n[Showing {start_idx + 1}-{end_idx} of {len(self.detected_sequences)}]"

        self.sequence_display.setText(display_text.strip())

        # Update navigation buttons
        self.prev_btn.setEnabled(self.sequence_offset > 0)
        self.next_btn.setEnabled(end_idx < len(self.detected_sequences))

    def show_previous(self):
        if self.sequence_offset > 0:
            self.sequence_offset = max(0, self.sequence_offset - 5)
            self.update_sequence_display()

    def show_next(self):
        if self.sequence_offset + 5 < len(self.detected_sequences):
            self.sequence_offset += 5
            self.update_sequence_display()

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"⚠ {error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video to Gloss - Recording Demo")
        self.setGeometry(100, 100, 900, 800)
        self.setStyleSheet(get_dark_stylesheet())

        widget = Video2GlossRecordWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
