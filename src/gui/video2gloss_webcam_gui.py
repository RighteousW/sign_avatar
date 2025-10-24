"""
Webcam to Gloss GUI
Live webcam feed with real-time gloss detection
"""

import sys
import threading
import pickle
import torch
import cv2
import time
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
    QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt
from PyQt6.QtGui import QImage, QPixmap

try:
    from ..constants import (
        GESTURE_MODEL_2_SKIP,
        GESTURE_MODEL_2_SKIP_METADATA_PATH,
        MEDIAPIPE_HAND_LANDMARKER_PATH,
        MEDIAPIPE_POSE_LANDMARKER_PATH,
    )
    from ..model_training import GestureRecognizerModel
    from ..video2gloss.inference_example import MediaPipeLandmarkExtractor
except ImportError:
    print("Import error: Ensure required modules are available")


class WebcamProcessor(QObject):
    """Processes webcam and detects glosses"""

    frame_ready = pyqtSignal(object)
    gloss_detected = pyqtSignal(str, float)
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_processing = False
        self.confidence_threshold = 0.7

        # Load model
        with open(str(GESTURE_MODEL_2_SKIP_METADATA_PATH), "rb") as f:
            self.model_info = pickle.load(f)

        self.model = GestureRecognizerModel(
            input_size=self.model_info["input_size"],
            num_classes=len(self.model_info["class_names"]),
            hidden_size=self.model_info["hidden_size"],
            dropout=self.model_info["dropout"],
        )

        checkpoint = torch.load(str(GESTURE_MODEL_2_SKIP), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize landmark extractor
        self.landmark_extractor = MediaPipeLandmarkExtractor(
            MEDIAPIPE_HAND_LANDMARKER_PATH,
            MEDIAPIPE_POSE_LANDMARKER_PATH,
            self.model_info["feature_info"],
        )

        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)
        self.detected_glosses = []
        self.last_prediction = ""
        self.current_confidence = 0.0

    def start_webcam(self):
        """Start webcam processing"""
        self.is_processing = True
        self.detected_glosses = []
        self.feature_queue.clear()
        self.landmark_extractor.frame_counter = 0
        self.landmark_extractor.frame_buffer = []
        self.landmark_extractor.last_processed_features = None
        self.last_prediction = ""

        thread = threading.Thread(target=self._process_webcam)
        thread.daemon = True
        thread.start()

    def _process_webcam(self):
        try:
            cap = cv2.VideoCapture(0)
            frame_times = deque(maxlen=30)

            if not cap.isOpened():
                self.error_occurred.emit("Could not open camera")
                return

            while self.is_processing and cap.isOpened():
                frame_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame using MediaPipeLandmarkExtractor (2_skip pattern)
                feature_list = self.landmark_extractor.process_frame(frame)

                # Add all returned features to queue
                for features in feature_list:
                    self.feature_queue.append(features)

                    # Run inference when queue is full
                    if len(self.feature_queue) == self.sequence_length:
                        prediction, confidence = self._run_inference()
                        self.current_confidence = confidence

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

                # Calculate FPS
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                avg_fps = (
                    1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                )

                # Add overlay info
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

                self.frame_ready.emit(frame)

            cap.release()
            self.processing_complete.emit(self.detected_glosses)

        except Exception as e:
            self.error_occurred.emit(f"Webcam error: {str(e)}")

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
        import numpy as np

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


class Webcam2GlossWidget(QWidget):
    """Main widget for webcam to gloss conversion"""

    update_gloss_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.processor = WebcamProcessor()

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Webcam display
        self.video_label = QLabel("Webcam feed will appear here")
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Controls
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_webcam)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_webcam)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Gloss output
        layout.addWidget(QLabel("Detected Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setMaximumHeight(100)
        layout.addWidget(self.gloss_output)

        self.setLayout(layout)

    def setup_connections(self):
        self.processor.frame_ready.connect(self.update_frame)
        self.processor.gloss_detected.connect(self.on_gloss_detected)
        self.processor.processing_complete.connect(self.on_processing_complete)
        self.processor.error_occurred.connect(self.on_error)
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def start_webcam(self):
        """Start webcam processing"""
        self.gloss_output.clear()
        self.update_status_signal.emit("Starting webcam...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.processor.start_webcam()

    def stop_webcam(self):
        """Stop webcam processing"""
        self.processor.stop_processing()
        self.update_status_signal.emit("Stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self, frame):
        """Update video display with webcam frame"""
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )

            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_label.setPixmap(scaled_pixmap)

    def on_gloss_detected(self, gloss, confidence):
        """Handle new gloss detection"""
        current_text = self.gloss_output.toPlainText()
        if current_text:
            self.update_gloss_signal.emit(f"{current_text} {gloss}")
        else:
            self.update_gloss_signal.emit(gloss)

        self.update_status_signal.emit(f"Detected: {gloss} ({confidence:.2f})")

    def on_processing_complete(self, glosses):
        """Handle completion"""
        self.update_status_signal.emit(f"Complete - {len(glosses)} glosses detected")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"Error: {error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam to Gloss")
        self.setGeometry(100, 100, 700, 700)

        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = Webcam2GlossWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
