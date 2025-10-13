"""
Unified Namibian Sign Language Translation System
- Pipeline 1: Audio → Glosses → Visualization
- Pipeline 2: Video → Glosses → Text → Audio
"""

import os
import sys
import threading
import pickle
import tempfile
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QTabWidget,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QKeyEvent, QPainter, QColor, QImage, QPixmap
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import cv2
import mediapipe as mp
import torch
from collections import deque

try:
    from ..audio2gloss import AudioToGlossConverter
    from ..gloss2visualization import GestureTransitionGenerator
    from ..gloss2audio import Gloss2Audio
    from ..constants import (
        REPRESENTATIVES_LEFT,
        GESTURE_MODEL_2_SKIP,
        GESTURE_MODEL_2_SKIP_METADATA_PATH,
        MEDIAPIPE_HAND_LANDMARKER_PATH,
        MEDIAPIPE_POSE_LANDMARKER_PATH,
    )
    from ..model_training import GestureRecognizerModel
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all modules are in the correct location")


class LandmarkCanvas(QWidget):
    """Widget for visualizing sign language landmarks"""

    def __init__(self):
        super().__init__()
        self.landmarks_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.setMinimumHeight(400)
        self.setStyleSheet("background-color: black;")

        # Colors
        self.hand_landmark_color = (0, 255, 0)
        self.hand_connection_color = (0, 200, 0)
        self.pose_landmark_color = (0, 0, 255)
        self.pose_connection_color = (0, 100, 255)

        # MediaPipe connections
        self.hand_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
        self.pose_connections = list(mp.solutions.pose.POSE_CONNECTIONS)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.status_text = "Ready"

    def load_landmarks(self, filepath):
        try:
            with open(filepath, "rb") as f:
                self.landmarks_data = pickle.load(f)

            if os.path.exists(filepath):
                os.remove(filepath)

            if (
                isinstance(self.landmarks_data, dict)
                and "frames" in self.landmarks_data
            ):
                self.total_frames = len(self.landmarks_data["frames"])
            elif isinstance(self.landmarks_data, list):
                self.total_frames = len(self.landmarks_data)
            else:
                self.total_frames = 0

            self.current_frame_idx = 0
            self.timer.start(33)  # ~30 fps
            self.status_text = f"Playing ({self.total_frames} frames)"
            self.update()
        except Exception as e:
            print(f"Error loading landmarks: {e}")
            self.status_text = f"Error: {str(e)}"
            self.update()

    def next_frame(self):
        if self.total_frames > 0:
            self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
            self.update()
        else:
            self.timer.stop()

    def stop_animation(self):
        self.timer.stop()
        self.landmarks_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.status_text = "Ready"
        self.update()

    def create_cv_frame(self, frame_data):
        frame_width = self.width()
        frame_height = self.height()
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Draw pose landmarks
        if frame_data.get("pose") and frame_data["pose"]:
            pose_landmarks = frame_data["pose"]["landmarks"]
            pose_points = []
            for landmark in pose_landmarks:
                x = int(landmark[0] * frame_width)
                y = int(landmark[1] * frame_height)
                pose_points.append((x, y))

            for connection in self.pose_connections:
                start_idx, end_idx = connection
                if (
                    start_idx < len(pose_points)
                    and end_idx < len(pose_points)
                    and 0 <= pose_points[start_idx][0] < frame_width
                    and 0 <= pose_points[start_idx][1] < frame_height
                    and 0 <= pose_points[end_idx][0] < frame_width
                    and 0 <= pose_points[end_idx][1] < frame_height
                ):
                    cv2.line(
                        frame,
                        pose_points[start_idx],
                        pose_points[end_idx],
                        self.pose_connection_color,
                        2,
                    )

            for x, y in pose_points:
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    cv2.circle(frame, (x, y), 3, self.pose_landmark_color, -1)

        # Draw hand landmarks
        if frame_data.get("hands"):
            for hand_idx, hand_data in enumerate(frame_data["hands"]):
                hand_landmarks = hand_data["landmarks"]
                handedness = hand_data.get("handedness", f"Hand_{hand_idx}")

                hand_points = []
                for landmark in hand_landmarks:
                    x = int(landmark[0] * frame_width)
                    y = int(landmark[1] * frame_height)
                    hand_points.append((x, y))

                for connection in self.hand_connections:
                    start_idx, end_idx = connection
                    if (
                        start_idx < len(hand_points)
                        and end_idx < len(hand_points)
                        and 0 <= hand_points[start_idx][0] < frame_width
                        and 0 <= hand_points[start_idx][1] < frame_height
                        and 0 <= hand_points[end_idx][0] < frame_width
                        and 0 <= hand_points[end_idx][1] < frame_height
                    ):
                        cv2.line(
                            frame,
                            hand_points[start_idx],
                            hand_points[end_idx],
                            self.hand_connection_color,
                            2,
                        )

                for x, y in hand_points:
                    if 0 <= x < frame_width and 0 <= y < frame_height:
                        cv2.circle(frame, (x, y), 4, self.hand_landmark_color, -1)

        return frame

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.landmarks_data is None or self.total_frames == 0:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self.status_text
            )
            return

        if isinstance(self.landmarks_data, dict) and "frames" in self.landmarks_data:
            frame_data = self.landmarks_data["frames"][self.current_frame_idx]
        else:
            frame_data = self.landmarks_data[self.current_frame_idx]

        cv_frame = self.create_cv_frame(frame_data)
        height, width, channel = cv_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            cv_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        q_image = q_image.rgbSwapped()

        pixmap = QPixmap.fromImage(q_image)
        painter.drawPixmap(0, 0, self.width(), self.height(), pixmap)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            10, 25, f"Frame: {self.current_frame_idx + 1}/{self.total_frames}"
        )


class AudioRecorder(QObject):
    """Audio recording handler"""

    recording_complete = pyqtSignal(object, int)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.sample_rate = 44100
        self.is_recording = False
        self.recorded_audio = None

    def start_recording(self):
        self.is_recording = True
        thread = threading.Thread(target=self._record)
        thread.daemon = True
        thread.start()

    def _record(self):
        try:
            duration = 30
            self.recorded_audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
            )

            start_time = 0
            while self.is_recording and start_time < duration:
                sd.sleep(100)
                start_time += 0.1

            sd.stop()

            if self.recorded_audio is not None:
                recorded_frames = int(start_time * self.sample_rate)
                audio_trimmed = self.recorded_audio[:recorded_frames]
                self.recording_complete.emit(audio_trimmed, self.sample_rate)

        except Exception as e:
            self.error_occurred.emit(f"Recording error: {str(e)}")

    def stop_recording(self):
        self.is_recording = False
        sd.stop()


class VideoProcessor(QObject):
    """Video processing for sign language recognition"""

    frame_ready = pyqtSignal(object)
    gloss_detected = pyqtSignal(str, float)
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, hand_model_path, pose_model_path):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_processing = False
        self.show_landmarks = True

        # Load gesture recognition model
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

        # Initialize MediaPipe
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)
        self.detected_glosses = []
        self.last_prediction = ""
        self.confidence_threshold = 0.7
        self.frame_counter = 0
        self.timestamp_ms = 0

    def process_video_file(self, video_path):
        """Process a video file"""
        self.is_processing = True
        self.detected_glosses = []
        self.feature_queue.clear()

        thread = threading.Thread(target=self._process_video_file, args=(video_path,))
        thread.daemon = True
        thread.start()

    def _process_video_file(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)

            while self.is_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self._process_frame(frame)

            cap.release()
            self.processing_complete.emit(self.detected_glosses)

        except Exception as e:
            self.error_occurred.emit(f"Video processing error: {str(e)}")

    def process_webcam(self):
        """Process webcam stream"""
        self.is_processing = True
        self.detected_glosses = []
        self.feature_queue.clear()

        thread = threading.Thread(target=self._process_webcam)
        thread.daemon = True
        thread.start()

    def _process_webcam(self):
        try:
            cap = cv2.VideoCapture(0)

            while self.is_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self._process_frame(frame)
                self.frame_ready.emit(processed_frame)

            cap.release()
            self.processing_complete.emit(self.detected_glosses)

        except Exception as e:
            self.error_occurred.emit(f"Webcam processing error: {str(e)}")

    def _process_frame(self, frame):
        """Process a single frame"""
        self.frame_counter += 1
        should_process = self.frame_counter % 3 == 1

        if should_process:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            frame_data = {"hands": [], "pose": None}

            # Hand detection
            hand_result = self.hand_landmarker.detect_for_video(
                mp_image, self.timestamp_ms
            )
            if hand_result and hand_result.hand_landmarks:
                for hand_landmarks in hand_result.hand_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                    frame_data["hands"].append({"landmarks": landmarks})

            # Pose detection
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

            # Extract features and run inference
            features = self._extract_features(frame_data)
            self.feature_queue.append(features)

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

            # Draw landmarks if enabled
            if self.show_landmarks:
                frame = self._draw_landmarks(frame, frame_data)

        return frame

    def _extract_features(self, frame_data):
        """Extract features from frame data"""
        features = []
        feature_info = self.model_info["feature_info"]

        # Hand landmarks
        if feature_info["hand_landmarks"] > 0:
            max_hands = feature_info["max_hands"]
            hand_dim_per_hand = feature_info["hand_landmarks_per_hand"]
            hand_features = np.zeros(feature_info["hand_landmarks"])

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
        if feature_info["pose_landmarks"] > 0:
            pose_features = np.zeros(feature_info["pose_landmarks"])
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

    def _run_inference(self):
        """Run model inference on current sequence"""
        if len(self.feature_queue) < self.sequence_length:
            return None, 0.0

        sequence = list(self.feature_queue)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.model_info["class_names"][predicted.item()]
            confidence_value = confidence.item()

        return predicted_class, confidence_value

    def _draw_landmarks(self, frame, frame_data):
        """Draw landmarks on frame"""
        # Draw pose
        if frame_data.get("pose") and frame_data["pose"]:
            h, w = frame.shape[:2]
            pose_landmarks = frame_data["pose"]["landmarks"]
            for landmark in pose_landmarks:
                x, y = int(landmark[0] * w), int(landmark[1] * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Draw hands
        if frame_data.get("hands"):
            h, w = frame.shape[:2]
            for hand_data in frame_data["hands"]:
                hand_landmarks = hand_data["landmarks"]
                for landmark in hand_landmarks:
                    x, y = int(landmark[0] * w), int(landmark[1] * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        return frame

    def stop_processing(self):
        self.is_processing = False


# Pipeline Tabs


class Audio2GlossTab(QWidget):
    """Audio → Glosses → Visualization pipeline"""

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.converter = AudioToGlossConverter()
        self.generator = GestureTransitionGenerator(REPRESENTATIVES_LEFT)
        self.is_recording = False

        self.setup_ui()
        self.setup_connections()
        self.load_converter()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Landmark visualization (top 60%)
        self.canvas = LandmarkCanvas()
        layout.addWidget(self.canvas, 60)

        # Controls and outputs (bottom 40%)
        controls_layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet(
            "padding: 5px; background-color: #333; color: white;"
        )
        controls_layout.addWidget(self.status_label)

        # Outputs
        outputs_layout = QHBoxLayout()

        speech_layout = QVBoxLayout()
        speech_layout.addWidget(QLabel("Recognized Speech:"))
        self.speech_output = QTextEdit()
        self.speech_output.setReadOnly(True)
        self.speech_output.setMaximumHeight(100)
        speech_layout.addWidget(self.speech_output)
        outputs_layout.addLayout(speech_layout)

        gloss_layout = QVBoxLayout()
        gloss_layout.addWidget(QLabel("Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setMaximumHeight(100)
        gloss_layout.addWidget(self.gloss_output)
        outputs_layout.addLayout(gloss_layout)

        controls_layout.addLayout(outputs_layout)

        # Instructions
        instructions = QLabel("Press and hold SPACE to record, release to process")
        instructions.setStyleSheet("padding: 5px; font-style: italic;")
        controls_layout.addWidget(instructions)

        layout.addLayout(controls_layout, 40)
        self.setLayout(layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setup_connections(self):
        self.recorder.recording_complete.connect(self.on_recording_complete)
        self.recorder.error_occurred.connect(self.on_error)

    def load_converter(self):
        thread = threading.Thread(target=self._load_converter)
        thread.daemon = True
        thread.start()

    def _load_converter(self):
        if self.converter.load_model():
            self.status_label.setText("Ready - Press SPACE to record")
        else:
            self.status_label.setText(
                "Error: Install spaCy model (python -m spacy download en_core_web_sm)"
            )

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if not self.is_recording:
                self.start_recording()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if self.is_recording:
                self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.canvas.stop_animation()
        self.status_label.setText("🔴 Recording... (release SPACE to stop)")
        self.recorder.start_recording()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.recorder.stop_recording()
            self.status_label.setText("⏳ Processing...")

    def on_recording_complete(self, audio_array, sample_rate):
        thread = threading.Thread(
            target=self.process_audio, args=(audio_array, sample_rate)
        )
        thread.daemon = True
        thread.start()

    def process_audio(self, audio_array, sample_rate):
        try:
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            text, clause_glosses = self.converter.numpy_to_glosses(
                audio_array, sample_rate
            )

            # Flatten glosses
            glosses = []
            for clause in clause_glosses:
                glosses.extend(clause)

            self.speech_output.setText(text)
            self.gloss_output.setText(" ".join(glosses))

            if glosses:
                self.status_label.setText("⏳ Generating gesture visualization...")
                landmark_file = tempfile.mktemp(suffix=".pkl")
                glosses_lower = [g.lower() for g in glosses]
                result = self.generator.generate_sequence(
                    glosses_lower, 4, landmark_file
                )

                if result["denied_glosses"]:
                    self.status_label.setText(
                        f"⚠️ Some glosses not found: {', '.join(result['denied_glosses'])}"
                    )

                self.canvas.load_landmarks(landmark_file)
                self.status_label.setText(
                    "✓ Playing gesture sequence (SPACE to record again)"
                )
            else:
                self.status_label.setText("⚠️ No glosses generated")

        except sr.UnknownValueError:
            self.status_label.setText("❌ Could not understand audio")
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")

    def on_error(self, error_msg):
        self.status_label.setText(f"❌ Error: {error_msg}")


class Video2AudioTab(QWidget):
    """Video → Glosses → Text → Audio pipeline"""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_processor = VideoProcessor(
            MEDIAPIPE_HAND_LANDMARKER_PATH, MEDIAPIPE_POSE_LANDMARKER_PATH
        )
        self.gloss2audio = Gloss2Audio(self.device)
        self.current_video_path = None

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls_layout = QHBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Webcam (Live)", "Record Video", "Select Video File"])
        controls_layout.addWidget(QLabel("Mode:"))
        controls_layout.addWidget(self.mode_combo)

        self.landmark_checkbox = QCheckBox("Show Landmarks")
        self.landmark_checkbox.setChecked(True)
        self.landmark_checkbox.stateChanged.connect(self.toggle_landmarks)
        controls_layout.addWidget(self.landmark_checkbox)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_processing)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        self.generate_audio_btn = QPushButton("Generate Audio")
        self.generate_audio_btn.clicked.connect(self.generate_audio)
        self.generate_audio_btn.setEnabled(False)
        controls_layout.addWidget(self.generate_audio_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Video display
        self.video_label = QLabel("Video feed will appear here")
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("Ready - Select mode and click Start")
        self.status_label.setStyleSheet(
            "padding: 5px; background-color: #333; color: white;"
        )
        layout.addWidget(self.status_label)

        # Outputs
        outputs_layout = QHBoxLayout()

        # Glosses
        gloss_layout = QVBoxLayout()
        gloss_layout.addWidget(QLabel("Detected Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setMaximumHeight(100)
        gloss_layout.addWidget(self.gloss_output)
        outputs_layout.addLayout(gloss_layout)

        # Translated text
        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("Translated Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(100)
        text_layout.addWidget(self.text_output)
        outputs_layout.addLayout(text_layout)

        layout.addLayout(outputs_layout)

        self.setLayout(layout)

    def setup_connections(self):
        self.video_processor.frame_ready.connect(self.update_frame)
        self.video_processor.gloss_detected.connect(self.on_gloss_detected)
        self.video_processor.processing_complete.connect(self.on_processing_complete)
        self.video_processor.error_occurred.connect(self.on_error)

    def toggle_landmarks(self, state):
        self.video_processor.show_landmarks = state == Qt.CheckState.Checked.value

    def start_processing(self):
        mode = self.mode_combo.currentText()

        if mode == "Select Video File":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            if not file_path:
                return
            self.current_video_path = file_path
            self.gloss_output.clear()
            self.text_output.clear()
            self.status_label.setText("⏳ Processing video file...")
            self.video_processor.process_video_file(file_path)

        elif mode == "Record Video":
            # TODO: Implement video recording functionality
            QMessageBox.information(
                self,
                "Coming Soon",
                "Video recording will be implemented in the next update",
            )
            return

        else:  # Webcam (Live)
            self.gloss_output.clear()
            self.text_output.clear()
            self.status_label.setText("⏳ Starting webcam...")
            self.video_processor.process_webcam()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.generate_audio_btn.setEnabled(False)

    def stop_processing(self):
        self.video_processor.stop_processing()
        self.status_label.setText("✓ Processing stopped")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.generate_audio_btn.setEnabled(True)

    def update_frame(self, frame):
        """Update video display with processed frame"""
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
            self.gloss_output.setText(f"{current_text} {gloss}")
        else:
            self.gloss_output.setText(gloss)

        self.status_label.setText(f"✓ Detected: {gloss} (confidence: {confidence:.2f})")

    def on_processing_complete(self, glosses):
        """Handle completion of video processing"""
        self.status_label.setText(
            f"✓ Processing complete - {len(glosses)} glosses detected"
        )
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.generate_audio_btn.setEnabled(True)

        if glosses:
            self.gloss_output.setText(" ".join(glosses))

    def generate_audio(self):
        """Generate audio from detected glosses"""
        glosses_text = self.gloss_output.toPlainText().strip()
        if not glosses_text:
            QMessageBox.warning(
                self,
                "No Glosses",
                "No glosses to translate. Please process a video first.",
            )
            return

        glosses = glosses_text.split()
        self.status_label.setText("⏳ Translating glosses to text...")

        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._generate_audio, args=(glosses,))
        thread.daemon = True
        thread.start()

    def _generate_audio(self, glosses):
        """Background thread for audio generation"""
        try:
            # Translate glosses to text
            text = self.gloss2audio.gloss2text.infer(glosses)
            text_str = " ".join(text)

            # Update text output
            self.text_output.setText(text_str)
            self.status_label.setText("⏳ Generating speech audio...")

            # Generate audio
            audio_path = tempfile.mktemp(suffix=".mp3")
            self.gloss2audio.infer_and_synthesize(glosses, audio_path)

            self.status_label.setText(f"✓ Audio generated: {audio_path}")

            # Play audio (platform-dependent)
            self._play_audio(audio_path)

        except Exception as e:
            self.status_label.setText(f"❌ Error generating audio: {str(e)}")
            print(f"Audio generation error: {e}")
            import traceback

            traceback.print_exc()

    def _play_audio(self, audio_path):
        """Play generated audio file"""
        try:
            import platform
            import subprocess

            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.call(["afplay", audio_path])
            elif system == "Windows":
                subprocess.call(["start", audio_path], shell=True)
            else:  # Linux
                subprocess.call(["xdg-open", audio_path])
        except Exception as e:
            print(f"Could not play audio: {e}")

    def on_error(self, error_msg):
        self.status_label.setText(f"❌ {error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Namibian Sign Language Translation System")
        self.setGeometry(100, 100, 1200, 800)

        # Create tab widget
        tabs = QTabWidget()

        # Add pipeline tabs
        self.audio2gloss_tab = Audio2GlossTab()
        self.video2audio_tab = Video2AudioTab()

        tabs.addTab(self.audio2gloss_tab, "Audio → Glosses → Visualization")
        tabs.addTab(self.video2audio_tab, "Video → Glosses → Text → Audio")

        # Set style
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: white;
                padding: 10px 20px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4a4a4a;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: white;
                border: 1px solid #444;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                margin-right: 5px;
            }
            QCheckBox {
                color: white;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #4a9eff;
            }
        """
        )

        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        """Handle application close"""
        # Stop any ongoing processing
        if hasattr(self, "video2audio_tab"):
            self.video2audio_tab.video_processor.stop_processing()

        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application-wide font
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
