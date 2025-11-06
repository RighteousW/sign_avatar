"""
Integrated NSL Translation System
Two main pipelines:
1. Speech-to-Visualization: Speech → Text → Gloss → Avatar
2. Video-to-Audio: Video → Gloss → Text → Audio
"""

import sys
import os
import threading
import tempfile
import pickle
import numpy as np
import torch
import cv2
import speech_recognition as sr
import sounddevice as sd
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
    QTabWidget,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent

try:
    from ..audio2gloss import AudioToGlossConverter
    from ..gloss2visualization import GestureTransitionGenerator
    from ..gloss2audio import Gloss2Text
    from ..constants import (
        REPRESENTATIVES_MANUAL,
        get_gesture_metadata_path,
        get_gesture_model_path,
    )
    from ..model_training import GestureRecognizerModel
    from ..landmark_extraction import LandmarkExtractor
    import mediapipe as mp
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure all required modules are available")


class AudioRecorder(QObject):
    """Handles audio recording"""

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
            max_duration = 30
            self.recorded_audio = sd.rec(
                int(max_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
            )

            start_time = 0
            while self.is_recording and start_time < max_duration:
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


class LandmarkCanvas(QWidget):
    """Widget for displaying animated landmarks"""

    def __init__(self):
        super().__init__()
        self.landmarks_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.setMinimumHeight(400)
        self.setStyleSheet("background-color: black;")

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

        # Draw pose
        if frame_data.get("pose") and frame_data["pose"]:
            pose_landmarks = frame_data["pose"]["landmarks"]
            pose_points = []
            for landmark in pose_landmarks:
                x = int(landmark[0] * frame_width)
                y = int(landmark[1] * frame_height)
                pose_points.append((x, y))

            for connection in self.pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(pose_points) and end_idx < len(pose_points):
                    cv2.line(
                        frame,
                        pose_points[start_idx],
                        pose_points[end_idx],
                        (0, 100, 255),
                        2,
                    )

            for x, y in pose_points:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Draw hands
        if frame_data.get("hands"):
            for hand_data in frame_data["hands"]:
                hand_landmarks = hand_data["landmarks"]
                hand_points = []
                for landmark in hand_landmarks:
                    x = int(landmark[0] * frame_width)
                    y = int(landmark[1] * frame_height)
                    hand_points.append((x, y))

                for connection in self.hand_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_points) and end_idx < len(hand_points):
                        cv2.line(
                            frame,
                            hand_points[start_idx],
                            hand_points[end_idx],
                            (0, 200, 0),
                            2,
                        )

                for x, y in hand_points:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        return frame

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor

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


class Speech2VisualizationWidget(QWidget):
    """Pipeline: Speech → Text → Gloss → Avatar Visualization"""

    update_text_signal = pyqtSignal(str)
    update_gloss_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)
    visualize_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.audio_converter = AudioToGlossConverter()
        self.generator = GestureTransitionGenerator(REPRESENTATIVES_MANUAL)
        self.is_recording = False

        self.setup_ui()
        self.setup_connections()
        self.load_converter()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Loading...")
        layout.addWidget(self.status_label)

        # Visualization canvas (top)
        self.canvas = LandmarkCanvas()
        layout.addWidget(self.canvas, 50)

        # Speech output
        layout.addWidget(QLabel("Transcribed Speech:"))
        self.speech_output = QTextEdit()
        self.speech_output.setReadOnly(True)
        self.speech_output.setMaximumHeight(60)
        layout.addWidget(self.speech_output)

        # Gloss output
        layout.addWidget(QLabel("Generated Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setMaximumHeight(60)
        layout.addWidget(self.gloss_output)

        # Instructions
        instructions = QLabel(
            "Press SPACE to record, release to stop and generate visualization"
        )
        layout.addWidget(instructions)

        self.setLayout(layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setup_connections(self):
        self.recorder.recording_complete.connect(self.on_recording_complete)
        self.recorder.error_occurred.connect(self.on_error)
        self.update_text_signal.connect(self.speech_output.setText)
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_status_signal.connect(self.status_label.setText)
        self.visualize_signal.connect(self.canvas.load_landmarks)

    def load_converter(self):
        thread = threading.Thread(target=self._load_converter)
        thread.daemon = True
        thread.start()

    def _load_converter(self):
        if self.audio_converter.load_model():
            self.update_status_signal.emit("Ready - Press SPACE to record")
        else:
            self.update_status_signal.emit(
                "Error: Run 'python3 -m spacy download en_core_web_sm'"
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
        self.update_status_signal.emit("Recording...")
        self.canvas.stop_animation()
        self.recorder.start_recording()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.recorder.stop_recording()
            self.update_status_signal.emit("Processing...")

    def on_recording_complete(self, audio_array, sample_rate):
        thread = threading.Thread(
            target=self.process_audio, args=(audio_array, sample_rate)
        )
        thread.daemon = True
        thread.start()

    def process_audio(self, audio_array, sample_rate):
        try:
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)
            text, clause_glosses = self.audio_converter.numpy_to_glosses(
                audio_array, sample_rate
            )

            glosses = []
            for clause in clause_glosses:
                glosses.extend(clause)

            self.update_text_signal.emit(text)
            self.update_gloss_signal.emit(" ".join(glosses))

            if glosses:
                self.update_status_signal.emit("Generating visualization...")
                self.generate_visualization(glosses)
            else:
                self.update_status_signal.emit("Ready - No glosses generated")

        except sr.UnknownValueError:
            self.update_status_signal.emit("Could not understand audio")
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")

    def generate_visualization(self, glosses):
        try:
            fd, landmark_file = tempfile.mkstemp(suffix=".pkl")
            os.close(fd)
            glosses_lower = [g.lower() for g in glosses]
            result = self.generator.generate_sequence(glosses_lower, 10, landmark_file)

            if result["denied_glosses"]:
                self.update_status_signal.emit(
                    f"Warning: Not found: {', '.join(result['denied_glosses'])}"
                )
            else:
                self.update_status_signal.emit("Playing visualization")

            self.visualize_signal.emit(landmark_file)

        except Exception as e:
            self.update_status_signal.emit(f"Visualization error: {str(e)}")

    def on_error(self, error_msg):
        self.update_status_signal.emit(error_msg)


class VideoProcessor(QObject):
    """Processes video file and detects glosses"""

    gloss_detected = pyqtSignal(str, float)
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_processing = False
        self.confidence_threshold = 0.7

        # Load gesture recognition model
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

        # Initialize landmark extractor
        self.landmark_extractor = LandmarkExtractor(use_pose=True)
        self.feature_info = self.model_info["feature_info"]
        self.frame_counter = 0
        self.last_processed_features = None

        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)
        self.detected_glosses = []
        self.last_prediction = ""

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
                        landmarks = np.array(hand_data["landmarks"][:21])[:, :3].flatten()
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
        """Process frame with 2_skip pattern"""
        frame_index = self.frame_counter
        self.frame_counter += 1

        should_process = frame_index % 3 == 0

        if should_process:
            frame_data = self.landmark_extractor.extract_landmarks_from_frame(frame)
            current_features = self.extract_features_from_frame_data(frame_data)
            self.last_processed_features = current_features
            return [current_features]
        else:
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

            while self.is_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

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
        """Padding logic"""
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


class Video2AudioWidget(QWidget):
    """Pipeline: Video → Gloss → Text → Audio"""

    update_gloss_signal = pyqtSignal(str)
    update_text_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.processor = VideoProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = None
        self.current_video_path = None

        self.setup_ui()
        self.setup_connections()
        self.load_translator()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Loading translator...")
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

        # Detected glosses
        layout.addWidget(QLabel("Detected Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setMaximumHeight(80)
        layout.addWidget(self.gloss_output)

        # Translated text
        layout.addWidget(QLabel("Translated Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(80)
        layout.addWidget(self.text_output)

        # Audio controls
        audio_layout = QHBoxLayout()
        self.play_audio_btn = QPushButton("Play Audio")
        self.play_audio_btn.clicked.connect(self.play_audio)
        self.play_audio_btn.setEnabled(False)
        audio_layout.addWidget(self.play_audio_btn)
        audio_layout.addStretch()
        layout.addLayout(audio_layout)

        self.setLayout(layout)

    def setup_connections(self):
        self.processor.gloss_detected.connect(self.on_gloss_detected)
        self.processor.processing_complete.connect(self.on_processing_complete)
        self.processor.error_occurred.connect(self.on_error)
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_text_signal.connect(self.text_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def load_translator(self):
        """Load translation model"""
        thread = threading.Thread(target=self._load_translator)
        thread.daemon = True
        thread.start()

    def _load_translator(self):
        try:
            self.translator = Gloss2Text(self.device)
            self.update_status_signal.emit("Ready - Select a video file")
        except Exception as e:
            self.update_status_signal.emit(f"Error loading translator: {str(e)}")

    def select_video(self):
        """Open file dialog to select video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.current_video_path = file_path
            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.process_btn.setEnabled(True)
            self.play_audio_btn.setEnabled(False)
            self.update_status_signal.emit("Ready to process")

    def process_video(self):
        """Start processing the selected video"""
        if not self.current_video_path:
            return

        self.gloss_output.clear()
        self.text_output.clear()
        self.update_status_signal.emit("Processing video...")
        self.select_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.play_audio_btn.setEnabled(False)

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
        self.update_status_signal.emit(
            f"Video complete - Translating {len(glosses)} glosses..."
        )

        if glosses and self.translator:
            thread = threading.Thread(target=self._translate_glosses, args=(glosses,))
            thread.daemon = True
            thread.start()
        else:
            self.update_status_signal.emit("No glosses detected")
            self.select_btn.setEnabled(True)
            self.process_btn.setEnabled(True)

    def _translate_glosses(self, glosses):
        """Translate glosses to text"""
        try:
            text = self.translator.infer(glosses)
            text_str = " ".join(text).replace("_", " ")

            self.update_text_signal.emit(text_str)
            self.update_status_signal.emit("Translation complete - Ready to play audio")
            self.play_audio_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.process_btn.setEnabled(True)

        except Exception as e:
            self.update_status_signal.emit(f"Translation error: {str(e)}")
            self.select_btn.setEnabled(True)
            self.process_btn.setEnabled(True)

    def play_audio(self):
        """Play translated text as audio"""
        text = self.text_output.toPlainText().strip()
        if not text:
            self.update_status_signal.emit("No text to play")
            return

        self.update_status_signal.emit("Playing audio...")
        thread = threading.Thread(target=self._play_audio, args=(text,))
        thread.daemon = True
        thread.start()

    def _play_audio(self, text):
        """Generate and play audio"""
        try:
            from gtts import gTTS
            import tempfile
            import os

            # Generate audio
            tts = gTTS(text=text, lang="en")
            fd, temp_audio = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            tts.save(temp_audio)

            # Play audio (cross-platform)
            if sys.platform == "linux":
                os.system(f"mpg123 -q {temp_audio}")
            elif sys.platform == "darwin":
                os.system(f"afplay {temp_audio}")
            elif sys.platform == "win32":
                os.system(f"start {temp_audio}")

            # Cleanup
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            self.update_status_signal.emit("Ready")

        except Exception as e:
            self.update_status_signal.emit(f"Audio error: {str(e)}")

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"Error: {error_msg}")
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)


class IntegratedNSLSystem(QTabWidget):
    """Main integrated system with two pipelines"""

    def __init__(self):
        super().__init__()
        self.setup_tabs()

    def setup_tabs(self):
        """Setup the two main pipeline tabs"""

        # Pipeline 1: Speech → Visualization
        self.speech2viz = Speech2VisualizationWidget()
        self.addTab(self.speech2viz, "Speech → Visualization")

        # Pipeline 2: Video → Audio
        self.video2audio = Video2AudioWidget()
        self.addTab(self.video2audio, "Video → Audio")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NSL Integrated Translation System")
        self.setGeometry(100, 100, 900, 700)

        # Apply stylesheet
        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            # Inline minimal dark theme
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                }
                QPushButton {
                    background-color: #0d7377;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #14a085;
                }
                QPushButton:disabled {
                    background-color: #2d2d2d;
                    color: #666;
                }
                QTextEdit {
                    background-color: #2b2b2b;
                    color: #e0e0e0;
                    border: 1px solid #3d3d3d;
                    border-radius: 4px;
                }
                QLabel {
                    color: #e0e0e0;
                }
            """
            )

        widget = IntegratedNSLSystem()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
