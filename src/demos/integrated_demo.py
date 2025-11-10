"""
Integrated NSL Translation System Demo
Two complete pipelines:
1. Audio → Text → Gloss → Visualization
2. Video → Gloss → Text → Audio
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
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QKeyEvent

try:
    from .demo_utils import get_dark_stylesheet, LandmarkCanvas, VideoDisplayLabel
    from ..audio2gloss import AudioToGlossConverter
    from ..gloss2visualization import GestureTransitionGenerator
    from ..gloss2audio import Gloss2Text
    from ..constants import (
        REPRESENTATIVES_MANUAL,
        get_gesture_metadata_path,
        get_gesture_model_path,
        FRAME_WIDTH,
        FRAME_HEIGHT,
    )
    from ..utils.interpolation import apply_frame_skipping
    from ..model_training import GestureRecognizerCNN, GestureRecognizerLSTM
    from ..landmark_extraction import LandmarkExtractor
except ImportError:
    print("Import error - ensure modules are available")


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


# ==================== Audio to Visualization Pipeline ====================
class Audio2VisualizationWidget(QWidget):
    """Pipeline: Audio → Text → Gloss → Visualization"""

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

        # Title
        title = QLabel("Audio → Text → Gloss → Visualization")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("Loading...")
        layout.addWidget(self.status_label)

        # Visualization canvas
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
            "Hold SPACE to record, release to stop and generate visualization"
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
            self.update_status_signal.emit("✓ Ready - Press SPACE to record")
        else:
            self.update_status_signal.emit(
                "⚠ Error: Run 'python3 -m spacy download en_core_web_sm'"
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
        self.update_status_signal.emit("🔴 Recording...")
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
                self.update_status_signal.emit("✓ Playing visualization")

            self.visualize_signal.emit(landmark_file)

        except Exception as e:
            self.update_status_signal.emit(f"Visualization error: {str(e)}")

    def on_error(self, error_msg):
        self.update_status_signal.emit(error_msg)


# ==================== Video Processor ====================
class VideoProcessor(QObject):
    """Processes video and detects glosses"""

    frame_ready = pyqtSignal(object)
    gloss_detected = pyqtSignal(str, float)
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_processing = False
        self.confidence_threshold = 0.7

        self.use_pose = False
        self.skip_pattern = 2
        self.model_type = "cnn"

        # Load model
        with open(str(get_gesture_metadata_path(self.use_pose, self.skip_pattern, self.model_type)), "rb") as f:
            self.model_info = pickle.load(f)

        self.model = (
            GestureRecognizerCNN(
                input_size=self.model_info["input_size"],
                num_classes=len(self.model_info["class_names"]),
                hidden_size=self.model_info["hidden_size"],
                dropout=self.model_info["dropout"],
            )
            if self.model_type == "cnn"
            else GestureRecognizerLSTM(
                input_size=self.model_info["input_size"],
                num_classes=len(self.model_info["class_names"]),
                hidden_size=self.model_info["hidden_size"],
                dropout=self.model_info["dropout"],
            )
        )
        checkpoint = torch.load(
            str(
                get_gesture_model_path(
                    self.use_pose, self.skip_pattern, self.model_type
                )
            ),
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.landmark_extractor = LandmarkExtractor(use_pose=True)
        self.feature_info = self.model_info["feature_info"]
        self.sequence_length = self.model_info["sequence_length"]
        self.feature_queue = deque(maxlen=self.sequence_length)
        self.detected_glosses = []
        self.last_prediction = ""

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

        if not hasattr(self, "frame_buffer"):
            self.frame_buffer = []

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

    def process_video(self, video_source, is_file=True):
        self.is_processing = True
        self.detected_glosses = []
        self.feature_queue.clear()
        self.last_prediction = ""
        self.frame_buffer = []

        thread = threading.Thread(
            target=self._process_video, args=(video_source, is_file)
        )
        thread.daemon = True
        thread.start()

    def _process_video(self, video_source, is_file):
        try:
            if is_file:
                cap = cv2.VideoCapture(video_source)
            else:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            while self.is_processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_ready.emit(frame)

                if is_file:
                    cv2.waitKey(33)  # Delay for file playback

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

                            for _ in range(int(0.8 * self.sequence_length)):
                                if self.feature_queue:
                                    self.feature_queue.popleft()

            cap.release()
            self.processing_complete.emit(self.detected_glosses)

        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")

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

    def stop_processing(self):
        self.is_processing = False


# ==================== Video to Audio Pipeline ====================
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

        # Title
        title = QLabel("Video → Gloss → Text → Audio")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("Loading translator...")
        layout.addWidget(self.status_label)

        # Video display
        self.video_display = VideoDisplayLabel("Select a video file to begin")
        layout.addWidget(self.video_display, 50)

        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label)

        self.select_btn = QPushButton("Select Video")
        self.select_btn.clicked.connect(self.select_video)
        file_layout.addWidget(self.select_btn)

        layout.addLayout(file_layout)

        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)

        # Detected glosses
        layout.addWidget(QLabel("Detected Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setMaximumHeight(60)
        layout.addWidget(self.gloss_output)

        # Translated text
        layout.addWidget(QLabel("Translated Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(60)
        layout.addWidget(self.text_output)

        # Audio controls
        self.play_audio_btn = QPushButton("Play Audio")
        self.play_audio_btn.clicked.connect(self.play_audio)
        self.play_audio_btn.setEnabled(False)
        layout.addWidget(self.play_audio_btn)

        self.setLayout(layout)

    def setup_connections(self):
        self.processor.frame_ready.connect(self.update_frame)
        self.processor.gloss_detected.connect(self.on_gloss_detected)
        self.processor.processing_complete.connect(self.on_processing_complete)
        self.processor.error_occurred.connect(self.on_error)

        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_text_signal.connect(self.text_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def load_translator(self):
        thread = threading.Thread(target=self._load_translator)
        thread.daemon = True
        thread.start()

    def _load_translator(self):
        try:
            self.translator = Gloss2Text(self.device)
            self.update_status_signal.emit("✓ Ready - Select video file")
        except Exception as e:
            self.update_status_signal.emit(f"⚠ Error loading translator: {str(e)}")

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "data/demo_videos",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if file_path:
            self.current_video_path = file_path
            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.process_btn.setEnabled(True)
            self.play_audio_btn.setEnabled(False)
            self.gloss_output.clear()
            self.text_output.clear()
            self.update_status_signal.emit("✓ Ready to process")

    def update_frame(self, frame):
        self.video_display.display_frame(frame)

    def process_video(self):
        if not self.current_video_path:
            return

        self.gloss_output.clear()
        self.text_output.clear()
        self.update_status_signal.emit("▶ Processing video...")
        self.select_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.play_audio_btn.setEnabled(False)

        self.processor.process_video(self.current_video_path, is_file=True)

    def on_gloss_detected(self, gloss, confidence):
        current_text = self.gloss_output.toPlainText()
        if current_text:
            self.update_gloss_signal.emit(f"{current_text} {gloss}")
        else:
            self.update_gloss_signal.emit(gloss)

    def on_processing_complete(self, glosses):
        self.update_status_signal.emit(
            f"Video complete - Translating {len(glosses)} glosses..."
        )

        if glosses and self.translator:
            thread = threading.Thread(target=self._translate_glosses, args=(glosses,))
            thread.daemon = True
            thread.start()
        else:
            self.update_status_signal.emit("✓ No glosses detected")
            self.select_btn.setEnabled(True)
            self.process_btn.setEnabled(True)

    def _translate_glosses(self, glosses):
        try:
            capitalized_glosses = [gloss.capitalize() for gloss in glosses]
            text = self.translator.infer(capitalized_glosses)
            text_str = " ".join(text).replace("_", " ")

            self.update_text_signal.emit(text_str)
            self.update_status_signal.emit(
                "✓ Translation complete - Ready to play audio"
            )
            self.play_audio_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.process_btn.setEnabled(True)

        except Exception as e:
            self.update_status_signal.emit(f"⚠ Translation error: {str(e)}")
            self.select_btn.setEnabled(True)
            self.process_btn.setEnabled(True)

    def play_audio(self):
        text = self.text_output.toPlainText().strip()
        if not text:
            return

        self.update_status_signal.emit("🔊 Playing audio...")
        thread = threading.Thread(target=self._play_audio, args=(text,))
        thread.daemon = True
        thread.start()

    def _play_audio(self, text):
        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang="en")
            fd, temp_audio = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            tts.save(temp_audio)

            if sys.platform == "linux":
                os.system(f"mpg123 -q {temp_audio}")
            elif sys.platform == "darwin":
                os.system(f"afplay {temp_audio}")
            elif sys.platform == "win32":
                os.system(f"start {temp_audio}")

            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            self.update_status_signal.emit("✓ Ready")

        except Exception as e:
            self.update_status_signal.emit(f"⚠ Audio error: {str(e)}")

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"⚠ {error_msg}")
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)


# ==================== Main Application ====================
class IntegratedNSLSystem(QTabWidget):
    """Main integrated system with two pipelines"""

    def __init__(self):
        super().__init__()
        self.setup_tabs()

    def setup_tabs(self):
        # Pipeline 1: Audio → Visualization
        self.audio2viz = Audio2VisualizationWidget()
        self.addTab(self.audio2viz, "Audio → Visualization")

        # Pipeline 2: Video → Audio
        self.video2audio = Video2AudioWidget()
        self.addTab(self.video2audio, "Video → Audio")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NSL Integrated Translation System")
        self.setGeometry(100, 100, 900, 800)
        self.setStyleSheet(get_dark_stylesheet())

        widget = IntegratedNSLSystem()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
