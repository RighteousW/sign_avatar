"""
Audio to Glosses GUI with Landmark Visualization
Space bar to record/stop
"""

import os
import sys
import threading
import pickle
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QKeyEvent, QPainter, QColor, QImage, QPixmap
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import cv2
import mediapipe as mp

from audio_to_gloss import AudioToGlossConverter
from interpolation_transition import GestureTransitionGenerator
from constants import REPRESENTATIVES_LEFT


class LandmarkCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.landmarks_data = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.setMinimumHeight(480)
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
            # delete the temp file
            os.remove(filepath)

            # Handle both old format (list) and new format (dict with 'frames')
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
        except FileNotFoundError:
            self.status_text = "Error: Landmark file not found"
            self.update()
        except EOFError:
            self.status_text = "Error: Landmark file is empty or corrupted"
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
        """Create a CV2 frame with landmarks"""
        frame_width = self.width()
        frame_height = self.height()

        # Create black background
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Draw pose landmarks first
        if frame_data.get("pose") and frame_data["pose"]:
            pose_landmarks = frame_data["pose"]["landmarks"]

            pose_points = []
            for landmark in pose_landmarks:
                x = int(landmark[0] * frame_width)
                y = int(landmark[1] * frame_height)
                pose_points.append((x, y))

            # Draw pose connections
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

            # Draw pose landmarks
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

                # Draw hand connections
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

                # Draw hand landmarks
                for x, y in hand_points:
                    if 0 <= x < frame_width and 0 <= y < frame_height:
                        cv2.circle(frame, (x, y), 4, self.hand_landmark_color, -1)

                # Add hand label
                if hand_points and len(hand_points) > 0:
                    wrist_x, wrist_y = hand_points[0]
                    if 0 <= wrist_x < frame_width and 10 <= wrist_y < frame_height:
                        cv2.putText(
                            frame,
                            handedness,
                            (wrist_x, wrist_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            self.hand_landmark_color,
                            1,
                        )

        return frame

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.landmarks_data is None or self.total_frames == 0:
            # Draw status text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self.status_text
            )
            return

        # Get current frame data
        if isinstance(self.landmarks_data, dict) and "frames" in self.landmarks_data:
            frame_data = self.landmarks_data["frames"][self.current_frame_idx]
        else:
            frame_data = self.landmarks_data[self.current_frame_idx]

        # Create CV frame
        cv_frame = self.create_cv_frame(frame_data)

        # Convert CV frame to QImage
        height, width, channel = cv_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            cv_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )
        q_image = q_image.rgbSwapped()  # Convert BGR to RGB

        # Draw the image
        pixmap = QPixmap.fromImage(q_image)
        painter.drawPixmap(0, 0, self.width(), self.height(), pixmap)

        # Draw frame counter
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            10, 25, f"Frame: {self.current_frame_idx + 1}/{self.total_frames}"
        )


class AudioRecorder(QObject):
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


class MainWidget(QWidget):
    update_text_signal = pyqtSignal(str)
    update_gloss_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)
    visualize_signal = pyqtSignal(str)

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
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top 80% - Landmark visualization with status overlay
        self.canvas = LandmarkCanvas()

        # Status label overlay (positioned on top of canvas)
        self.status_label = QLabel(self.canvas)
        self.status_label.setText("Loading...")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
        )
        self.status_label.setStyleSheet(
            "padding: 10px; font-size: 14px; color: white; background-color: rgba(0, 0, 0, 150);"
        )
        self.status_label.setGeometry(0, 0, 800, 50)
        self.status_label.raise_()

        main_layout.addWidget(self.canvas, 80)

        # Bottom 20% - Split horizontally
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(5, 5, 5, 5)
        bottom_layout.setSpacing(5)

        # Left - Recognized speech
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Speech:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        left_layout.addWidget(self.text_output)
        bottom_layout.addLayout(left_layout, 50)

        # Right - Glosses
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        right_layout.addWidget(self.gloss_output)
        bottom_layout.addLayout(right_layout, 50)

        main_layout.addWidget(bottom_widget, 20)

        self.setLayout(main_layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setup_connections(self):
        self.recorder.recording_complete.connect(self.on_recording_complete)
        self.recorder.error_occurred.connect(self.on_error)

        self.update_text_signal.connect(self.text_output.setText)
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_status_signal.connect(self.status_label.setText)
        self.visualize_signal.connect(self.canvas.load_landmarks)

    def load_converter(self):
        thread = threading.Thread(target=self._load_converter)
        thread.daemon = True
        thread.start()

    def _load_converter(self):
        if self.converter.load_model():
            self.update_status_signal.emit(
                "Press SPACE to record, release SPACE to stop and generate"
            )
        else:
            self.update_status_signal.emit("Error: Install spaCy model")

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
        self.update_status_signal.emit("Recording... (release SPACE to stop)")
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
            # Clean audio array (remove NaN and inf values)
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Convert audio to glosses
            text, glosses = self.converter.numpy_to_glosses(audio_array, sample_rate)

            self.update_text_signal.emit(text)
            self.update_gloss_signal.emit(" ".join(glosses))

            # Generate gesture transitions
            if glosses:
                self.update_status_signal.emit("Generating gestures...")
                landmark_file = "temp_landmarks.pkl"
                # Convert glosses to lowercase for matching with saved data
                glosses_lower = [g.lower() for g in glosses]
                denied_glosses = self.generator.generate_sequence(glosses_lower, 4, landmark_file)["denied_glosses"]
                if denied_glosses:
                    self.update_status_signal.emit(
                        f"Warning: Some glosses not found: {', '.join(denied_glosses)}"
                    )

                # Load and visualize
                self.visualize_signal.emit(landmark_file)
                self.update_status_signal.emit(
                    "Playing gesture sequence (SPACE to record again)"
                )
            else:
                self.update_status_signal.emit(
                    "No glosses generated (SPACE to try again)"
                )

        except sr.UnknownValueError:
            self.update_status_signal.emit("Could not understand audio")
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")
            print(f"Full error: {e}")
            import traceback

            traceback.print_exc()

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"Error: {error_msg}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio to Glosses")
        self.setGeometry(100, 100, 800, 600)

        widget = MainWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
