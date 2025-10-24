"""
Gloss to Visualization GUI
Enter glosses, see animated sign language visualization
"""

import sys
import os
import pickle
import tempfile
import threading
import numpy as np
import cv2
import mediapipe as mp
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor, QImage, QPixmap

try:
    from ..gloss2visualization import GestureTransitionGenerator
    from ..constants import REPRESENTATIVES_LEFT
except ImportError:
    print("Import error: Ensure gloss2visualization module is available")


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


class Gloss2VisualizationWidget(QWidget):
    """Main widget"""

    update_status_signal = pyqtSignal(str)
    visualize_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.generator = GestureTransitionGenerator(REPRESENTATIVES_LEFT)
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Visualization canvas
        self.canvas = LandmarkCanvas()
        layout.addWidget(self.canvas)

        # Input section
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Enter Glosses (space-separated):"))

        gloss_layout = QHBoxLayout()
        self.gloss_input = QTextEdit()
        self.gloss_input.setPlaceholderText("Example: TOMORROW I STORE GO")
        self.gloss_input.setMaximumHeight(60)
        gloss_layout.addWidget(self.gloss_input)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.generate_visualization)
        gloss_layout.addWidget(self.generate_btn)

        input_layout.addLayout(gloss_layout)
        layout.addLayout(input_layout)

        self.setLayout(layout)

    def setup_connections(self):
        self.update_status_signal.connect(self.status_label.setText)
        self.visualize_signal.connect(self.canvas.load_landmarks)

    def generate_visualization(self):
        glosses_text = self.gloss_input.toPlainText().strip()
        if not glosses_text:
            self.update_status_signal.emit("Error: Enter glosses first")
            return

        glosses = glosses_text.split()
        self.canvas.stop_animation()
        self.update_status_signal.emit("Generating...")

        thread = threading.Thread(target=self._generate, args=(glosses,))
        thread.daemon = True
        thread.start()

    def _generate(self, glosses):
        try:
            fd, landmark_file = tempfile.mkstemp(suffix=".pkl")
            os.close(fd)
            glosses_lower = [g.lower() for g in glosses]
            result = self.generator.generate_sequence(glosses_lower, 4, landmark_file)

            if result["denied_glosses"]:
                self.update_status_signal.emit(
                    f"Warning: Not found: {', '.join(result['denied_glosses'])}"
                )
            else:
                self.update_status_signal.emit("Playing")

            self.visualize_signal.emit(landmark_file)

        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")
            print(f"Generation error: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gloss to Visualization")
        self.setGeometry(100, 100, 700, 600)

        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = Gloss2VisualizationWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
