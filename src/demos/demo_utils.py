"""
Shared utilities for NSL demonstration apps
"""

import pickle
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor

try:
    import mediapipe as mp
except ImportError:
    mp = None


def get_dark_stylesheet():
    """Returns dark theme stylesheet for all demo apps"""
    return """
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
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #14a085;
        }
        QPushButton:disabled {
            background-color: #2d2d2d;
            color: #666;
        }
        QTextEdit, QLineEdit, QPlainTextEdit {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
            font-size: 13px;
        }
        QLabel {
            color: #e0e0e0;
            font-size: 13px;
        }
        QRadioButton {
            color: #e0e0e0;
            font-size: 13px;
        }
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2b2b2b;
            color: #e0e0e0;
            padding: 8px 15px;
            border: 1px solid #3d3d3d;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background-color: #0d7377;
        }
    """


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
        if mp:
            self.hand_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
            self.pose_connections = list(mp.solutions.pose.POSE_CONNECTIONS)
        else:
            self.hand_connections = []
            self.pose_connections = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.status_text = "Ready"

    def load_landmarks(self, filepath):
        try:
            with open(filepath, "rb") as f:
                self.landmarks_data = pickle.load(f)

            import os

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


class VideoDisplayLabel(QLabel):
    """Label for displaying video frames"""

    def __init__(self, text=""):
        super().__init__(text)
        self.setMinimumHeight(400)
        self.setStyleSheet("background-color: black; color: gray;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)

    def display_frame(self, frame):
        """Display a cv2 frame"""
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )

            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled_pixmap)

    def clear_display(self):
        """Clear the video display"""
        self.clear()
        self.setText("Ready")
