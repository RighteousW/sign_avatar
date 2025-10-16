"""
Record Video to Gloss GUI - Simple Version
Record video, then process it using the file processor
"""

import sys
import threading
import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, QObject, Qt
from PyQt6.QtGui import QImage, QPixmap

try:
    from ..constants import FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT
    from ..data_creation.video_recording import VideoRecorder, FrameTimer
    from .video2gloss_file_gui import VideoFileProcessor
except ImportError:
    print("Import error: Ensure required modules are available")


class VideoRecordingCapture(QObject):
    """Handles video recording from webcam"""

    frame_ready = pyqtSignal(object)
    recording_saved = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.video_recorder = None
        self.frame_timer = None

    def start_recording(self, word):
        """Start recording"""
        self.is_recording = True
        self.video_recorder = VideoRecorder(word)
        self.frame_timer = FrameTimer(FRAME_RATE)

        thread = threading.Thread(target=self._record)
        thread.daemon = True
        thread.start()

    def _record(self):
        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                self.error_occurred.emit("Could not open camera")
                return

            # Setup camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Start video recording
            self.video_recorder.start_recording(actual_width, actual_height, FRAME_RATE)

            while self.is_recording and cap.isOpened():
                # Wait for proper frame timing
                self.frame_timer.wait_for_next_frame()

                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame
                self.video_recorder.add_frame(frame)

                # Display frame
                display_frame = frame.copy()
                rec_text = f"REC [{self.video_recorder.frame_count}]"
                cv2.putText(
                    display_frame,
                    rec_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                self.frame_ready.emit(display_frame)

            cap.release()

            # Save recording
            if self.video_recorder:
                saved_paths = self.video_recorder.stop_recording()
                self.recording_saved.emit(saved_paths)

        except Exception as e:
            self.error_occurred.emit(f"Recording error: {str(e)}")

    def stop_recording(self):
        self.is_recording = False


class RecordVideo2GlossWidget(QWidget):
    """Main widget - record then process"""

    update_gloss_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recorder = VideoRecordingCapture()
        self.processor = VideoFileProcessor()
        self.recorded_video_path = None

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Video display
        self.video_label = QLabel("Video feed will appear here")
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Word input
        word_layout = QHBoxLayout()
        word_layout.addWidget(QLabel("Word:"))
        self.word_input = QTextEdit()
        self.word_input.setPlaceholderText("Enter word to record...")
        self.word_input.setText("sign")
        self.word_input.setMaximumHeight(30)
        word_layout.addWidget(self.word_input)
        layout.addLayout(word_layout)

        # Controls
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Recording")
        self.start_btn.clicked.connect(self.start_recording)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop Recording")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        controls_layout.addWidget(self.process_btn)

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
        # Recording connections
        self.recorder.frame_ready.connect(self.update_frame)
        self.recorder.recording_saved.connect(self.on_recording_saved)
        self.recorder.error_occurred.connect(self.on_error)

        # Processing connections
        self.processor.gloss_detected.connect(self.on_gloss_detected)
        self.processor.processing_complete.connect(self.on_processing_complete)
        self.processor.error_occurred.connect(self.on_error)

        # UI update signals
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def start_recording(self):
        """Start recording"""
        word = self.word_input.toPlainText().strip()
        if not word:
            QMessageBox.warning(self, "No Word", "Please enter a word to record.")
            return

        # Clean word
        cleaned_word = "".join(
            c for c in word if c.isalnum() or c in ("-", "_")
        ).lower()
        if not cleaned_word:
            QMessageBox.warning(self, "Invalid Word", "Please enter a valid word.")
            return

        self.update_status_signal.emit("Recording...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.process_btn.setEnabled(False)

        self.recorder.start_recording(cleaned_word)

    def stop_recording(self):
        """Stop recording"""
        self.recorder.stop_recording()
        self.update_status_signal.emit("Saving...")
        self.stop_btn.setEnabled(False)

    def process_video(self):
        """Process recorded video"""
        if not self.recorded_video_path:
            return

        self.gloss_output.clear()
        self.update_status_signal.emit("Processing video...")
        self.process_btn.setEnabled(False)

        self.processor.process_video_file(self.recorded_video_path)

    def update_frame(self, frame):
        """Update video display"""
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

    def on_recording_saved(self, saved_paths):
        """Handle video saved"""
        import os

        self.recorded_video_path = saved_paths[0]  # Use original (not flipped)

        paths_str = "\n".join([f"  - {os.path.basename(p)}" for p in saved_paths])
        self.update_status_signal.emit(
            "Video saved! Click 'Process Video' to extract glosses"
        )

        QMessageBox.information(
            self,
            "Recording Saved",
            f"Video recordings saved:\n{paths_str}\n\nClick 'Process Video' to detect glosses",
        )

        self.start_btn.setEnabled(True)
        self.process_btn.setEnabled(True)

    def on_gloss_detected(self, gloss, confidence):
        """Handle gloss detection"""
        current_text = self.gloss_output.toPlainText()
        if current_text:
            self.update_gloss_signal.emit(f"{current_text} {gloss}")
        else:
            self.update_gloss_signal.emit(gloss)

        self.update_status_signal.emit(f"Detected: {gloss} ({confidence:.2f})")

    def on_processing_complete(self, glosses):
        """Handle processing complete"""
        self.update_status_signal.emit(f"Complete - {len(glosses)} glosses detected")
        self.start_btn.setEnabled(True)
        self.process_btn.setEnabled(True)

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"Error: {error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Record Video to Gloss")
        self.setGeometry(100, 100, 700, 750)

        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = RecordVideo2GlossWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
