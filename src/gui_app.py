"""
Minimal Audio to Glosses GUI
Space bar to record/stop
"""

import sys
import threading
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QKeyEvent
import speech_recognition as sr
import sounddevice as sd
import numpy as np

from audio_to_gloss import AudioToGlossConverter


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

    def __init__(self):
        super().__init__()

        self.recorder = AudioRecorder()
        self.converter = AudioToGlossConverter()
        self.is_recording = False

        self.setup_ui()
        self.setup_connections()
        self.load_converter()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top 80% - Status overlay
        self.status_widget = QWidget()
        self.status_widget.setMinimumHeight(480)
        status_layout = QVBoxLayout(self.status_widget)

        self.status_label = QLabel("Loading...")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
        )
        self.status_label.setStyleSheet("padding: 10px; font-size: 14px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        main_layout.addWidget(self.status_widget, 80)

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

    def load_converter(self):
        thread = threading.Thread(target=self._load_converter)
        thread.daemon = True
        thread.start()

    def _load_converter(self):
        if self.converter.load_model():
            self.update_status_signal.emit(
                "Press SPACE to record, release SPACE to stop recording and get glosses"
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
            text, glosses = self.converter.numpy_to_glosses(audio_array, sample_rate)

            self.update_text_signal.emit(text)
            self.update_gloss_signal.emit(" ".join(glosses))
            self.update_status_signal.emit(
                "Press SPACE to record, release SPACE to stop recording and get glosses"
            )

        except sr.UnknownValueError:
            self.update_status_signal.emit("Could not understand audio")
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")

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
