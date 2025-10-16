"""
Audio to Glosses GUI - Simple Version
Press SPACE to record, release to convert to glosses
"""

import sys
import threading
import numpy as np
import speech_recognition as sr
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QKeyEvent

try:
    from ..audio2gloss import AudioToGlossConverter
except ImportError:
    print("Import error: Ensure audio2gloss module is available")


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


class Audio2GlossWidget(QWidget):
    """Main widget"""

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
        layout = QVBoxLayout()

        # Status
        self.status_label = QLabel("Loading...")
        layout.addWidget(self.status_label)

        # Speech output
        layout.addWidget(QLabel("Speech:"))
        self.speech_output = QTextEdit()
        self.speech_output.setReadOnly(True)
        layout.addWidget(self.speech_output)

        # Gloss output
        layout.addWidget(QLabel("Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        layout.addWidget(self.gloss_output)

        # Instructions
        instructions = QLabel("Press SPACE to record, release to stop")
        layout.addWidget(instructions)

        self.setLayout(layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setup_connections(self):
        self.recorder.recording_complete.connect(self.on_recording_complete)
        self.recorder.error_occurred.connect(self.on_error)
        self.update_text_signal.connect(self.speech_output.setText)
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def load_converter(self):
        thread = threading.Thread(target=self._load_converter)
        thread.daemon = True
        thread.start()

    def _load_converter(self):
        if self.converter.load_model():
            self.update_status_signal.emit("Ready")
        else:
            self.update_status_signal.emit(
                "Error: Run 'python -m spacy download en_core_web_sm'"
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
            text, clause_glosses = self.converter.numpy_to_glosses(
                audio_array, sample_rate
            )

            glosses = []
            for clause in clause_glosses:
                glosses.extend(clause)

            self.update_text_signal.emit(text)
            self.update_gloss_signal.emit(" ".join(glosses))
            self.update_status_signal.emit("Ready")

        except sr.UnknownValueError:
            self.update_status_signal.emit("Could not understand audio")
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")

    def on_error(self, error_msg):
        self.update_status_signal.emit(error_msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio to Glosses")
        self.setGeometry(100, 100, 600, 400)

        # Load stylesheet
        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass  # Use default style if styles module not found

        widget = Audio2GlossWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
