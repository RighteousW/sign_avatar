"""
Audio2Text Demo
Live audio recording (spacebar) -> Text transcription
"""

import sys
import threading
import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QKeyEvent

try:
    from .demo_utils import get_dark_stylesheet
    from ..audio2gloss import AudioToGlossConverter
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


class Audio2TextWidget(QWidget):
    """Main widget for audio to text conversion"""

    update_text_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.audio_converter = (
            AudioToGlossConverter()
        )
        self.is_recording = False

        self.setup_ui()
        self.setup_connections()
        self.load_converter()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Audio to Text Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Instructions
        instructions = QLabel(
            "Hold SPACE to record audio, release to stop and transcribe"
        )
        instructions.setStyleSheet("font-size: 12px; color: #888; padding: 5px;")
        layout.addWidget(instructions)

        # Recording button (for tab compatibility)
        self.record_btn = QPushButton("Start Recording (or hold SPACE)")
        self.record_btn.pressed.connect(self.start_recording)
        self.record_btn.released.connect(self.stop_recording)
        layout.addWidget(self.record_btn)

        # Transcribed text output
        layout.addWidget(QLabel("Transcribed Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Transcribed speech will appear here...")
        layout.addWidget(self.text_output)

        self.setLayout(layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setup_connections(self):
        self.recorder.recording_complete.connect(self.on_recording_complete)
        self.recorder.error_occurred.connect(self.on_error)
        self.update_text_signal.connect(self.text_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def load_converter(self):
        """Load the audio converter model"""
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
        self.update_status_signal.emit("🔴 Recording... (release SPACE to stop)")
        self.recorder.start_recording()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.recorder.stop_recording()
            self.update_status_signal.emit("Processing audio...")

    def on_recording_complete(self, audio_array, sample_rate):
        thread = threading.Thread(
            target=self.transcribe_audio, args=(audio_array, sample_rate)
        )
        thread.daemon = True
        thread.start()

    def transcribe_audio(self, audio_array, sample_rate):
        try:
            # Clean audio data
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Use AudioToGlossConverter's robust transcription
            text, clause_glosses = self.audio_converter.numpy_to_glosses(
                audio_array, sample_rate
            )

            self.update_text_signal.emit(text)
            self.update_status_signal.emit(
                "✓ Transcription complete - Press SPACE to record again"
            )

        except Exception as e:
            error_msg = str(e)
            if "Could not understand" in error_msg or "UnknownValueError" in error_msg:
                self.update_text_signal.emit("[Could not understand audio]")
                self.update_status_signal.emit(
                    "⚠ Could not understand audio - Try again"
                )
            elif "RequestError" in error_msg:
                self.update_text_signal.emit(f"[API Error: {error_msg}]")
                self.update_status_signal.emit("⚠ Recognition service error")
            else:
                self.update_text_signal.emit(f"[Error: {error_msg}]")
                self.update_status_signal.emit(f"⚠ Error: {error_msg}")

    def on_error(self, error_msg):
        self.update_status_signal.emit(f"⚠ {error_msg}")

    def showEvent(self, event):
        """Called when tab becomes visible - ensure we have focus for keyboard events"""
        super().showEvent(event)
        self.setFocus()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio to Text Demo")
        self.setGeometry(100, 100, 700, 500)
        self.setStyleSheet(get_dark_stylesheet())

        widget = Audio2TextWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
