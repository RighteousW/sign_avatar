"""
Text2Speech Demo
Text input -> Audio generation and playback
"""

import sys
import os
import threading
import tempfile
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
)
from PyQt6.QtCore import pyqtSignal

try:
    from demo_utils import get_dark_stylesheet
except ImportError:
    from .demo_utils import get_dark_stylesheet


class Text2SpeechWidget(QWidget):
    """Main widget for text to speech conversion"""

    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Text to Speech Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("✓ Ready - Enter text and click Play")
        self.status_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Input text
        layout.addWidget(QLabel("Input Text:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to convert to speech...")
        layout.addWidget(self.text_input)

        # Play button
        self.play_btn = QPushButton("Play Audio")
        self.play_btn.clicked.connect(self.play_audio)
        layout.addWidget(self.play_btn)

        # Instructions
        instructions = QLabel(
            "Text will be converted to speech using Google TTS. "
        )
        instructions.setStyleSheet("font-size: 11px; color: #888; padding: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self.setLayout(layout)

    def setup_connections(self):
        self.update_status_signal.connect(self.status_label.setText)

    def play_audio(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.update_status_signal.emit("⚠ Please enter some text")
            return

        self.update_status_signal.emit("Generating audio...")
        self.play_btn.setEnabled(False)

        thread = threading.Thread(target=self._play_audio, args=(text,))
        thread.daemon = True
        thread.start()

    def _play_audio(self, text):
        try:
            from gtts import gTTS

            # Generate audio
            tts = gTTS(text=text, lang="en")
            fd, temp_audio = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            tts.save(temp_audio)

            self.update_status_signal.emit("🔊 Playing audio...")

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

            self.update_status_signal.emit("✓ Audio playback complete")

        except ImportError:
            self.update_status_signal.emit("⚠ Error: Install gtts (pip install gtts)")
        except Exception as e:
            self.update_status_signal.emit(f"⚠ Audio error: {str(e)}")

        self.play_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text to Speech Demo")
        self.setGeometry(100, 100, 700, 500)
        self.setStyleSheet(get_dark_stylesheet())

        widget = Text2SpeechWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
