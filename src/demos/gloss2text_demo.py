"""
Gloss2Text Demo
Gloss input -> Text translation
"""

import sys
import threading
import torch
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
)
from PyQt6.QtCore import pyqtSignal

try:
    from demo_utils import get_dark_stylesheet
    from ..gloss2audio import Gloss2Text
except ImportError:
    try:
        from .demo_utils import get_dark_stylesheet
        from ..gloss2audio import Gloss2Text
    except ImportError:
        print("Import error - ensure modules are available")


class Gloss2TextWidget(QWidget):
    """Main widget for gloss to text translation"""

    update_text_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = None

        self.setup_ui()
        self.setup_connections()
        self.load_translator()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Gloss to Text Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("Loading translator...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Input glosses
        layout.addWidget(QLabel("Input Glosses (space-separated):"))
        self.gloss_input = QLineEdit()
        self.gloss_input.setPlaceholderText("e.g., hello my name john")
        layout.addWidget(self.gloss_input)

        # Translate button
        self.translate_btn = QPushButton("Translate to Text")
        self.translate_btn.clicked.connect(self.translate_glosses)
        self.translate_btn.setEnabled(False)
        layout.addWidget(self.translate_btn)

        # Output text
        layout.addWidget(QLabel("Translated Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Translation will appear here...")
        layout.addWidget(self.text_output)

        # Instructions
        instructions = QLabel(
            "Enter glosses in uppercase or lowercase, separated by spaces."
        )
        instructions.setStyleSheet("font-size: 11px; color: #888; padding: 5px;")
        layout.addWidget(instructions)

        self.setLayout(layout)

    def setup_connections(self):
        self.update_text_signal.connect(self.text_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def load_translator(self):
        thread = threading.Thread(target=self._load_translator)
        thread.daemon = True
        thread.start()

    def _load_translator(self):
        try:
            self.translator = Gloss2Text(self.device)
            self.update_status_signal.emit(
                "✓ Ready - Enter glosses and click Translate"
            )
            self.translate_btn.setEnabled(True)
        except Exception as e:
            self.update_status_signal.emit(f"⚠ Error loading translator: {str(e)}")

    def translate_glosses(self):
        gloss_text = self.gloss_input.text().strip()
        if not gloss_text:
            self.update_status_signal.emit("⚠ Please enter some glosses")
            return

        glosses = gloss_text.split()
        self.update_status_signal.emit(f"Translating {len(glosses)} glosses...")
        self.translate_btn.setEnabled(False)

        thread = threading.Thread(target=self._translate_glosses, args=(glosses,))
        thread.daemon = True
        thread.start()

    def _translate_glosses(self, glosses):
        try:
            text = self.translator.infer(glosses)
            text_str = " ".join(text).replace("_", " ")

            self.update_text_signal.emit(text_str)
            self.update_status_signal.emit("✓ Translation complete")

        except Exception as e:
            self.update_text_signal.emit(f"[Error: {str(e)}]")
            self.update_status_signal.emit(f"⚠ Translation error: {str(e)}")

        self.translate_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gloss to Text Demo")
        self.setGeometry(100, 100, 700, 500)
        self.setStyleSheet(get_dark_stylesheet())

        widget = Gloss2TextWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
