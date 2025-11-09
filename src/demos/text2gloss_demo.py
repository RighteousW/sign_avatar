"""
Text2Gloss Demo
Text input -> Gloss output
"""

import sys
import threading
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
    from ..audio2gloss import AudioToGlossConverter
except ImportError:
    try:
        from .demo_utils import get_dark_stylesheet
        from ..audio2gloss import AudioToGlossConverter
    except ImportError:
        print("Import error - ensure modules are available")


class Text2GlossWidget(QWidget):
    """Main widget for text to gloss conversion"""

    update_gloss_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.converter = AudioToGlossConverter()
        self.setup_ui()
        self.setup_connections()
        self.load_converter()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Text to Gloss Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("Loading converter...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Input text
        layout.addWidget(QLabel("Input Text:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter English text here...")
        self.text_input.setMaximumHeight(150)
        layout.addWidget(self.text_input)

        # Convert button
        self.convert_btn = QPushButton("Convert to Glosses")
        self.convert_btn.clicked.connect(self.convert_text)
        self.convert_btn.setEnabled(False)
        layout.addWidget(self.convert_btn)

        # Output glosses
        layout.addWidget(QLabel("Generated Glosses:"))
        self.gloss_output = QTextEdit()
        self.gloss_output.setReadOnly(True)
        self.gloss_output.setPlaceholderText("Glosses will appear here...")
        layout.addWidget(self.gloss_output)

        self.setLayout(layout)

    def setup_connections(self):
        self.update_gloss_signal.connect(self.gloss_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def load_converter(self):
        thread = threading.Thread(target=self._load_converter)
        thread.daemon = True
        thread.start()

    def _load_converter(self):
        if self.converter.load_model():
            self.update_status_signal.emit("✓ Ready - Enter text and click Convert")
            self.convert_btn.setEnabled(True)
        else:
            self.update_status_signal.emit(
                "⚠ Error: Run 'python3 -m spacy download en_core_web_sm'"
            )

    def convert_text(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.update_status_signal.emit("⚠ Please enter some text")
            return

        self.update_status_signal.emit("Converting...")
        self.convert_btn.setEnabled(False)

        thread = threading.Thread(target=self._convert_text, args=(text,))
        thread.daemon = True
        thread.start()

    def _convert_text(self, text):
        try:
            clause_glosses = self.converter.text_to_glosses(text)

            # Flatten glosses
            glosses = []
            for clause in clause_glosses:
                glosses.extend(clause)

            gloss_str = " ".join(glosses)
            self.update_gloss_signal.emit(gloss_str)
            self.update_status_signal.emit(
                f"✓ Conversion complete ({len(glosses)} glosses)"
            )

        except Exception as e:
            self.update_gloss_signal.emit(f"[Error: {str(e)}]")
            self.update_status_signal.emit(f"⚠ Error: {str(e)}")

        self.convert_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text to Gloss Demo")
        self.setGeometry(100, 100, 700, 600)
        self.setStyleSheet(get_dark_stylesheet())

        widget = Text2GlossWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
