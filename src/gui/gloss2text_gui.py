"""
Gloss to Text GUI
Enter glosses, get translated English text
"""

import sys
import threading
import torch
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
from PyQt6.QtCore import pyqtSignal

try:
    from ..gloss2audio import Gloss2Text
except ImportError:
    print("Import error: Ensure gloss2audio module is available")


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

        # Status
        self.status_label = QLabel("Loading...")
        layout.addWidget(self.status_label)

        # Input section
        layout.addWidget(QLabel("Enter Glosses (space-separated):"))
        
        input_layout = QHBoxLayout()
        self.gloss_input = QTextEdit()
        self.gloss_input.setPlaceholderText("Example: TOMORROW I STORE GO")
        self.gloss_input.setMaximumHeight(80)
        input_layout.addWidget(self.gloss_input)
        
        self.translate_btn = QPushButton("Translate")
        self.translate_btn.clicked.connect(self.translate_glosses)
        self.translate_btn.setEnabled(False)
        input_layout.addWidget(self.translate_btn)
        
        layout.addLayout(input_layout)

        # Output section
        layout.addWidget(QLabel("Translated Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        layout.addWidget(self.text_output)

        self.setLayout(layout)

    def setup_connections(self):
        self.update_text_signal.connect(self.text_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def load_translator(self):
        """Load the translation model"""
        thread = threading.Thread(target=self._load_translator)
        thread.daemon = True
        thread.start()

    def _load_translator(self):
        try:
            self.translator = Gloss2Text(self.device)
            self.update_status_signal.emit("Ready")
            self.translate_btn.setEnabled(True)
        except Exception as e:
            self.update_status_signal.emit(f"Error loading model: {str(e)}")

    def translate_glosses(self):
        """Translate glosses to text"""
        glosses_text = self.gloss_input.toPlainText().strip()
        if not glosses_text:
            self.update_status_signal.emit("Error: Enter glosses first")
            return

        glosses = glosses_text.split()
        self.update_status_signal.emit("Translating...")
        self.translate_btn.setEnabled(False)
        
        thread = threading.Thread(target=self._translate, args=(glosses,))
        thread.daemon = True
        thread.start()

    def _translate(self, glosses):
        """Background translation"""
        try:
            text = self.translator.infer(glosses)
            text_str = " ".join(text)
            
            self.update_text_signal.emit(text_str)
            self.update_status_signal.emit("Ready")
            self.translate_btn.setEnabled(True)
            
        except Exception as e:
            self.update_status_signal.emit(f"Error: {str(e)}")
            self.translate_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gloss to Text")
        self.setGeometry(100, 100, 600, 400)
        
        try:
            from .styles import get_dark_stylesheet
            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = Gloss2TextWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()