"""
Video to Text Unified GUI - Simple Version
Combines all 3 video2gloss modes + gloss2text
"""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QComboBox,
    QTabWidget,
)
from PyQt6.QtCore import pyqtSignal
import torch

try:
    from .video2gloss_file_gui import VideoFile2GlossWidget
    from .video2gloss_webcam_gui import Webcam2GlossWidget
    from .video2gloss_record_gui import RecordVideo2GlossWidget
    from ..gloss2audio import Gloss2Text
except ImportError:
    print("Import error: Ensure required modules are available")


class Video2TextWidget(QWidget):
    """Main unified widget"""

    update_text_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = Gloss2Text(self.device)

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))

        self.mode_tabs = QTabWidget()

        # Tab 1: File
        self.file_widget = VideoFile2GlossWidget()
        self.mode_tabs.addTab(self.file_widget, "Select File")

        # Tab 2: Webcam
        self.webcam_widget = Webcam2GlossWidget()
        self.mode_tabs.addTab(self.webcam_widget, "Webcam Live")

        # Tab 3: Record
        self.record_widget = RecordVideo2GlossWidget()
        self.mode_tabs.addTab(self.record_widget, "Record Video")

        layout.addWidget(self.mode_tabs, 70)

        # Translation section
        translation_layout = QVBoxLayout()

        # Status and translate button
        controls_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)

        self.translate_btn = QPushButton("Translate to Text")
        self.translate_btn.clicked.connect(self.translate_glosses)
        controls_layout.addWidget(self.translate_btn)
        controls_layout.addStretch()

        translation_layout.addLayout(controls_layout)

        # Translated text output
        translation_layout.addWidget(QLabel("Translated Text:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(100)
        translation_layout.addWidget(self.text_output)

        layout.addLayout(translation_layout, 30)

        self.setLayout(layout)

    def setup_connections(self):
        self.update_text_signal.connect(self.text_output.setText)
        self.update_status_signal.connect(self.status_label.setText)

    def translate_glosses(self):
        """Translate glosses from current tab"""
        # Get glosses from active tab
        current_widget = self.mode_tabs.currentWidget()
        glosses_text = current_widget.gloss_output.toPlainText().strip()

        if not glosses_text:
            self.update_status_signal.emit("Error: No glosses to translate")
            return

        glosses = glosses_text.split()
        self.update_status_signal.emit("Translating...")
        self.translate_btn.setEnabled(False)

        import threading

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
        self.setWindowTitle("Video to Text - Unified")
        self.setGeometry(100, 100, 800, 800)

        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = Video2TextWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
