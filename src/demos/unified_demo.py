"""
Unified Sign Language Translation Demo
All-in-one interface for sign language processing pipeline
"""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTabWidget,
)
from PyQt6.QtCore import pyqtSignal, Qt

try:
    from demo_utils import get_dark_stylesheet
    from .audio2text_demo import Audio2TextWidget
    from .text2gloss_demo import Text2GlossWidget
    from .gloss2visualization_demo import Gloss2VisualizationWidget
    from .video2gloss_webcam_demo import Webcam2GlossWidget
    from .gloss2text_demo import Gloss2TextWidget
    from .text2speech_demo import Text2SpeechWidget
except ImportError:
    try:
        from .demo_utils import get_dark_stylesheet
        from .audio2text_demo import Audio2TextWidget
        from .text2gloss_demo import Text2GlossWidget
        from .gloss2visualization_demo import Gloss2VisualizationWidget
        from .video2gloss_webcam_demo import Webcam2GlossWidget
        from .gloss2text_demo import Gloss2TextWidget
        from .text2speech_demo import Text2SpeechWidget
    except ImportError:
        print("Import error - ensure modules are available")


class Video2GlossTabWidget(QWidget):
    """Video2Gloss tab with live webcam"""

    glosses_detected_signal = pyqtSignal(
        list
    )  # Signal to share glosses with other tabs

    def __init__(self):
        super().__init__()
        self.current_widget = None
        self.detected_glosses = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Directly add webcam widget
        self.current_widget = Webcam2GlossWidget()

        # Connect gloss detection signal
        if hasattr(self.current_widget.processor, "gloss_detected"):
            self.current_widget.processor.gloss_detected.connect(self.on_gloss_detected)

        layout.addWidget(self.current_widget)
        self.setLayout(layout)

    def on_gloss_detected(self, gloss, confidence):
        """Collect detected glosses to share with other tabs"""
        if gloss not in self.detected_glosses:
            self.detected_glosses.append(gloss)
            self.glosses_detected_signal.emit(self.detected_glosses)


class UnifiedDemoWidget(QWidget):
    """Main unified demo widget with tabs"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Sign Language Translation System")
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Tab widget
        self.tabs = QTabWidget()

        # Tab 1: Speech2Text (Audio2Text)
        self.audio2text_tab = Audio2TextWidget()
        self.tabs.addTab(self.audio2text_tab, "Speech → Text")

        # Tab 2: Text2Gloss
        self.text2gloss_tab = Text2GlossWidget()
        self.tabs.addTab(self.text2gloss_tab, "Text → Gloss")

        # Tab 3: Gloss2Visualization
        self.gloss2vis_tab = Gloss2VisualizationWidget()
        self.tabs.addTab(self.gloss2vis_tab, "Gloss → Visualization")

        # Tab 4: Video2Gloss (with sub-modes)
        self.video2gloss_tab = Video2GlossTabWidget()
        self.tabs.addTab(self.video2gloss_tab, "Video → Gloss")

        # Tab 5: Gloss2Text
        self.gloss2text_tab = Gloss2TextWidget()
        self.tabs.addTab(self.gloss2text_tab, "Gloss → Text")

        # Tab 6: Text2Speech
        self.text2speech_tab = Text2SpeechWidget()
        self.tabs.addTab(self.text2speech_tab, "Text → Speech")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def setup_connections(self):
        """Setup cross-tab communication for pipeline"""

        # Audio2Text → Text2Gloss
        if hasattr(self.audio2text_tab, "update_text_signal"):
            self.audio2text_tab.update_text_signal.connect(
                lambda text: self.text2gloss_tab.text_input.setPlainText(text)
            )

        # Text2Gloss → Gloss2Visualization
        if hasattr(self.text2gloss_tab, "update_gloss_signal"):
            self.text2gloss_tab.update_gloss_signal.connect(
                lambda glosses: self.gloss2vis_tab.gloss_input.setText(glosses)
            )

        # Text2Gloss → Gloss2Text (for verification)
        if hasattr(self.text2gloss_tab, "update_gloss_signal"):
            self.text2gloss_tab.update_gloss_signal.connect(
                lambda glosses: self.gloss2text_tab.gloss_input.setText(glosses)
            )

        # Video2Gloss → Gloss2Text
        self.video2gloss_tab.glosses_detected_signal.connect(
            lambda glosses: self.gloss2text_tab.gloss_input.setText(" ".join(glosses))
        )

        # Video2Gloss → Gloss2Visualization
        self.video2gloss_tab.glosses_detected_signal.connect(
            lambda glosses: self.gloss2vis_tab.gloss_input.setText(" ".join(glosses))
        )

        # Gloss2Text → Text2Speech
        if hasattr(self.gloss2text_tab, "update_text_signal"):
            self.gloss2text_tab.update_text_signal.connect(
                lambda text: self.text2speech_tab.text_input.setPlainText(text)
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Sign Language Translation Demo")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet(get_dark_stylesheet())

        widget = UnifiedDemoWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
