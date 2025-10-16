"""
Unified NSL Translation GUI - Simple Version
All pipelines in one interface with tabs
"""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
)

try:
    from .audio2gloss_gui import Audio2GlossWidget
    from .gloss2visualization_gui import Gloss2VisualizationWidget
    from .audio2visualization_gui import Audio2VisualizationWidget
    from .gloss2text_gui import Gloss2TextWidget
    from .video2text_unified_gui import Video2TextWidget
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure all GUI modules are available")


class UnifiedNSLWidget(QTabWidget):
    """Main unified widget with all pipelines"""

    def __init__(self):
        super().__init__()
        self.setup_tabs()

    def setup_tabs(self):
        """Setup all pipeline tabs"""

        # Pipeline 1: Audio → Glosses
        self.audio2gloss = Audio2GlossWidget()
        self.addTab(self.audio2gloss, "Audio → Glosses")

        # Pipeline 2: Glosses → Visualization
        self.gloss2viz = Gloss2VisualizationWidget()
        self.addTab(self.gloss2viz, "Glosses → Visualization")

        # Pipeline 3: Audio → Visualization
        self.audio2viz = Audio2VisualizationWidget()
        self.addTab(self.audio2viz, "Audio → Visualization")

        # Pipeline 4: Glosses → Text
        self.gloss2text = Gloss2TextWidget()
        self.addTab(self.gloss2text, "Glosses → Text")

        # Pipeline 5: Video → Text (Unified)
        self.video2text = Video2TextWidget()
        self.addTab(self.video2text, "Video → Text")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Namibian Sign Language - Unified Translation System")
        self.setGeometry(100, 100, 900, 800)

        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = UnifiedNSLWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
