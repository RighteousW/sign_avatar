"""
Audio to Visualization GUI
Combines existing Audio2Gloss + Gloss2Visualization widgets
Press SPACE to record, see animated sign language
"""

import sys
import tempfile
import threading
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal

try:
    from .audio2gloss_gui import Audio2GlossWidget
    from .gloss2visualization_gui import LandmarkCanvas
    from ..gloss2visualization import GestureTransitionGenerator
    from ..constants import REPRESENTATIVES_LEFT
except ImportError:
    print("Import error: Ensure required modules are available")


class Audio2VisualizationWidget(QWidget):
    """Combines audio2gloss and visualization widgets"""

    visualize_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.generator = GestureTransitionGenerator(REPRESENTATIVES_LEFT)
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Visualization canvas (top 60%)
        self.canvas = LandmarkCanvas()
        layout.addWidget(self.canvas, 60)

        # Audio2Gloss widget (bottom 40%)
        self.audio2gloss = Audio2GlossWidget()
        layout.addWidget(self.audio2gloss, 40)

        self.setLayout(layout)

    def setup_connections(self):
        # When glosses are updated, generate visualization
        self.audio2gloss.update_gloss_signal.connect(self.on_glosses_updated)
        self.visualize_signal.connect(self.canvas.load_landmarks)

    def on_glosses_updated(self, glosses_text):
        """Generate visualization when new glosses arrive"""
        glosses = glosses_text.split()
        if not glosses or glosses_text == "(No glosses generated)":
            return

        self.canvas.stop_animation()
        self.audio2gloss.update_status_signal.emit("Generating visualization...")

        thread = threading.Thread(target=self._generate, args=(glosses,))
        thread.daemon = True
        thread.start()

    def _generate(self, glosses):
        try:
            landmark_file = tempfile.mktemp(suffix=".pkl")
            glosses_lower = [g.lower() for g in glosses]
            result = self.generator.generate_sequence(glosses_lower, 4, landmark_file)

            if result["denied_glosses"]:
                self.audio2gloss.update_status_signal.emit(
                    f"Warning: Not found: {', '.join(result['denied_glosses'])}"
                )
            else:
                self.audio2gloss.update_status_signal.emit("Playing")

            self.visualize_signal.emit(landmark_file)

        except Exception as e:
            self.audio2gloss.update_status_signal.emit(f"Error: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio to Visualization")
        self.setGeometry(100, 100, 700, 700)

        try:
            from .styles import get_dark_stylesheet

            self.setStyleSheet(get_dark_stylesheet())
        except ImportError:
            pass

        widget = Audio2VisualizationWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
