"""
Gloss2Visualization Demo
Gloss input -> Avatar visualization
"""

import sys
import os
import tempfile
import threading
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)
from PyQt6.QtCore import pyqtSignal

try:
    from demo_utils import get_dark_stylesheet, LandmarkCanvas
    from ..gloss2visualization import GestureTransitionGenerator
    from ..constants import REPRESENTATIVES_MANUAL
except ImportError:
    try:
        from .demo_utils import get_dark_stylesheet, LandmarkCanvas
        from ..gloss2visualization import GestureTransitionGenerator
        from ..constants import REPRESENTATIVES_MANUAL
    except ImportError:
        print("Import error - ensure modules are available")


class Gloss2VisualizationWidget(QWidget):
    """Main widget for gloss to visualization conversion"""

    visualize_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.generator = GestureTransitionGenerator(REPRESENTATIVES_MANUAL)
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Gloss to Visualization Demo")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Status
        self.status_label = QLabel("✓ Ready - Enter glosses to visualize")
        self.status_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Visualization canvas
        self.canvas = LandmarkCanvas()
        layout.addWidget(self.canvas, 60)

        # Input glosses
        layout.addWidget(QLabel("Input Glosses (space-separated):"))
        self.gloss_input = QLineEdit()
        self.gloss_input.setPlaceholderText("e.g., hello my name john")
        layout.addWidget(self.gloss_input)

        # Visualize button
        self.visualize_btn = QPushButton("Generate Visualization")
        self.visualize_btn.clicked.connect(self.generate_visualization)
        layout.addWidget(self.visualize_btn)

        # Instructions
        instructions = QLabel(
            "Enter glosses separated by spaces. Available glosses depend on your trained models."
        )
        instructions.setStyleSheet("font-size: 11px; color: #888; padding: 5px;")
        layout.addWidget(instructions)

        self.setLayout(layout)

    def setup_connections(self):
        self.visualize_signal.connect(self.canvas.load_landmarks)
        self.update_status_signal.connect(self.status_label.setText)

    def generate_visualization(self):
        gloss_text = self.gloss_input.text().strip()
        if not gloss_text:
            self.update_status_signal.emit("⚠ Please enter some glosses")
            return

        glosses = gloss_text.lower().split()
        self.update_status_signal.emit(
            f"Generating visualization for {len(glosses)} glosses..."
        )
        self.visualize_btn.setEnabled(False)
        self.canvas.stop_animation()

        thread = threading.Thread(target=self._generate_visualization, args=(glosses,))
        thread.daemon = True
        thread.start()

    def _generate_visualization(self, glosses):
        try:
            fd, landmark_file = tempfile.mkstemp(suffix=".pkl")
            os.close(fd)

            result = self.generator.generate_sequence(glosses, 10, landmark_file)

            if result["denied_glosses"]:
                warning = f"⚠ Not found: {', '.join(result['denied_glosses'])}"
                self.update_status_signal.emit(warning)
            else:
                self.update_status_signal.emit("✓ Playing visualization")

            self.visualize_signal.emit(landmark_file)

        except Exception as e:
            self.update_status_signal.emit(f"⚠ Visualization error: {str(e)}")

        self.visualize_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gloss to Visualization Demo")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet(get_dark_stylesheet())

        widget = Gloss2VisualizationWidget()
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
