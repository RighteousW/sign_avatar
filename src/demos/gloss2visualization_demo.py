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
    QHBoxLayout,
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
        self.current_landmark_file = None
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
        layout.addWidget(self.canvas, 50)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.play_pause_btn = QPushButton("⏸ Pause")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setEnabled(False)
        controls_layout.addWidget(self.play_pause_btn)

        self.restart_btn = QPushButton("⏮ Restart")
        self.restart_btn.clicked.connect(self.restart_animation)
        self.restart_btn.setEnabled(False)
        controls_layout.addWidget(self.restart_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Input glosses
        layout.addWidget(QLabel("Input Glosses (space-separated):"))
        self.gloss_input = QLineEdit()
        self.gloss_input.setPlaceholderText("e.g., SICK WOMAN VOMIT")
        layout.addWidget(self.gloss_input)

        # Visualize button
        self.visualize_btn = QPushButton("Generate Visualization")
        self.visualize_btn.clicked.connect(self.generate_visualization)
        layout.addWidget(self.visualize_btn)

        # Instructions
        instructions = QLabel(
            "Enter glosses in uppercase separated by spaces. Available glosses depend on your trained models."
        )
        instructions.setStyleSheet("font-size: 11px; color: #888; padding: 5px;")
        layout.addWidget(instructions)

        self.setLayout(layout)

    def setup_connections(self):
        self.visualize_signal.connect(self.on_visualization_ready)
        self.update_status_signal.connect(self.status_label.setText)

    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if self.canvas.is_animating:
            self.canvas.pause_animation()
            self.play_pause_btn.setText("▶ Play")
            self.update_status_signal.emit("⏸ Paused")
        else:
            self.canvas.resume_animation()
            self.play_pause_btn.setText("⏸ Pause")
            self.update_status_signal.emit("▶ Playing")

    def restart_animation(self):
        """Restart animation from the beginning"""
        if self.current_landmark_file and os.path.exists(self.current_landmark_file):
            self.canvas.load_landmarks(self.current_landmark_file)
            self.play_pause_btn.setText("⏸ Pause")
            self.update_status_signal.emit("⏮ Restarted - Playing")

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
        self.play_pause_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
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

            self.current_landmark_file = landmark_file
            self.visualize_signal.emit(landmark_file)

        except Exception as e:
            self.update_status_signal.emit(f"⚠ Visualization error: {str(e)}")

        self.visualize_btn.setEnabled(True)

    def on_visualization_ready(self, landmark_file):
        """Called when visualization is ready to play"""
        self.canvas.load_landmarks(landmark_file)
        self.play_pause_btn.setEnabled(True)
        self.play_pause_btn.setText("⏸ Pause")
        self.restart_btn.setEnabled(True)

    def showEvent(self, event):
        """Called when tab becomes visible - resume animation if paused"""
        super().showEvent(event)
        if self.canvas.is_animating:
            # Already playing, do nothing
            pass
        elif (
            self.current_landmark_file
            and hasattr(self.canvas, "landmarks")
            and self.canvas.landmarks
        ):
            # Was paused, resume
            self.canvas.resume_animation()
            self.play_pause_btn.setText("⏸ Pause")

    def hideEvent(self, event):
        """Called when tab is hidden - pause animation"""
        super().hideEvent(event)
        if self.canvas.is_animating:
            self.canvas.pause_animation()
            self.play_pause_btn.setText("▶ Play")


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
