"""
GUI Styles Module
Centralized styling for all GUI components with responsive design
"""


def get_dark_stylesheet():
    """Returns modern dark theme stylesheet with responsive design"""
    return """
        /* Main containers - no fixed sizes */
        QMainWindow, QWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Labels - flexible sizing */
        QLabel {
            color: #e0e0e0;
            padding: 2px;
            font-size: 10pt;
        }
        
        /* Buttons - flexible with min sizes */
        QPushButton {
            background-color: #0d7377;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 9pt;
            min-height: 25px;
            min-width: 60px;
        }
        QPushButton:hover {
            background-color: #14a085;
        }
        QPushButton:pressed {
            background-color: #0a5d61;
        }
        QPushButton:disabled {
            background-color: #2d2d2d;
            color: #666;
        }
        
        /* Text Edit - responsive */
        QTextEdit {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            padding: 5px;
            font-size: 9pt;
            selection-background-color: #0d7377;
        }
        QTextEdit:focus {
            border: 1px solid #0d7377;
        }
        
        /* Combo Box - flexible */
        QComboBox {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #3d3d3d;
            padding: 5px 8px;
            border-radius: 4px;
            min-height: 25px;
        }
        QComboBox:hover {
            border: 1px solid #0d7377;
        }
        QComboBox::drop-down {
            border: none;
            width: 25px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #e0e0e0;
            margin-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #2b2b2b;
            color: #e0e0e0;
            selection-background-color: #0d7377;
            border: 1px solid #3d3d3d;
            padding: 2px;
        }
        
        /* Check Box - compact */
        QCheckBox {
            color: #e0e0e0;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #3d3d3d;
            border-radius: 3px;
            background-color: #2b2b2b;
        }
        QCheckBox::indicator:hover {
            border: 1px solid #0d7377;
        }
        QCheckBox::indicator:checked {
            background-color: #0d7377;
            border: 1px solid #0d7377;
        }
        
        /* Group Box - responsive */
        QGroupBox {
            border: 1px solid #3d3d3d;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #14a085;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            left: 8px;
        }
        
        /* Tab Widget - flexible */
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            border-radius: 4px;
            background-color: #1e1e1e;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #2b2b2b;
            color: #e0e0e0;
            padding: 8px 15px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 80px;
        }
        QTabBar::tab:selected {
            background-color: #0d7377;
            color: white;
        }
        QTabBar::tab:hover:!selected {
            background-color: #3d3d3d;
        }
        
        /* Scroll Bar - compact */
        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 10px;
            border-radius: 5px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #0d7377;
            border-radius: 5px;
            min-height: 20px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #14a085;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        
        QScrollBar:horizontal {
            background-color: #2b2b2b;
            height: 10px;
            border-radius: 5px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #0d7377;
            border-radius: 5px;
            min-width: 20px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #14a085;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* Status indicators */
        QLabel[status="recording"] {
            color: #ff4444;
            font-weight: bold;
        }
        QLabel[status="processing"] {
            color: #ffaa00;
            font-weight: bold;
        }
        QLabel[status="ready"] {
            color: #00ff88;
            font-weight: bold;
        }
    """


def get_light_stylesheet():
    """Returns modern light theme stylesheet with responsive design"""
    return """
        /* Main containers - no fixed sizes */
        QMainWindow, QWidget {
            background-color: #f5f5f5;
            color: #2d2d2d;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Labels - flexible sizing */
        QLabel {
            color: #2d2d2d;
            padding: 2px;
            font-size: 10pt;
        }
        
        /* Buttons - flexible with min sizes */
        QPushButton {
            background-color: #0d7377;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 9pt;
            min-height: 25px;
            min-width: 60px;
        }
        QPushButton:hover {
            background-color: #14a085;
        }
        QPushButton:pressed {
            background-color: #0a5d61;
        }
        QPushButton:disabled {
            background-color: #d0d0d0;
            color: #888;
        }
        
        /* Text Edit - responsive */
        QTextEdit {
            background-color: white;
            color: #2d2d2d;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            padding: 5px;
            font-size: 9pt;
            selection-background-color: #0d7377;
        }
        QTextEdit:focus {
            border: 1px solid #0d7377;
        }
        
        /* Combo Box - flexible */
        QComboBox {
            background-color: white;
            color: #2d2d2d;
            border: 1px solid #d0d0d0;
            padding: 5px 8px;
            border-radius: 4px;
            min-height: 25px;
        }
        QComboBox:hover {
            border: 1px solid #0d7377;
        }
        QComboBox::drop-down {
            border: none;
            width: 25px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #2d2d2d;
            margin-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: white;
            color: #2d2d2d;
            selection-background-color: #0d7377;
            border: 1px solid #d0d0d0;
            padding: 2px;
        }
        
        /* Check Box - compact */
        QCheckBox {
            color: #2d2d2d;
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #d0d0d0;
            border-radius: 3px;
            background-color: white;
        }
        QCheckBox::indicator:hover {
            border: 1px solid #0d7377;
        }
        QCheckBox::indicator:checked {
            background-color: #0d7377;
            border: 1px solid #0d7377;
        }
        
        /* Group Box - responsive */
        QGroupBox {
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #0d7377;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            left: 8px;
        }
        
        /* Tab Widget - flexible */
        QTabWidget::pane {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            background-color: #f5f5f5;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            color: #2d2d2d;
            padding: 8px 15px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 80px;
        }
        QTabBar::tab:selected {
            background-color: #0d7377;
            color: white;
        }
        QTabBar::tab:hover:!selected {
            background-color: #d0d0d0;
        }
        
        /* Scroll Bar - compact */
        QScrollBar:vertical {
            background-color: #e0e0e0;
            width: 10px;
            border-radius: 5px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #0d7377;
            border-radius: 5px;
            min-height: 20px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #14a085;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        
        QScrollBar:horizontal {
            background-color: #e0e0e0;
            height: 10px;
            border-radius: 5px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #0d7377;
            border-radius: 5px;
            min-width: 20px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #14a085;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* Status indicators */
        QLabel[status="recording"] {
            color: #cc0000;
            font-weight: bold;
        }
        QLabel[status="processing"] {
            color: #ff8800;
            font-weight: bold;
        }
        QLabel[status="ready"] {
            color: #00aa44;
            font-weight: bold;
        }
    """


def get_theme_colors():
    """Returns color palette for programmatic use"""
    return {
        "dark": {
            "primary": "#0d7377",
            "primary_hover": "#14a085",
            "primary_pressed": "#0a5d61",
            "background": "#1e1e1e",
            "surface": "#2b2b2b",
            "border": "#3d3d3d",
            "text": "#e0e0e0",
            "text_secondary": "#999",
            "error": "#ff4444",
            "warning": "#ffaa00",
            "success": "#00ff88",
        },
        "light": {
            "primary": "#0d7377",
            "primary_hover": "#14a085",
            "primary_pressed": "#0a5d61",
            "background": "#f5f5f5",
            "surface": "#ffffff",
            "border": "#d0d0d0",
            "text": "#2d2d2d",
            "text_secondary": "#666",
            "error": "#cc0000",
            "warning": "#ff8800",
            "success": "#00aa44",
        },
    }


def apply_responsive_sizing(widget, is_main_window=False):
    """
    Apply responsive sizing policies to widgets

    Args:
        widget: The QWidget to apply sizing to
        is_main_window: Whether this is the main window (default: False)
    """
    from PyQt6.QtWidgets import QSizePolicy

    if is_main_window:
        # Main window should be resizable
        widget.setMinimumSize(800, 600)
        widget.resize(1024, 768)
    else:
        # Other widgets should expand to fill available space
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
