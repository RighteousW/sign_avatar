"""
GUI Styles Module
Centralized styling for all GUI components
"""


def get_dark_stylesheet():
    """Returns modern dark theme stylesheet"""
    return """
        /* Main containers */
        QMainWindow, QWidget {
            background-color: #1e1e1e;
            color: #e0e0e0;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Labels */
        QLabel {
            color: #e0e0e0;
            padding: 5px;
            font-size: 11pt;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #0d7377;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 10pt;
            min-height: 30px;
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
        
        /* Text Edit */
        QTextEdit {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 2px solid #3d3d3d;
            border-radius: 6px;
            padding: 8px;
            font-size: 10pt;
            selection-background-color: #0d7377;
        }
        QTextEdit:focus {
            border: 2px solid #0d7377;
        }
        
        /* Combo Box */
        QComboBox {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 2px solid #3d3d3d;
            padding: 8px;
            border-radius: 6px;
            min-height: 30px;
        }
        QComboBox:hover {
            border: 2px solid #0d7377;
        }
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #e0e0e0;
            margin-right: 10px;
        }
        QComboBox QAbstractItemView {
            background-color: #2b2b2b;
            color: #e0e0e0;
            selection-background-color: #0d7377;
            border: 2px solid #3d3d3d;
        }
        
        /* Check Box */
        QCheckBox {
            color: #e0e0e0;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #3d3d3d;
            border-radius: 4px;
            background-color: #2b2b2b;
        }
        QCheckBox::indicator:hover {
            border: 2px solid #0d7377;
        }
        QCheckBox::indicator:checked {
            background-color: #0d7377;
            border: 2px solid #0d7377;
        }
        
        /* Group Box */
        QGroupBox {
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #14a085;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
        }
        
        /* Tab Widget */
        QTabWidget::pane {
            border: 2px solid #3d3d3d;
            border-radius: 6px;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2b2b2b;
            color: #e0e0e0;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            min-width: 100px;
        }
        QTabBar::tab:selected {
            background-color: #0d7377;
            color: white;
        }
        QTabBar::tab:hover:!selected {
            background-color: #3d3d3d;
        }
        
        /* Scroll Bar */
        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #0d7377;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #14a085;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
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
    """Returns modern light theme stylesheet"""
    return """
        /* Main containers */
        QMainWindow, QWidget {
            background-color: #f5f5f5;
            color: #2d2d2d;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        /* Labels */
        QLabel {
            color: #2d2d2d;
            padding: 5px;
            font-size: 11pt;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #0d7377;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 10pt;
            min-height: 30px;
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
        
        /* Text Edit */
        QTextEdit {
            background-color: white;
            color: #2d2d2d;
            border: 2px solid #d0d0d0;
            border-radius: 6px;
            padding: 8px;
            font-size: 10pt;
            selection-background-color: #0d7377;
        }
        QTextEdit:focus {
            border: 2px solid #0d7377;
        }
        
        /* Combo Box */
        QComboBox {
            background-color: white;
            color: #2d2d2d;
            border: 2px solid #d0d0d0;
            padding: 8px;
            border-radius: 6px;
            min-height: 30px;
        }
        QComboBox:hover {
            border: 2px solid #0d7377;
        }
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #2d2d2d;
            margin-right: 10px;
        }
        QComboBox QAbstractItemView {
            background-color: white;
            color: #2d2d2d;
            selection-background-color: #0d7377;
            border: 2px solid #d0d0d0;
        }
        
        /* Check Box */
        QCheckBox {
            color: #2d2d2d;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #d0d0d0;
            border-radius: 4px;
            background-color: white;
        }
        QCheckBox::indicator:hover {
            border: 2px solid #0d7377;
        }
        QCheckBox::indicator:checked {
            background-color: #0d7377;
            border: 2px solid #0d7377;
        }
        
        /* Group Box */
        QGroupBox {
            border: 2px solid #d0d0d0;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #0d7377;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
        }
        
        /* Tab Widget */
        QTabWidget::pane {
            border: 2px solid #d0d0d0;
            border-radius: 6px;
            background-color: #f5f5f5;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            color: #2d2d2d;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            min-width: 100px;
        }
        QTabBar::tab:selected {
            background-color: #0d7377;
            color: white;
        }
        QTabBar::tab:hover:!selected {
            background-color: #d0d0d0;
        }
        
        /* Scroll Bar */
        QScrollBar:vertical {
            background-color: #e0e0e0;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #0d7377;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #14a085;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
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
