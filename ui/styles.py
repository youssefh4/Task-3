"""UI styling constants and functions."""

APPLICATION_STYLESHEET = """
    QWidget {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10pt;
    }
    QPushButton {
        background-color: #0078d4;
        color: white;
        border: none;
        border-radius: 3px;
        padding: 4px 5px;
        font-weight: 500;
        min-height: 20px;
        font-size: 8.5pt;
        max-width: 220px;
    }
    QPushButton:hover {
        background-color: #106ebe;
    }
    QPushButton:pressed {
        background-color: #005a9e;
    }
    QPushButton:disabled {
        background-color: #cccccc;
        color: #666666;
    }
    QScrollArea {
        border: 1px solid #cccccc;
        border-radius: 4px;
        background-color: #f8f8f8;
    }
"""

GROUPBOX_STYLESHEET = """
    QGroupBox {
        font-weight: bold;
        font-size: 10px;
        border: 1px solid #cccccc;
        border-radius: 3px;
        margin-top: 8px;
        padding-top: 8px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 3px;
    }
"""

