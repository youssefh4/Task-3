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
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 500;
        min-height: 28px;
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
        font-size: 11px;
        border: 2px solid #cccccc;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
"""

