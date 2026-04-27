from PySide6.QtWidgets import QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice
import os


class Heatmap(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._load_ui()
        self._load_style()
        self._setup_connections()

    def _load_ui(self):
        loader = QUiLoader()
        ui_path = os.path.join(os.path.dirname(__file__), "heatmap.ui")
        ui_file = QFile(ui_path)
        if ui_file.open(QIODevice.ReadOnly):
            self.ui = loader.load(ui_file, self)
            ui_file.close()

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "heatmap.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())

    def _setup_connections(self):
        pass
