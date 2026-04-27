from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget
import os

from .pages import Clusterisation, Configuration, Extraction, Resultat, Visualisation
from .components.sidebar import Sidebar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Application")
        self.resize(1280, 800)
        self._load_global_style()
        self._setup_ui()

    def _load_global_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "style", "global.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())

    def _setup_ui(self):
        # Widget central contenant sidebar + contenu
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self.navigate_to)
        root_layout.addWidget(self.sidebar)

        # Zone de contenu (pages)
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack)

        self.pages = {
            "clusterisation": Clusterisation(),
            "configuration":  Configuration(),
            "extraction":     Extraction(),
            "resultat":       Resultat(),
            "visualisation":  Visualisation(),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        # Page par défaut
        self.navigate_to("extraction")

    def navigate_to(self, page_name: str):
        if page_name in self.pages:
            self.stack.setCurrentWidget(self.pages[page_name])
            self.sidebar._set_active(page_name)
