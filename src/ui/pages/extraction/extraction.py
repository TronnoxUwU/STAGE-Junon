from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
import os
import toml

from ...components.extraction_departement import ExtractionDepartement

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.toml")


class Extraction(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self.dept_widgets = {}
        self._setup_ui()
        self._load_style()
        self._load_config_if_exists()
        self.scroll.setFixedHeight(600)

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        self.title = QLabel("Départements :")
        self.title.setObjectName("page_title")
        root.addWidget(self.title)

        # Zone scrollable horizontale
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setFixedHeight(420)

        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(16)
        self.cards_layout.addStretch()

        self.scroll.setWidget(self.cards_container)
        root.addWidget(self.scroll)
        root.addStretch()

    def _load_config_if_exists(self):
        if os.path.exists(CONFIG_PATH):
            try:
                self.apply_config(toml.load(CONFIG_PATH))
            except Exception:
                pass

    def apply_config(self, cfg: dict):
        self.config = cfg
        departements = cfg.get("pipeline", {}).get("departements", [])
        self._refresh_departments(departements)

    def _refresh_departments(self, departements: list):
        # Supprime les anciennes cartes
        for w in self.dept_widgets.values():
            self.cards_layout.removeWidget(w)
            w.deleteLater()
        self.dept_widgets.clear()

        for dept in departements:
            widget = ExtractionDepartement(dept, self.config)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, widget)
            self.dept_widgets[dept] = widget

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "extraction.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())