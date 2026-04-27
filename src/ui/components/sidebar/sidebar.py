from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSpacerItem, QSizePolicy, QLabel
from PySide6.QtCore import Signal, Qt, QSize, QEvent
from PySide6.QtGui import QIcon
import os

from PySide6.QtGui import QPixmap, QPainter, QColor, QIcon


IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "img")

class Sidebar(QWidget):
    page_changed = Signal(str)  # Signal émis avec le nom de la page

    PAGES = [
        ("extraction",     "Extraction",     "cloud-arrow-down.svg"),
        ("visualisation",  "Visualisation",  "globe-europe-africa.svg"),
        ("clusterisation", "Clusterisation", "graph.svg"),
        ("resultat",       "Résultats",      "file-earmark-ruled.svg"),
    ]

    PAGES_BOTTOM = [
        ("configuration", "Configuration", "gear.svg"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self._buttons = {}
        self._setup_ui()
        self._load_style()

    def _update_icon_color(self, btn, color):
        pixmap = QPixmap(btn._icon_path)

        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(color))
        painter.end()

        btn.setIcon(QIcon(pixmap))

    def eventFilter(self, obj, event):
        if isinstance(obj, QPushButton) and obj.objectName() == "sidebar_btn":
            if event.type() == QEvent.Type.Enter:
                self._update_icon_color(obj, "#cdd6f4")
            elif event.type() == QEvent.Type.Leave:
                color = "#89b4fa" if obj.isChecked() else "#a6adc8"
                self._update_icon_color(obj, color)
        return super().eventFilter(obj, event)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("Faut trouver un nom")
        title.setObjectName("sidebar_title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(16)

        # Pages principales
        for page_name, label, icon_file in self.PAGES:
            self._add_button(layout, page_name, label, icon_file)

        # Pousse configuration vers le bas
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Pages du bas
        for page_name, label, icon_file in self.PAGES_BOTTOM:
            self._add_button(layout, page_name, label, icon_file)

        layout.addSpacing(8)
        self.setLayout(layout)
        self._set_active(self.PAGES[0][0])

    def _add_button(self, layout, page_name, label, icon_file):
        btn = QPushButton(label)
        btn.setObjectName("sidebar_btn")
        btn.setCheckable(True)
        btn.installEventFilter(self)
        icon_path = os.path.join(IMG_DIR, icon_file)
        btn._icon_path = icon_path
        if os.path.exists(icon_path):
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(20, 20))
        btn.clicked.connect(lambda checked, p=page_name: self._on_click(p))
        layout.addWidget(btn)
        self._buttons[page_name] = btn

    def _on_click(self, page_name: str):
        self._set_active(page_name)
        self.page_changed.emit(page_name)

    def _set_active(self, page_name: str):
        for name, btn in self._buttons.items():
            checked = name == page_name
            btn.setChecked(checked)

            color = "#89b4fa" if checked else "#a6adc8"
            self._update_icon_color(btn, color)

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "sidebar.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())
