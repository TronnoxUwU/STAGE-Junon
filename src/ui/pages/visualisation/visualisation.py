from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QStackedWidget
)
from PySide6.QtCore import Qt, QSize, QEvent
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
import os
import toml

from ...components.liste_nappes import ListeNappes
from ...components.visualisation_carte import VisualisationCarte
from ...components.visualisation_donnee import VisualisationDonnee

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.toml")


class Visualisation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self._setup_ui()
        self._load_style()
        self.load_config_if_exists()

    def _update_icon_color(self, btn, color):
        if not hasattr(btn, "_icon_path"):
            return
        pixmap = QPixmap(btn._icon_path)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(color))
        painter.end()
        btn.setIcon(QIcon(pixmap))

    def eventFilter(self, obj, event):
        # On intercepte le survol de la souris pour les boutons de toggle
        if obj in [self.btn_carte, self.btn_donnees]:
            if event.type() == QEvent.Type.Enter:
                self._update_icon_color(obj, "#cdd6f4") # Couleur survol
            elif event.type() == QEvent.Type.Leave:
                color = "#89b4fa" if obj.isChecked() else "#a6adc8"
                self._update_icon_color(obj, color)
        return super().eventFilter(obj, event)

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Panneau gauche — liste des nappes
        self.liste_nappes = ListeNappes()
        self.liste_nappes.nappe_selected.connect(self._on_nappe_selected)
        root.addWidget(self.liste_nappes)

        # Panneau droit
        right = QWidget()
        right.setAttribute(Qt.WA_StyledBackground, True)
        right.setObjectName("visu_right")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Barre de toggle carte / données
        toggle_bar = QWidget()
        toggle_bar.setObjectName("toggle_bar")
        toggle_layout = QHBoxLayout(toggle_bar)
        toggle_layout.setContentsMargins(16, 8, 16, 8)
        toggle_layout.setSpacing(8)
        icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "img")

        self.btn_carte = QPushButton(" Carte")
        self.btn_donnees = QPushButton(" Données")

        # Chemins des icônes
        self.btn_carte._icon_path = os.path.join(icon_path, "map.svg")
        self.btn_donnees._icon_path = os.path.join(icon_path, "clipboard-data.svg")

        # Installation du filtre pour le hover (survol)
        self.btn_carte.installEventFilter(self)
        self.btn_donnees.installEventFilter(self)

        self.btn_carte.setIconSize(QSize(20, 20))
        self.btn_donnees.setIconSize(QSize(20, 20))
        
        self.btn_carte.setCheckable(True)
        self.btn_donnees.setCheckable(True)
        self.btn_carte.setChecked(True)
        self.btn_donnees.setChecked(False)
        
        # Initialisation de la couleur par défaut (Actif pour carte, Inactif pour données)
        self._update_icon_color(self.btn_carte, "#89b4fa")
        self._update_icon_color(self.btn_donnees, "#a6adc8")

        self.btn_carte.clicked.connect(lambda: self._switch_view(0))
        self.btn_donnees.clicked.connect(lambda: self._switch_view(1))
        self.btn_carte.setObjectName("btn_toggle_active")
        self.btn_donnees.setObjectName("btn_toggle")

        toggle_layout.addWidget(self.btn_carte)
        toggle_layout.addWidget(self.btn_donnees)
        toggle_layout.addStretch()
        right_layout.addWidget(toggle_bar)

        # Stack carte / données
        self.stack = QStackedWidget()
        self.visu_carte   = VisualisationCarte()
        self.visu_donnee  = VisualisationDonnee()
        self.stack.addWidget(self.visu_carte)
        self.stack.addWidget(self.visu_donnee)
        right_layout.addWidget(self.stack)

        root.addWidget(right)

    def _switch_view(self, index: int):
        self.stack.setCurrentIndex(index)
        
        # Mise à jour de l'état "checked"
        self.btn_carte.setChecked(index == 0)
        self.btn_donnees.setChecked(index == 1)
        
        # Mise à jour des couleurs d'icônes
        self._update_icon_color(self.btn_carte, "#89b4fa" if index == 0 else "#a6adc8")
        self._update_icon_color(self.btn_donnees, "#89b4fa" if index == 1 else "#a6adc8")
        
        # Conservation de tes ObjectName pour le QSS
        self.btn_carte.setObjectName("btn_toggle_active" if index == 0 else "btn_toggle")
        self.btn_donnees.setObjectName("btn_toggle_active" if index == 1 else "btn_toggle")
        
        # Forcer le rafraîchissement du style
        self.btn_carte.setStyle(self.btn_carte.style())
        self.btn_donnees.setStyle(self.btn_donnees.style())

    def _on_nappe_selected(self, filepath: str):
        self.visu_carte.highlight_nappe(filepath)
        self.visu_donnee.load_nappe(filepath)

    def load_config_if_exists(self):
        if os.path.exists(CONFIG_PATH):
            try:
                self.apply_config(toml.load(CONFIG_PATH))
            except Exception:
                pass

    def apply_config(self, cfg: dict):
        self.config = cfg
        d = cfg.get("dossier", {})

        folder_in = d.get("dossier_nappe_inertielle", "data/clusterisation/inertielle")
        folder_re = d.get("dossier_nappe_reactive",   "data/clusterisation/reactive")
        folder_fusion = d.get("dossier_fusion", "data/fusion")

        def has_csv(folder):
            return (
                os.path.exists(folder) and
                any(f.endswith(".csv") for f in os.listdir(folder))
            )

        # Si clusterisation faite → on l'utilise, sinon → fusion
        if has_csv(folder_in) or has_csv(folder_re):
            folders = [folder_in, folder_re]
        else:
            folders = [folder_fusion]

        self.liste_nappes.load_nappes(folders)
        self.visu_carte.load_nappes(folders)

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "visualisation.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())