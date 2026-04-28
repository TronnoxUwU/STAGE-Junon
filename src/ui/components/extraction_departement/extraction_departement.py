from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPainter, QColor, QConicalGradient, QPen
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class ExtractionWorker(QObject):
    log = Signal(str)
    progress = Signal(int)   # 0-100
    finished = Signal()
    error = Signal(str)

    # Étapes avec leur poids relatif
    STEPS = [
        ("communes",       15),
        ("mailles",        10),
        ("impermeabilite", 25),
        ("etp",            20),
        ("meteo",          20),
        ("nappe",          10),
    ]

    def __init__(self, dept: int, config: dict):
        super().__init__()
        self.dept = dept
        self.config = config

    def run(self):
        try:
            from data.extraction import (
                download_communes_csv, download_maille_csv,
                process_impermeabilite, process_etp, process_meteo, process_nappe
            )

            d = self.config.get("dossier", {})
            n = self.config
            names = {
                "communes_name_extraction":      d.get("communes_name_extraction", "communes"),
                "mailles_name_extraction":       d.get("mailles_name_extraction", "maille"),
                "impermeabilite_name_extraction":d.get("impermeabilite_name_extraction", "impermeabilite"),
                "etp_name_extraction":           d.get("etp_name_extraction", "etp"),
                "meteo_name_extraction":         d.get("meteo_name_extraction", "meteo"),
                "nappe_name_extraction":         d.get("nappe_name_extraction", "nappes"),
            }
            output_folder = d.get("dossier_extraction", "data/extraction")
            tmp_folder    = d.get("dossier_extraction_tmp", "data/extraction/tmp")

            cumulative = 0

            self.log.emit(f"[{self.dept}] Téléchargement communes...")
            download_communes_csv(output_folder, names["communes_name_extraction"])
            cumulative += 15
            self.progress.emit(cumulative)

            self.log.emit(f"[{self.dept}] Téléchargement mailles...")
            download_maille_csv(output_folder, names["mailles_name_extraction"])
            cumulative += 10
            self.progress.emit(cumulative)

            self.log.emit(f"[{self.dept}] Traitement imperméabilité...")
            process_impermeabilite(
                dataset_id="697b4f4ceea77fb452ba9d6d",
                tmp_folder=tmp_folder,
                output_folder=output_folder,
                communes_file=f"{output_folder}/{names['communes_name_extraction']}.csv",
                name=names["impermeabilite_name_extraction"],
                departements=[self.dept]
            )
            cumulative += 25
            self.progress.emit(cumulative)

            self.log.emit(f"[{self.dept}] Traitement ETP...")
            process_etp(
                dataset_id="667eae35510cd549fc7722c1",
                tmp_folder=tmp_folder,
                output_folder=output_folder,
                name=names["etp_name_extraction"],
                maille_file=f"{output_folder}/{names['mailles_name_extraction']}.csv"
            )
            cumulative += 20
            self.progress.emit(cumulative)

            self.log.emit(f"[{self.dept}] Traitement météo...")
            process_meteo(
                dataset_id="6569b3d7d193b4daf2b43edc",
                tmp_folder=tmp_folder,
                output_folder=output_folder,
                name=names["meteo_name_extraction"],
                departements=[self.dept]
            )
            cumulative += 20
            self.progress.emit(cumulative)

            self.log.emit(f"[{self.dept}] Traitement nappes...")
            process_nappe(
                output_folder=output_folder,
                name=names["nappe_name_extraction"],
                departements=[self.dept]
            )
            cumulative += 10
            self.progress.emit(cumulative)

            self.log.emit(f"[{self.dept}] ✓ Extraction terminée.")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


class SpeedometerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self.setFixedSize(180, 180)

    def set_value(self, v: int):
        self._value = max(0, min(100, v))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        margin = 20
        rect_size = min(w, h) - margin * 2

        cx = w // 2
        cy = h // 2

        # Arc de 270° démarrant à 225° (en bas à gauche)
        # Qt: 0° = 3h, sens antihoraire, angles en 1/16°
        START_ANGLE = 225    # degrés, en bas à gauche
        SPAN        = 270    # degrés, 3/4 de cercle

        x = cx - rect_size // 2
        y = cy - rect_size // 2

        # Fond (arc gris)
        pen = QPen(QColor("#313244"), 12, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(x, y, rect_size, rect_size,
                        START_ANGLE * 16, -SPAN * 16)

        # Arc de progression
        if self._value > 0:
            t = self._value / 100
            r = int(0x89 + (0xa6 - 0x89) * t)
            g = int(0xb4 + (0xe3 - 0xb4) * t)
            b = int(0xfa + (0xa1 - 0xfa) * t)
            pen = QPen(QColor(r, g, b), 12, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            span = int(self._value * SPAN / 100)
            painter.drawArc(x, y, rect_size, rect_size,
                            START_ANGLE * 16, -span * 16)

        # Texte pourcentage au centre
        from PySide6.QtGui import QFont
        painter.setPen(QColor("#cdd6f4"))
        font = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font)
        painter.drawText(0, 0, w, h, Qt.AlignCenter, f"{self._value}%")


class ExtractionDepartement(QWidget):
    def __init__(self, dept: int, config: dict, parent=None):
        super().__init__(parent)
        self.dept = dept
        self.config = config
        self._thread = None
        self._worker = None
        self.setFixedWidth(280)
        self._setup_ui()
        self._load_style()
        # self.setAttribute(Qt.WA_StyledBackground, True)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # En-tête
        lbl = QLabel(f"Département {self.dept:02d}")
        lbl.setObjectName("dept_title")
        layout.addWidget(lbl)

        # Compteur
        self.speedometer = SpeedometerWidget()
        layout.addWidget(self.speedometer, alignment=Qt.AlignHCenter)

        # Console
        self.console = QTextEdit()
        self.console.setObjectName("dept_console")
        self.console.setReadOnly(True)
        self.console.setFixedHeight(120)
        layout.addWidget(self.console)

        # Boutons
        self.btn_extract = QPushButton("Extraire")
        self.btn_extract.setObjectName("btn_extract")

        self.btn_use_existing = QPushButton("📂")
        self.btn_use_existing.setObjectName("btn_existing")
        self.btn_use_existing.setFixedSize(28, 28)
        self.btn_use_existing.setToolTip("Utiliser les données existantes")
        self.btn_use_existing.clicked.connect(self._use_existing)

        self.btn_fuse = QPushButton("Fusionner")
        self.btn_fuse.setObjectName("btn_fuse")
        self.btn_fuse.setEnabled(False)
        layout.addWidget(self.btn_extract)
        layout.addWidget(self.btn_use_existing)
        layout.addWidget(self.btn_fuse)

        self.btn_extract.clicked.connect(self._start_extraction)
        self.btn_fuse.clicked.connect(self._start_fusion)

        self.setObjectName("dept_card")
        # self.setAttribute(Qt.WA_StyledBackground, True)

    def _start_extraction(self):
        if self._thread and self._thread.isRunning():
            return
        self.console.clear()
        self.speedometer.set_value(0)
        self.btn_extract.setEnabled(False)
        self.btn_fuse.setEnabled(False)

        self._worker = ExtractionWorker(self.dept, self.config)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._on_log)
        self._worker.progress.connect(self.speedometer.set_value)
        self._worker.finished.connect(self._on_extraction_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_extraction_done(self):
        self.btn_extract.setEnabled(True)
        self.btn_fuse.setEnabled(True) 

    def _use_existing(self):
        output_folder = self.config.get("dossier", {}).get("dossier_extraction", "data/extraction")

        if not os.path.exists(output_folder):
            self.console.append(f"<span style='color:#f38ba8'>Dossier introuvable : {output_folder}</span>")
            return

        files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]

        if not files:
            self.console.append(f"<span style='color:#f38ba8'>Aucun fichier trouvé dans : {output_folder}</span>")
            return

        self.console.clear()
        self.speedometer.set_value(100)
        self.console.append(f"<span style='color:#89b4fa'>{len(files)} fichier(s) détecté(s) :</span>")
        for f in files:
            self.console.append(f"<span style='color:#a6adc8'>  • {f}</span>")
        self.console.append(f"<span style='color:#a6e3a1'>✓ Prêt pour la fusion.</span>")
        self.btn_fuse.setEnabled(True)

    def _start_fusion(self):
        if self._thread and self._thread.isRunning():
            return
        self.console.append("── Fusion ──")
        self.btn_fuse.setEnabled(False)
        self.btn_extract.setEnabled(False)

        self._worker = FusionWorker(self.dept, self.config)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_fusion_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_fusion_done(self):
        self.btn_extract.setEnabled(True)
        self.btn_fuse.setEnabled(True)

    def _on_log(self, msg: str):
        self.console.append(msg)

    def _on_done(self):
        self.btn_start.setEnabled(True)

    def _on_error(self, msg: str):
        self.console.append(f"<span style='color:#f38ba8'>Erreur : {msg}</span>")
        self.btn_start.setEnabled(True)

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "extraction_departement.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())

class FusionWorker(QObject):
    log = Signal(str)
    progress = Signal(int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, dept: int, config: dict):
        super().__init__()
        self.dept = dept
        self.config = config

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = _LogRedirect(self.log)
        try:
            from pipeline import fusion

            d = self.config.get("dossier", {})
            names = {
                "meteo_name_extraction":          d.get("meteo_name_extraction", "meteo"),
                "etp_name_extraction":            d.get("etp_name_extraction", "etp"),
                "impermeabilite_name_extraction": d.get("impermeabilite_name_extraction", "impermeabilite"),
                "nappe_name_extraction":          d.get("nappe_name_extraction", "nappes"),
            }
            methodes = {
                "PRELIQ_Q": self.config.get("fusion", {}).get("PRELIQ_Q", "sum")
            }

            fusion(
                output_folder=d.get("dossier_fusion", "data/fusion"),
                input_folder=d.get("dossier_extraction", "data/extraction"),
                names=names,
                methodes=methodes
            )
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout  

class _LogRedirect:
    """Redirige stdout vers un Signal PySide6."""
    def __init__(self, signal):
        self._signal = signal

    def write(self, text):
        text = text.strip()
        if text:
            self._signal.emit(text)

    def flush(self):
        pass