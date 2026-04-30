from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QAbstractItemView,
    QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QMimeData, QSize
from PySide6.QtGui import QDrag
import os
import toml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.toml")


class ClusterisationWorker(QObject):
    log      = Signal(str)
    finished = Signal()
    error    = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        try:
            from pipeline import clusterisations
            import sys

            class _Redirect:
                def __init__(self, sig): self._sig = sig
                def write(self, t):
                    t = t.strip()
                    if t: self._sig.emit(t)
                def flush(self): pass

            old = sys.stdout
            sys.stdout = _Redirect(self.log)

            dossiers = self.config["dossier"]
            clusterisations(
                dossiers["dossier_fusion"],
                dossiers["dossier_nappe_inertielle"],
                dossiers["dossier_nappe_reactive"]
            )

            sys.stdout = old
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


from PySide6.QtCharts import QChart, QChartView, QLineSeries
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtCore import QMargins


class NappePreviewWidget(QWidget):
    """Widget affiché dans chaque item de la liste : nom + mini graphe."""
    def __init__(self, name: str, filepath: str, parent=None):
        super().__init__(parent)
        self.name     = name
        self.filepath = filepath
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)

        lbl = QLabel(self.name)
        lbl.setObjectName("nappe_name")
        lbl.setFixedWidth(160)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # Mini chart
        self.chart = QChart()
        self.chart.setBackgroundBrush(QColor("#181825"))
        self.chart.setPlotAreaBackgroundBrush(QColor("#181825"))
        self.chart.setPlotAreaBackgroundVisible(True)
        self.chart.legend().hide()
        self.chart.setMargins(QMargins(0, 0, 0, 0))
        self.chart.layout().setContentsMargins(0, 0, 0, 0)

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setFixedSize(200, 70)
        self.chart_view.setBackgroundBrush(QColor("#181825"))
        self.chart_view.setEnabled(False)
        self.chart_view.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.chart_view)

    def _load_data(self):
        try:
            import pandas as pd
            from PySide6.QtCharts import QAreaSeries, QDateTimeAxis, QValueAxis
            from PySide6.QtGui import QLinearGradient, QGradient, QFont

            df = pd.read_csv(self.filepath, sep=";")

            if "niveau_nappe_eau" not in df.columns or "time" not in df.columns:
                return

            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

            if df.empty:
                return

            # 👉 Construire des segments (sans NaN)
            segments = []
            current = []

            for _, row in df.iterrows():
                t = row["time"]
                v = row["niveau_nappe_eau"]

                if pd.isna(v):
                    if current:
                        segments.append(current)
                        current = []
                    continue

                ms = int(t.timestamp() * 1000)
                current.append((ms, float(v)))

            if current:
                segments.append(current)

            if not segments:
                return

            # Bornes Y
            all_vals = [v for seg in segments for _, v in seg]
            vmin = min(all_vals)
            vmax = max(all_vals)
            vrange = vmax - vmin if vmax != vmin else 1.0

            y_min = vmin - vrange * 0.05
            y_max = vmax + vrange * 0.05

            # 👉 Créer UNE série par segment
            for seg in segments:
                upper = QLineSeries()
                pen = QPen(QColor("#89b4fa"))
                pen.setWidth(2)
                upper.setPen(pen)

                lower = QLineSeries()

                for ms, v in seg:
                    upper.append(ms, v)
                    lower.append(ms, y_min)

                upper.setVisible(True)
                lower.setVisible(True)

                # Rendre les séries de base transparentes
                pen_invisible = QPen(Qt.transparent)
                pen_invisible.setWidth(0)
                upper.setPen(pen_invisible)
                lower.setPen(pen_invisible)

                self.chart.addSeries(upper)
                self.chart.addSeries(lower)

                # Area propre (pas de pont entre segments)
                area = QAreaSeries(upper, lower)

                gradient = QLinearGradient(0, 0, 0, 1)
                gradient.setCoordinateMode(QGradient.ObjectBoundingMode)
                gradient.setColorAt(0.0, QColor(137, 180, 250, 120))
                gradient.setColorAt(1.0, QColor(137, 180, 250, 0))
                area.setBrush(gradient)

                pen_area = QPen(QColor("#89b4fa"))
                pen_area.setWidth(2)
                area.setPen(pen_area)

                self.chart.addSeries(area)

            for i in range(len(segments) - 1):
                end_ms   = segments[i][-1][0]
                start_ms = segments[i + 1][0][0]

                gap_upper = QLineSeries()
                gap_upper.append(end_ms,   y_max)
                gap_upper.append(start_ms, y_max)
                gap_lower = QLineSeries()
                gap_lower.append(end_ms,   y_min)
                gap_lower.append(start_ms, y_min)

                pen_invisible = QPen(Qt.transparent)
                pen_invisible.setWidth(0)
                gap_upper.setPen(pen_invisible)
                gap_lower.setPen(pen_invisible)

                self.chart.addSeries(gap_upper)
                self.chart.addSeries(gap_lower)

                gap_area = QAreaSeries(gap_upper, gap_lower)
                gap_gradient = QLinearGradient(0, 0, 0, 1)
                gap_gradient.setCoordinateMode(QGradient.ObjectBoundingMode)
                gap_gradient.setColorAt(0.0, QColor(243, 139, 168, 100))
                gap_gradient.setColorAt(1.0, QColor(243, 139, 168, 30))
                gap_area.setBrush(gap_gradient)
                pen_gap = QPen(QColor(243, 139, 168, 150))
                pen_gap.setWidth(1)
                pen_gap.setStyle(Qt.DashLine)
                gap_area.setPen(pen_gap)
                self.chart.addSeries(gap_area)

            # Axes
            axis_x = QDateTimeAxis()
            axis_x.setFormat("MM/yy")
            axis_x.setLabelsColor(QColor("#6c7086"))
            axis_x.setGridLineColor(QColor("#313244"))
            axis_x.setMinorGridLineColor(QColor("#24273a"))
            axis_x.setMinorGridLineVisible(True)

            axis_y = QValueAxis()
            axis_y.setLabelsColor(QColor("#6c7086"))
            axis_y.setGridLineColor(QColor("#313244"))
            axis_y.setMinorGridLineColor(QColor("#24273a"))
            axis_y.setMinorGridLineVisible(True)
            axis_y.setTickCount(4)
            axis_y.setRange(y_min, y_max)

            font = QFont("Segoe UI", 7)
            axis_x.setLabelsFont(font)
            axis_y.setLabelsFont(font)

            self.chart.addAxis(axis_x, Qt.AlignBottom)
            self.chart.addAxis(axis_y, Qt.AlignLeft)

            all_times = [ms for seg in segments for ms, _ in seg]
            axis_x.setRange(
                pd.to_datetime(min(all_times), unit="ms"),
                pd.to_datetime(max(all_times), unit="ms")
            )

            for s in self.chart.series():
                s.attachAxis(axis_x)
                s.attachAxis(axis_y)

            self.chart.setPlotAreaBackgroundBrush(QColor("#11111b"))
            self.chart.setPlotAreaBackgroundVisible(True)
        
            for axis in self.chart.axes():
                axis.setLabelsColor(QColor("#6c7086"))
                axis.setGridLineColor(QColor("#313244"))
                axis.setLinePen(QPen(QColor("#313244")))

        except Exception:
            import traceback, sys as _sys
            traceback.print_exc(file=_sys.__stdout__)


class NappeList(QListWidget):
    """Liste drag & drop d'un type de nappe."""
    type_changed = Signal(str, str)  # (filepath, new_type)

    def __init__(self, nappe_type: str, dest_folder: str, parent=None):
        super().__init__(parent)
        self.nappe_type  = nappe_type
        self.dest_folder = dest_folder
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSpacing(2)
        self.setObjectName("nappe_list")

    def dropEvent(self, event):
        widgets = {}
        for i in range(self.count()):
            item = self.item(i)
            filepath = item.data(Qt.UserRole)
            if filepath:
                widgets[filepath] = self.itemWidget(item)

        # drop standard
        super().dropEvent(event)

        # 🔥 2. Réassigner les widgets
        for i in range(self.count()):
            item = self.item(i)
            filepath = item.data(Qt.UserRole)

            widget = widgets.get(filepath)

            if widget is None:
                # fallback si jamais
                widget = NappePreviewWidget(
                    os.path.basename(filepath).replace(".csv", ""),
                    filepath
                )

            self.setItemWidget(item, widget)

            # gestion déplacement fichier
            if filepath:
                target = os.path.join(self.dest_folder, os.path.basename(filepath))
                if filepath != target and os.path.exists(filepath):
                    os.rename(filepath, target)
                    item.setData(Qt.UserRole, target)
                    self.type_changed.emit(target, self.nappe_type)


class ClusterColumn(QWidget):
    """Colonne affichant un type de nappe avec flèches de transfert."""
    transfer = Signal(str, str)  # (filepath, target_type)

    def __init__(self, nappe_type: str, folder: str, other_type: str, parent=None):
        super().__init__(parent)
        self.nappe_type  = nappe_type
        self.other_type  = other_type
        self.folder      = folder
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(f"{'Inertielle' if self.nappe_type == 'inertielle' else 'Réactive'}")
        title.setObjectName("cluster_title")
        layout.addWidget(title)

        self.count_label = QLabel("0 nappes")
        self.count_label.setObjectName("cluster_count")
        layout.addWidget(self.count_label)

        self.list_widget = NappeList(self.nappe_type, self.folder)
        self.list_widget.setSpacing(4)
        self.list_widget.setUniformItemSizes(False)
        layout.addWidget(self.list_widget)   # ← manquait

        arrow = "→  Passer en réactive" if self.nappe_type == "inertielle" else "←  Passer en inertielle"
        self.btn_transfer = QPushButton(arrow)
        self.btn_transfer.setObjectName("btn_transfer")
        self.btn_transfer.clicked.connect(self._transfer_selected)
        layout.addWidget(self.btn_transfer)

    def load_nappes(self):
        self.list_widget.clear()
        if not os.path.exists(self.folder):
            return
        files = sorted([f for f in os.listdir(self.folder) if f.endswith(".csv")])
        for f in files:
            filepath = os.path.join(self.folder, f)
            item = QListWidgetItem()
            widget = NappePreviewWidget(f.replace(".csv", ""), filepath)
            item.setSizeHint(QSize(400, 90))
            item.setData(Qt.UserRole, filepath)   # pour le drag & drop
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
        self._update_count()

    def _update_count(self):
        self.count_label.setText(f"{self.list_widget.count()} nappe(s)")

    def _transfer_selected(self):
        item = self.list_widget.currentItem()
        if item is None:
            return
        filepath = item.data(Qt.UserRole)
        if filepath:
            self.transfer.emit(filepath, self.other_type)
            self.list_widget.takeItem(self.list_widget.row(item))
            self._update_count()


class Clusterisation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self._thread = None
        self._worker = None
        self._setup_ui()
        self._load_style()
        self._load_config_if_exists()

    def _setup_ui(self):
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(24, 24, 24, 24)
        self.root_layout.setSpacing(16)

        title = QLabel("Clusterisation")
        title.setObjectName("page_title")
        self.root_layout.addWidget(title)

        # Zone bouton lancement
        self.launch_zone = QWidget()
        lz = QVBoxLayout(self.launch_zone)
        lz.setAlignment(Qt.AlignCenter)

        self.btn_launch = QPushButton("Lancer la clusterisation")
        self.btn_launch.setObjectName("btn_primary")
        self.btn_launch.setFixedWidth(240)
        self.btn_launch.clicked.connect(self._start_clustering)
        lz.addWidget(self.btn_launch, alignment=Qt.AlignCenter)

        self.log_label = QLabel("")
        self.log_label.setObjectName("cluster_log")
        self.log_label.setAlignment(Qt.AlignCenter)
        lz.addWidget(self.log_label)

        self.root_layout.addWidget(self.launch_zone)

        # Zone résultats (cachée par défaut)
        self.result_zone = QWidget()
        self.result_zone.setVisible(False)
        rz = QHBoxLayout(self.result_zone)
        rz.setSpacing(16)

        self.col_inertielle = ClusterColumn(
            "inertielle", "", "reactive"
        )
        self.col_reactive = ClusterColumn(
            "reactive", "", "inertielle"
        )
        self.col_inertielle.transfer.connect(self._on_transfer)
        self.col_reactive.transfer.connect(self._on_transfer)

        rz.addWidget(self.col_inertielle)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setObjectName("cluster_sep")
        rz.addWidget(sep)

        rz.addWidget(self.col_reactive)
        self.root_layout.addWidget(self.result_zone)

    def _load_config_if_exists(self):
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

        # Met à jour les dossiers des colonnes
        self.col_inertielle.folder = folder_in
        self.col_inertielle.list_widget.dest_folder = folder_in
        self.col_reactive.folder   = folder_re
        self.col_reactive.list_widget.dest_folder   = folder_re

        self._check_existing()

    def _check_existing(self):
        d = self.config.get("dossier", {})
        folder_in = d.get("dossier_nappe_inertielle", "data/clusterisation/inertielle")
        folder_re = d.get("dossier_nappe_reactive",   "data/clusterisation/reactive")

        has_in = os.path.exists(folder_in) and any(f.endswith(".csv") for f in os.listdir(folder_in)) if os.path.exists(folder_in) else False
        has_re = os.path.exists(folder_re) and any(f.endswith(".csv") for f in os.listdir(folder_re)) if os.path.exists(folder_re) else False

        if has_in or has_re:
            self._show_results()

    def _show_results(self):
        self.launch_zone.setVisible(False)
        self.result_zone.setVisible(True)
        self.col_inertielle.load_nappes()
        self.col_reactive.load_nappes()

    def _start_clustering(self):
        self.btn_launch.setEnabled(False)
        self.log_label.setText("Clusterisation en cours...")

        self._worker = ClusterisationWorker(self.config)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(lambda msg: self.log_label.setText(msg))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_done(self):
        self._show_results()

    def _on_error(self, msg: str):
        self.log_label.setText(f"Erreur : {msg}")
        self.btn_launch.setEnabled(True)

    def _on_transfer(self, filepath: str, target_type: str):
        d = self.config.get("dossier", {})
        if target_type == "inertielle":
            dest_folder = d.get("dossier_nappe_inertielle", "data/clusterisation/inertielle")
            dest_col    = self.col_inertielle
            src_col     = self.col_reactive
        else:
            dest_folder = d.get("dossier_nappe_reactive", "data/clusterisation/reactive")
            dest_col    = self.col_reactive
            src_col     = self.col_inertielle

        dest = os.path.join(dest_folder, os.path.basename(filepath))
        if os.path.exists(filepath):
            os.makedirs(dest_folder, exist_ok=True)
            os.rename(filepath, dest)

        item = QListWidgetItem()
        widget = NappePreviewWidget(os.path.basename(dest).replace(".csv", ""), dest)
        item.setSizeHint(QSize(400, 90))
        item.setData(Qt.UserRole, dest)
        dest_col.list_widget.addItem(item)
        dest_col.list_widget.setItemWidget(item, widget)
        dest_col._update_count()
        src_col._update_count()

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "clusterisation.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())