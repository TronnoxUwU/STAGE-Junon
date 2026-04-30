import pandas as pd
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame, QGridLayout
)
from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QAreaSeries,
    QDateTimeAxis, QValueAxis
)
from PySide6.QtCore import Qt, QMargins
from PySide6.QtGui import QPainter, QPen, QColor, QLinearGradient, QFont, QGradient

# Couleurs pour les données (Thème Catppuccin)
SERIES_COLORS = {
    "niveau_nappe_eau": "#89b4fa",  # Bleu
    "ETP_Q":            "#a6e3a1",  # Vert
    "PRELIQ_Q":         "#fab387",  # Orange
    "T_Q":              "#f38ba8",  # Rouge
}

# Couleur pour les zones de données manquantes (Pink/Red)
GAP_COLOR_TOP = QColor(243, 139, 168, 100)
GAP_COLOR_BOTTOM = QColor(243, 139, 168, 30)

CONSTANTS = ["lon", "lat", "surface_imp", "surface_totale"]

class MiniChart(QWidget):
    """Graphe individuel avec détection de trous (exactement comme en clusterisation)."""
    def __init__(self, col: str, color_hex: str, parent=None):
        super().__init__(parent)
        self.col = col
        self.color = QColor(color_hex)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl = QLabel(self.col)
        self.lbl.setObjectName("chart_label")
        layout.addWidget(self.lbl)

        self.chart = QChart()
        self.chart.setBackgroundBrush(QColor("#181825"))
        self.chart.setPlotAreaBackgroundBrush(QColor("#11111b"))
        self.chart.setPlotAreaBackgroundVisible(True)
        self.chart.legend().hide()
        self.chart.setMargins(QMargins(0, 0, 0, 0))
        self.chart.layout().setContentsMargins(0, 0, 0, 0)

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setMinimumHeight(180)
        self.chart_view.setBackgroundBrush(QColor("#181825"))
        layout.addWidget(self.chart_view)

    def load(self, df: pd.DataFrame):
        self.chart.removeAllSeries()
        for ax in self.chart.axes():
            self.chart.removeAxis(ax)

        # 1. Préparation des données
        data = df[["time", self.col]].sort_values("time")
        
        segments = []
        current = []

        for _, row in data.iterrows():
            if pd.isna(row[self.col]):
                if current:
                    segments.append(current)
                    current = []
                continue
            
            ms = int(row["time"].timestamp() * 1000)
            current.append((ms, float(row[self.col])))
        
        if current:
            segments.append(current)

        if not segments:
            return

        # 2. Calcul des limites Y (Verticales)
        all_vals = [v for seg in segments for _, v in seg]
        vmin, vmax = min(all_vals), max(all_vals)
        vrange = vmax - vmin if vmax != vmin else 1.0
        y_min = vmin - vrange * 0.05
        y_max = vmax + vrange * 0.05

        # 3. Calcul des limites X (Horizontales) - CRUCIAL pour tout voir
        all_times = [ms for seg in segments for ms, _ in seg]
        min_ts = min(all_times)
        max_ts = max(all_times)

        # 4. Ajout des segments de données
        for seg in segments:
            upper = QLineSeries()
            lower = QLineSeries()
            for ms, v in seg:
                upper.append(ms, v)
                lower.append(ms, y_min)

            pen_invis = QPen(Qt.transparent)
            upper.setPen(pen_invis)
            lower.setPen(pen_invis)
            self.chart.addSeries(upper)
            self.chart.addSeries(lower)

            area = QAreaSeries(upper, lower)
            grad = QLinearGradient(0, 0, 0, 1)
            grad.setCoordinateMode(QGradient.ObjectBoundingMode)
            grad.setColorAt(0.0, QColor(self.color.red(), self.color.green(), self.color.blue(), 120))
            grad.setColorAt(1.0, QColor(self.color.red(), self.color.green(), self.color.blue(), 0))
            
            area.setBrush(grad)
            area.setPen(QPen(self.color, 2))
            self.chart.addSeries(area)

        # 5. Ajout des Gaps (le principe du bloc rose)
        for i in range(len(segments) - 1):
            end_ms = segments[i][-1][0]
            start_ms = segments[i + 1][0][0]

            gap_upper = QLineSeries()
            gap_lower = QLineSeries()
            gap_upper.append(end_ms, y_max)
            gap_upper.append(start_ms, y_max)
            gap_lower.append(end_ms, y_min)
            gap_lower.append(start_ms, y_min)

            gap_upper.setPen(QPen(Qt.transparent))
            gap_lower.setPen(QPen(Qt.transparent))
            self.chart.addSeries(gap_upper)
            self.chart.addSeries(gap_lower)

            gap_area = QAreaSeries(gap_upper, gap_lower)
            gap_grad = QLinearGradient(0, 0, 0, 1)
            gap_grad.setCoordinateMode(QGradient.ObjectBoundingMode)
            gap_grad.setColorAt(0.0, GAP_COLOR_TOP)
            gap_grad.setColorAt(1.0, GAP_COLOR_BOTTOM)
            
            gap_area.setBrush(gap_grad)
            gap_area.setPen(QPen(QColor(243, 139, 168, 150), 1, Qt.DashLine))
            self.chart.addSeries(gap_area)

        # 6. Configuration des Axes avec Range forcé
        axis_x = QDateTimeAxis()
        axis_x.setFormat("MM/yyyy")
        axis_x.setLabelsColor(QColor("#6c7086"))
        axis_x.setGridLineColor(QColor("#313244"))
        # On force l'axe X à couvrir du début du 1er segment à la fin du dernier
        axis_x.setRange(pd.to_datetime(min_ts, unit='ms'), pd.to_datetime(max_ts, unit='ms'))

        axis_y = QValueAxis()
        axis_y.setRange(y_min, y_max)
        axis_y.setTickCount(4)
        axis_y.setLabelsColor(QColor("#6c7086"))
        axis_y.setGridLineColor(QColor("#313244"))

        self.chart.addAxis(axis_x, Qt.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignLeft)

        # 7. Attachement de TOUTES les séries aux axes
        for s in self.chart.series():
            s.attachAxis(axis_x)
            s.attachAxis(axis_y)

class VisualisationDonnee(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._charts = {}
        self._setup_ui()
        self._load_style()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # Section Constantes (lon, lat, etc.)
        self.constants_widget = QWidget()
        self.constants_widget.setObjectName("constants_card")
        cw_layout = QGridLayout(self.constants_widget)
        cw_layout.setSpacing(16)
        self._const_labels = {}
        for i, col in enumerate(CONSTANTS):
            lbl_name = QLabel(col)
            lbl_name.setObjectName("const_name")
            lbl_val  = QLabel("—")
            lbl_val.setObjectName("const_value")
            cw_layout.addWidget(lbl_name, 0, i)
            cw_layout.addWidget(lbl_val,  1, i)
            self._const_labels[col] = lbl_val
        root.addWidget(self.constants_widget)

        # Scroll Area pour les graphiques
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(20)

        for col, color in SERIES_COLORS.items():
            chart = MiniChart(col, color)
            container_layout.addWidget(chart)
            self._charts[col] = chart

        container_layout.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)

    def load_nappe(self, filepath: str):
        try:
            df = pd.read_csv(filepath, sep=";")
            if "time" not in df.columns:
                return
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"])

            # Mise à jour des constantes
            for col in CONSTANTS:
                if col in df.columns:
                    valid_data = df[col].dropna()
                    val = valid_data.iloc[0] if not valid_data.empty else None
                    self._const_labels[col].setText(str(round(val, 4)) if val is not None else "—")

            # Mise à jour des graphiques
            for col, chart in self._charts.items():
                if col in df.columns:
                    chart.load(df)

        except Exception:
            import traceback, sys
            traceback.print_exc(file=sys.stdout)

    def _load_style(self):
        # Charge ton fichier QSS s'il existe
        qss_path = os.path.join(os.path.dirname(__file__), "visualisation_donnee.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())