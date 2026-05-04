from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor, QPainter, QFont
import os
import toml
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.toml")

# ─────────────────────────────────────────────
#  Worker
# ─────────────────────────────────────────────
class CompletionWorker(QObject):
    log      = Signal(str)
    progress = Signal(int, int)   # (current, total)
    finished = Signal()
    error    = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run(self):
        import sys

        class _Redirect:
            def __init__(self, sig): self._sig = sig
            def write(self, t):
                t = t.strip()
                if t: self._sig.emit(t)
            def flush(self): pass

        old = sys.stdout
        sys.stdout = _Redirect(self.log)
        try:
            from pipeline import methodes_completion

            cfg     = self.config
            dossiers = []

            types = cfg["pipeline"].get("type", [])
            if "reactive" in types:
                dossiers.append((
                    cfg["dossier"]["dossier_nappe_reactive"],
                    cfg["dossier"]["dossier_completion_reactive"],
                    "reactive"
                ))
            if "inertielle" in types:
                dossiers.append((
                    cfg["dossier"]["dossier_nappe_inertielle"],
                    cfg["dossier"]["dossier_completion_inertielle"],
                    "inertielle"
                ))
            if not types:
                dossiers.append((
                    cfg["dossier"]["dossier_fusion"],
                    cfg["dossier"]["dossier_completion"],
                    ""
                ))

            summary = None
            for input_f, output_f, cluster in dossiers:
                summary = methodes_completion(
                    input_f, output_f,
                    cfg["completion"],
                    cluster,
                    cfg["dossier"]["dossier_model"],
                    cfg["dossier"]["dossier_scaler"],
                    cfg["entrainement_model"]["window_size"],
                    0.2, 1990, 1990,
                    summary
                )

            out_path = f"{cfg['dossier']['dossier_summary']}/{cfg['dossier']['summary_name']}"
            os.makedirs(cfg["dossier"]["dossier_summary"], exist_ok=True)
            pd.DataFrame(summary).to_csv(out_path, sep=";", index=False)

            self.finished.emit()
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())
        finally:
            sys.stdout = old


# ─────────────────────────────────────────────
#  Heatmap cell
# ─────────────────────────────────────────────
class HeatmapCell(QLabel):
    PALETTE = [
        (0.0,  (17,  17,  27)),    # #11111b  très mauvais
        (0.33, (137, 180, 250)),   # #89b4fa
        (0.66, (166, 227, 161)),   # #a6e3a1
        (1.0,  (30,  215, 96)),    # vert vif
    ]

    def __init__(self, value: float | None, parent=None):
        super().__init__(parent)
        self.value = value
        self.setFixedSize(80, 36)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_StyledBackground, True)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            self.setText("—")
            self.setStyleSheet("background:#181825; color:#45475a; border:1px solid #313244; border-radius:4px;")
        else:
            txt_color, bg = self._color(value)
            self.setText(f"{value:.3f}")
            self.setStyleSheet(
                f"background:rgb({bg[0]},{bg[1]},{bg[2]});"
                f"color:{'#11111b' if sum(bg)>380 else '#cdd6f4'};"
                f"border:1px solid #313244; border-radius:4px; font-size:11px;"
            )

    def _color(self, v: float):
        # Interpole dans la palette (0 = bon, 1 = mauvais → on inverse)
        t = max(0.0, min(1.0, 1.0 - v))
        p = self.PALETTE
        for i in range(len(p) - 1):
            t0, c0 = p[i]
            t1, c1 = p[i + 1]
            if t0 <= t <= t1:
                f  = (t - t0) / (t1 - t0)
                bg = tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3))
                return None, bg
        return None, p[-1][1]


# ─────────────────────────────────────────────
#  Heatmap widget (une variable)
# ─────────────────────────────────────────────
class HeatmapWidget(QWidget):
    def __init__(self, variable: str, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.variable  = variable
        self.df        = df
        self._expanded = True
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("heatmap_card")
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header cliquable
        header = QWidget()
        header.setObjectName("heatmap_header")
        header.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(12, 8, 12, 8)
        self.chevron = QLabel("▼")
        self.chevron.setObjectName("chevron")
        title = QLabel(self.variable)
        title.setObjectName("heatmap_title")
        hl.addWidget(self.chevron)
        hl.addWidget(title)
        hl.addStretch()
        root.addWidget(header)

        # Corps
        self.body = QWidget()
        self.body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(12, 8, 12, 12)
        body_layout.setSpacing(4)
        self._build_table(body_layout)
        root.addWidget(self.body)

        header.mousePressEvent = lambda e: self._toggle()

    def _build_table(self, layout):
        # Colonnes = datasets (code_bss), lignes = méthodes
        methods = [c.replace(f"_{self.variable}", "")
                   for c in self.df.columns
                   if c.endswith(f"_{self.variable}")]
        datasets = self.df["code_bss"].tolist() if "code_bss" in self.df.columns else []

        if not methods or not datasets:
            layout.addWidget(QLabel("Aucune donnée"))
            return

        # Grille
        grid_widget = QWidget()
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout(grid_widget)
        grid.setSpacing(2)
        grid.setContentsMargins(0, 0, 0, 0)

        # En-tête datasets (tronqué)
        grid.addWidget(QLabel(""), 0, 0)
        for j, ds in enumerate(datasets):
            lbl = QLabel(ds[:12] + "…" if len(ds) > 12 else ds)
            lbl.setObjectName("heatmap_header_cell")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(80)
            grid.addWidget(lbl, 0, j + 1)

        # Lignes méthodes
        for i, method in enumerate(methods):
            lbl = QLabel(method)
            lbl.setObjectName("heatmap_row_label")
            lbl.setFixedWidth(120)
            grid.addWidget(lbl, i + 1, 0)
            for j, ds in enumerate(datasets):
                row = self.df[self.df["code_bss"] == ds]
                val = row[f"{method}_{self.variable}"].iloc[0] if not row.empty else None
                grid.addWidget(HeatmapCell(val), i + 1, j + 1)

        # Scroll horizontal si beaucoup de datasets
        grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_widget.adjustSize()
        layout.addWidget(grid_widget)

    def _toggle(self):
        self._expanded = not self._expanded
        self.body.setVisible(self._expanded)
        self.chevron.setText("▼" if self._expanded else "▶")


# ─────────────────────────────────────────────
#  Groupe (inertielle / reactive / "")
# ─────────────────────────────────────────────
class GroupeWidget(QWidget):
    def __init__(self, label: str, df: pd.DataFrame, variables: list, parent=None):
        super().__init__(parent)
        self.label     = label
        self._expanded = True
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("groupe_card")
        self._setup_ui(df, variables)

    def _setup_ui(self, df, variables):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("groupe_header")
        header.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 10, 16, 10)
        self.chevron = QLabel("▼")
        self.chevron.setObjectName("groupe_chevron")
        icon = "🔵" if self.label == "inertielle" else ("🟢" if self.label == "reactive" else "📊")
        title = QLabel(f"{icon}  {self.label.capitalize() or 'Résultats'}")
        title.setObjectName("groupe_title")
        hl.addWidget(self.chevron)
        hl.addWidget(title)
        hl.addStretch()
        root.addWidget(header)

        # Corps — une heatmap par variable
        self.body = QWidget()
        self.body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(8, 8, 8, 8)
        bl.setSpacing(8)
        for var in variables:
            bl.addWidget(HeatmapWidget(var, df))
        root.addWidget(self.body)

        header.mousePressEvent = lambda e: self._toggle()

    def _toggle(self):
        self._expanded = not self._expanded
        self.body.setVisible(self._expanded)
        self.chevron.setText("▼" if self._expanded else "▶")


# ─────────────────────────────────────────────
#  Page Résultat
# ─────────────────────────────────────────────
class Resultat(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config  = {}
        self._thread = None
        self._worker = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._setup_ui()
        self._load_style()
        self._load_config_if_exists()

    def _setup_ui(self):
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(0)

        title = QLabel("Résultats & Complétion")
        title.setObjectName("page_title")

        # Tout dans un seul scroll vertical
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll_container = QWidget()
        self.scroll_container.setAttribute(Qt.WA_StyledBackground, True)
        inner = QVBoxLayout(self.scroll_container)
        inner.setContentsMargins(24, 24, 24, 24)
        inner.setSpacing(16)

        inner.addWidget(title)

        # Zone lancement
        self.launch_zone = QWidget()
        lz = QVBoxLayout(self.launch_zone)
        lz.setAlignment(Qt.AlignCenter)
        lz.setSpacing(12)
        self.btn_launch = QPushButton("Lancer la complétion")
        self.btn_launch.setObjectName("btn_primary")
        self.btn_launch.setFixedWidth(240)
        self.btn_launch.clicked.connect(self._start)
        lz.addWidget(self.btn_launch, alignment=Qt.AlignCenter)
        self.log_label = QLabel("")
        self.log_label.setObjectName("log_label")
        self.log_label.setAlignment(Qt.AlignCenter)
        self.log_label.setWordWrap(True)
        lz.addWidget(self.log_label)
        inner.addWidget(self.launch_zone)

        # Zone résultats (plus de QScrollArea ici)
        self.result_zone = QWidget()
        self.result_zone.setVisible(False)
        self.result_layout = QVBoxLayout(self.result_zone)
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_layout.setSpacing(12)
        self.result_layout.addStretch()
        inner.addWidget(self.result_zone)

        #inner.addStretch()
        self.scroll.setWidget(self.scroll_container)
        self.root.addWidget(self.scroll)

    def _load_config_if_exists(self):
        if os.path.exists(CONFIG_PATH):
            try:
                self.apply_config(toml.load(CONFIG_PATH))
            except Exception:
                pass

    def apply_config(self, cfg: dict):
        self.config = cfg
        self._check_existing()

    def _check_existing(self):
        cfg = self.config
        d   = cfg.get("dossier", {})
        types = cfg.get("pipeline", {}).get("type", [])

        folders = []
        if "reactive" in types:
            folders.append((d.get("dossier_completion_reactive", ""), "reactive"))
        if "inertielle" in types:
            folders.append((d.get("dossier_completion_inertielle", ""), "inertielle"))
        if not types:
            folders.append((d.get("dossier_completion", ""), ""))

        has_data = any(
            os.path.exists(f) and any(fn.endswith(".csv") for fn in os.listdir(f))
            for f, _ in folders if f
        )

        summary_path = f"{d.get('dossier_summary','')}/{d.get('summary_name','summary.csv')}"

        if has_data and os.path.exists(summary_path):
            self._show_results(summary_path, types)

    def _show_results(self, summary_path: str, types: list):
        self.launch_zone.setVisible(False)
        self.result_zone.setVisible(True)

        # Vide l'ancien contenu
        while self.result_layout.count() > 1:
            item = self.result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            df = pd.read_csv(summary_path, sep=";")
        except Exception:
            return

        # Détecte les variables à partir des colonnes
        variables = set()
        methods_cols = [c for c in df.columns
                        if c not in ["code_bss", "lat", "lon", "cluster"]]
        for c in methods_cols:
            parts = c.rsplit("_", 1)
            if len(parts) == 2:
                variables.add(parts[1])
        variables = sorted(variables)

        if not variables:
            return

        # Groupe par cluster
        if "cluster" in df.columns and df["cluster"].nunique() > 1:
            for cluster_val in sorted(df["cluster"].unique()):
                sub = df[df["cluster"] == cluster_val].reset_index(drop=True)
                grp = GroupeWidget(str(cluster_val), sub, variables)
                self.result_layout.insertWidget(self.result_layout.count() - 1, grp)
        else:
            grp = GroupeWidget("", df, variables)
            self.result_layout.insertWidget(self.result_layout.count() - 1, grp)

    def _start(self):
        self.btn_launch.setEnabled(False)
        self.log_label.setText("Complétion en cours...")

        self._worker = CompletionWorker(self.config)
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
        self.btn_launch.setEnabled(True)
        cfg   = self.config
        d     = cfg.get("dossier", {})
        types = cfg.get("pipeline", {}).get("type", [])
        summary_path = f"{d.get('dossier_summary','')}/{d.get('summary_name','summary.csv')}"
        self._show_results(summary_path, types)

    def _on_error(self, msg: str):
        self.log_label.setText(f"Erreur :\n{msg}")
        self.btn_launch.setEnabled(True)

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "resultat.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())