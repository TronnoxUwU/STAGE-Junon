from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QPushButton, QLineEdit, QCheckBox, QSpinBox, QLabel,
    QScrollArea, QFrame, QGroupBox, QFileDialog, QComboBox
)
from PySide6.QtCore import Qt, Signal
import os
import toml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.toml")


class Configuration(QWidget):

    config_saved = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_style()
        self._load_config()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("Configuration")
        title.setObjectName("page_title")
        root.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("config_tabs")
        root.addWidget(self.tabs)

        self._build_tab_dossiers()
        self._build_tab_pipeline()
        self._build_tab_entrainement()
        self._build_tab_fusion_completion()

        # Boutons bas
        btn_row = QHBoxLayout()
        self.btn_import = QPushButton("Importer")
        self.btn_import.setObjectName("btn_secondary")
        self.btn_cancel = QPushButton("Annuler")
        self.btn_cancel.setObjectName("btn_secondary")
        self.btn_save = QPushButton("Sauvegarder")
        self.btn_save.setObjectName("btn_primary")
        btn_row.addWidget(self.btn_import)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_save)
        root.addLayout(btn_row)

        self.btn_import.clicked.connect(self._import_config)
        self.btn_save.clicked.connect(self._save_config)
        self.btn_cancel.clicked.connect(self._load_config)
        self.btn_cancel.setVisible(False)
        self._connect_change_signals()
    
    def _connect_change_signals(self):
        for widget in self.findChildren(QLineEdit):
            widget.textChanged.connect(self._on_modified)
        for widget in self.findChildren(QCheckBox):
            widget.stateChanged.connect(self._on_modified)
        for widget in self.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self._on_modified)
        for widget in self.findChildren(QSpinBox):
            widget.valueChanged.connect(self._on_modified)

    def _on_modified(self):
        self.btn_cancel.setVisible(True)

    # ------------------------------------------------------------------ #
    #  TAB 1 — Dossiers                                                    #
    # ------------------------------------------------------------------ #
    def _build_tab_dossiers(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)

        # Extraction
        grp_ext = QGroupBox("Extraction")
        form_ext = QFormLayout(grp_ext)
        self.f_dossier_extraction          = self._path_field(form_ext, "Dossier extraction")
        self.f_dossier_extraction_tmp      = self._path_field(form_ext, "Dossier tmp")
        self.f_meteo_name                  = self._line(form_ext, "Nom météo")
        self.f_nappe_name                  = self._line(form_ext, "Nom nappes")
        self.f_etp_name                    = self._line(form_ext, "Nom ETP")
        self.f_impermeabilite_name         = self._line(form_ext, "Nom imperméabilité")
        self.f_mailles_name                = self._line(form_ext, "Nom mailles")
        self.f_communes_name               = self._line(form_ext, "Nom communes")
        layout.addWidget(grp_ext)

        # Fusion / Clusterisation / Completion
        grp_other = QGroupBox("Fusion / Clusterisation / Complétion")
        form_other = QFormLayout(grp_other)
        self.f_dossier_fusion              = self._path_field(form_other, "Dossier fusion")
        self.f_dossier_nappe_inertielle    = self._path_field(form_other, "Nappe inertielle")
        self.f_dossier_nappe_reactive      = self._path_field(form_other, "Nappe réactive")
        self.f_dossier_completion          = self._path_field(form_other, "Complétion")
        self.f_dossier_completion_in       = self._path_field(form_other, "Complétion inertielle")
        self.f_dossier_completion_re       = self._path_field(form_other, "Complétion réactive")
        layout.addWidget(grp_other)

        # Summary / Modèles
        grp_model = QGroupBox("Summary / Modèles")
        form_model = QFormLayout(grp_model)
        self.f_dossier_summary             = self._path_field(form_model, "Dossier summary")
        self.f_summary_name                = self._line(form_model, "Nom fichier summary")
        self.f_dossier_model               = self._path_field(form_model, "Dossier modèles")
        self.f_dossier_scaler              = self._path_field(form_model, "Dossier scaler")
        layout.addWidget(grp_model)

        layout.addStretch()
        scroll.setWidget(container)
        self.tabs.addTab(scroll, "Dossiers")

    # ------------------------------------------------------------------ #
    #  TAB 2 — Pipeline                                                    #
    # ------------------------------------------------------------------ #
    def _build_tab_pipeline(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        grp_params = QGroupBox("Paramètres")
        form = QFormLayout(grp_params)

        # Départements
        self.f_departements = QLineEdit()
        self.f_departements.setPlaceholderText("ex: 45, 75, 69")
        form.addRow("Départements", self.f_departements)

        # Qualité
        self.f_qualite = QComboBox()
        self.f_qualite.addItems(["0", "5", "10", "20", "30"])
        form.addRow("Qualité (années)", self.f_qualite)

        # Type de nappe
        type_row = QHBoxLayout()
        self.f_type_reactive   = QCheckBox("Réactive")
        self.f_type_inertielle = QCheckBox("Inertielle")
        type_row.addWidget(self.f_type_reactive)
        type_row.addWidget(self.f_type_inertielle)
        type_row.addStretch()
        form.addRow("Type de nappe", type_row)

        layout.addWidget(grp_params)

        grp_etapes = QGroupBox("Étapes actives")
        form2 = QFormLayout(grp_etapes)
        self.f_extraction     = self._checkbox(form2, "Extraction")
        self.f_fusion         = self._checkbox(form2, "Fusion")
        self.f_clusterisation = self._checkbox(form2, "Clusterisation")
        self.f_entrainement   = self._checkbox(form2, "Entraînement")
        self.f_completion     = self._checkbox(form2, "Complétion")
        self.f_affichage      = self._checkbox(form2, "Affichage")
        layout.addWidget(grp_etapes)

        layout.addStretch()
        self.tabs.addTab(container, "Pipeline")

    # ------------------------------------------------------------------ #
    #  TAB 3 — Entraînement                                                #
    # ------------------------------------------------------------------ #
    def _build_tab_entrainement(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        grp = QGroupBox("Modèle")
        form = QFormLayout(grp)

        self.f_window_size = QSpinBox()
        self.f_window_size.setRange(1, 9999)
        form.addRow("Window size", self.f_window_size)

        models_row = QHBoxLayout()
        self.f_model_cnn    = QCheckBox("CNN")
        self.f_model_lstm   = QCheckBox("LSTM")
        self.f_model_bilstm = QCheckBox("BILSTM")
        models_row.addWidget(self.f_model_cnn)
        models_row.addWidget(self.f_model_lstm)
        models_row.addWidget(self.f_model_bilstm)
        models_row.addStretch()
        form.addRow("Modèles", models_row)

        self.f_global     = self._checkbox(form, "Global")
        self.f_fine_tune  = self._checkbox(form, "Fine-tune")
        self.f_local      = self._checkbox(form, "Local")

        layout.addWidget(grp)
        layout.addStretch()
        self.tabs.addTab(container, "Entraînement")

    # ------------------------------------------------------------------ #
    #  TAB 4 — Fusion & Complétion                                         #
    # ------------------------------------------------------------------ #
    def _build_tab_fusion_completion(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # Fusion
        grp_fusion = QGroupBox("Fusion")
        form_fusion = QFormLayout(grp_fusion)
        self.f_preliq_fusion = QComboBox()
        self.f_preliq_fusion.addItems(["sum", "mean"])
        form_fusion.addRow("PRELIQ_Q", self.f_preliq_fusion)
        layout.addWidget(grp_fusion)

        # Complétion — chaque variable
        self.completion_fields = {}
        for var in ["niveau_nappe_eau", "PRELIQ_Q", "T_Q", "ETP_Q"]:
            grp = QGroupBox(f"Complétion — {var}")
            form = QFormLayout(grp)
            chk = QCheckBox("Réaliser")
            form.addRow("", chk)
            methodes = QLineEdit()
            methodes.setPlaceholderText('ex: knn_nappe, lineaire')
            form.addRow("Méthodes", methodes)
            layout.addWidget(grp)
            self.completion_fields[var] = (chk, methodes)

        layout.addStretch()
        scroll.setWidget(container)
        self.tabs.addTab(scroll, "Fusion & Complétion")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _line(self, form, label):
        w = QLineEdit()
        form.addRow(label, w)
        return w

    def _checkbox(self, form, label):
        w = QCheckBox()
        form.addRow(label, w)
        return w

    def _path_field(self, form, label):
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.setObjectName("btn_browse")
        btn.clicked.connect(lambda: self._browse(edit))
        hl.addWidget(edit)
        hl.addWidget(btn)
        form.addRow(label, row)
        return edit

    def _browse(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Choisir un dossier")
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------ #
    #  Load / Save                                                         #
    # ------------------------------------------------------------------ #
    def _load_config(self):
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            cfg = toml.load(CONFIG_PATH)
        except Exception:
            return

        d = cfg.get("dossier", {})
        self.f_dossier_extraction.setText(d.get("dossier_extraction", ""))
        self.f_dossier_extraction_tmp.setText(d.get("dossier_extraction_tmp", ""))
        self.f_meteo_name.setText(d.get("meteo_name_extraction", ""))
        self.f_nappe_name.setText(d.get("nappe_name_extraction", ""))
        self.f_etp_name.setText(d.get("etp_name_extraction", ""))
        self.f_impermeabilite_name.setText(d.get("impermeabilite_name_extraction", ""))
        self.f_mailles_name.setText(d.get("mailles_name_extraction", ""))
        self.f_communes_name.setText(d.get("communes_name_extraction", ""))
        self.f_dossier_fusion.setText(d.get("dossier_fusion", ""))
        self.f_dossier_nappe_inertielle.setText(d.get("dossier_nappe_inertielle", ""))
        self.f_dossier_nappe_reactive.setText(d.get("dossier_nappe_reactive", ""))
        self.f_dossier_completion.setText(d.get("dossier_completion", ""))
        self.f_dossier_completion_in.setText(d.get("dossier_completion_inertielle", ""))
        self.f_dossier_completion_re.setText(d.get("dossier_completion_reactive", ""))
        self.f_dossier_summary.setText(d.get("dossier_summary", ""))
        self.f_summary_name.setText(d.get("summary_name", ""))
        self.f_dossier_model.setText(d.get("dossier_model", ""))
        self.f_dossier_scaler.setText(d.get("dossier_scaler", ""))

        p = cfg.get("pipeline", {})
        self.f_departements.setText(", ".join(str(x) for x in p.get("departements", [])))
        self.f_qualite.setCurrentText(str(p.get("qualite", 30)))
        types = p.get("type", [])
        self.f_type_reactive.setChecked("reactive" in types)
        self.f_type_inertielle.setChecked("inertielle" in types)
        self.f_extraction.setChecked(p.get("extraction", False))
        self.f_fusion.setChecked(p.get("fusion", False))
        self.f_clusterisation.setChecked(p.get("clusterisation", False))
        self.f_entrainement.setChecked(p.get("entrainement", False))
        self.f_completion.setChecked(p.get("completion", False))
        self.f_affichage.setChecked(p.get("affichage", True))

        e = cfg.get("entrainement_model", {})
        self.f_window_size.setValue(e.get("window_size", 120))
        models = e.get("models", [])
        self.f_model_cnn.setChecked("CNN" in models)
        self.f_model_lstm.setChecked("LSTM" in models)
        self.f_model_bilstm.setChecked("BILSTM" in models)
        self.f_global.setChecked(e.get("global", True))
        self.f_fine_tune.setChecked(e.get("fine_tune", True))
        self.f_local.setChecked(e.get("local", False))

        f = cfg.get("fusion", {})
        self.f_preliq_fusion.setCurrentText(f.get("PRELIQ_Q", "sum"))

        comp = cfg.get("completion", {})
        for var, (chk, methodes) in self.completion_fields.items():
            section = comp.get(var, {})
            chk.setChecked(section.get("realiser", False))
            methodes.setText(", ".join(section.get("methodes", [])))

        self.btn_cancel.setVisible(False)

    def _import_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Importer une configuration",
            "",
            "Fichiers TOML (*.toml)"
        )
        if not path:
            return
        try:
            cfg = toml.load(path)
            # Copie vers CONFIG_PATH puis recharge les champs
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                toml.dump(cfg, f)
            self._load_config()
            self.config_saved.emit(cfg)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier :\n{e}")

    def _save_config(self):
        types = []
        if self.f_type_reactive.isChecked():   types.append("reactive")
        if self.f_type_inertielle.isChecked(): types.append("inertielle")

        models = []
        if self.f_model_cnn.isChecked():    models.append("CNN")
        if self.f_model_lstm.isChecked():   models.append("LSTM")
        if self.f_model_bilstm.isChecked(): models.append("BILSTM")

        deps_raw = self.f_departements.text().split(",")
        deps = [int(x.strip()) for x in deps_raw if x.strip().isdigit()]

        comp = {}
        for var, (chk, methodes) in self.completion_fields.items():
            raw = methodes.text()
            comp[var] = {
                "realiser": chk.isChecked(),
                "methodes": [m.strip() for m in raw.split(",") if m.strip()]
            }

        cfg = {
            "dossier": {
                "dossier_extraction":           self.f_dossier_extraction.text(),
                "dossier_extraction_tmp":       self.f_dossier_extraction_tmp.text(),
                "meteo_name_extraction":        self.f_meteo_name.text(),
                "nappe_name_extraction":        self.f_nappe_name.text(),
                "etp_name_extraction":          self.f_etp_name.text(),
                "impermeabilite_name_extraction": self.f_impermeabilite_name.text(),
                "mailles_name_extraction":      self.f_mailles_name.text(),
                "communes_name_extraction":     self.f_communes_name.text(),
                "dossier_fusion":               self.f_dossier_fusion.text(),
                "dossier_nappe_inertielle":     self.f_dossier_nappe_inertielle.text(),
                "dossier_nappe_reactive":       self.f_dossier_nappe_reactive.text(),
                "dossier_completion":           self.f_dossier_completion.text(),
                "dossier_completion_inertielle": self.f_dossier_completion_in.text(),
                "dossier_completion_reactive":  self.f_dossier_completion_re.text(),
                "dossier_summary":              self.f_dossier_summary.text(),
                "summary_name":                 self.f_summary_name.text(),
                "dossier_model":                self.f_dossier_model.text(),
                "dossier_scaler":               self.f_dossier_scaler.text(),
            },
            "pipeline": {
                "departements":   deps,
                "qualite":        int(self.f_qualite.currentText()),
                "type":           types,
                "extraction":     self.f_extraction.isChecked(),
                "fusion":         self.f_fusion.isChecked(),
                "clusterisation": self.f_clusterisation.isChecked(),
                "entrainement":   self.f_entrainement.isChecked(),
                "completion":     self.f_completion.isChecked(),
                "affichage":      self.f_affichage.isChecked(),
            },
            "entrainement_model": {
                "window_size": self.f_window_size.value(),
                "models":      models,
                "global":      self.f_global.isChecked(),
                "fine_tune":   self.f_fine_tune.isChecked(),
                "local":       self.f_local.isChecked(),
            },
            "fusion": {
                "PRELIQ_Q": self.f_preliq_fusion.currentText(),
            },
            "completion": comp,
        }

        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            toml.dump(cfg, f)
        
        self.config_saved.emit(cfg)
        self.btn_cancel.setVisible(False)

    def _load_style(self):
        qss_path = os.path.join(os.path.dirname(__file__), "configuration.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())