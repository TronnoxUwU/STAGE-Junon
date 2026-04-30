from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
import os
import pandas as pd
import json


class VisualisationCarte(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._nappes_data = []   # [{name, lat, lon, filepath, type}]
        self._selected    = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.web = QWebEngineView()
        layout.addWidget(self.web)
        self._render_map()

    def load_nappes(self, folders: list):
        self._nappes_data.clear()
        for folder in folders:
            nappe_type = "inertielle" if "inertielle" in folder else "reactive"
            if not os.path.exists(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if not f.endswith(".csv"):
                    continue
                filepath = os.path.join(folder, f)
                try:
                    df  = pd.read_csv(filepath, sep=";", nrows=1)
                    lat = float(df["lat"].iloc[0])  if "lat"  in df.columns else None
                    lon = float(df["lon"].iloc[0])  if "lon"  in df.columns else None
                    if lat and lon:
                        self._nappes_data.append({
                            "name":     f.replace(".csv", ""),
                            "lat":      lat,
                            "lon":      lon,
                            "filepath": filepath,
                            "type":     nappe_type,
                        })
                except Exception:
                    pass
        self._render_map()

    def highlight_nappe(self, filepath: str):
        self._selected = filepath
        self._render_map()

    def _render_map(self):
        import json
        markers_js = json.dumps(self._nappes_data)
        selected   = json.dumps(self._selected)

        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    body, html {{ margin:0; padding:0; background: #11111b; }}
    #map {{ width:100%; height:100vh; background: #242424; }}

    .nappe-tooltip {{
        background-color: #313244 !important;
        color: #cdd6f4 !important;
        border: 1px solid #89b4fa !important;
        border-radius: 6px;
        padding: 5px 10px;
        font-family: "Segoe UI", sans-serif;
        font-size: 12px;
    }}
</style>
</head>
<body>
<div id="map"></div>
<script>
const map = L.map('map', {{
    center: [46.8, 2.3],
    zoom: 6,
    zoomControl: false
}});

// --- FOND ARCGIS DARK GRAY CANVAS ---
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
    maxZoom: 16
}}).addTo(map);

// Ajout des labels par-dessus (optionnel, pour que les noms de villes soient lisibles)
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
    attribution: 'Esri, HERE, Garmin',
    maxZoom: 16
}}).addTo(map);

const markers = {markers_js};
const selected = {selected};

markers.forEach(m => {{
    const isSelected = selected && m.filepath === selected;
    const color = isSelected ? '#f38ba8' : (m.type === 'inertielle' ? '#89b4fa' : '#a6e3a1');
    
    const circle = L.circleMarker([m.lat, m.lon], {{
        radius: isSelected ? 11 : 7,
        fillColor: color,
        color: isSelected ? '#ffffff' : '#11111b', 
        weight: isSelected ? 2.5 : 1.5,
        opacity: 1,
        fillOpacity: isSelected ? 1.0 : 0.8
    }}).addTo(map);

    circle.bindTooltip(m.name, {{
        permanent: false,
        direction: 'top',
        className: 'nappe-tooltip',
        offset: [0, -8]
    }});

    if (isSelected) circle.bringToFront();
}});

if (selected) {{
    const sel = markers.find(m => m.filepath === selected);
    if (sel) map.setView([sel.lat, sel.lon], 11, {{animate: true}});
}}
</script>
</body>
</html>
"""
        self.web.setHtml(html)