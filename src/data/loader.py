import pandas as pd

import os
import glob

def charger_fichier(fichier_nappe):
    df = pd.read_csv(fichier_nappe, sep=";")
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time')

def liste_fichiers(folder_path):
    pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"❌ Aucun fichier CSV trouvé dans : {folder_path}")
        return []

    return files