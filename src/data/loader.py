import pandas as pd

import os
os.environ["KERAS_BACKEND"] = "torch"
import glob

def charger_fichier(
    fichier_nappe:str
) -> pd.DataFrame:
    """Charge un fichier csv

    Args:
        fichier_nappe (str): Le chemin du fichier à charger

    Returns:
        DataFrame: Le fichier ouvert sous forme d'un Dataframe
    """    
    df = pd.read_csv(fichier_nappe, sep=";")
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time')

def liste_fichiers(
    folder_path:str
) -> list[str]:
    """Liste tout les fichiers csv dans un dossier

    Args:
        folder_path (str): Le chemin du dossier

    Returns:
        list[str]: La liste des chemins des fichier csv 
    """    
    pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"❌ Aucun fichier CSV trouvé dans : {folder_path}")
        return []

    return files

def charger_dossier(
    dossier_path:str
) -> pd.DataFrame:
    """Charge un dossier de fichiers csv

    Args:
        dossier_path (str): Le chemin du dossier à charger

    Returns:
        DataFrame: Le dossier ouvert sous forme d'un Dataframe
    """   
    fichiers_csv = liste_fichiers(dossier_path)
    liste_df = [pd.read_csv(f, sep=';') for f in fichiers_csv]
    df = pd.concat(liste_df, ignore_index=True)

    df['time'] = pd.to_datetime(df['time'])

    df = df.sort_values(['code_bss', 'time'])
    return df