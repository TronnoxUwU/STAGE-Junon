import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "torch"
from src.methodes import *
from src.data import *
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

def nmae(
    prediction:np.ndarray,
    reelle:np.ndarray
) -> float:
    """calcul la MAE normalisé

    Args:
        prediction (np.ndarray): valeur prédite
        reelle (np.ndarray): valeur réelle

    Returns:
        float: Resultat
    """    
    range_val = np.max(reelle)
    if range_val == 0: return 0 # Sécurité si le signal est plat
    return np.mean(np.abs(reelle - prediction)) / range_val

def nrmse(
    prediction:np.ndarray,
    reelle:np.ndarray
) -> float:
    """calcul la RMSE normalisé

    Args:
        prediction (np.ndarray): valeur prédite
        reelle (np.ndarray): valeur réelle

    Returns:
        float: Resultat
    """    
    range_val = np.max(reelle)
    if range_val == 0: return 0
    # On calcule la RMSE standard puis on normalise
    return np.sqrt(np.mean((reelle - prediction) ** 2)) / range_val

def nse(
    prediction:np.ndarray,
    reelle:np.ndarray
) -> float:
    """calcul la NSE normalisé

    Args:
        prediction (np.ndarray): valeur prédite
        reelle (np.ndarray): valeur réelle

    Returns:
        float: Resultat
    """     
    numerator = np.sum((reelle - prediction) ** 2)
    denominator = np.sum((reelle - np.mean(reelle)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    return 1 - (numerator / denominator)

def compute_interpolations(
    df:pd.DataFrame, 
    valeur_de_travail:str, 
    models:Sequential, 
    mon_scaler:MinMaxScaler
) -> Dict[str, np.ndarray]:
    """Calcul complete les troues d'un data frame de toute les manière

    Args:
        df (pd.DataFrame): Les données à compléter
        valeur_de_travail (str): Valeur que l'on veut compléter
        models (Sequential): La liste des models d'ia
        mon_scaler (MinMaxScaler): Le scaler pour les NN

    Returns:
        Dict[str, np.ndarray]: Resultat pour chaque méthodes
    """    

    df['time_num'] = df['time'].astype('int64') // 10**9

    features = ["niveau_nappe_eau","lon","lat","time_num","ETP_Q","PRELIQ_Q","T_Q","surface_imp","surface_totale"]
    
    methods = {
        "Linear": interpolation_lineaire_array(df, valeur_de_travail),
        "PCHIP": interpolation_pchip_array(df, valeur_de_travail),
        "Akima": interpolation_akima_array(df, valeur_de_travail),
        "Cubic": interpolation_cubique_array(df, valeur_de_travail),
        "Poly": interpolation_polynomiale_array(df, valeur_de_travail),
        "B-Spline": interpolation_spline_array(df, valeur_de_travail),
        "KNN": knn_impute(df, valeur_de_travail),
        "Bootstrap saisonier": bootstrap_saisonnier_impute(df, valeur_de_travail),
        "Random forest": random_forest_delta_array(df, valeur_de_travail),
        "LSTM" : lstm_predict_array(df, models["LSTM"], mon_scaler, features, window_size=60, target_col=valeur_de_travail),
        "BILSTM" : lstm_predict_array(df, models["BILSTM"], mon_scaler, features, window_size=6, target_col=valeur_de_travail),
        "LSTM2" : lstm_predict_array(df, models["LSTM2"], mon_scaler, features, window_size=60, target_col=valeur_de_travail),
        "CNN" : lstm_predict_array(df, models["CNN"], mon_scaler, features, window_size=6, target_col=valeur_de_travail),
    }


    return methods

def evaluate_methods(
    methods_dict:Dict[str, np.ndarray], 
    y_full:np.ndarray, 
    ds_name:str, 
    remove_pct:float
) -> Dict[str, np.ndarray]:
    """Calcul les erreurs des méthodes

    Args:
        methods_dict (Dict[str, np.ndarray]): dictionnaire des méthodes et leurs resultats
        y_full (np.ndarray): vrai resultat
        ds_name (str): nom du fichier sur lequel le travail a été fait
        remove_pct (float): pourcentage de données retrirées

    Returns:
        Dict[str, np.ndarray]: Resultat de chaques méthodes
    """    
    rows = []
    
    y_full = y_full[6:]

    for method_name, arr in methods_dict.items():
        arr = arr[6:]
        rows.append({
            'dataset': ds_name,
            'method': method_name,
            'pct_removed': remove_pct,
            'NMAE': nmae(arr, y_full),
            'NRMSE': nrmse(arr, y_full)
        })

    return rows

def evaluate_all_files(
    folder_path:str, 
    valeur_de_travail:str,
    mon_scaler:MinMaxScaler,
    remove_pct_list:List[float]=[0.1],
    random_state:int=42,
    max_files:int=1000,
    path:str="../"
) -> pd.DataFrame:
    """evalue les manière de complété les fichiers

    Args:
        folder_path (str): dossier contenant les fichier que l'on veut compléter
        valeur_de_travail (str): valeur que l'on veut compler
        mon_scaler (MinMaxScaler): scaler pour les NN
        remove_pct_list (List[float], optional): liste de pourcentage de données à retirer. Defaults to [0.1].
        random_state (int, optional): seed du random. Defaults to 42.
        max_files (int, optional): nombre max de fichier. Defaults to 1000.
        path (str, optional): chemin vers la racine du projet. Defaults to "../".

    Returns:
        pd.DataFrame: resultat 
    """    

    files = liste_fichiers(folder_path)
    if not files:
        return None

    rng = np.random.default_rng(random_state)
    all_rows = []
    processed_count = 0

    models = {
        "LSTM" : load_model(path +"/models/LSTM.keras", custom_objects={'masked_mse': masked_mse}),
        "BILSTM" : load_model(path +"/models/BILSTM.keras", custom_objects={'masked_mse': masked_mse}),
        "LSTM2" : load_model(path +"/models/LSTM2.keras", custom_objects={'masked_mse': masked_mse}),
        "CNN" : load_model(path + "/models/CNN.keras", custom_objects={'masked_mse': masked_mse})
    }


    for remove_pct in remove_pct_list:
        for file in files:
            if processed_count >= max_files:
                break

            try:
                prepared = generate_missing_data(charger_fichier(file), file, valeur_de_travail, remove_pct, rng)
                if prepared is None:
                    continue

                df, y_full, ds_name = prepared

                methods = compute_interpolations(df, valeur_de_travail, models, mon_scaler)

                rows = evaluate_methods(methods, y_full, ds_name, remove_pct)
                all_rows.extend(rows)

                processed_count += 1

            except Exception as e:
                print(f"💥 Erreur sur {file} : {e}")
                continue

    if not all_rows:
        print("🛑 Aucun fichier traité.")
        return None

    return pd.DataFrame(all_rows)
