import numpy as np
from src.methodes import *
from src.data import *
from keras.models import load_model

def nmae(prediction, reelle):
    range_val = np.max(reelle)
    if range_val == 0: return 0 # Sécurité si le signal est plat
    return np.mean(np.abs(reelle - prediction)) / range_val

def nrmse(prediction, reelle):
    range_val = np.max(reelle)
    if range_val == 0: return 0
    # On calcule la RMSE standard puis on normalise
    return np.sqrt(np.mean((reelle - prediction) ** 2)) / range_val

def nse(prediction, reelle):    
    numerator = np.sum((reelle - prediction) ** 2)
    denominator = np.sum((reelle - np.mean(reelle)) ** 2)
    
    if denominator == 0:
        return np.nan
    
    return 1 - (numerator / denominator)

def compute_interpolations(df, y, valeur_de_travail, models, mon_scaler):
    """
    Calcule plusieurs interpolations/completion pour une colonne donnée
    et renvoie un dictionnaire {méthode: np.array}.
    """
    # On crée une copie avec les trous simulés
    df_with_holes = df.copy()
    df_with_holes[valeur_de_travail] = y
    df_with_holes['time_num'] = df['time'].astype('int64') // 10**9

    features = ["niveau_nappe_eau","lon","lat","time_num","ETP_Q","PRELIQ_Q","T_Q","surface_imp","surface_totale"]
    
    methods = {
        "Linear": interpolation_lineaire_array(df_with_holes, valeur_de_travail),
        "PCHIP": interpolation_pchip_array(df_with_holes, valeur_de_travail),
        "Akima": interpolation_akima_array(df_with_holes, valeur_de_travail),
        "Cubic": interpolation_cubique_array(df_with_holes, valeur_de_travail),
        "Poly": interpolation_polynomiale_array(df_with_holes, valeur_de_travail),
        "B-Spline": interpolation_spline_array(df_with_holes, valeur_de_travail),
        "KNN": knn_impute(df_with_holes, valeur_de_travail),
        "Bootstrap saisonier": bootstrap_saisonnier_impute(df_with_holes, valeur_de_travail),
        "Random forest": random_forest_delta_array(df_with_holes, valeur_de_travail),
        "LSTM" : lstm_predict_array(df_with_holes, models["LSTM"], mon_scaler, features, window_size=60, target_col=valeur_de_travail),
        "BILSTM" : lstm_predict_array(df_with_holes, models["BILSTM"], mon_scaler, features, window_size=6, target_col=valeur_de_travail),
        "LSTM2" : lstm_predict_array(df_with_holes, models["LSTM2"], mon_scaler, features, window_size=60, target_col=valeur_de_travail),
        "CNN" : lstm_predict_array(df_with_holes, models["CNN"], mon_scaler, features, window_size=6, target_col=valeur_de_travail),
    }


    return methods

def evaluate_methods(methods_dict, y_full, ds_name, remove_pct):
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

def evaluate_all_files(folder_path, valeur_de_travail, mon_scaler,
                       remove_pct_list=[0.1],
                       random_state=42,
                       max_files=1000,
                       path="../"):

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
                prepared = prepare_dataset(file, valeur_de_travail, remove_pct, rng)
                if prepared is None:
                    continue

                df, y_full, y, ds_name = prepared

                methods = compute_interpolations(df, y, valeur_de_travail, models, mon_scaler)

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
