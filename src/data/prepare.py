import numpy as np
import pandas as pd

import os
import glob
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_dataset(file, valeur_de_travail, remove_pct, rng, annee_deb = 0, annee_fin = 0):
    ds_name = os.path.basename(file)

    df = pd.read_csv(file, sep=';')

    if valeur_de_travail not in df.columns:
        print(f"⚠️ {ds_name} ignoré : colonne absente.")
        return None

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    y_full = df[valeur_de_travail].to_numpy()

    if np.isnan(y_full).all():
        print(f"⚠️ {ds_name} ignoré : uniquement des NaN.")
        return None

    n_points = len(df)

    y = y_full.copy()
    if annee_deb != annee_fin and annee_deb < annee_fin :
        mask = (df['time'].dt.year >= annee_deb) & (df['time'].dt.year <= annee_fin)
        y[mask] = np.nan


    if remove_pct > 0:
        n_remove = int(n_points * remove_pct)
        remove_idx = rng.choice(n_points, size=n_remove, replace=False)
        y[remove_idx] = np.nan

    if np.count_nonzero(~np.isnan(y)) < 4:
        print(f"⚠️ {ds_name} ignoré : pas assez de points valides.")
        return None

    return df, y_full, y, ds_name

def charger_et_preparer_donnees(dossier_path, features, scaler_path="../../scalers"):
    # 1. Chargement global (comme tu faisais)
    fichiers_csv = glob.glob(os.path.join(dossier_path, "*.csv"))
    liste_df = [pd.read_csv(f, sep=';') for f in fichiers_csv]
    df_final = pd.concat(liste_df, ignore_index=True)
    
    # 2. Nettoyage et Tri
    df_final['time'] = pd.to_datetime(df_final['time'])
    # On trie par station PUIS par temps pour que les chronologies soient respectées
    df_final = df_final.sort_values(['code_bss', 'time'])
    
    df_travail = df_final.copy()
    # Conversion du temps pour le scaler
    df_travail['time_num'] = df_travail['time'].astype('int64') // 10**9
    
    # On ajuste la liste des features si besoin (on utilise time_num au lieu de time)
    features_scaler = [f if f != 'time' else 'time_num' for f in features]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_travail[features_scaler] = scaler.fit_transform(df_travail[features_scaler])
    joblib.dump(scaler, scaler_path)
    
    return df_travail, scaler

def creer_sequences_par_bss(df, features, window_size):
    X_list, y_list = [], []
    
    # On groupe par code_bss pour traiter chaque piézomètre séparément
    for _, group in df.groupby('code_bss'):
        data = group[features].values
        
        # On ne crée des fenêtres que si la station a assez de données
        if len(data) > window_size:
            for i in range(window_size, len(data)):
                X_list.append(data[i-window_size:i, :])
                y_list.append(data[i, :])
    
    return np.array(X_list), np.array(y_list)


def train_data(dossier_nappe, window_size, scaler_path="../"):
    # 1. Préparation
    features = ["niveau_nappe_eau","lon","lat","time","ETP_Q","PRELIQ_Q","T_Q","surface_imp","surface_totale"]
    df_norm, mon_scaler = charger_et_preparer_donnees(dossier_nappe, features, scaler_path)

    # 2. Création des fenêtres étanches (6 mois ici)
    features_pour_ia = [f if f != 'time' else 'time_num' for f in features]
    X, y = creer_sequences_par_bss(df_norm, features_pour_ia, window_size=window_size)

    # 3. Gestion des NaN pour la Masked Loss
    X = np.nan_to_num(X, nan=-999.0)
    y = np.nan_to_num(y, nan=-999.0)

    # 4. Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, X_val, y_train, y_val, mon_scaler

def lstm_predict_array(df_entree, model, scaler, features, window_size, target_col="T_Q"):
    df_travail = df_entree.copy()
    
    # 1. Préparation du temps : On écrase 'time' par sa version numérique
    # pour que le nom de la colonne soit identique à celui vu au 'fit'
    if 'time' in df_travail.columns:
        df_travail['time'] = pd.to_datetime(df_travail['time']).astype('int64') // 10**9
    
    # 2. On s'assure de ne passer QUE les colonnes du scaler, dans le bon ordre
    # Le scaler attend les noms : ["niveau_nappe_eau", ..., "time", ...]
    df_pour_scaler = df_travail[features]
    
    # 3. Transformation (maintenant les noms correspondent)
    values_norm = scaler.transform(df_pour_scaler)
    
    # 4. Gestion des NaN pour le modèle
    values_masked = np.nan_to_num(values_norm, nan=-999.0)
    
    # 5. Création des séquences
    X_all = []
    for i in range(window_size, len(values_masked)):
        X_all.append(values_masked[i-window_size:i, :])
    X_all = np.array(X_all, dtype='float32')
    
    if len(X_all) == 0:
        return np.full(len(df_entree), np.nan)
    
    # 6. Prédiction
    y_pred_norm = model.predict(X_all, verbose=0)
    
    # 7. Inverse Transform
    y_pred_final = scaler.inverse_transform(y_pred_norm)
    
    # 8. Extraction avec l'index dynamique
    target_idx = features.index(target_col)
    
    # 9. Reconstruction
    full_array = np.full(len(df_entree), np.nan)
    full_array[window_size:] = y_pred_final[:, target_idx]
    
    return full_array