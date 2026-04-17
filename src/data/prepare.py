import numpy as np
import pandas as pd

import os
import joblib

from typing import Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential

def generate_missing_data(
    df:pd.DataFrame, 
    file:str, 
    valeur_de_travail:str, 
    remove_pct:float, 
    rng:np.random.Generator, 
    annee_deb:int = 0, 
    annee_fin:int = 0
) -> Optional[Tuple[pd.DataFrame, np.ndarray, str]]:
    """Fonction qui réalise des trous dans les données sur la valeur demandé

    Args:
        df (pd.DataFrame): Dataframe sur lequel on veut faire des troues
        file (str): Nom du fichier
        valeur_de_travail (str): Valeur sur laquel on veut faire des trous
        remove_pct (float): Pourcentage de trous que l'on veut atteindre
        rng (np.random.Generator): Generateur d'aléatoire
        annee_deb (int, optional): Debut d'un trou. Defaults to 0.
        annee_fin (int, optional): Fin d'un trou. Defaults to 0.

    Returns:
        Optional[Tuple[pd.DataFrame, np.ndarray, str]]: tuple composé de 
    """    
    ds_name = os.path.basename(file)

    y_full = df.copy()[valeur_de_travail].to_numpy()

    n_points = len(df)

    if annee_deb != annee_fin and annee_deb < annee_fin :
        mask = (df['time'].dt.year >= annee_deb) & (df['time'].dt.year <= annee_fin)
        df.loc[mask, valeur_de_travail] = np.nan

    if remove_pct > 0:
        n_remove = int(n_points * remove_pct)
        remove_idx = rng.choice(n_points, size=n_remove, replace=False)
        df.loc[remove_idx, valeur_de_travail] = np.nan

    if df[valeur_de_travail].notna().sum() < 4:
        print(f"⚠️ {ds_name} ignoré : pas assez de points valides.")
        return None

    return df, y_full, ds_name

def generate_missing_data_NN(
    df:pd.DataFrame, 
    valeur_de_travail:str, 
    remove_pct:float,
    rng:np.random.Generator,
    taille:tuple[int]=(1,1),  
) -> Optional[Tuple[pd.DataFrame, np.ndarray, str]]:
    """Fonction qui réalise des trous dans les données sur la valeur demandé

    Args:
        df (pd.DataFrame): Dataframe sur lequel on veut faire des troues
        file (str): Nom du fichier
        valeur_de_travail (str): Valeur sur laquel on veut faire des trous
        remove_pct (float): Pourcentage de trous que l'on veut atteindre
        rng (np.random.Generator): Generateur d'aléatoire
        taille (tuple[int]): Taille des troues créer

    Returns:
        Optional[Tuple[pd.DataFrame, np.ndarray, str]]: tuple composé de 
    """    

    df = df.copy()
    y_full = df[valeur_de_travail].to_numpy()

    n_points = len(df)
    n_remove_target = int(n_points * remove_pct)
    
    removed = 0
    removed_indices = set()

    while removed < n_remove_target:
        # 1. choisir une taille de trou
        hole_size = rng.integers(taille[0], taille[1] + 1)

        # 2. choisir un point de départ valide
        start = rng.integers(0, n_points - hole_size + 1)

        # 3. indices du trou
        indices = range(start, start + hole_size)

        # éviter de supprimer deux fois les mêmes points
        new_indices = [i for i in indices if i not in removed_indices]

        if not new_indices:
            continue

        # 4. appliquer suppression
        df.iloc[new_indices, df.columns.get_loc(valeur_de_travail)] = -1

        removed_indices.update(new_indices)
        removed += len(new_indices)

    if df[valeur_de_travail].notna().sum() < 4:
        return None

    return df, y_full

def preparer_donnees(
    df:pd.DataFrame, 
    features:List[str], 
    scaler_path:str="../../scalers",
    scaler:MinMaxScaler = None
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Normalise un DataFrame

    Args:
        df (pd.DataFrame): DataFrame que l'on veut normaliser
        features (List[str]): features que l'on veut normaliser
        scaler_path (str, optional): chemin ou l'on veut sauvegarder le fichier. Defaults to "../../scalers".
        scaler(MinMaxScaler, optional): Scaler si deja existant

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]: tuple composé du dataFrame et du scaler utilisé
    """    
    # Conversion du temps pour le scaler
    df['time_num'] = df['time'].astype('int64') // 10**9
    
    # On ajuste la liste des features si besoin (on utilise time_num au lieu de time)
    features_scaler = [f if f != 'time' else 'time_num' for f in features]
    
    if scaler is None :
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[features_scaler] = scaler.fit_transform(df[features_scaler])
        joblib.dump(scaler, scaler_path)
    else :
        df[features_scaler] = scaler.transform(df[features_scaler])

    
    return df, scaler

def creer_sequences_par_bss(
    df:pd.DataFrame, 
    features:List[str], 
    window_size:int,
    croissant:bool=True
) -> np.ndarray:
    """créer des séquences de données 

    Args:
        df (pd.DataFrame): DataFrame sur lequel on va se baser
        features (List[str]): Features
        window_size (int): taille des séquences que l'on veut faire
        croissant (bool, optional): True si dans l'ordre croissant

    Returns:
        ndarray: liste de séquences.
    """    
    X_list, y_list = [], []
    
    # On groupe par code_bss pour traiter chaque piézomètre séparément
    for _, group in df.groupby('code_bss'):
        group = group.sort_values(by='time_num', ascending=croissant)
        data = group[features].values
        
        # On ne crée des fenêtres que si la station a assez de données
        if len(data) > window_size:
            for i in range(window_size, len(data)):
                X_list.append(data[i-window_size:i, :])
                y_list.append(data[i, :])
    
    return np.array(X_list), np.array(y_list)

def creer_sequences_par_bss_cnn(
    df:pd.DataFrame, 
    features:List[str], 
    window_size:int,
    rng: np.random.Generator,
    remove_pct: float = 0.1,
    taille: tuple[int, int] = (1, 5),
    croissant: bool = True
) -> np.ndarray:
    """créer des séquences de données 

    Args:
        df (pd.DataFrame): DataFrame sur lequel on va se baser
        features (List[str]): Features
        window_size (int): taille des séquences que l'on veut faire
        croissant (bool, optional): True si dans l'ordre croissant

    Returns:
        Tuple[ndarray]: liste de séquences.
    """    
    X_list, y_list = [], []

    for _, group in df.groupby('code_bss'):
        group = group.sort_values(by='time_num', ascending=croissant).copy()

        # On garde une version complète pour y
        group_full = group.copy()

        # On applique les trous sur UNE feature (ou plusieurs si tu veux)
        for feature in features:
            result = generate_missing_data_NN(
                group,
                valeur_de_travail=feature,
                remove_pct=remove_pct,
                rng=rng,
                taille=taille
            )
            if result is None:
                continue
            group, _ = result

        data_with_holes = group[features].values
        data_full = group_full[features].values

        if len(data_with_holes) > window_size + 1:
            for i in range(window_size, len(data_with_holes)):
                # X = données avec trous
                X_list.append(data_with_holes[i-window_size:i, :])
                
                # y = données complètes (vérité terrain)
                y_list.append(data_full[i-window_size:i, :])

    return np.array(X_list), np.array(y_list)

def train_data(
    df:pd.DataFrame, 
    window_size:int, 
    scaler_path:str = "../", 
    scaler:MinMaxScaler = None,
    croissant:bool = True,
    saine:bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Normalise un DataFrame

    Args:
        df (pd.DataFrame): DataFrame que l'on veut normaliser
        features (List[str]): features que l'on veut normaliser
        scaler_path (str, optional): chemin ou l'on veut sauvegarder le fichier. Defaults to "../../scalers".
        scaler(MinMaxScaler, optional): Scaler si deja existant
        croissant (bool, optional): True si dans l'ordre croissant
        saine (bool, optional): False si les données que l'on veut faire soit a partir de données non saine

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: tuple composé des données pour réalisé l'entrainement de l'IA et du scaler pour les données.
    """    

    if saine :
        df_norm.dropna()
    
    # 1. Préparation
    features = ["niveau_nappe_eau","lon","lat","time","ETP_Q","PRELIQ_Q","T_Q","surface_imp","surface_totale"]
    df_norm, mon_scaler = preparer_donnees(df, features, scaler_path, scaler)


    # 2. Création des fenêtres étanches (6 mois ici)
    features_pour_ia = [f if f != 'time' else 'time_num' for f in features]
    X, y = creer_sequences_par_bss(df_norm, features_pour_ia, window_size=window_size, croissant=croissant)

    # 3. Gestion des NaN pour la Masked Loss
    X = np.nan_to_num(X, nan=-999.0)
    y = np.nan_to_num(y, nan=-999.0)

    # 4. Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)
    return X_train, X_val, y_train, y_val, mon_scaler

def train_data_cnn(
    df:pd.DataFrame, 
    window_size:int, 
    scaler_path:str = "../", 
    scaler:MinMaxScaler = None,
    croissant:bool = True,
    saine:bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Normalise un DataFrame

    Args:
        df (pd.DataFrame): DataFrame que l'on veut normaliser
        features (List[str]): features que l'on veut normaliser
        scaler_path (str, optional): chemin ou l'on veut sauvegarder le fichier. Defaults to "../../scalers".
        scaler(MinMaxScaler, optional): Scaler si deja existant
        croissant (bool, optional): True si dans l'ordre croissant
        saine (bool, optional): False si les données que l'on veut faire soit a partir de données non saine

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: tuple composé des données pour réalisé l'entrainement de l'IA et du scaler pour les données.
    """    
    # 1. Préparation
    if saine :
        df = df.dropna()

    features = ["niveau_nappe_eau","lon","lat","time","ETP_Q","PRELIQ_Q","T_Q","surface_imp","surface_totale"]
    
    df_norm, mon_scaler = preparer_donnees(df, features, scaler_path, scaler)
    df_norm = df_norm.fillna(-1)

    # 2. Création des fenêtres étanches (6 mois ici)
    features_pour_ia = [f if f != 'time' else 'time_num' for f in features]

    split_idx = int(len(df_norm) * 0.8)
    df_train = df_norm.iloc[:split_idx]
    df_val = df_norm.iloc[split_idx:]

    rng = np.random.default_rng(42)

    X_train, y_train = creer_sequences_par_bss_cnn(
        df_train, features_pour_ia, window_size=window_size, rng=rng, croissant=croissant
    )
    
    X_val, y_val = creer_sequences_par_bss_cnn(
        df_val, features_pour_ia, window_size=window_size, rng=rng, croissant=croissant
    )
    
    return X_train, X_val, y_train, y_val, mon_scaler

def lstm_predict_array(
    df:pd.DataFrame,
    model:Sequential, 
    scaler:MinMaxScaler, 
    features:List[str], 
    window_size:int, 
    target_col:str = "T_Q"
) -> np.ndarray:
    """Predit le resultat d'une valeur demander via un model

    Args:
        df (pd.DataFrame): DataFrame de travail
        model (Sequential): Model que l'on veut utilisé
        scaler (MinMaxScaler): Scaler pour les données
        features (List[str]): Features pour le model
        window_size (int): Taille des fenetres
        target_col (str, optional): Colonne sur laquel on veut travailler. Defaults to "T_Q".

    Returns:
        np.ndarray: Colonne complété.
    """    
    df_travail = df.copy()
    
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
    values_masked = np.nan_to_num(values_norm, nan=-1)
    
    # 5. Création des séquences
    X_all = []
    for i in range(window_size, len(values_masked)):
        X_all.append(values_masked[i-window_size:i, :])
    X_all = np.array(X_all, dtype='float32')
    
    if len(X_all) == 0:
        return np.full(len(df), np.nan)
    
    # 6. Prédiction
    y_pred_norm = model.predict(X_all, verbose=0)
    
    # 7. Inverse Transform
    y_pred_final = scaler.inverse_transform(y_pred_norm)
    
    # 8. Extraction avec l'index dynamique
    target_idx = features.index(target_col)
    
    # 9. Reconstruction
    full_array = np.full(len(df), np.nan)
    full_array[:window_size] = y_pred_final[:, target_idx]
    full_array[window_size:] = y_pred_final[:, target_idx]
    
    return full_array

def cnn_predict_array(
    df:pd.DataFrame,
    model:Sequential, 
    scaler:MinMaxScaler, 
    features:List[str], 
    window_size:int, 
    target_col:str = "T_Q"
) -> np.ndarray:
    
    df_travail = df.copy()
    
    # 1. Préparation du temps
    if 'time' in df_travail.columns:
        df_travail['time'] = pd.to_datetime(df_travail['time']).astype('int64') // 10**9
    
    # 2. Scaling
    df_pour_scaler = df_travail[features]
    values_norm = scaler.transform(df_pour_scaler)
    values_masked = np.nan_to_num(values_norm, nan=-1.0)
    
    # 3. Création des fenêtres
    X_all = []
    for i in range(window_size, len(values_masked)):
        X_all.append(values_masked[i-window_size:i, :])
    X_all = np.array(X_all, dtype='float32')
    
    if len(X_all) == 0:
        return np.full(len(df), np.nan)
    
    # 4. Prédiction (Sortie: [Nb_fenetres, 120, 13])
    y_pred_norm = model.predict(X_all, verbose=0)
    
    # 5. Extraction de la dernière valeur de chaque fenêtre (Many-to-Many -> Many-to-One)
    # On prend [toutes les fenêtres, le dernier pas de temps (index -1), toutes les features]
    y_pred_last_step = y_pred_norm[:, -1, :] 

    n_points = len(df)
    sum_predictions = np.zeros((n_points, len(features)))
    counts = np.zeros(n_points)

    # On itère sur chaque fenêtre prédite
    # La fenêtre i commence à l'index i et finit à i + window_size
    for i in range(len(y_pred_norm)):
        start_idx = i
        end_idx = i + window_size
        
        sum_predictions[start_idx:end_idx] += y_pred_norm[i]
        counts[start_idx:end_idx] += 1

    # 6. Éviter la division par zéro et calculer la moyenne
    # Les premiers et derniers points auront moins de contributeurs (counts < window_size)
    counts[counts == 0] = np.nan 
    avg_pred_norm = sum_predictions / counts[:, np.newaxis]

    # 7. Inverse Transform sur les données moyennées
    # On ne garde que les lignes où on a au moins une prédiction
    mask = ~np.isnan(counts)
    y_pred_final_full = np.full((n_points, len(features)), np.nan)
    
    y_pred_final_full[mask] = scaler.inverse_transform(avg_pred_norm[mask])
    
    # 8. Extraction de la colonne cible
    target_idx = features.index(target_col)
    values_target = y_pred_final_full[:, target_idx]

    return values_target