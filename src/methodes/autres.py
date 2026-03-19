import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def random_forest_delta_array(df:pd.DataFrame, valeur_de_travail:str, features:list[str]=None)->np.ndarray:
    """
    Interpole une colonne cible d’un DataFrame en utilisant un modèle 
    Random Forest pour prédire le taux de variation journalier et 
    reconstruire la série temporelle de manière itérative.

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé
        features (list[str], optional): Liste des colonnes à utiliser comme 
            caractéristiques pour entraîner le modèle de prédiction des deltas.
            Par défaut ['ETP_Q', 'T_Q'].

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """

    if features is None:
        features = ['ETP_Q', 'T_Q']

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Calcul précis du temps écoulé en jours (même si c'est mensuel)
    # total_seconds / 86400 donne le nombre exact de jours entre deux relevés
    df['days_diff'] = df['time'].diff().dt.total_seconds() / 86400
    df['delta_niveaux'] = df[valeur_de_travail].diff()
    df['delta_per_day'] = df['delta_niveaux'] / df['days_diff']

    # Nettoyage pour l'entraînement
    # On ne garde que les lignes où on a à la fois la cible et les features
    train_df = df.dropna(subset=[valeur_de_travail, 'delta_per_day'] + features)

    if len(train_df) < 3:
        # Si trop peu de données, on renvoie des NaN (le script passera au suivant)
        return np.full(len(df), np.nan)

    # Modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[features], train_df['delta_per_day'])

    # Prédiction des deltas (on remplace les features manquantes par la moyenne)
    X_all = df[features].fillna(train_df[features].mean())
    pred_delta_per_day = model.predict(X_all)

    # Reconstruction itérative du signal
    y_reconstruit = df[valeur_de_travail].to_numpy().copy()

    # Si la 1ère valeur est manquante, on cherche la 1ère connue pour reculer ou on prend la moyenne
    if np.isnan(y_reconstruit[0]):
        first_valid_idx = df[valeur_de_travail].first_valid_index()
        if first_valid_idx is not None:
            y_reconstruit[0] = df[valeur_de_travail].iloc[first_valid_idx]
        else:
            y_reconstruit[0] = train_df[valeur_de_travail].mean()

    for i in range(1, len(y_reconstruit)):
        if np.isnan(y_reconstruit[i]):
            # On applique : Valeur_précédente + (Pente_prédite * Nb_jours_réels)
            d_days = df['days_diff'].iloc[i]
            y_reconstruit[i] = y_reconstruit[i-1] + (pred_delta_per_day[i] * d_days)

    return y_reconstruit

def knn_impute(df:pd.DataFrame, valeur_de_travail:str, k:int=5, past_only:bool=False)->np.ndarray:
    """
    Impute les valeurs manquantes d'une colonne cible en utilisant la méthode
    des K plus proches voisins (KNN) avec normalisation des caractéristiques.
    
    La fonction peut restreindre l'imputation aux valeurs passées uniquement 
    (option `past_only`) pour respecter l'ordre temporel.

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé
        k (int, optional): Nombre de voisins à considérer 
            Par défaut : 5.
        past_only (bool, optional): Si True, seuls les points temporellement 
            antérieurs à la ligne à remplir sont utilisés pour l'imputation. 
            Par défaut : False.

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """
    df = df.copy()
    
    # Préparation des features
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    
    df = df.sort_values(["code_bss", "time"])
    
    features = ["lon", "lat", "year", "month_sin", "month_cos", 
                ]

    # Séparation et Nettoyage
    # On définit 'complete' : les lignes où on a la cible ET toutes les features
    # C'est crucial pour le StandardScaler
    mask_complete = df[valeur_de_travail].notna() & df[features].notna().all(axis=1)
    complete = df[mask_complete].copy()
    missing = df[df[valeur_de_travail].isna()].copy()

    # Sécurité : Si pas assez de points pour le KNN
    if len(complete) < k:
        # On remplit avec la moyenne simple du fichier pour éviter le crash
        return df.fillna({valeur_de_travail: df[valeur_de_travail].mean()})

    # Normalisation
    scaler = StandardScaler()
    # .values pour éviter le warning "feature names"
    complete_scaled = scaler.fit_transform(complete[features].values)

    # Imputation
    for idx in missing.index:
        # On récupère les features de la ligne à remplir
        row_data = df.loc[[idx], features]
        
        # Si la ligne manque de features (ex: pas de lag), on passe ou on met la moyenne
        if row_data.isna().any().any():
            df.loc[idx, valeur_de_travail] = complete[valeur_de_travail].mean()
            continue
            
        row_scaled = scaler.transform(row_data.values)
        
        # Filtrage temporel si demandé
        if past_only:
            time_mask = complete["time"] < df.loc[idx, "time"]
            valid_subset = complete[time_mask]
            valid_scaled_subset = complete_scaled[time_mask.values]
            
            if len(valid_subset) < k: # Pas assez de passé ? On prend ce qu'on a
                if len(valid_subset) == 0: continue
                current_k = len(valid_subset)
            else:
                current_k = k
        else:
            valid_subset = complete
            valid_scaled_subset = complete_scaled
            current_k = k
        
        # Calcul de la distance Euclidienne : $d = \sqrt{\sum (x_i - y_i)^2}$
        distances = np.sqrt(((valid_scaled_subset - row_scaled)**2).sum(axis=1))
        k_idx = np.argsort(distances)[:current_k]
        
        # Moyenne des K plus proches voisins
        df.loc[idx, valeur_de_travail] = valid_subset.iloc[k_idx][valeur_de_travail].mean()
    
    return df[valeur_de_travail].to_numpy()

def bootstrap_saisonnier_impute(df:pd.DataFrame, valeur_de_travail:str)->np.ndarray:
    """
    Impute les valeurs manquantes d'une colonne temporelle en utilisant un profil 
    saisonnier basé sur la médiane historique par jour de l'année.

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """
        
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    
    # Extraire le jour et le mois pour définir la "saison"
    df['day_of_year'] = df['time'].dt.dayofyear
    
    # Créer le profil historique (la signature de la station)
    # On calcule la médiane pour chaque jour de l'année sur toutes les années dispos
    profil_saisonnier = df.groupby('day_of_year')[valeur_de_travail].median()
    
    # Sécurité : si certains jours n'ont AUCUNE donnée historique, 
    # on lisse le profil avec une moyenne glissante
    profil_saisonnier = profil_saisonnier.interpolate(method='linear').fillna(df[valeur_de_travail].mean())
    
    # Imputation
    # On crée une Series qui contient la valeur du profil pour chaque ligne du DF original
    df['valeur_type'] = df['day_of_year'].map(profil_saisonnier)
    
    # On ne remplit que les vides
    df[valeur_de_travail] = df[valeur_de_travail].fillna(df['valeur_type'])
    
    return df[valeur_de_travail].to_numpy()