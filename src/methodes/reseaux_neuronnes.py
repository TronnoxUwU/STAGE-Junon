import os
os.environ["KERAS_BACKEND"] = "torch"
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Conv1D, MaxPooling1D
import torch
import keras

@keras.saving.register_keras_serializable()
def masked_mse(y_true, y_pred):
    # On crée un masque : True là où la donnée n'est pas la valeur sentinelle
    mask = torch.logical_not(torch.eq(y_true, -999.0))
    
    # On ne garde que les valeurs valides
    y_true_masked = torch.masked_select(y_true, mask)
    y_pred_masked = torch.masked_select(y_pred, mask)
    
    # Calcul de l'erreur classique (MSE) sur ce qu'il reste
    return torch.mean(torch.square(y_true_masked - y_pred_masked))

def CNN(X_train, y_train, X_val, y_val ):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=3,
                                         restore_best_weights=True)


    model = Sequential()
    model.add(Conv1D(64, 3, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(LSTM(128))
    model.add(Dense(X_train.shape[2])) 

    model.compile(optimizer='adam', loss=masked_mse)

    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_val, y_val),
        callbacks=[callback],
        verbose=1
    )

    model.save("models/CNN.keras")

def lstm(X_train, y_train, X_val, y_val):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=5,
                                            restore_best_weights=True)

    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation="tanh"))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="tanh"))
    model.add(Dense(12, activation="tanh"))
    model.add(Dense(X_train.shape[2])) 

    model.compile(optimizer='adam', loss=masked_mse)

    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_val, y_val),
        callbacks=[callback],
        verbose=1
    )

    model.save("models/LSTM.keras")

def bilstm(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(X_train.shape[2])) 

    model.compile(optimizer='adam', loss=masked_mse)

    history = model.fit(
        X_train, y_train, 
        epochs=30, 
        batch_size=128, 
        validation_data=(X_val, y_val),
        verbose=1
    )

    model.save("models/BILSTM.keras")


# def charger_et_preparer_donnees(dossier_path, features):
#     # 1. Chargement global (comme tu faisais)
#     fichiers_csv = glob.glob(os.path.join(dossier_path, "*.csv"))
#     liste_df = [pd.read_csv(f, sep=';') for f in fichiers_csv]
#     df_final = pd.concat(liste_df, ignore_index=True)
    
#     # 2. Nettoyage et Tri
#     df_final['time'] = pd.to_datetime(df_final['time'])
#     # On trie par station PUIS par temps pour que les chronologies soient respectées
#     df_final = df_final.sort_values(['code_bss', 'time'])
    
#     df_travail = df_final.copy()
#     # Conversion du temps pour le scaler
#     df_travail['time_num'] = df_travail['time'].astype('int64') // 10**9
    
#     # On ajuste la liste des features si besoin (on utilise time_num au lieu de time)
#     features_scaler = [f if f != 'time' else 'time_num' for f in features]
    
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     df_travail[features_scaler] = scaler.fit_transform(df_travail[features_scaler])
    
#     return df_travail, scaler

# def creer_sequences_par_bss(df, features, window_size):
#     X_list, y_list = [], []
    
#     # On groupe par code_bss pour traiter chaque piézomètre séparément
#     for _, group in df.groupby('code_bss'):
#         data = group[features].values
        
#         # On ne crée des fenêtres que si la station a assez de données
#         if len(data) > window_size:
#             for i in range(window_size, len(data)):
#                 X_list.append(data[i-window_size:i, :])
#                 y_list.append(data[i, :])
    
#     return np.array(X_list), np.array(y_list)

# # --- EXECUTION ---

# # 1. Préparation
# features = ["niveau_nappe_eau","lon","lat","time","ETP_Q","PRELIQ_Q","T_Q","surface_imp","surface_totale"]
# df_norm, mon_scaler = charger_et_preparer_donnees(dossier_nappe, features)

# # 2. Création des fenêtres étanches (6 mois ici)
# features_pour_ia = [f if f != 'time' else 'time_num' for f in features]
# X, y = creer_sequences_par_bss(df_norm, features_pour_ia, window_size=24)

# # 3. Gestion des NaN pour la Masked Loss
# X = np.nan_to_num(X, nan=-999.0)
# y = np.nan_to_num(y, nan=-999.0)

# # 4. Split Train/Val
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

# def lstm_predict_array(df_entree, model, scaler, features, window_size, target_col="T_Q"):
#     df_travail = df_entree.copy()
    
#     # 1. Préparation du temps : On écrase 'time' par sa version numérique
#     # pour que le nom de la colonne soit identique à celui vu au 'fit'
#     if 'time' in df_travail.columns:
#         df_travail['time'] = pd.to_datetime(df_travail['time']).astype('int64') // 10**9
    
#     # 2. On s'assure de ne passer QUE les colonnes du scaler, dans le bon ordre
#     # Le scaler attend les noms : ["niveau_nappe_eau", ..., "time", ...]
#     df_pour_scaler = df_travail[features]
    
#     # 3. Transformation (maintenant les noms correspondent)
#     values_norm = scaler.transform(df_pour_scaler)
    
#     # 4. Gestion des NaN pour le modèle
#     values_masked = np.nan_to_num(values_norm, nan=-999.0)
    
#     # 5. Création des séquences
#     X_all = []
#     for i in range(window_size, len(values_masked)):
#         X_all.append(values_masked[i-window_size:i, :])
#     X_all = np.array(X_all, dtype='float32')
    
#     if len(X_all) == 0:
#         return np.full(len(df_entree), np.nan)
    
#     # 6. Prédiction
#     y_pred_norm = model.predict(X_all, verbose=0)
    
#     # 7. Inverse Transform
#     y_pred_final = scaler.inverse_transform(y_pred_norm)
    
#     # 8. Extraction avec l'index dynamique
#     target_idx = features.index(target_col)
    
#     # 9. Reconstruction
#     full_array = np.full(len(df_entree), np.nan)
#     full_array[window_size:] = y_pred_final[:, target_idx]
    
#     return full_array