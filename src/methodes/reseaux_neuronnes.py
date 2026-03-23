import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping, History
import torch
import keras
from numpy import ndarray

@keras.saving.register_keras_serializable()
def masked_mse(
    y_true:ndarray, 
    y_pred:ndarray
)->torch.Tensor:
    """Fonction qui calcul la MSE d'un model pour son entrainement.

    Args:
        y_true (ndarray): Résultats réelles attendus
        y_pred (ndarray): Résultats prédits par le model

    Returns:
        Tensor: Tensor de MSE
    """    
    # On crée un masque : True là où la donnée n'est pas la valeur sentinelle
    mask = torch.logical_not(torch.eq(y_true, -999.0))
    
    # On ne garde que les valeurs valides
    y_true_masked = torch.masked_select(y_true, mask)
    y_pred_masked = torch.masked_select(y_pred, mask)
    
    # Calcul de l'erreur classique (MSE) sur ce qu'il reste
    return torch.mean(torch.square(y_true_masked - y_pred_masked))

def fit(
    model:Sequential, 
    X_train:ndarray,
    y_train:ndarray, 
    X_val:ndarray, 
    y_val:ndarray, 
    callback:EarlyStopping=None
) -> History:
    """Fonction d'entrainement d'un model

    Args:
        model (Sequential): Le model à entrainer
        X_train (ndarray): Données d'entré d'entrainement
        y_train (ndarray): Resultat réelles attendus d'entrainement
        X_val (ndarray): Données d'entré d'évalutation
        y_val (ndarray): Resultat réelles attendus d'évalutation
        callback (EarlyStopping, optional): EarlyStopping pour evité l'Over fitting. Defaults to None.

    Returns:
        History: Historique de l'entrainement du model
    """    
    if callback is None :
        callback = EarlyStopping(monitor='val_loss',
                                patience=3,
                                restore_best_weights=True)
    return model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_val, y_val),
        callbacks=[callback],
        verbose=1
    )

def cnn(
    X_train:ndarray,
    y_train:ndarray,
    X_val:ndarray,
    y_val:ndarray,
    dossier_model:str
) -> History:
    """Génère un model cnn

    Args:
        X_train (ndarray): Données d'entré d'entrainement
        y_train (ndarray): Resultat réelles attendus d'entrainement
        X_val (ndarray): Données d'entré d'évalutation
        y_val (ndarray): Resultat réelles attendus d'évalutation
        dossier_model (str): Dossier ou sera enregistrer le model

    Returns:
        History: Historique de l'entrainement du model
    """    
    model = Sequential()
    model.add(Conv1D(64, 3, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(LSTM(128))
    model.add(Dense(X_train.shape[2])) 

    model.compile(optimizer='adam', loss=masked_mse)

    history = fit(model, X_train, y_train, X_val, y_val)

    model.save(dossier_model + "./CNN.keras")

    return history

def lstm(
    X_train:ndarray, 
    y_train:ndarray, 
    X_val:ndarray, 
    y_val:ndarray, 
    dossier_model:str
) -> History:
    """Génère un model lstm

    Args:
        X_train (ndarray): Données d'entré d'entrainement
        y_train (ndarray): Resultat réelles attendus d'entrainement
        X_val (ndarray): Données d'entré d'évalutation
        y_val (ndarray): Resultat réelles attendus d'évalutation
        dossier_model (str): Dossier ou sera enregistrer le model

    Returns:
        History: Historique de l'entrainement du model
    """    
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(X_train.shape[2])) 

    model.compile(optimizer='adam', loss=masked_mse)

    history = fit(model, X_train, y_train, X_val, y_val)

    model.save(dossier_model + "LSTM.keras")

    return history

def bilstm(
    X_train:ndarray, 
    y_train:ndarray, 
    X_val:ndarray, 
    y_val:ndarray, 
    dossier_model:str
) -> History:
    """Génère un model bilstm

    Args:
        X_train (ndarray): Données d'entré d'entrainement
        y_train (ndarray): Resultat réelles attendus d'entrainement
        X_val (ndarray): Données d'entré d'évalutation
        y_val (ndarray): Resultat réelles attendus d'évalutation
        dossier_model (str): Dossier ou sera enregistrer le model

    Returns:
        History: Historique de l'entrainement du model
    """    
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(X_train.shape[2])) 

    model.compile(optimizer='adam', loss=masked_mse)

    history = fit(model, X_train, y_train, X_val, y_val)

    model.save(dossier_model + "BILSTM.keras")

    return history
