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