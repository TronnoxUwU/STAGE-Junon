import matplotlib.pyplot as plt

def afficher_apprentissage(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('Évolution de l\'erreur pendant l\'entraînement')
    plt.xlabel('Époques')
    plt.ylabel('Erreur (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def affiche_prediction(df, fonction, valeur_de_travail):
    plt.figure(figsize=(12,6))
    plt.plot(df['time'], fonction(df, valeur_de_travail), label="Prédiction")
    plt.plot(df['time'], df[valeur_de_travail], label="Mesure")
    plt.xlabel("Date")
    plt.ylabel(valeur_de_travail)
    plt.legend()
    plt.show()