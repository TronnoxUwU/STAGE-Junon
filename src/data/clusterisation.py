import numpy as np

def classifier_nappe_fluctuation(df, col="niveau_nappe_eau"):

    niveau = df[col].dropna()

    # variations mensuelles
    variations = niveau.diff().dropna()

    variation_moy = variations.abs().mean()
    variabilite = variations.std()

    # pente moyenne (tendance globale)
    x = np.arange(len(niveau))

    # indice de dynamique
    indice = variation_moy + variabilite

    # classification
    if indice > 1.0:
        type_nappe = "nappe réactive"
    elif indice > 0.4:
        type_nappe = "nappe intermédiaire"
    else:
        type_nappe = "nappe inertielle"

    return {
        "indice_dynamique": indice,
        "type_nappe": type_nappe
    }