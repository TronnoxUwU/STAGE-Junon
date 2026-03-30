import numpy as np

def classifier_nappe_fluctuation(df, col="niveau_nappe_eau"):

    niveau = df[col].dropna()

    # variations mensuelles
    variations = niveau.diff().dropna()

    variation_moy = variations.abs().mean()
    variabilite = variations.std()

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

def classifier_par_duree(df, col="niveau_nappe_eau"):
    df = df = df.copy().dropna(subset=[col])
    
    grouped = df.groupby("code_bss")
    
    resultats = {
        ">30 ans": 0,
        ">20 ans": 0,
        ">10 ans": 0,
        "<10 ans": 0
    }
    
    for _, group in grouped:
        duree = (group["time"].max() - group["time"].min()).days / 365.25
        
        if duree > 30:
            resultats[">30 ans"] += 1
        elif duree > 20:
            resultats[">20 ans"] += 1
        elif duree > 10:
            resultats[">10 ans"] += 1
        else:
            resultats["<10 ans"] += 1

    return resultats

def classifier_par_consecutif(df, col="niveau_nappe_eau"):
    df = df.copy().dropna(subset=[col])
    
    grouped = df.groupby("code_bss")
    
    resultats = {
        ">30 ans": 0,
        ">20 ans": 0,
        ">10 ans": 0,
        "<10 ans": 0
    }
    
    for _, group in grouped:
        dates = group["time"].sort_values().drop_duplicates()
        
        if len(dates) == 0:
            continue
        
        max_streak = 1
        current_streak = 1
        
        for i in range(1, len(dates)):
            diff = (dates.iloc[i].year - dates.iloc[i-1].year) * 12 + \
                   (dates.iloc[i].month - dates.iloc[i-1].month)
            
            if diff == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1

        if max_streak > 30:
            resultats[">30 ans"] += 1
        elif max_streak > 20:
            resultats[">20 ans"] += 1
        elif max_streak > 10:
            resultats[">10 ans"] += 1
        else:
            resultats["<10 ans"] += 1

    
    return resultats