import pandas as pd
import numpy as np

def interpolation_lineaire_array(df:pd.DataFrame, valeur_de_travail:str)->np.ndarray:
    """Interpole linéairement la colonne `valeur_de_travail`

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='linear', limit_direction='both').to_numpy()

def interpolation_cubique_array(df:pd.DataFrame, valeur_de_travail:str)->np.ndarray:
    """Interpole cubiquement la colonne `valeur_de_travail`

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='cubic').to_numpy()

def interpolation_polynomiale_array(df:pd.DataFrame, valeur_de_travail:str, deg:int=2)->np.ndarray:
    """Interpole polynomialement la colonne `valeur_de_travail`

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé
        deg (int): Degré de la polynomiale

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method="polynomial", order=deg).to_numpy()

def interpolation_spline_array(df:pd.DataFrame, valeur_de_travail:str, deg:int=3)->np.ndarray:
    """Interpole via une spline polynomiale par moindres carrés la colonne `valeur_de_travail`

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé
        deg (int): Degré de la spline

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method="spline", order=deg).to_numpy()

def interpolation_pchip_array(df:pd.DataFrame, valeur_de_travail:str)->np.ndarray:
    """Interpole via PCHIP la colonne `valeur_de_travail`

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='pchip').to_numpy()

def interpolation_akima_array(df:pd.DataFrame, valeur_de_travail:str)->np.ndarray:
    """Interpole via Akima la colonne `valeur_de_travail`

    Args:
        df (pd.DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        np.ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='akima').to_numpy()