from numpy import ndarray
from pandas import DataFrame

def interpolation_lineaire_array(
    df:DataFrame, 
    valeur_de_travail:str
)->ndarray:
    """Interpole linéairement la colonne `valeur_de_travail`

    Args:
        df (DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='linear', limit_direction='both').to_numpy()

def interpolation_cubique_array(
    df:DataFrame, 
    valeur_de_travail:str
)->ndarray:
    """Interpole cubiquement la colonne `valeur_de_travail`

    Args:
        df (DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='cubic').to_numpy()

def interpolation_polynomiale_array(
    df:DataFrame, 
    valeur_de_travail:str, 
    deg:int=2
)->ndarray:
    """Interpole polynomialement la colonne `valeur_de_travail`

    Args:
        df (DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé
        deg (int): Degré de la polynomiale

    Returns:
        ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method="polynomial", order=deg).to_numpy()

def interpolation_spline_array(
    df:DataFrame, 
    valeur_de_travail:str, 
    deg:int=3
)->ndarray:
    """Interpole via une spline polynomiale par moindres carrés la colonne `valeur_de_travail`

    Args:
        df (DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé
        deg (int): Degré de la spline

    Returns:
        ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method="spline", order=deg).to_numpy()

def interpolation_pchip_array(
    df:DataFrame, 
    valeur_de_travail:str
)->ndarray:
    """Interpole via PCHIP la colonne `valeur_de_travail`

    Args:
        df (DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='pchip').to_numpy()

def interpolation_akima_array(
    df:DataFrame, 
    valeur_de_travail:str
)->ndarray:
    """Interpole via Akima la colonne `valeur_de_travail`

    Args:
        df (DataFrame): Dataset à compléter
        valeur_de_travail (str): Valeur que l'on veut interpolé

    Returns:
        ndarray: Le dataset complété sur la valeur demandée
    """

    return df[valeur_de_travail].interpolate(method='akima').to_numpy()