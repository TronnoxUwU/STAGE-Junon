import pandas as pd

def load_dataset(fichier_nappe):
    df = pd.read_csv(fichier_nappe, sep=";")
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time')