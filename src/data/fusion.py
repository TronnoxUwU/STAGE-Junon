import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import os

def nearest_point_valid(df_points, df_target, value_column):
    df_valid = df_points.dropna(subset=[value_column])

    if len(df_valid) == 0:
        return None, None

    tree = cKDTree(df_valid[["lat", "lon"]].values)
    _, idx = tree.query(df_target[["lat", "lon"]].values)

    return df_valid, idx

def load_meteo(path):
    df = pd.read_csv(path, sep=";")

    df = df[["AAAAMM", "RR", "TMM", "ETP", "LAT", "LON"]]
    df = df.rename(columns={
        "AAAAMM": "time",
        "RR": "PRELIQ_Q",
        "TMM": "T_Q",
        "LAT": "lat",
        "LON": "lon"
    })

    df["time"] = pd.to_datetime(df["time"].astype(str), format="%Y%m")
    # df["PRELIQ_Q"] /= df["time"].dt.days_in_month on veut le total au mois et non la moyenne.

    return df.sort_values("time").reset_index(drop=True)


def load_etp(path):
    df = pd.read_csv(path, sep=";")

    df["time"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d")

    df = df.rename(columns={
        "lat_dg": "lat",
        "lon_dg": "lon",
        "ETP_Q_H0175": "ETP_Q"
    })

    df["month"] = df["time"].dt.to_period("M")

    df_month = df.groupby(["lat", "lon", "month"]).agg({
        "ETP_Q": "sum"
    }).reset_index()

    df_month["time"] = df_month["month"].dt.to_timestamp()

    return df_month.drop(columns=["month"])


def load_imperm(path):
    df = pd.read_csv(path, sep=";")
    df["time"] = pd.to_datetime(df["time"].astype(int).astype(str) + "-01-01")
    return df

def preprocess_nappe(df):
    df = df.rename(columns={"date_mesure": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.to_period("M")

    df_month = df.groupby(["code_bss", "month"]).agg({
        "niveau_nappe_eau": "mean",
        "lon": "first",
        "lat": "first"
    }).reset_index()

    df_month["time"] = df_month["month"].dt.to_timestamp()
    df_month = df_month.drop(columns="month")

    return complete_time_series(df_month)


def complete_time_series(df):
    date_min = df["time"].min()
    date_max = df["time"].max()
    all_months = pd.date_range(start=date_min, end=date_max, freq='MS')

    bss_code = df["code_bss"].iloc[0]
    lat_ref = df["lat"].iloc[0]
    lon_ref = df["lon"].iloc[0]

    df = df.set_index("time").reindex(all_months).reset_index()
    df = df.rename(columns={"index": "time"})

    df["code_bss"] = bss_code
    df["lat"] = df["lat"].fillna(lat_ref)
    df["lon"] = df["lon"].fillna(lon_ref)

    return df

def merge_spatial_data(nappe_month, meteo, etp, imperm):
    nappe_month["ETP_Q"] = np.nan
    nappe_month["PRELIQ_Q"] = np.nan
    nappe_month["T_Q"] = np.nan

    for d in nappe_month["time"].unique():
        mask = nappe_month["time"] == d
        df_day = nappe_month[mask]

        etp_day = etp[etp["time"] == d]
        meteo_day = meteo[meteo["time"] == d]

        # --- ETP ---
        if len(etp_day) > 0:
            df_valid, idx = nearest_point_valid(etp_day, df_day, "ETP_Q")
            if df_valid is not None:
                nappe_month.loc[mask, "ETP_Q"] = df_valid.iloc[idx]["ETP_Q"].values

        # --- METEO ---
        if len(meteo_day) > 0:
            for var in ["PRELIQ_Q", "T_Q"]:
                df_valid, idx = nearest_point_valid(meteo_day, df_day, var)
                if df_valid is not None:
                    nappe_month.loc[mask, var] = df_valid.iloc[idx][var].values

        # --- IMPERM ---
        year = pd.Timestamp(d).year
        ref_year = 2020 if year < 2023 else 2023

        imperm_year = imperm[imperm["time"].dt.year == ref_year]

        if len(imperm_year) > 0:
            df_valid, idx = nearest_point_valid(imperm_year, df_day, "surface_imp")

            if df_valid is not None:
                nappe_month.loc[mask, "surface_imp"] = df_valid.iloc[idx]["surface_imp"].values
                nappe_month.loc[mask, "surface_totale"] = df_valid.iloc[idx]["surface"].values

    return nappe_month

def process_nappe_file(filepath, meteo, etp, imperm):

    nappe = pd.read_csv(filepath, sep=";")
    nappe_month = preprocess_nappe(nappe)

    nappe_month = merge_spatial_data(nappe_month, meteo, etp, imperm)

    return nappe_month


def save_output(df, output_folder):
    code = df["code_bss"].iloc[0].replace("/", "_")
    filename = f"data_{code}.csv"

    df.to_csv(os.path.join(output_folder, filename), sep=";", index=False)

if __name__ == "__main__":
    
    FICHIER_METEO = "../../data/extraction/meteo.csv"
    FICHIER_ETP = "../../data/extraction/etp.csv"
    FICHIER_IMPERM = "../../data/extraction/impermeabilite.csv"
    DOSSIER_NAPPE = "../../data/extraction/nappes/"
    OUTPUT_FOLDER = "../../data/fusion"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    meteo = load_meteo(FICHIER_METEO)
    etp = load_etp(FICHIER_ETP)
    imperm = load_imperm(FICHIER_IMPERM)

    total = len(os.listdir(DOSSIER_NAPPE))
    fichiers_csv = [f for f in os.listdir(DOSSIER_NAPPE) if f.endswith(".csv")]

    for i, fichier in enumerate(fichiers_csv, start=1):
        
        filepath = os.path.join(DOSSIER_NAPPE, fichier)

        print(f"[TRAITEMENT][{i},{total}]", filepath)

        nappe_month = process_nappe_file(filepath, meteo, etp, imperm)
        save_output(nappe_month, OUTPUT_FOLDER)
