import os
import requests
import pandas as pd
import gzip
import zipfile
import time

BASE_URL = "https://www.data.gouv.fr/api/1/datasets"

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)


def fetch_dataset_resources(dataset_id:str):
    url = f"{BASE_URL}/{dataset_id}/"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("resources", [])


def download_files(resources, dest_folder, formats=("csv",), filename=None):
    files = []

    for r in resources:
        url = r.get("url")
        fmt = r.get("format", "").lower()

        if fmt in formats:
            # Si un nom est fourni → on l'utilise, sinon nom original
            if filename:
                final_name = filename
            else:
                final_name = url.split("/")[-1]

            filepath = os.path.join(dest_folder, final_name)

            print(f"[DOWNLOAD] {url}")
            resp = requests.get(url)
            resp.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(resp.content)

            files.append(filepath)

    return files


def read_csv_flexible(path, sep=";"):
    """
    Gère:
    - csv
    - csv.gz (gz ou zip mal encodé)
    """
    if path.endswith(".csv"):
        return pd.read_csv(path, sep=sep)

    if path.endswith(".csv.gz"):
        try:
            with zipfile.ZipFile(path, "r") as z:
                for name in z.namelist():
                    if name.endswith(".csv"):
                        with z.open(name) as f:
                            return pd.read_csv(f, sep=sep)
        except:
            with gzip.open(path, "rt") as f:
                return pd.read_csv(f, sep=sep)

    return None

def process_impermeabilite(dataset_id, tmp_folder, output_folder, communes_file, name, departements):

    folder = f"{tmp_folder}/{name}"
    ensure_dir(folder)

    resources = fetch_dataset_resources(dataset_id)
    download_files(resources, folder, formats=("csv",))

    dfs = []
    for file in os.listdir(folder):
        if file.endswith("commune.csv"):
            path = os.path.join(folder, file)
            df = pd.read_csv(path, dtype={"commune_code": str})
            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    communes = pd.read_csv(communes_file, dtype={"Code Insee": str})
    
    dfs = []
    for num in departements:
        dfs.append(communes[communes["Code postal"].astype(int).between(num*1000, (num+1)*1000 - 1)])
    
    communes = pd.concat(dfs, ignore_index=True)
    communes = communes[["Code Insee", "latitude", "longitude"]]

    df_all["commune_code"] = df_all["commune_code"].str.zfill(5)
    communes["Code Insee"] = communes["Code Insee"].str.zfill(5)

    df = df_all.merge(
        communes,
        left_on="commune_code",
        right_on="Code Insee",
        how="inner"
    )

    # reshape propre avec melt
    df1 = df.rename(columns={
        "millesimes_1": "time",
        "surface_imper_1": "surface_imp"
    })[["latitude", "longitude", "time", "surface_imp", "commune_surface"]]

    df2 = df.rename(columns={
        "millesimes_2": "time",
        "surface_imper_2": "surface_imp"
    })[["latitude", "longitude", "time", "surface_imp", "commune_surface"]]

    df_final = pd.concat([df1, df2], ignore_index=True)
    df_final.columns = ["lat", "lon", "time", "surface_imp", "surface"]

    df_final = df_final.sort_values(["time", "lat", "lon"])

    out = f"{output_folder}/{name}.csv"
    df_final.to_csv(out, sep=";", index=False)

    print(f"✔ {name} : {len(df_final)} lignes")


def process_etp(dataset_id, tmp_folder, output_folder, name, maille_file):

    folder = f"{tmp_folder}/{name}"
    ensure_dir(folder)

    resources = fetch_dataset_resources(dataset_id)
    files = download_files(resources, folder, formats=("csv", "csv.gz"))

    maille_df = pd.read_csv(maille_file, sep=";")

    mailles = set(maille_df["num_maille"])
    maille_xy = maille_df[["lambx", "lamby", "lat_dg", "lon_dg"]]

    dfs = []

    for file in files:
        df = read_csv_flexible(file)

        if df is None:
            continue

        if "NUM_MAILLE" in df.columns:
            df = df[df["NUM_MAILLE"].isin(mailles)]

        elif {"LAMBX", "LAMBY"}.issubset(df.columns):
            df = df.merge(
                maille_xy,
                left_on=["LAMBX", "LAMBY"],
                right_on=["lambx", "lamby"],
                how="inner"
            )

        dfs.append(df)

    if not dfs:
        print("❌ Aucun fichier ETP")
        return

    df = pd.concat(dfs, ignore_index=True)
    df = df[["DATE", "ETP_Q_H0175", "lat_dg", "lon_dg"]].sort_values("DATE")

    out = f"{output_folder}/{name}.csv"
    df.to_csv(out, sep=";", index=False)

    print(f"✔ {name} : {len(df)} lignes")

def process_meteo(dataset_id, tmp_folder, output_folder, name, departements):

    folder = f"{tmp_folder}/{name}"
    ensure_dir(folder)

    resources = fetch_dataset_resources(dataset_id)

    urls = [r["url"] for r in resources for num in departements if f"_{num}" in (r["title"] + r["url"]).lower()]

    if not urls:
        print("❌ Aucun fichier météo")
        return

    dfs = []

    for i, url in enumerate(urls):
        print(f"[DOWNLOAD] {i+1}/{len(urls)}")

        file = os.path.join(folder, f"tmp_{i}.csv.gz")

        r = requests.get(url)
        r.raise_for_status()

        with open(file, "wb") as f:
            f.write(r.content)

        df = pd.read_csv(file, sep=";", compression="gzip")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    out = f"{output_folder}/{name}.csv"
    df.to_csv(out, sep=";", index=False)

    print(f"✔ {name} : {len(df)} lignes")

def process_nappe(output_folder, name, departements):
    folder = f"{output_folder}/{name}"
    ensure_dir(folder)

    url_stations = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations"

    codes = []
    for departement in departements :
        page = 1
        # Récupération paginée des stations du département
        while True:
            params = {
                "code_departement": departement,
                "size": 200,
                "page": page
            }
            
            r = requests.get(url_stations, params=params)
            data = r.json()["data"]
            
            if not data:
                break

            # Stockage des codes des stations et des coordonnées
            codes.extend([(s["code_bss"],s["x"],s["y"]) for s in data])
            print(f"Page {page} : {len(data)} stations")
            page += 1

    # Suppression des doublons
    codes = list(set(codes))
    print(f"Total stations trouvées : {len(codes)}")

    url_chroniques = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques"

    # Téléchargement des chroniques pour chaque station
    for i, (code, lat, lon) in enumerate(codes):
        params = {
            "code_bss": code,
            "size": 20000
        }
        
        try:
            r = requests.get(url_chroniques, params=params)
            data = r.json()["data"]
            
            if not data:
                print(f"[{i+1}/{len(codes)}] {code} : aucune donnée")
                continue
            
            df = pd.DataFrame(data)
            df["lat"] = lat
            df["lon"] = lon
            
            # Nettoyage du nom pour le fichier
            safe_code = code.replace("/", "_")
            file_path = os.path.join(f"{output_folder}/{name}", f"{name}_{safe_code}.csv")
            
            df.to_csv(file_path, sep=";", index=False)
            print(f"[DOWNLOAD][{i+1}/{len(codes)}] {code} : {len(df)} lignes")
            
            # Pause pour éviter de surcharger l'API
            time.sleep(0.2)
            
        except Exception as e:
            print(f"[{i+1}/{len(codes)}] Erreur {code} : {e}")
    print(f"✔ {name} : {len(codes)} lignes")


def download_communes_csv(dest_folder: str, name:str = "communes"):
    ensure_dir(dest_folder)

    # Identifiant du dataset (dataset id sur data.gouv.fr)
    dataset_id = "6769a388c3b64f95ea639e27"

    # Récupération des ressources via la fonction existante
    resources = fetch_dataset_resources(dataset_id)

    # Télécharger uniquement les fichiers CSV
    download_files(resources, dest_folder, formats=("csv",), filename=f"{name}.csv")

def download_maille_csv(folder:str, name:str = "maille"):
    url = "https://donneespubliques.meteofrance.fr/client/document/metadonnees_swi_276.csv"
    print(f"[DOWNLOAD] {url}")
    df = pd.read_csv(url, sep=";", skiprows=4)

    df.columns = df.columns.str.replace("#", "").str.strip()

    # Sauvegarde propre
    df.to_csv(f"{folder}/{name}.csv", sep=";", index=False)


if __name__ == "__main__":
    # Dossier de sortie pour les données finales
    output_folder = "data/extraction/"

    # Dossier temporaire pour les fichiers téléchargés / compressés
    tmp_folder = "data/extraction/tmp/"

    # Préfixes utilisés pour nommer les différents jeux de données
    meteo_name = "meteo"
    nappe_name = "nappes"
    etp_name = "etp"
    impermeabilite_name = "impermeabilite"

    download_communes_csv("./data")
    download_maille_csv("./data")

    process_impermeabilite(
        dataset_id="697b4f4ceea77fb452ba9d6d",
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        communes_file="data/communes.csv",
        name=impermeabilite_name,
        departements=[45]
    )

    process_etp(
        dataset_id="667eae35510cd549fc7722c1",
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        name=etp_name,
        maille_file="data/maille.csv"
    )

    process_meteo(
        dataset_id="6569b3d7d193b4daf2b43edc",
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        name=meteo_name,
        departements=[45]
    )

    process_nappe(
        output_folder=output_folder,
        name=nappe_name,
        departements=[45]
    )
