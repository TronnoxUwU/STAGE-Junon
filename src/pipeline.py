from data.extraction import *
from data.fusion import *
from data.clusterisation import classifier_nappe_fluctuation
from data.loader import charger_fichier, liste_fichiers
from data.prepare import *
from methodes import *
from evaluations import *



import sys
import tomllib

import shutil

def chargement_config():
    pass

def extraction(output_folder, tmp_folder, departements, names):
    download_communes_csv(output_folder, names["communes_name_extraction"])
    download_maille_csv(output_folder, names["mailles_name_extraction"])


    process_impermeabilite(
        dataset_id="697b4f4ceea77fb452ba9d6d",
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        communes_file=f"{output_folder}/{names["communes_name_extraction"]}.csv",
        name=names["impermeabilite_name_extraction"],
        departements=departements
    )

    process_etp(
        dataset_id="667eae35510cd549fc7722c1",
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        name=names["etp_name_extraction"],
        maille_file=f"{output_folder}/{names["mailles_name_extraction"]}.csv"
    )

    process_meteo(
        dataset_id="6569b3d7d193b4daf2b43edc",
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        name=names["meteo_name_extraction"],
        departements=departements
    )

    process_nappe(
        output_folder=output_folder,
        name=names["nappe_name_extraction"],
        departements=departements
    )

def fusion(output_folder,input_folder, names):
    os.makedirs(output_folder, exist_ok=True)

    print(f"[CHARGEMENT] {input_folder}/{names["meteo_name_extraction"]}.csv")
    meteo = load_meteo(f"{input_folder}/{names["meteo_name_extraction"]}.csv")

    print(f"[CHARGEMENT] {input_folder}/{names["etp_name_extraction"]}.csv")
    etp = load_etp(f"{input_folder}/{names["etp_name_extraction"]}.csv")

    print(f"[CHARGEMENT] {input_folder}/{names["impermeabilite_name_extraction"]}.csv")
    imperm = load_imperm(f"{input_folder}/{names["impermeabilite_name_extraction"]}.csv")

    
    total = len(os.listdir(f"{input_folder}/{names["nappe_name_extraction"]}"))
    fichiers_csv = [f for f in os.listdir(f"{input_folder}/{names["nappe_name_extraction"]}") if f.endswith(".csv")]

    for i, fichier in enumerate(fichiers_csv, start=1):
        filepath = os.path.join(f"{input_folder}/{names["nappe_name_extraction"]}", fichier)

        print(f"[TRAITEMENT][{i}/{total}]", filepath)

        nappe_month = process_nappe_file(filepath, meteo, etp, imperm)
        save_output(nappe_month, output_folder)

def clusterisations(input_folder, dossier_nappe_inertielle, dossier_nappe_reactive):
    dfs = {fichier:charger_fichier(fichier) for fichier in liste_fichiers(input_folder)}

    shutil.rmtree(dossier_nappe_inertielle, ignore_errors=True)
    shutil.rmtree(dossier_nappe_reactive, ignore_errors=True)

    os.makedirs(dossier_nappe_inertielle, exist_ok=True)
    os.makedirs(dossier_nappe_reactive, exist_ok=True)

    for nom, df in dfs.items():
        if classifier_nappe_fluctuation(df)["indice_dynamique"]>0.4 :
            df.to_csv(os.path.join(dossier_nappe_reactive, nom.split("\\")[1]), sep=";", index=False)
        else :
            df.to_csv(os.path.join(dossier_nappe_inertielle, nom.split("\\")[1]), sep=";", index=False)


def methodes_completion(input_folder, ouput_folder, travail, cluster, remove_pct, troue_deb, troue_fin, summary=None):

    os.makedirs(ouput_folder, exist_ok=True)
    
    if summary is None :
        summary = pd.DataFrame(columns=["code_bss","lat","lon","cluster"])

    for file in liste_fichiers(input_folder):
        df = charger_fichier(file)

        ligne = {
            "code_bss": df["code_bss"].iloc[0],
            "lat": df["lat"].iloc[0],
            "lon": df["lon"].iloc[0],
            "cluster": cluster
        }
        
        for valeur_de_travail, methodes in travail.items():
            if methodes["methodes"] == [] or not methodes["realiser"]:
                continue
            
            df_error = df.copy()

            res = generate_missing_data(df_error, file, valeur_de_travail, remove_pct, np.random.default_rng(), troue_deb, troue_fin)
            
            if res is None :
                continue

            df_error, y_full, _ = res

            result = {"vrai": {},
                    "erreur": {}}

            if "lineaire" in methodes["methodes"]:
                result["vrai"]["lineaire"] = interpolation_lineaire_array(df, valeur_de_travail)
                result["erreur"]["lineaire"] = interpolation_lineaire_array(df_error, valeur_de_travail)
            if "cubique" in methodes["methodes"]:
                result["vrai"]["cubique"] = interpolation_cubique_array(df, valeur_de_travail)
                result["erreur"]["cubique"] = interpolation_cubique_array(df_error, valeur_de_travail)
            if "spline" in methodes["methodes"]:
                result["vrai"]["spline"] = interpolation_spline_array(df, valeur_de_travail)
                result["erreur"]["spline"] = interpolation_spline_array(df_error, valeur_de_travail)
            if "polynome" in methodes["methodes"]:
                result["vrai"]["polynome"] = interpolation_polynomiale_array(df, valeur_de_travail)
                result["erreur"]["polynome"] = interpolation_polynomiale_array(df_error, valeur_de_travail)
            if "pchip" in methodes["methodes"]:
                result["vrai"]["pchip"] = interpolation_pchip_array(df, valeur_de_travail)
                result["erreur"]["pchip"] = interpolation_pchip_array(df_error, valeur_de_travail)
            if "akima" in methodes["methodes"]:
                result["vrai"]["akima"] = interpolation_akima_array(df, valeur_de_travail)
                result["erreur"]["akima"] = interpolation_akima_array(df_error, valeur_de_travail)

            for m, res in result["erreur"].items():
                ligne[f"{m}_{valeur_de_travail}"] = nrmse(y_full,res)

                df[f"{m}_{valeur_de_travail}"] = result["vrai"][m]

            summary.loc[len(summary)] = ligne
        output_file = df["code_bss"].iloc[0].replace("/", "_")   # supposé unique par fichier
        output_file = f"data_{output_file}.csv"
        df.to_csv(f"{ouput_folder}/{output_file}", sep=";", index=False)
    return summary




def entrainement_création_NNs():
    pass

def load_config(path):
    print(f"[CHARGEMENT] {path}")
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return config

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)

    if config["pipeline"]["extraction"]:
        print("="*100)
        print("Extraction des données")
        print("="*100)
        extraction(
            config["dossier"]["dossier_extraction"],
            config["dossier"]["dossier_extraction_tmp"],
            config["pipeline"]["departements"],
            config["dossier"]
        )
        
    if config["pipeline"]["fusion"]:
        print("="*100)
        print("Fusion des données")
        print("="*100)
        fusion(
            config["dossier"]["dossier_fusion"],
            config["dossier"]["dossier_extraction"],
            config["dossier"]
        )

    if config["pipeline"]["clusterisation"]:
        print("="*100)
        print("Clusterisation des données")
        print("="*100)
        clusterisations(
            config["dossier"]["dossier_fusion"],
            config["dossier"]["dossier_nappe_inertielle"],
            config["dossier"]["dossier_nappe_reactive"]
        )

    if config["pipeline"]["completion"]:
        dossiers = []
        print("="*100)
        print("Complétion des données")
        print("="*100)
        match config["pipeline"]["type"]:
            case "reactive":
                dossiers.append(config["dossier"]["dossier_nappe_reactive"])
            case "inertielle":
                dossiers.append(config["dossier"]["dossier_nappe_inertielle"])
            case _ :
                dossiers.append(config["dossier"]["dossier_nappe_reactive"])
                dossiers.append(config["dossier"]["dossier_nappe_inertielle"])

        summary = None
        for dossier in dossiers:
            summary = methodes_completion(
                dossier,
                config["dossier"]["dossier_completion"],
                config["completion"],
                config["pipeline"]["type"],
                0.1,
                1990,1990,
                summary
            )
