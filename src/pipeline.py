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


def methodes_completion(input_folder, ouput_folder, travail, cluster, dossier_model, fichier_scaler, window_size, remove_pct, troue_deb, troue_fin, summary=None):
    model = {}
    features = ["niveau_nappe_eau","lon","lat","time_num","ETP_Q","PRELIQ_Q","T_Q","surface_imp","surface_totale"]
    for valeur_de_travail, methodes in travail.items():
        model[valeur_de_travail] = {}
        for m in methodes["methodes"]:
            if "/" in m:
                model[valeur_de_travail][m] = load_model(f"{dossier_model}/{m}.keras", custom_objects={'masked_mse': masked_mse})

    mon_scaler = joblib.load(f"{fichier_scaler}/scaler.save")

    df_all = charger_dossier(input_folder)

    os.makedirs(ouput_folder, exist_ok=True)
    
    if summary is None :
        summary = []

    for file in liste_fichiers(input_folder):
        df = charger_fichier(file)
        df['time_num'] = df['time'].astype('int64') // 10**9

        df.sort_values(by='time_num', ascending=False)

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
            if "rf" in methodes["methodes"]:
                result["vrai"]["rf"] = random_forest_delta_array(df, valeur_de_travail)
                result["erreur"]["rf"] = random_forest_delta_array(df_error, valeur_de_travail)
            if "knn" in methodes["methodes"]:
                result["vrai"]["knn"] = knn_impute(df, valeur_de_travail)
                result["erreur"]["knn"] = knn_impute(df_error, valeur_de_travail)
            if "knn_nappe" in methodes["methodes"]:
                result["vrai"]["knn"] = knn_nappe(df, df_all, valeur_de_travail)
                result["erreur"]["knn"] = knn_nappe(df_error, df_all, valeur_de_travail)
            if "bss" in methodes["methodes"]:
                result["vrai"]["bss"] = bootstrap_saisonnier_impute(df, valeur_de_travail)
                result["erreur"]["bss"] = bootstrap_saisonnier_impute(df_error, valeur_de_travail)
            # if model[valeur_de_travail] != {} or not window_size<=len(df):
                
            #     for name, m in model[valeur_de_travail].items():
            #         result["vrai"][name] = lstm_predict_array(df, m, mon_scaler, features, window_size=window_size, target_col=valeur_de_travail)
            #         result["erreur"][name] = lstm_predict_array(df_error, m, mon_scaler, features, window_size=window_size, target_col=valeur_de_travail)

            for m, res in result["erreur"].items():
                ligne[f"{m}_{valeur_de_travail}"] = nrmse(res, y_full)

                print(res, y_full)

                df[f"{m}_{valeur_de_travail}"] = result["vrai"][m]

            summary.append(ligne)
        output_file = df["code_bss"].iloc[0].replace("/", "_")   # supposé unique par fichier
        output_file = f"data_{output_file}.csv"
        df.to_csv(f"{ouput_folder}/{output_file}", sep=";", index=False) 
    return summary

def entrainement_création_NNs(dossier_nappe, window_size, fichier_scaler, dossier_model, models):
    os.makedirs(dossier_model, exist_ok=True)
    if os.path.exists(f"{fichier_scaler}/scaler.save"):
        scaler =  joblib.load(f"{fichier_scaler}/scaler.save")
    else :
        scaler = None
    X_train, X_val, y_train, y_val, scaler = train_data(charger_dossier(dossier_nappe), window_size, fichier_scaler, scaler, croissant=False)

    if "CNN" in models:
        cnn(X_train, y_train, X_val, y_val, dossier_model)
    if "LSTM" in models:
        lstm(X_train, y_train, X_val, y_val, dossier_model)
    if "BILSTM" in models:
        bilstm(X_train, y_train, X_val, y_val, dossier_model)

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

    if config["pipeline"]["entrainement"]:
        if config["entrainement_model"]["global"]:
            entrainement_création_NNs(config["dossier"]["dossier_fusion"], 
                                      config["entrainement_model"]["window_size"], 
                                      config["dossier"]["dossier_scaler"], 
                                      f"{config["dossier"]["dossier_model"]}/global/", 
                                      config["entrainement_model"]["models"])
        
        if config["entrainement_model"]["fine_tune"]:
            entrainement_création_NNs(config["dossier"]["dossier_nappe_inertielle"], 
                                      config["entrainement_model"]["window_size"], 
                                      config["dossier"]["dossier_scaler"], 
                                      f"{config["dossier"]["dossier_model"]}/inertielle/", 
                                      config["entrainement_model"]["models"])
            
            entrainement_création_NNs(config["dossier"]["dossier_nappe_reactive"], 
                                      config["entrainement_model"]["window_size"], 
                                      config["dossier"]["dossier_scaler"], 
                                      f"{config["dossier"]["dossier_model"]}/reactive/", 
                                      config["entrainement_model"]["models"])


    if config["pipeline"]["completion"]:
        dossiers = []
        print("="*100)
        print("Complétion des données")
        print("="*100)
        if "reactive" in config["pipeline"]["type"]:
            dossiers.append((config["dossier"]["dossier_nappe_reactive"],
                                config["dossier"]["dossier_completion_reactive"],
                                "reactive"))
        if "inertielle" in config["pipeline"]["type"]:
            dossiers.append((config["dossier"]["dossier_nappe_inertielle"],
                                config["dossier"]["dossier_completion_inertielle"],
                                "inertielle"))
        if config["pipeline"]["type"] == [] :
            dossiers.append((config["dossier"]["dossier_fusion"],
                                config["dossier"]["dossier_completion"],
                                ""))

        summary = None
        for input,ouput,type in dossiers:
            summary = methodes_completion(
                input,
                ouput,
                config["completion"],
                type,
                config["dossier"]["dossier_model"],
                config["dossier"]["dossier_scaler"],
                config["entrainement_model"]["window_size"],
                0.1,
                1990,1990,
                summary
            )
        pd.DataFrame(summary).to_csv(f"{config["dossier"]["dossier_summary"]}/{config["dossier"]["summary_name"]}", sep=";", index=False)

        
