from .data.extraction import *

def chargement_config():
    pass

def extraction():
    process_impermeabilite(
        dataset_id="697b4f4ceea77fb452ba9d6d",
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        communes_file="data/communesdefrancev2.csv",
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
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        name=nappe_name,
        departements=[45]
    )

    download_communes_csv("./data")
    download_maille_csv("./data")

def clusterisations():
    pass

def interpolations():
    pass

def methodes_non_naives():
    pass

def entrainement_création_NNs():
    pass

def resultat_NNs():
    pass



if __name__ == "__main__":
