# PROJET - Junon : Récupération et interpolation de données

Réalisé par Baptiste RICHARD dans le cadre du Projet Junon.

## Liens utiles

- Trello : https://trello.com/b/efFEyp9q/stage-junon

## Commandes utiles

Création du venv :

```bash
python -m venv ./venv 
venv\Scripts\activate 
```

Installation des dépandances :

```bash
pip install -r ./requirements.txt
```

Ajout de dépandances :

```bash
pip freeze > requirements.txt
```

Autres :

```bash
jupyter nbconvert --to script
```

## Données à réccupérer

- [x] niveau_nappe_eau : https://ades.eaufrance.fr/recherche
- [x] PRELIQ_Q : https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles
- [x] T_Q : https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles
- [x] ETP_Q : https://www.data.gouv.fr/datasets/etp-fao-hargreaves

## Correspondances mailles

https://donneespubliques.meteofrance.fr/client/document/metadonnees_swi_276.csv

### Calcul ETP

https://www.campbellsci.fr/blog/evapotranspiration-101
Problème -> Demande les données météos.

ou

https://www.drias-climat.fr/accompagnement/sections/310