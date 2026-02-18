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

y a une api https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques?code_bss=03287X0018/S1&size=3921

- [x] niveau_nappe_eau : https://ades.eaufrance.fr/recherche
- [x] PRELIQ_Q : https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=115&id_rubrique=38 ou https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles
- [x] T_Q : https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=115&id_rubrique=38 ou https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles
- [x] ETP_Q : https://www.data.gouv.fr/datasets/etp-fao-hargreaves
- [ ] Peff_All :

### Calcul ETP

https://www.campbellsci.fr/blog/evapotranspiration-101
Problème -> Demande les données météos.

ou

https://www.drias-climat.fr/accompagnement/sections/310

https://seaborn.pydata.org/generated/seaborn.heatmap.html

heat map pour afiché les donée manquante

par fichier et par paquet de 10

et aussi affiché les pourcentage de completion d'un fichier.