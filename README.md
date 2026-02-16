# PROJET - Junon : Récupération et interpolation de données

Réalisé par Baptiste RICHARD dans le cadre du Projet Junon.

## Liens utiles

- Trello : https://trello.com/b/efFEyp9q/stage-junon
- Ades : https://ades.eaufrance.fr/recherche
- ERA 5 : https://data.europa.eu/data/datasets/d08cd288-a2c5-4c8d-a621-eedc33fab449~~1?locale=fr

## Commandes utiles

```bash
python -m venv ./venv 
venv\Scripts\activate 

pip install -r ./requirements.txt

pip freeze > requirements.txt

jupyter nbconvert --to script
```

## Données à réccupérer


- [x] niveau_nappe_eau : https://ades.eaufrance.fr/recherche
- [x] PRELIQ_Q : https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=115&id_rubrique=38
- [x] T_Q : https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=115&id_rubrique=38
- [x] ETP_Q : https://www.data.gouv.fr/datasets/etp-fao-hargreaves
- [ ] Peff_All : peut se calculer

### Calcul ETP

https://www.campbellsci.fr/blog/evapotranspiration-101
Problème -> Demande les données météos.

ou

https://www.drias-climat.fr/accompagnement/sections/310
