# Dossier data — Données brutes, intermédiaires et nettoyées

Ce dossier contient l’ensemble des fichiers de données utilisés dans le projet de classification des émotions dans les dialogues de jeux vidéo.

Il regroupe à la fois les données brutes, les versions nettoyées, ainsi que les fichiers intermédiaires produits lors du prétraitement.

---

## Objectifs du dossier

- Centraliser toutes les données utilisées par le pipeline
- Séparer clairement les données brutes des données traitées
- Faciliter la reproductibilité du projet
- Fournir un corpus propre pour l’entraînement et l’évaluation du modèle

---

## Contenu du dossier

```text
data/
│
├── rawcorpus.txt              # Corpus brut extrait des scripts de jeux vidéo
├── dialogues_clean.csv        # Corpus nettoyé (version principale)
├── dialogues_clean_v2.csv     # Variante nettoyée (tests, ajustements)
└── README.md                  # Documentation du dossier
