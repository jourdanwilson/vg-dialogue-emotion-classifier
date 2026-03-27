# DOssier Preprocessing — Pipeline d’extraction et de nettoyage des dialogues

Ce dossier contient les scripts responsables de l’extraction, du nettoyage et de la préparation des dialogues utilisés pour l’entraînement du classificateur d’émotions.

L’objectif est de transformer des scripts bruts de jeux vidéo en un corpus propre, cohérent et exploitable pour les modèles de traitement automatique du langage.

---

## Objectifs du préprocessing

- Extraire automatiquement les lignes de dialogue à partir de fichiers texte bruts
- Éliminer les métadonnées, balises, noms de personnages et autres éléments non linguistiques
- Nettoyer les doublons, les lignes trop courtes ou non pertinentes
- Générer un fichier final `dialogues_clean.csv` prêt pour l’entraînement du modèle

---

## Contenu du dossier

```text
preprocessing/
│
├── data_extract.py     # Extraction des dialogues depuis les fichiers .txt
├── data_clean.py       # Nettoyage, filtrage et normalisation des dialogues
└── README.md           # Documentation du dossier
