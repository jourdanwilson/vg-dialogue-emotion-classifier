# Dossier annotation — Annotations humaines et évaluation du modèle

Ce dossier contient les fichiers liés à l’annotation manuelle des émotions et à l’évaluation des performances du modèle de classification.

Il permet de comparer les prédictions du modèle avec les annotations humaines afin d’analyser la qualité des résultats et mesurer l’accord entre annotateurs.

---

## Objectifs du dossier

- Stocker les annotations humaines des dialogues
- Comparer les prédictions du modèle avec les annotations
- Mesurer l’accord entre annotateurs humains
- Évaluer la fiabilité du modèle
- Fournir des données pour l’analyse qualitative et quantitative

---

## Contenu du dossier

```text
annotation/
│
├── annotation_emotions.xlsx - Annotation.csv   # annotations humaines des émotions
├── predictions.csv                             # émotions prédites par le modèle
├── annotation_scores.png                       # métriques d’accord annotateurs / modèle
└── README.md
```

## Description des fichiers

### annotation_emotions.xlsx - Annotation.csv
Fichier contenant les annotations manuelles des dialogues.  
Chaque ligne correspond à un dialogue associé à une émotion attribuée par un annotateur humain.  

Permet de créer une référence (*gold standard*) pour l’évaluation du modèle.

---

### predictions.csv
Fichier contenant les émotions prédites par le modèle pour les dialogues annotés.  

Utilisé pour comparer les prédictions automatiques aux annotations humaines.

---

### annotation_scores.png
Figure résumant les résultats de l’évaluation :

- nombre de dialogues annotés
- accord entre annotateurs humains
- accord entre le modèle et les annotateurs
- score de confiance moyen du modèle

Permet d’évaluer la cohérence des annotations et la performance globale du modèle.

---

## Résultat attendu

Ce dossier permet :

- d’évaluer la qualité des prédictions du modèle
- de mesurer l’accord entre annotateurs humains
- d’identifier les biais éventuels du modèle
- d’améliorer le processus d’annotation et d’entraînement
