# Dossier models 

Ce dossier contient les scripts liés au modèle de classification des émotions appliqué aux dialogues de jeux vidéo.

Le modèle utilise un Transformer pré-entraîné (DistilBERT) fine-tuné pour la détection d’émotions selon les catégories d’Ekman :
joy, sadness, anger, fear, surprise, disgust, neutral.

---

## Objectifs du dossier

- Charger un modèle de classification d’émotions pré-entraîné
- Prédire l’émotion dominante d’un dialogue
- Analyser un corpus complet de dialogues
- Fournir une interface interactive pour tester le modèle
- Produire des visualisations pour l’analyse des résultats

---

## Contenu du dossier

```text
models/
│
├── emotion_predictor.py   # classe principale pour prédire les émotions
├── analyze_dataset.py     # script d’analyse statistique du dataset
├── app.py                 # interface web pour tester le modèle
└── README.md              # documentation du dossier
```
---

## Description des scripts

### emotion_predictor.py
Implémente la classe `EmotionPredictor`, qui utilise un modèle Transformer pré-entraîné (DistilBERT) pour prédire l’émotion dominante d’un dialogue.
Permet la prédiction sur un texte unique, une liste de textes ou un fichier CSV.
Retourne l’émotion prédite, le score de confiance et la distribution des probabilités pour chaque classe.

### analyze_dataset.py
Script d’analyse du dataset permettant de générer des prédictions sur un fichier CSV et de produire des visualisations :
distribution des émotions, scores de confiance et exemples représentatifs.
Les résultats sont enregistrés dans un dossier de sortie.

### app.py
Interface web interactive développée avec Gradio.
Permet de tester le modèle sur un dialogue unique ou un fichier CSV et d’afficher les résultats sous forme de graphiques et tableaux.
