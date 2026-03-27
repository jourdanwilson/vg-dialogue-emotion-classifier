# Dossier figures — Visualisations des résultats

Ce dossier contient les graphiques générés lors de l’évaluation du modèle de classification des émotions appliqué aux dialogues de jeux vidéo.

Ces figures permettent d’analyser la performance du modèle et la distribution des émotions dans le corpus.

---

## Objectifs du dossier

- Visualiser la distribution des émotions prédites
- Analyser la confiance du modèle selon chaque émotion
- Identifier des exemples représentatifs pour chaque catégorie émotionnelle
- Fournir des figures pour le rapport final

---

## Contenu du dossier

```text
figures/
│
├── distribution_emotions.png        # fréquence et proportion de chaque émotion
├── confiance_par_emotion.png        # distribution des scores de confiance du modèle
├── top_dialogues_par_emotion.png    # exemples les plus représentatifs par émotion
└── README.md
```

## Description des figures

### distribution_emotions.png
Distribution des émotions prédites dans le corpus.
Montre la fréquence absolue et la proportion de chaque émotion.

### confiance_par_emotion.png
Distribution des scores de confiance du modèle pour chaque émotion.
Permet d’évaluer la fiabilité des prédictions.

### top_dialogues_par_emotion.png
Exemples de dialogues les plus représentatifs pour chaque émotion.
Correspond aux prédictions ayant les scores de confiance les plus élevés.
