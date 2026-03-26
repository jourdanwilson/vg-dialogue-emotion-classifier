# 🎮 Classificateur d’émotions dans les dialogues de jeux vidéo
Un modèle de réseau de neurones pour la classification des émotions dans les dialogues de RPG

## Présentation
Ce projet porte sur la classification des émotions dans les dialogues de jeux vidéo à l’aide de réseaux de neurones.

Nous entraînons un classificateur supervisé capable de prédire l’une des sept émotions de base d’Ekman :

- Joie
- Tristesse
- Colère
- Peur
- Surprise
- Dégoût
- Neutre

Le projet comprend :

- Un pipeline de prétraitement personnalisé pour extraire et nettoyer les dialogues à partir de scripts bruts de jeux vidéo
- Un corpus construit à partir de plusieurs jeux de la série *Final Fantasy*
- Un classificateur neuronal entraîné sur le corpus nettoyé
- Une étape d’annotation humaine pour évaluer les performances du modèle
- (Optionnel) une API web et une interface pour des prédictions en temps réel

Ce travail a été réalisé dans le cadre du cours de Réseaux de Neurones.

---

## Jeu de données

Notre jeu de données est dérivé du dépôt **VideoGameDialogueCorpusPublic** :

https://github.com/seannyD/VideoGameDialogueCorpusPublic

Nous avons utilisé des dialogues extraits de sous-ensembles de scripts.


## Objectifs

- Collecter et préparer un corpus de dialogues de jeux vidéo
- Générer ou attribuer des étiquettes d’émotion aux dialogues
- Entraîner un modèle de classification des émotions
- Évaluer les performances du modèle


## Pipeline de prétraitement

Nous avons combiné deux scripts complémentaires en un seul pipeline d’extraction et de nettoyage :

### 1. Extraction à partir de scripts `.txt` bruts

- Extraire les lignes de dialogue entre guillemets
- Supprimer les métadonnées (`SYSTEM`, `CHOICE`, `ACTION`, etc.)
- Filtrer les titres, identifiants et textes qui ne correspondent pas à du dialogue
- Supprimer les doublons tout en conservant l’ordre original

### 2. Nettoyage et normalisation

- Supprimer à nouveau les doublons (étape de sécurité)
- Supprimer les lignes contenant moins de 3 mots
- Supprimer les espaces inutiles
- Exporter vers `dialogues_clean.csv`

## Modèle

Le classificateur prédit l’une des sept émotions à l’aide de :

- Tokenisation (WordPiece / BPE ou embeddings simples)
- Une architecture de réseau de neurones (LSTM, GRU, CNN ou Transformer selon l’implémentation)
- Une couche de sortie Softmax sur 7 classes

L’évaluation comprend :

- Accuracy (exactitude)
- F1-score
- Matrice de confusion
- Accord entre annotations humaines et prédictions du modèle


## Annotation humaine

Afin de valider le modèle, nous avons annoté manuellement un sous-ensemble de **75 lignes de dialogue** :

- Deux annotateurs humains (A1, A2)
- Prédictions du modèle
- Accord inter-annotateurs
- Accord entre le modèle et les annotateurs humains

Cette étape a permis d’identifier certains biais systématiques (par exemple, une tendance à sur-prédire la classe neutre).


## Extension : API web et interface

Un membre de l’équipe a étendu le projet avec :

- Une API REST permettant d’utiliser le modèle entraîné
- Une interface web pour la prédiction d’émotions en temps réel

Cette extension illustre une application pratique du modèle au-delà du cadre du projet académique.


## Auteurs

- Mickaëla Mastrodicasa
- Jourdan Wilson
