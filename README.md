#  🎮 Classificateur d’émotions dans les dialogues de jeux vidéo

Modèle de réseau de neurones pour la classification automatique des émotions dans les dialogues de RPG.

---

## Présentation

Ce projet propose un système de classification automatique des émotions dans des dialogues issus de jeux vidéo, en particulier des RPG.  

Le modèle prédit l’une des sept émotions fondamentales définies par Ekman :

- Joie
- Tristesse
- Colère
- Peur
- Surprise
- Dégoût
- Neutre

Le projet comprend :

- un pipeline de prétraitement permettant d’extraire et nettoyer des dialogues à partir de scripts bruts
- la constitution d’un corpus à partir de plusieurs jeux de la série *Final Fantasy*
- l’utilisation d’un modèle Transformer pré-entraîné pour la classification d’émotions
- une étape d’annotation humaine pour évaluer la qualité des prédictions
- une analyse quantitative et qualitative des performances
- une interface web permettant de tester le modèle en temps réel

Ce travail a été réalisé dans le cadre du cours de **Réseaux de Neurones**.

---

## Jeu de données

Le corpus est construit à partir du dépôt :

VideoGameDialogueCorpusPublic  
https://github.com/seannyD/VideoGameDialogueCorpusPublic

Seules les répliques de personnages ont été conservées.  
Les métadonnées, descriptions narratives et identifiants techniques ont été supprimés afin d’obtenir un corpus linguistique exploitable pour l’apprentissage automatique.

---

## Objectifs du projet

- constituer un corpus propre de dialogues de jeux vidéo
- associer une étiquette d’émotion à chaque dialogue
- comparer les prédictions du modèle avec des annotations humaines
- analyser les biais éventuels du modèle
- explorer une application concrète du NLP dans le domaine du jeu vidéo

---

## Pipeline de prétraitement

Le corpus est obtenu grâce à un pipeline en deux étapes :

### 1. Extraction des dialogues

Extraction automatique des lignes de dialogue à partir de fichiers texte bruts :

- identification des segments entre guillemets
- suppression des balises et métadonnées (`SYSTEM`, `ACTION`, `CHOICE`, etc.)
- filtrage des titres, identifiants et contenus non dialogués
- suppression des doublons en conservant l’ordre d’apparition

### 2. Nettoyage et normalisation

Nettoyage linguistique du corpus :

- suppression des doublons restants
- suppression des lignes contenant moins de 3 mots
- normalisation des espaces
- export du corpus final vers `dialogues_clean.csv`

Le résultat est un jeu de données cohérent et exploitable pour l’entraînement du modèle.

---

## Modèle

La classification des émotions est réalisée à l’aide d’un modèle Transformer pré-entraîné :

https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

Caractéristiques :

- tokenisation automatique du texte
- représentation contextuelle des phrases
- classification probabiliste via une couche Softmax
- prédiction parmi 7 catégories émotionnelles

Pour chaque dialogue, le modèle produit :

- l’émotion prédite
- un score de confiance
- la distribution de probabilité sur l’ensemble des émotions

---

## Évaluation

Les performances du modèle sont évaluées à l’aide de plusieurs indicateurs :

- accuracy
- F1-score
- matrice de confusion
- analyse de la distribution des émotions
- score de confiance moyen
- comparaison avec annotations humaines

Des visualisations permettent d’analyser :

- la répartition des émotions dans le corpus
- la confiance du modèle selon chaque émotion
- les dialogues les plus représentatifs de chaque catégorie

---

## Annotation humaine

Un sous-ensemble de 75 dialogues a été annoté manuellement afin d’évaluer la qualité des prédictions.

Le protocole comprend :

- deux annotateurs humains indépendants
- comparaison des annotations humaines entre elles
- comparaison entre annotations humaines et prédictions du modèle

Mesures analysées :

- accord inter-annotateurs
- accord modèle / humain
- confiance moyenne du modèle

Cette étape permet d’identifier les limites du modèle, notamment une tendance à sur-prédire la classe *neutre*.

---

## Interface web

Une interface interactive permettant :

- d’entrer un dialogue libre
- d’obtenir l’émotion prédite
- de visualiser les probabilités associées à chaque émotion
- d’analyser un fichier complet de dialogues

---

## Structure du projet

```text
data/           corpus brut et nettoyé  
preprocessing/  scripts d’extraction et de nettoyage  
models/         modèle de classification et scripts d’analyse  
annotation/     annotations humaines et métriques d’évaluation  
figures/        visualisations des résultats
rapport
```
---

## Liste des jeux utilisés et explorés

Dans le cadre de ce projet, plusieurs jeux vidéo ont été mobilisés 

### 1. Jeux issus du corpus général

Cette  liste regroupe les jeux mentionnés ou explorés au cours du projet

- Super Mario RPG  
- Star Wars: Knights of the Old Republic  
- Phoenix Wright: Ace Attorney  
- Chrono Trigger  
- Dragon Age II  
- Dragon Age: Origins  
- The Elder Scrolls V: Skyrim  
- Hades  
- Horizon Zero Dawn  
- The Secret of Monkey Island  
- Persona 3  
- Persona 4  
- Persona 5  
- Mass Effect  
- Mass Effect 2  
- Mass Effect 3  
- Kingdom Hearts  
- Kingdom Hearts II  
- Kingdom Hearts III  
- Kingdom Hearts Birth by Sleep  
- Kingdom Hearts Dream Drop Distance  
- King’s Quest II  
- King’s Quest III  
- King’s Quest IV  
- King’s Quest V  
- King’s Quest VI  
- Final Fantasy II  
- Final Fantasy III  
- Final Fantasy IV  
- Final Fantasy V  
- Final Fantasy VI  
- Final Fantasy VII  
- Final Fantasy VIII  
- Final Fantasy X

---

## Auteurs

Mickaëla Mastrodicasa  
Jourdan Wilson
