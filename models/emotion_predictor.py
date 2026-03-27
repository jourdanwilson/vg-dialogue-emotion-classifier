"""
module de prédiction automatique des émotions dans des dialogues de jeux vidéo.

ce module implémente un système de classification supervisée basé sur un modèle
transformer pré-entraîné (distilroberta). le modèle a été fine-tuné sur un corpus
annoté selon les catégories émotionnelles d'ekman. l'objectif est d'associer à
chaque ligne de dialogue une émotion dominante ainsi qu'un ensemble de scores
de probabilité pour l'ensemble des classes.

le module fournit :
- une classe emotionpredictor permettant la prédiction unitaire ou en lot
- des outils pour appliquer la prédiction à un fichier csv
- des fonctions de synthèse statistique des résultats

les émotions prises en charge sont : joy, sadness, anger, fear, surprise,
disgust et neutral.
"""


# importation des bibliothèques nécessaires:
# pipeline permet de charger un modèle transformer prêt à l'emploi
# pandas manipulation de tableaux de données
# pathlib gestion de chemins de fichiers
from transformers import pipeline
import pandas as pd
from pathlib import Path


# définition du nom du modèle utilisé pour la classification des émotions
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# dictionnaire associant chaque émotion à un emoji pour faciliter l'affichage
EMOTION_EMOJIS = {
    "joy":      "😄",
    "sadness":  "😢",
    "anger":    "😠",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢",
    "neutral":  "😐",
}

# dictionnaire de traduction des émotions en français
EMOTION_FR = {
    "joy":      "joie",
    "sadness":  "tristesse",
    "anger":    "colère",
    "fear":     "peur",
    "surprise": "surprise",
    "disgust":  "dégoût",
    "neutral":  "neutre",
}


class EmotionPredictor:
    """
    classe responsable de la prédiction d'émotions à partir d'un texte.

    cette classe encapsule une pipeline transformers qui gère automatiquement
    la tokenisation, l'inférence et la production des scores de classification.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        # initialisation de la pipeline de classification
        # top_k=None permet de récupérer les scores pour toutes les émotions
        self._pipeline = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,
            truncation=True,
            max_length=512,
        )

    def predict(self, text: str) -> dict:
        """
        prédit l'émotion dominante d'un texte.
        """

        # gestion des cas où le texte est vide ou ne contient que des espaces
        if not text or not text.strip():
            return self._empty_result(text)

        # exécution du modèle sur le texte
        raw = self._pipeline(text)[0]

        # transformation de la sortie en dictionnaire {émotion: score}
        scores = {item["label"]: round(item["score"], 4) for item in raw}

        # sélection de l'émotion ayant le score le plus élevé
        top = max(scores, key=scores.get)

        # construction du dictionnaire de sortie
        return {
            "text":       text,
            "emotion":    top,
            "emotion_fr": EMOTION_FR.get(top, top),
            "emoji":      EMOTION_EMOJIS.get(top, ""),
            "confidence": scores[top],
            "scores":     scores,
        }

    def predict_batch(self, texts: list[str], verbose: bool = False) -> list[dict]:
        """
        applique la prédiction à une liste de textes.
        """

        # initialisation de la liste des résultats
        results = []
        total = len(texts)

        # boucle sur chaque texte
        for i, text in enumerate(texts):
            # affichage optionnel de la progression
            if verbose and i % 100 == 0:
                print(f"  traitement : {i}/{total} dialogues...")
            results.append(self.predict(text))

        # affichage final si verbose
        if verbose:
            print(f"  traitement : {total}/{total} dialogues. terminé.")

        return results

    def predict_csv(
        self,
        csv_path: str,
        text_column: str = "dialogue",
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """
        applique la prédiction à un fichier csv contenant une colonne de textes.
        """

        # chargement du fichier csv
        df = pd.read_csv(csv_path)

        # vérification que la colonne contenant les textes existe
        if text_column not in df.columns:
            raise ValueError(
                f"colonne '{text_column}' introuvable. "
                f"colonnes disponibles : {list(df.columns)}"
            )

        # extraction des textes et remplacement des valeurs manquantes
        texts = df[text_column].fillna("").tolist()
        print(f"[emotionpredictor] analyse de {len(texts)} dialogues...")

        # prédiction en lot
        predictions = self.predict_batch(texts, verbose=True)

        # ajout des colonnes principales au dataframe
        df["emotion"]    = [p["emotion"]    for p in predictions]
        df["emotion_fr"] = [p["emotion_fr"] for p in predictions]
        df["confidence"] = [p["confidence"] for p in predictions]
        df["emoji"]      = [p["emoji"]      for p in predictions]

        # ajout des scores détaillés pour chaque émotion
        emotions = list(EMOTION_EMOJIS.keys())
        for emo in emotions:
            df[f"score_{emo}"] = [p["scores"].get(emo, 0.0) for p in predictions]

        # sauvegarde éventuelle du fichier enrichi
        if output_path:
            df.to_csv(output_path, index=False)

        return df

    def summary(self, df: pd.DataFrame):
        """
        génère un tableau récapitulatif des émotions prédites.
        """

        # comptage des occurrences de chaque émotion
        counts = df["emotion"].value_counts().reset_index()

        # renommage des colonnes pour plus de clarté
        counts.columns = ["emotion", "count"]

        # calcul du pourcentage d'occurrences
        counts["pct"]  = (counts["count"] / len(df) * 100).round(1)

        # ajout des traductions et emojis
        counts["emotion_fr"] = counts["emotion"].map(EMOTION_FR)
        counts["emoji"]      = counts["emotion"].map(EMOTION_EMOJIS)

        return counts

    def _empty_result(self, text: str):
        """
        retourne un résultat par défaut si le texte est vide.
        """

        # renvoie une prédiction neutre avec un score maximal
        return {
            "text":       text,
            "emotion":    "neutral",
            "emotion_fr": "neutre",
            "emoji":      "😐",
            "confidence": 1.0,
            "scores":     {e: 0.0 for e in EMOTION_EMOJIS},
        }


if __name__ == "__main__":
    # exemple d'utilisation simple du module
    predictor = EmotionPredictor()

    # liste de dialogues pour démonstration
    exemples = [
        "ha ha ha ha. i'm so lonely. will you play with me?",
        "stop. you don't know what you're doing.",
        "thanks for the help. you really got me out of a jam.",
        "we have to find malak. hurry.",
        "the kath hounds seem to have calmed down.",
    ]

    # affichage des prédictions pour chaque exemple
    for texte in exemples:
        r = predictor.predict(texte)
        print(f"{r['emoji']} [{r['emotion_fr']:12s} {r['confidence']:.0%}] {texte[:60]}")

