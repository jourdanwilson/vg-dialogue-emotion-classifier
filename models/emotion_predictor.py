"""
Module de prédiction d'émotions sur des dialogues de jeux vidéo.
Utilise un réseau de neurones Transformer (DistilBERT) pré-entraîné
fine-tuné sur la détection d'émotions (dataset GoEmotions / Ekman).

Modèle : j-hartmann/emotion-english-distilbert-base-uncased
Émotions détectées : joy, sadness, anger, fear, surprise, disgust, neutral
"""

from transformers import pipeline
import pandas as pd
from pathlib import Path


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

EMOTION_EMOJIS = {
    "joy":      "😄",
    "sadness":  "😢",
    "anger":    "😠",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢",
    "neutral":  "😐",
}

EMOTION_FR = {
    "joy":      "Joie",
    "sadness":  "Tristesse",
    "anger":    "Colère",
    "fear":     "Peur",
    "surprise": "Surprise",
    "disgust":  "Dégoût",
    "neutral":  "Neutre",
}


class EmotionPredictor:
    """
    Prédicateur d'émotions basé sur DistilBERT.

    Architecture du réseau de neurones :
    - Tokenizer WordPiece (vocabulaire de 30 522 tokens)
    - 6 couches Transformer (DistilBERT = version distillée de BERT)
    - Chaque couche : multi-head self-attention (12 têtes) + FFN
    - Tête de classification : couche dense 768 → 7 (une sortie par émotion)
    - Activation finale : softmax → probabilités normalisées

    """

    def __init__(self, model_name: str = MODEL_NAME):
        self._pipeline = pipeline(
            task="text-classification",
            model=model_name,
            top_k=None,          # retourne les scores pour TOUTES les émotions
            truncation=True,
            max_length=512,
        )

    def predict(self, text: str) -> dict:
        """
        Prédit les émotions d'un texte.

        Args:
            text : le dialogue à analyser (str)

        Returns:
            dict avec :
              - 'text'       : texte original
              - 'emotion'    : émotion dominante (str)
              - 'confidence' : score de confiance de l'émotion dominante (float)
              - 'scores'     : dict {émotion: score} pour toutes les émotions
              - 'emoji'      : emoji de l'émotion dominante (str)
              - 'emotion_fr' : traduction française de l'émotion dominante (str)
        """
        if not text or not text.strip():
            return self._empty_result(text)

        raw = self._pipeline(text)[0]
        scores = {item["label"]: round(item["score"], 4) for item in raw}
        top = max(scores, key=scores.get)

        return {
            "text":       text,
            "emotion":    top,
            "emotion_fr": EMOTION_FR.get(top, top),
            "emoji":      EMOTION_EMOJIS.get(top, "❓"),
            "confidence": scores[top],
            "scores":     scores,
        }

    def predict_batch(self, texts: list[str], verbose: bool = False) -> list[dict]:
        """
        Prédit les émotions sur une liste de textes.
        """
        results = []
        total = len(texts)
        for i, text in enumerate(texts):
            if verbose and i % 100 == 0:
                print(f"  Traitement : {i}/{total} dialogues...")
            results.append(self.predict(text))
        if verbose:
            print(f"  Traitement : {total}/{total} dialogues. Terminé.")
        return results

    def predict_csv(
        self,
        csv_path: str,
        text_column: str = "dialogue",
        output_path: str | None = None,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(
                f"Colonne '{text_column}' introuvable. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

        texts = df[text_column].fillna("").tolist()
        print(f"[EmotionPredictor] Analyse de {len(texts)} dialogues...")
        predictions = self.predict_batch(texts, verbose=True)

        df["emotion"]    = [p["emotion"]    for p in predictions]
        df["emotion_fr"] = [p["emotion_fr"] for p in predictions]
        df["confidence"] = [p["confidence"] for p in predictions]
        df["emoji"]      = [p["emoji"]      for p in predictions]

        emotions = list(EMOTION_EMOJIS.keys())
        for emo in emotions:
            df[f"score_{emo}"] = [p["scores"].get(emo, 0.0) for p in predictions]

        if output_path:
            df.to_csv(output_path, index=False)
        return df

    def summary(self, df: pd.DataFrame):
        counts = df["emotion"].value_counts().reset_index()
        counts.columns = ["emotion", "count"]
        counts["pct"]  = (counts["count"] / len(df) * 100).round(1)
        counts["emotion_fr"] = counts["emotion"].map(EMOTION_FR)
        counts["emoji"]      = counts["emotion"].map(EMOTION_EMOJIS)
        return counts

    def _empty_result(self, text: str):
        return {
            "text":       text,
            "emotion":    "neutral",
            "emotion_fr": "Neutre",
            "emoji":      "😐",
            "confidence": 1.0,
            "scores":     {e: 0.0 for e in EMOTION_EMOJIS},
        }

if __name__ == "__main__":
    predictor = EmotionPredictor()

    exemples = [
        "Ha ha ha ha. I'm so lonely. Will you play with me?",
        "Stop! You don't know what you're doing!",
        "Thanks for the help. You really got me out of a jam.",
        "We have to find Malak! Hurry!",
        "The kath hounds seem to have calmed down.",
    ]

    for texte in exemples:
        r = predictor.predict(texte)
        print(f"{r['emoji']} [{r['emotion_fr']:12s} {r['confidence']:.0%}] {texte[:60]}")
