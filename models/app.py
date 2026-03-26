"""
app.py
------
Interface web Gradio pour la prédiction d'émotions sur des dialogues de jeux vidéo.

Lancer avec :
    uv run app.py

L'interface s'ouvre automatiquement dans le navigateur (http://localhost:7860).
"""

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

from emotion_predictor import EmotionPredictor, EMOTION_EMOJIS, EMOTION_FR

# on initialise le modèle une seule fois au démarrage
predictor = EmotionPredictor()

# on a généré une palette de couleurs par émotion 
COLORS = {
    "joy":      "#FFD166",
    "sadness":  "#118AB2",
    "anger":    "#EF233C",
    "fear":     "#7B2D8B",
    "surprise": "#06D6A0",
    "disgust":  "#8D99AE",
    "neutral":  "#ADB5BD",
}

# quelques dialogues d'exemple
EXEMPLES = [
    "Ha ha ha ha. I'm so lonely. Will you play with me?",
    "Stop! Hold it right there! You don't know what you're doing!",
    "Thanks for the help! You really got me out of a jam.",
    "We have to find Malak! A victory won't mean anything if he gets away!",
    "Please, don't disturb me; I have pressing matters at hand.",
    "They swarmed out and over us. There was no way we could stop them.",
    "Due to an explosion at Mako Reactor 1, an emergency schedule is now in effect.",
    "I'm not looking for any trouble with you. Please, just leave me alone.",
]


# prédiction texte libre 

def predict_single(text: str):
    """Appelée par l'onglet 'Dialogue unique'."""
    if not text or not text.strip():
        return "Entrez un dialogue.", None

    result = predictor.predict(text)
    scores = result["scores"]

    # texte de résultat 
    label = (
        f"## {result['emoji']} {result['emotion_fr'].upper()}  "
        f"— confiance : **{result['confidence']:.1%}**\n\n"
        f"> *{text}*"
    )


    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    emotions = list(scores.keys())
    values   = [scores[e] for e in emotions]
    colors   = [COLORS.get(e, "#888") for e in emotions]
    labels   = [f"{EMOTION_EMOJIS[e]} {EMOTION_FR[e]}" for e in emotions]

    # on trie par score décroissant
    sorted_pairs = sorted(zip(values, labels, colors), reverse=True)
    values, labels, colors = zip(*sorted_pairs)

    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)

    # on annote les barres
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left",
            color="white", fontsize=9
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score de probabilité", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    plt.tight_layout(pad=0.8)

    return label, fig


# analyse du csv

def analyze_csv(file):
    df = pd.read_csv(file.name)

    # on limite à 200 lignes pour la démo et éviter le timeout
    sample = df.head(200).copy()
    texts  = sample["dialogue"].fillna("").tolist()

    predictions = predictor.predict_batch(texts, verbose=False)
    sample["emotion"]    = [p["emotion"]    for p in predictions]
    sample["emotion_fr"] = [p["emotion_fr"] for p in predictions]
    sample["confidence"] = [p["confidence"] for p in predictions]
    sample["emoji"]      = [p["emoji"]      for p in predictions]

    summary = predictor.summary(sample)
    summary_md = "### Distribution des émotions\n\n"
    summary_md += "| Émotion | Count | % |\n|---|---|---|\n"
    for _, row in summary.iterrows():
        summary_md += f"| {row['emoji']} {row['emotion_fr']} | {row['count']} | {row['pct']}% |\n"
    summary_md += f"\n*Analyse sur les {len(sample)} premiers dialogues.*"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")

    pie_labels  = [f"{EMOTION_EMOJIS[e]} {EMOTION_FR[e]}" for e in summary["emotion"]]
    pie_values  = summary["count"].tolist()
    pie_colors  = [COLORS.get(e, "#888") for e in summary["emotion"]]

    wedges, texts_, autotexts = ax1.pie(
        pie_values, labels=None, colors=pie_colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 1.5},
        pctdistance=0.75,
    )
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(pie_colors, pie_labels)
    ]
    ax1.legend(handles=legend_patches, loc="lower left", fontsize=8,
               labelcolor="white", framealpha=0, bbox_to_anchor=(-0.15, -0.15))
    ax1.set_title("Répartition des émotions", color="white", fontsize=11, pad=10)

    bar_emos   = [EMOTION_FR[e] for e in summary["emotion"]]
    bar_vals   = summary["count"].tolist()
    bar_colors = [COLORS.get(e, "#888") for e in summary["emotion"]]

    bars = ax2.barh(bar_emos, bar_vals, color=bar_colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, bar_vals):
        ax2.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", color="white", fontsize=9
        )
    ax2.set_title("Nombre de dialogues par émotion", color="white", fontsize=11, pad=10)
    ax2.tick_params(colors="white", labelsize=9)
    ax2.spines[:].set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.set_xlim(0, max(bar_vals) * 1.18)

    plt.tight_layout(pad=1.2)

    preview = sample[["dialogue", "emoji", "emotion_fr", "confidence"]].head(20).copy()
    preview.columns = ["Dialogue", "Ém.", "Émotion", "Confiance"]
    preview["Confiance"] = preview["Confiance"].map(lambda x: f"{x:.1%}")
    preview["Dialogue"]  = preview["Dialogue"].str[:70] + "…"

    return summary_md, fig, preview


# Construction de l'interface

with gr.Blocks(
    title="🎮 Game Dialogue Emotion Detector",
    theme=gr.themes.Base(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        .gradio-container { max-width: 900px !important; }
        h1 { text-align: center; margin-bottom: 0.2em; }
        .subtitle { text-align: center; color: #888; margin-bottom: 1.5em; font-size: 0.95em; }
    """,
) as demo:

    gr.Markdown("# 🎮 Game Dialogue Emotion Detector")
    gr.Markdown(
        "<p class='subtitle'>Analyse des sentiments sur des dialogues de jeux vidéo "
        "— Modèle : <b>DistilBERT</b> fine-tuné sur 7 émotions (Ekman)</p>"
    )

    with gr.Tabs():

        # Onglet 1 : dialogue unique
        with gr.Tab("💬 Dialogue unique"):
            with gr.Row():
                with gr.Column(scale=3):
                    txt_input = gr.Textbox(
                        label="Entrez un dialogue de jeu vidéo",
                        placeholder="Ex: I'm so lonely. Will you play with me?",
                        lines=3,
                    )
                    btn = gr.Button("🔍 Analyser", variant="primary")
                    gr.Examples(
                        examples=[[e] for e in EXEMPLES],
                        inputs=[txt_input],
                        label="Exemples",
                    )

            result_label = gr.Markdown()
            result_plot  = gr.Plot(label="Scores par émotion")

            btn.click(predict_single, inputs=[txt_input], outputs=[result_label, result_plot])
            txt_input.submit(predict_single, inputs=[txt_input], outputs=[result_label, result_plot])

        # Onglet 2 : analyse CSV
        with gr.Tab("📊 Analyse CSV"):
            gr.Markdown(
                "Déposez votre fichier CSV (colonne `dialogue` requise). "
                "Analyse limitée aux 200 premières lignes pour la démo."
            )
            file_input   = gr.File(label="Fichier CSV", file_types=[".csv"])
            btn_csv      = gr.Button("🚀 Analyser le dataset", variant="primary")
            summary_out  = gr.Markdown()
            chart_out    = gr.Plot(label="Distribution des émotions")
            table_out    = gr.Dataframe(label="Aperçu des prédictions (20 premières lignes)")

            btn_csv.click(
                analyze_csv,
                inputs=[file_input],
                outputs=[summary_out, chart_out, table_out],
            )

    gr.Markdown(
        "<p class='subtitle' style='margin-top:2em;font-size:0.8em;'>"
        "Projet TAL — Analyse de sentiment sur dialogues de jeux vidéo | "
        "Modèle : <code>j-hartmann/emotion-english-distilbert-base-uncased</code></p>"
    )


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
