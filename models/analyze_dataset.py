"""
analyse du dataset de dialogues de jeux vidéo. ce script permet de charger un
fichier csv, d'appliquer la prédiction d'émotions à chaque dialogue, puis de
générer plusieurs visualisations statistiques.

on peut lancer le script avec :
   uv run analyze_dataset.py --csv dialogues_sample_v1.csv
   uv run analyze_dataset.py --csv dialogues_sample_v1.csv --output results/
"""

# importation des bibliothèques nécessaires :
# argparse gestion des arguments en ligne de commande
# os manipulation de chemins et dossiers
# collections.counter comptage rapide d'éléments
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# importation du prédicteur d'émotions et des dictionnaires associés
from emotion_predictor import EmotionPredictor, EMOTION_EMOJIS, EMOTION_FR

# palette de couleurs utilisée pour les graphiques
COLORS = {
    "joy":      "#FFD166",
    "sadness":  "#118AB2",
    "anger":    "#EF233C",
    "fear":     "#7B2D8B",
    "surprise": "#06D6A0",
    "disgust":  "#8D99AE",
    "neutral":  "#ADB5BD",
}


# configuration du style graphique sombre

def setup_dark_style():
    # mise à jour des paramètres matplotlib pour un thème sombre homogène
    plt.rcParams.update({
        "figure.facecolor":  "#1a1a2e",
        "axes.facecolor":    "#1a1a2e",
        "text.color":        "white",
        "axes.labelcolor":   "white",
        "xtick.color":       "white",
        "ytick.color":       "white",
        "axes.edgecolor":    "#444",
        "grid.color":        "#333",
        "font.family":       "sans-serif",
    })


# graphique : distribution des émotions

def plot_distribution(df_results: pd.DataFrame, output_dir: str):
    """distribution des émotions dans le dataset."""

    # comptage des émotions prédites
    summary = df_results["emotion"].value_counts()
    emotions = summary.index.tolist()
    counts   = summary.values.tolist()

    # préparation des couleurs et labels
    colors   = [COLORS.get(e, "#888") for e in emotions]
    labels   = [f"{EMOTION_EMOJIS[e]} {EMOTION_FR[e]}" for e in emotions]

    # création d'une figure avec deux sous-graphiques : barres + camembert
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "distribution des émotions dans les dialogues de jeux vidéo",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    # graphique en barres horizontales
    ax = axes[0]
    bars = ax.barh(labels, counts, color=colors, edgecolor="none", height=0.65)

    # annotation des barres
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center", color="white", fontsize=10
        )

    ax.set_xlabel("nombre de dialogues", color="white")
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_title("fréquence absolue", color="white", fontsize=11)

    # graphique en camembert
    ax2 = axes[1]
    wedges, _, autotexts = ax2.pie(
        counts, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.78,
        textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2},
    )

    # légende du camembert
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in zip(colors, labels)
    ]
    ax2.legend(
        handles=legend_patches, loc="lower left", fontsize=8,
        labelcolor="white", framealpha=0, bbox_to_anchor=(-0.1, -0.15)
    )
    ax2.set_title("répartition relative", color="white", fontsize=11)

    # sauvegarde du graphique
    plt.tight_layout()
    path = os.path.join(output_dir, "distribution_emotions.png")
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"  ✓ sauvegardé : {path}")
    return path


# graphique : distribution des scores de confiance

def plot_confidence(df_results: pd.DataFrame, output_dir: str):
    """distribution des scores de confiance par émotion."""

    # création d'une figure unique
    fig, ax = plt.subplots(figsize=(10, 4))

    # extraction des émotions présentes dans le dataset
    emotions_present = df_results["emotion"].unique()

    # préparation des données pour les boxplots
    data = [
        df_results[df_results["emotion"] == e]["confidence"].values
        for e in emotions_present
    ]
    colors_list = [COLORS.get(e, "#888") for e in emotions_present]
    labels_list = [f"{EMOTION_EMOJIS[e]} {EMOTION_FR[e]}" for e in emotions_present]

    # tracé du boxplot
    bp = ax.boxplot(
        data, patch_artist=True, notch=False,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "#aaa"},
        capprops={"color": "#aaa"},
        flierprops={"marker": ".", "markerfacecolor": "#888", "markersize": 3}
    )

    # application des couleurs aux boîtes
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # configuration des axes
    ax.set_xticklabels(labels_list, fontsize=9, color="white")
    ax.set_ylabel("score de confiance", color="white")
    ax.set_title(
        "distribution de la confiance du modèle par émotion",
        color="white", fontsize=12
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="#555", linestyle="--", linewidth=0.8)

    # sauvegarde
    plt.tight_layout()
    path = os.path.join(output_dir, "confiance_par_emotion.png")
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"  ✓ sauvegardé : {path}")
    return path


# graphique : top dialogues les plus représentatifs

def plot_top_dialogues(df_results: pd.DataFrame, output_dir: str):
    """sélectionne les dialogues ayant la plus haute confiance pour chaque émotion."""

    emotions = list(EMOTION_EMOJIS.keys())

    # création d'une figure avec une ligne par émotion
    fig, axes = plt.subplots(len(emotions), 1, figsize=(11, len(emotions) * 1.5))
    fig.suptitle(
        "dialogues les plus représentatifs par émotion",
        color="white", fontsize=13, fontweight="bold"
    )

    # boucle sur chaque émotion
    for ax, emo in zip(axes, emotions):

        # sélection du dialogue le plus confiant pour cette émotion
        subset = df_results[df_results["emotion"] == emo].nlargest(1, "confidence")

        # configuration du fond et des axes
        ax.set_facecolor(COLORS.get(emo, "#888") + "22")
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # affichage du dialogue si disponible
        if not subset.empty:
            row = subset.iloc[0]
            text = str(row["dialogue"])[:120] + ("…" if len(str(row["dialogue"])) > 120 else "")
            label = f"{EMOTION_EMOJIS[emo]} {EMOTION_FR[emo].upper()}  ({row['confidence']:.0%})"

            ax.text(
                0.01, 0.75, label, transform=ax.transAxes,
                color=COLORS.get(emo, "white"), fontsize=9, fontweight="bold"
            )
            ax.text(
                0.01, 0.2, f'"{text}"', transform=ax.transAxes,
                color="white", fontsize=8.5, style="italic", wrap=True
            )

        # si aucun dialogue n'existe pour cette émotion
        else:
            ax.text(
                0.5, 0.5, f"{EMOTION_EMOJIS[emo]} {EMOTION_FR[emo]} — aucun exemple",
                transform=ax.transAxes, ha="center", va="center",
                color="#888", fontsize=9
            )

    # sauvegarde
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, "top_dialogues_par_emotion.png")
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"  ✓ sauvegardé : {path}")
    return path


# fonction principale : chargement, prédiction, visualisation

def main():
    # définition des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="analyse des émotions dans un csv de dialogues")
    parser.add_argument("--csv",    default="dialogues_sample_v1.csv", help="chemin vers le csv")
    parser.add_argument("--output", default="results/",               help="dossier de sortie")
    parser.add_argument("--limit",  type=int, default=None,           help="limiter le nombre de lignes")
    args = parser.parse_args()

    # création du dossier de sortie si nécessaire
    os.makedirs(args.output, exist_ok=True)

    # activation du thème sombre
    setup_dark_style()

    # initialisation du modèle
    predictor = EmotionPredictor()

    # chargement du csv
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    # prédiction via la méthode intégrée du prédicteur
    df_results = predictor.predict_csv(
        args.csv if not args.limit else None,
        output_path=os.path.join(args.output, "predictions.csv"),
    )

    # si un limit est appliqué, on doit recalculer manuellement les prédictions
    if args.limit:
        texts = df["dialogue"].fillna("").tolist()
        preds = predictor.predict_batch(texts, verbose=True)
        df["emotion"]    = [p["emotion"]    for p in preds]
        df["emotion_fr"] = [p["emotion_fr"] for p in preds]
        df["confidence"] = [p["confidence"] for p in preds]
        df["emoji"]      = [p["emoji"]      for p in preds]
        df_results = df

    # génération des graphiques
    plot_distribution(df_results, args.output)
    plot_confidence(df_results, args.output)
    plot_top_dialogues(df_results, args.output)


# exécution du script si appelé directement
if __name__ == "__main__":
    main()
