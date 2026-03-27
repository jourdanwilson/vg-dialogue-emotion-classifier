import re
import pandas as pd
from pathlib import Path

# définition des chemins d'entrée et de sortie
# input_file fichier texte brut contenant l'ensemble du corpus exporté
# output_file fichier csv qui contiendra uniquement les lignes de dialogue extraites
input_file = Path("corpus.txt")
output_file = Path("dialogues_clean.csv")

# lecture du fichier texte complet
text = input_file.read_text(encoding="utf-8")

# extraction de toutes les chaînes entre guillemets
# l'expression régulière capture tout texte entre " ... " sans sauter de ligne
quoted = re.findall(r'"([^"\n]+)"', text)

cleaned = []
for line in quoted:
    # suppression des espaces inutiles
    line = line.strip()

    # filtrage des lignes vides ou correspondant à des métadonnées évidentes
    if not line:
        continue
    if line in {
        "aliases", "SYSTEM", "CHOICE", "ACTION",
        "Game", "dialogue", "text"
    }:
        continue

    # filtrage des lignes très courtes ressemblant à des titres ou noms de personnages
    # critère : moins de 5 mots et absence de ponctuation de phrase
    if len(line.split()) <= 4 and not re.search(r"[.!?]", line):
        continue

    # filtrage des lignes qui ressemblent à des identifiants ou en-têtes
    # critère : uniquement lettres/chiffres/symboles simples et pas de ponctuation de phrase
    if re.fullmatch(r"[A-Za-z0-9 _:'()/-]+", line) and not re.search(r"[.!?]", line):
        continue

    # si la ligne ressemble à un vrai dialogue, on la conserve
    if len(line) >= 3:
        cleaned.append(line)

# suppression des doublons tout en conservant l'ordre d'apparition
seen = set()
dialogues = []
for line in cleaned:
    if line not in seen:
        seen.add(line)
        dialogues.append(line)

# création d'un dataframe pandas contenant les dialogues extraits
df = pd.DataFrame({"dialogue": dialogues})

# sauvegarde du fichier final au format csv
df.to_csv(output_file, index=False, encoding="utf-8")

# affichage d'un résumé simple
print(f"Saved {len(df)} dialogue lines to {output_file}")
print(df.head(20))
