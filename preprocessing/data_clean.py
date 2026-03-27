import pandas as pd

# chargement du fichier csv contenant les dialogues nettoyés
df = pd.read_csv("dialogues_clean.csv")

# suppression des doublons éventuels dans le dataset
df = df.drop_duplicates()

# suppression des lignes trop courtes (moins de 3 mots)
# cette étape permet d'éliminer les fragments non informatifs
df = df[df.dialogue.str.split().str.len() >= 3]

# suppression des espaces inutiles en début et fin de ligne
df.dialogue = df.dialogue.str.strip()

# sauvegarde du fichier nettoyé sous un nouveau nom
df.to_csv("dialogues_clean.csv", index=False)

# affichage du nombre de lignes restantes après nettoyage
print(len(df), "lines remaining")
