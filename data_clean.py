import pandas as pd

df = pd.read_csv("dialogues_clean.csv")

# remove duplicates
df = df.drop_duplicates()

# remove very short lines (optional but recommended)
df = df[df.dialogue.str.split().str.len() >= 3]

# strip spaces
df.dialogue = df.dialogue.str.strip()

df.to_csv("dialogues_clean-new.csv", index=False)

print(len(df), "lines remaining")