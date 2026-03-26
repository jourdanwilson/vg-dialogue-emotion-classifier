import re
import pandas as pd
from pathlib import Path

input_file = Path("corpus.txt")   # put your exported text file here
output_file = Path("dialogues_clean.csv")

text = input_file.read_text(encoding="utf-8")

# Extract all quoted strings
quoted = re.findall(r'"([^"\n]+)"', text)

cleaned = []
for line in quoted:
    line = line.strip()

    # Skip obvious metadata / labels
    if not line:
        continue
    if line in {
        "aliases", "SYSTEM", "CHOICE", "ACTION",
        "Game", "dialogue", "text"
    }:
        continue

    # Skip likely speaker/game/category labels:
    # short title-like strings without sentence punctuation
    if len(line.split()) <= 4 and not re.search(r"[.!?]", line):
        continue

    # Skip lines that look like identifiers / headings
    if re.fullmatch(r"[A-Za-z0-9 _:'()/-]+", line) and not re.search(r"[.!?]", line):
        continue

    # Keep actual dialogue-looking lines
    if len(line) >= 3:
        cleaned.append(line)

# Remove duplicates while preserving order
seen = set()
dialogues = []
for line in cleaned:
    if line not in seen:
        seen.add(line)
        dialogues.append(line)

df = pd.DataFrame({"dialogue": dialogues})
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Saved {len(df)} dialogue lines to {output_file}")
print(df.head(20))