"""Build the substitution list for evaluate_weights.py from Recipe1MSubs plus the manual pairs.

Recipe1MSubs (Fatemi et al. 2023) has 70k substitutions mined from recipe comments.
Download its three pickles into data/eval/recipe1msubs/:
  https://dl.fbaipublicfiles.com/gismo/{train,val,test}_comments_subs.pkl
Each record is {"id", "ingredients", "subs": (ingredient, substitute)}.
We only use the (ingredient, substitute) pairs, from all three splits, since nothing is trained on them.
"""

import pickle
from collections import Counter

import pandas as pd

from normalisation import normalise

SUBS_DIR = "data/eval/recipe1msubs"
MANUAL = "substitutions.csv"
INGREDIENTS = "data/processed/ingredients.csv"
OUTPUT = "data/eval/substitutions_all.csv"
MIN_SUPPORT = 3   # a pair must appear in at least this many comments; at 2, pairs like eggplant -> pasta remain


class DataOnlyUnpickler(pickle.Unpickler):
    """The pickles are plain lists and dicts; refuse anything that would import code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


records = []
for split in ["train", "val", "test"]:
    with open(f"{SUBS_DIR}/{split}_comments_subs.pkl", "rb") as f:
        records += DataOnlyUnpickler(f).load()

# Count each pair after normalising the names the same way as the graph ("baby_spinach" -> "baby spinach")
support = Counter()
for r in records:
    a, b = (normalise(name.replace("_", " ")) for name in r["subs"])
    if a and b and a != b:
        support[(a, b)] += 1

vocab = set(pd.read_csv(INGREDIENTS)["name"])
rows = [(a, b, n, "recipe1msubs") for (a, b), n in support.items()
        if n >= MIN_SUPPORT and a in vocab and b in vocab]

# The manual pairs are kept whatever their support
manual = pd.read_csv(MANUAL)
for a, b in zip(manual["ingredient"], manual["substitute"]):
    a, b = normalise(a), normalise(b)
    if a in vocab and b in vocab and a != b:
        rows.append((a, b, support.get((a, b), 0), "manual"))

subs = (pd.DataFrame(rows, columns=["ingredient", "substitute", "support", "source"])
        .sort_values("support", ascending=False, kind="stable")
        .drop_duplicates(["ingredient", "substitute"]))   # a manual pair also in Recipe1MSubs keeps that row
subs.to_csv(OUTPUT, index=False)

print(f"{len(records):,} Recipe1MSubs records, {len(support):,} unique pairs after normalising")
print(f"kept {len(subs):,} pairs over {subs['ingredient'].nunique()} ingredients "
      f"({(subs['source'] == 'manual').sum()} only from the manual list) -> {OUTPUT}")
