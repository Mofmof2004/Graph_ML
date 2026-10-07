"""Ingredient and title normalisation used to build the recipe graph."""

import re
import unicodedata

# Normalisation rules
SYNONYMS = {"chili": "chilli", "chile": "chilli", "oleo": "margarine"}
IRREGULAR = {"leaves": "leaf", "halves": "half", "loaves": "loaf"}
KEEP_AS_IS = {"molasses", "hummus", "couscous", "asparagus", "swiss", "grits", "lemongrass"}

LEADING_DROP = {
    "fresh", "frozen", "grated", "shredded", "chopped", "diced", "sliced", "minced",
    "drained", "toasted", "mashed", "cooked", "softened", "melted", "beaten",
    "boneless", "skinless", "skinned", "large", "small", "medium", "handful",
    "very", "warm", "cold", "tap",
}
TRAILING_DROP = {"slices", "slice", "pieces", "piece", "bits", "mixed", "chunks", "cubes"}


def singularise(word):
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word in KEEP_AS_IS or len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "oes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalise(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"[^a-z\s-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    words = name.split()
    while len(words) > 1 and words[0] in LEADING_DROP:
        words = words[1:]
    while len(words) > 1 and words[-1] in TRAILING_DROP:
        words = words[:-1]
    if not words:
        return None
    words[-1] = singularise(words[-1])
    name = " ".join(words)
    return SYNONYMS.get(name, name)


def norm_title(title):
    title = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    title = title.lower()
    title = re.sub(r"[^a-z\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()
