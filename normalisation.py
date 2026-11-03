"""Ingredient and title normalisation used to build the recipe graph."""

import re
import unicodedata

# Normalisation rules
# Only labels for the same ingredient are merged. Kept apart on purpose: canola oil and plain
# cream (substitution test ingredients), ground ginger / coriander / clove (the fresh form or
# another meaning), and red, white, sweet and green onion. "red pepper" stays ambiguous
# (cayenne or the vegetable).
SYNONYMS = {
    "chili": "chilli", "chile": "chilli", "oleo": "margarine",
    "green chile": "green chilli", "green chili": "green chilli", "red chile": "red chilli",
    # salt
    "kosher salt": "salt", "coarse salt": "salt", "coarse kosher salt": "salt", "sea salt": "salt",
    "coarse sea salt": "salt", "course sea salt": "salt", "flaky sea salt": "salt",
    "ground salt": "salt", "ground sea salt": "salt", "freshly ground salt": "salt",
    "table salt": "salt",
    # butter
    "unsalted butter": "butter", "salted butter": "butter", "sweet butter": "butter",
    "sweet cream butter": "butter", "sweet unsalted butter": "butter",
    # black and white pepper; plain "peppers" is caught by PLURAL_SYNONYMS first
    "pepper": "black pepper", "ground black pepper": "black pepper",
    "freshly ground black pepper": "black pepper", "fresh-ground black pepper": "black pepper",
    "ground pepper": "black pepper", "freshly ground pepper": "black pepper",
    "cracked black pepper": "black pepper", "cracked pepper": "black pepper",
    "freshly cracked black pepper": "black pepper", "freshly cracked pepper": "black pepper",
    "coarse black pepper": "black pepper", "coarse ground black pepper": "black pepper",
    "coarse ground pepper": "black pepper", "black ground pepper": "black pepper",
    "grnd black pepper": "black pepper", "grnd pepper": "black pepper",
    "freshly grnd black pepper": "black pepper", "freshly grnd pepper": "black pepper",
    "ground white pepper": "white pepper", "freshly ground white pepper": "white pepper",
    # bell peppers, one per colour
    "green pepper": "green bell pepper", "sweet green pepper": "green bell pepper",
    "green sweet pepper": "green bell pepper",
    "sweet red pepper": "red bell pepper", "red sweet pepper": "red bell pepper",
    "sweet red bell pepper": "red bell pepper", "red sweet bell pepper": "red bell pepper",
    "yellow pepper": "yellow bell pepper", "sweet yellow pepper": "yellow bell pepper",
    "yellow sweet pepper": "yellow bell pepper", "sweet yellow bell pepper": "yellow bell pepper",
    "orange pepper": "orange bell pepper",
    "sweet pepper": "bell pepper", "sweet bell pepper": "bell pepper",
    "jalepeno pepper": "jalapeno pepper",
    # oils
    "extra-virgin olive oil": "olive oil", "extra virgin olive oil": "olive oil",
    "extravirgin olive oil": "olive oil", "extra-virgin extra virgin olive oil": "olive oil",
    "extra virgin extra virgin olive oil": "olive oil", "virgin olive oil": "olive oil",
    "light olive oil": "olive oil", "extra light olive oil": "olive oil",
    "oil": "vegetable oil", "cooking oil": "vegetable oil", "salad oil": "vegetable oil",
    "wesson oil": "vegetable oil", "crisco oil": "vegetable oil", "veg oil": "vegetable oil",
    "veggie oil": "vegetable oil", "neutral oil": "vegetable oil",
    "neutral-tasting oil": "vegetable oil", "light oil": "vegetable oil",
    "frying oil": "vegetable oil",
    # sugars
    "white sugar": "sugar", "granulated sugar": "sugar", "granulated white sugar": "sugar",
    "white granulated sugar": "sugar",
    "confectioners sugar": "powdered sugar", "icing sugar": "powdered sugar",
    "light brown sugar": "brown sugar", "dark brown sugar": "brown sugar",
    "packed brown sugar": "brown sugar", "firmly packed brown sugar": "brown sugar",
    "light corn syrup": "corn syrup",
    # cream, spices, others
    "whipping cream": "heavy cream", "heavy whipping cream": "heavy cream",
    "ground cinnamon": "cinnamon", "ground nutmeg": "nutmeg", "ground cumin": "cumin",
    "yellow onion": "onion",
    "clove garlic": "garlic", "cloves garlic": "garlic", "garlic clove": "garlic",
    "stalk celery": "celery", "stalks celery": "celery", "rib celery": "celery",
    "ribs celery": "celery",
    "hot water": "water",
    "lean ground beef": "ground beef", "extra lean ground beef": "ground beef",
    "semi-sweet chocolate chip": "chocolate chip", "semisweet chocolate chip": "chocolate chip",
    "semi sweet chocolate chip": "chocolate chip",
    "apple cider vinegar": "cider vinegar",
    "mushroom soup": "cream of mushroom soup",
}
# Whole names matched before singularising, where the plural means something else
PLURAL_SYNONYMS = {"peppers": "bell pepper"}     # "pepper" alone is the spice
IRREGULAR = {"leaves": "leaf", "halves": "half", "loaves": "loaf",
             "chilies": "chilli", "chillies": "chilli", "chiles": "chilli"}   # not "chily"
KEEP_AS_IS = {"molasses", "hummus", "couscous", "asparagus", "swiss", "grits", "lemongrass"}

LEADING_DROP = {
    "fresh", "frozen", "grated", "shredded", "chopped", "diced", "sliced", "minced",
    "drained", "toasted", "mashed", "cooked", "softened", "melted", "beaten",
    "boneless", "skinless", "skinned", "large", "small", "medium", "handful",
    "very", "warm", "cold", "tap", "boiling", "chilled", "lukewarm",
    # units left in front of the name ("3 stalks celery", "T butter"); not rib or clove,
    # which start real names (rib roast, clove oil) and are handled in SYNONYMS
    "stalk", "stalks", "sprig", "sprigs", "dash", "dashes", "pinch", "knob", "drizzle",
    "stick", "sticks", "t", "c",
}
TRAILING_DROP = {"slices", "slice", "pieces", "piece", "bits", "mixed", "chunks", "cubes",
                 "stalk", "stalks"}


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
    if " ".join(words) in PLURAL_SYNONYMS:
        return PLURAL_SYNONYMS[" ".join(words)]
    words[-1] = singularise(words[-1])
    name = " ".join(words)
    return SYNONYMS.get(name, name)


def norm_title(title):
    title = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    title = title.lower()
    title = re.sub(r"[^a-z\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()
