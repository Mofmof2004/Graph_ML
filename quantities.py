"""Rough conversion of RecipeNLG ingredient lines to grams.

Example: "2 (16 oz.) pkg. frozen corn" -> 907 g, "1/2 c. brown sugar" -> 106 g, "3 eggs" -> 150 g.
Volumes are turned into grams with the density from ingredient_properties.csv,
counts ("3 eggs") with the weight of one piece. Not precise, but good enough to tell
a main ingredient from a pinch of spice.
"""

import re

import pandas as pd

PROPERTIES = "ingredient_properties.csv"

# Grams per unit
MASS = {
    "g": 1, "gram": 1, "grams": 1, "kg": 1000,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.6, "lbs": 453.6, "pound": 453.6, "pounds": 453.6,
}
# Millilitres per unit, multiplied by the ingredient density
VOLUME = {
    "tsp": 4.93, "tsps": 4.93, "teaspoon": 4.93, "teaspoons": 4.93,
    "tbsp": 14.79, "tbsps": 14.79, "tbs": 14.79, "tablespoon": 14.79, "tablespoons": 14.79,
    "c": 236.6, "cup": 236.6, "cups": 236.6,
    "pt": 473, "pint": 473, "pints": 473,
    "qt": 946, "quart": 946, "quarts": 946,
    "gal": 3785, "gallon": 3785, "gallons": 3785,
    "ml": 1, "l": 1000, "liter": 1000, "litre": 1000,
}
# Packages: use the size in brackets if given, else the ingredient's package_g, else this
CONTAINER = {
    "can": 400, "cans": 400, "pkg": 250, "pkgs": 250, "package": 250, "packages": 250,
    "box": 300, "boxes": 300, "jar": 350, "jars": 350, "bag": 350, "bags": 350,
    "carton": 450, "cartons": 450, "container": 450, "containers": 450, "tub": 450,
    "bottle": 500, "bottles": 500, "envelope": 30, "envelopes": 30, "packet": 30, "packets": 30,
}
# Fixed grams per unit, independent of the ingredient
FIXED = {
    "stick": 113, "sticks": 113, "clove": 5, "cloves": 5, "slice": 25, "slices": 25,
    "strip": 25, "strips": 25, "head": 600, "heads": 600, "bunch": 100, "bunches": 100,
    "stalk": 40, "stalks": 40, "rib": 40, "ribs": 40, "sprig": 1, "sprigs": 1,
    "sq": 28, "square": 28, "squares": 28, "loaf": 450, "loaves": 450,
    "dash": 0.5, "dashes": 0.5, "pinch": 0.35, "pinches": 0.35, "drop": 0.05, "drops": 0.05,
    "handful": 30, "scoop": 70, "scoops": 70, "ear": 100, "ears": 100, "ring": 30, "rings": 30,
}
# Units that multiply the weight of one piece
PIECE = {"doz": 12, "dozen": 12, "whole": 1}
# "6 pieces chicken" means portions, not whole items: weight of one piece, at most this
PORTION = {"piece": 150, "pieces": 150}
SIZE = {"large": 1.3, "medium": 1.0, "small": 0.7, "heaping": 1.3, "rounded": 1.2,
        "level": 1.0, "scant": 0.9, "generous": 1.2, "big": 1.3}

TO_TASTE_GRAMS = 1.0
MAX_GRAMS = 5000.0     # clip obvious parsing errors, such as "1 (50 lb.) bag"

FRACTIONS = {"½": " 1/2", "¼": " 1/4", "¾": " 3/4", "⅓": " 1/3", "⅔": " 2/3", "⅛": " 1/8"}
NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d*\.\d+|\d+)"
QTY = re.compile(rf"^\s*({NUM})(?:\s*(?:-|to|or)\s*({NUM}))?")
PAREN = re.compile(r"^\s*\(([^)]*)\)")
WORD = re.compile(r"^\s*([A-Za-z]+)\.?")
AMOUNT = re.compile(rf"({NUM})\s*-?\s*([A-Za-z]+)")


def singular(word):
    """Simple plural stripping, enough to find 'chicken breast' in '4 chicken breasts'."""
    if word in ("leaves", "halves", "loaves"):
        return word[:-3] + "f"
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def number(s):
    total = 0.0
    for part in s.split():
        if "/" in part:
            a, b = part.split("/")
            total += float(a) / float(b) if float(b) else 0.0
        else:
            total += float(part)
    return total


def unit_grams(unit, amount, density):
    """Grams for an amount in a mass or volume unit, or None for any other unit."""
    if unit in MASS:
        return amount * MASS[unit]
    if unit in VOLUME:
        return amount * VOLUME[unit] * density
    return None


class Properties:
    """Density, piece weight and package weight per ingredient, looked up by name."""

    def __init__(self, path=PROPERTIES):
        table = pd.read_csv(path).set_index("key")
        self.table = table
        self.default = table.loc["default"]
        self.cache = {}

    def match(self, name):
        """Exact name, else the longest key that ends the name ('cheddar cheese' -> 'cheese'),
        else the longest key contained in it as whole words ('cream of chicken soup' -> 'soup')."""
        if name in self.cache:
            return self.cache[name]
        key = name if name in self.table.index else None
        if key is None:
            words = name.split()
            for k in range(1, len(words)):
                if " ".join(words[k:]) in self.table.index:
                    key = " ".join(words[k:])
                    break
        if key is None:
            padded = f" {name} "
            hits = [k for k in self.table.index if k != "default" and f" {k} " in padded]
            key = max(hits, key=len) if hits else "default"
        self.cache[name] = key
        return key

    def refine(self, name, line):
        """A more specific key from the raw line, e.g. 'chicken' -> 'chicken breast' for
        '4 chicken breast halves', or 'pepper' -> 'jalapeno pepper'. The new key must contain
        the old one, unless the name had no match at all."""
        key = self.match(name)
        if line is None:
            return key
        words = re.sub(r"[^a-z\s-]", " ", line.lower()).split()
        text = " " + " ".join(singular(w) for w in words) + " "
        hits = [k for k in self.table.index if k != "default" and len(k) > len(key)
                and (key == "default" or key in k) and f" {k} " in text]
        return max(hits, key=len) if hits else key

    def get(self, name, line=None):
        row = self.table.loc[self.refine(name, line)]
        density = row["density_g_per_ml"] if pd.notna(row["density_g_per_ml"]) else self.default["density_g_per_ml"]
        package = row["package_g"] if pd.notna(row["package_g"]) else None
        # Packaged goods without a piece weight are counted in packages: "1 chocolate cake mix" -> 500 g
        if pd.notna(row["piece_g"]):
            piece = row["piece_g"]
        else:
            piece = package if package is not None else self.default["piece_g"]
        return float(density), float(piece), package


def line_to_grams(line, name, props):
    """Grams of ingredient `name` described by one raw ingredient line, or None if not parseable."""
    density, piece, package = props.get(name, line)
    text = line
    for k, v in FRACTIONS.items():
        text = text.replace(k, v)
    text = re.sub(r"(\d)-(\d+/\d+)", r"\1 \2", text)     # "1-1/2" -> "1 1/2"
    low = text.lower()

    m = QTY.match(text)
    if m:
        qty = number(m.group(1))
        if m.group(2):
            qty = (qty + number(m.group(2))) / 2         # "4 to 6 oz." -> 5 oz.
        rest = text[m.end():]
    else:
        if "to taste" in low or "as needed" in low or "optional" in low:
            return TO_TASTE_GRAMS
        qty, rest = 1.0, text                            # "dash of vanilla", "pinch of salt"

    # Package size in brackets: "2 (16 oz.) pkg. corn", "1 (1 lb. 2 oz.) jar Tang"
    size = None
    p = PAREN.match(rest)
    if p:
        grams = [unit_grams(u.lower(), number(a), density) for a, u in AMOUNT.findall(p.group(1))]
        grams = [g for g in grams if g is not None]
        if grams:
            size = sum(grams)
        rest = rest[p.end():]

    mult = 1.0
    unit = None
    while True:
        w = WORD.match(rest)
        if not w:
            break
        token = w.group(1)
        # "T." is a tablespoon and "t." a teaspoon in old American recipes
        word = {"T": "tbsp", "t": "tsp"}.get(token, token.lower())
        if word in SIZE:
            mult *= SIZE[word]
            rest = rest[w.end():]
            continue
        if word == "fl" or word == "fluid":                # "8 fl. oz." is about 8 oz. of water
            rest = rest[w.end():]
            continue
        unit = word
        break

    if unit in MASS or unit in VOLUME:
        if not m:
            return None
        grams = unit_grams(unit, qty, density)
    elif size is not None:
        grams = qty * size
    elif unit in CONTAINER:
        grams = qty * (package if package is not None else CONTAINER[unit]) * mult
    elif unit in FIXED:
        grams = qty * FIXED[unit] * (mult if m else 1.0)
    elif unit in PIECE and m:
        grams = qty * PIECE[unit] * piece * mult
    elif unit in PORTION and m:
        grams = qty * min(piece, PORTION[unit]) * mult
    elif m:
        grams = qty * piece * mult                        # "3 eggs", "2 large onions"
    else:
        return None
    return float(min(grams, MAX_GRAMS)) if grams > 0 else None


def recipe_grams(lines, ner, normalise, props):
    """Grams per normalised ingredient name for one recipe. Names without a parseable
    quantity map to None. Lines and NER names usually correspond by position; when the
    lists differ in length, each name is matched to the first line containing it."""
    out = {}
    same_length = len(lines) == len(ner)
    for k, raw in enumerate(ner):
        name = normalise(raw)
        if not name:
            continue
        if same_length:
            line = lines[k]
        else:
            line = next((l for l in lines if raw.lower() in l.lower()), None)
        g = line_to_grams(line, name, props) if line is not None else None
        if g is None:
            out.setdefault(name, None)
        else:
            out[name] = (out.get(name) or 0.0) + g
    return out
