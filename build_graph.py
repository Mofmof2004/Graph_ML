import json
import random
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer

from normalisation import normalise, norm_title
from quantities import Properties, recipe_grams

RAW = "data/raw/full_dataset.csv"
OUT = "data/interim"
SLICE_SIZE = 10_000          # number of rows read from the CSV
MIN_INGS = 2
MIN_ING_COUNT = 5            # ingredient must appear in at least this many recipes
DROP_INGREDIENTS = set()     # hub ingredients to remove, e.g. {"salt", "water"}, empty for now
CHECK_KEYWORDS = ["vindaloo", "carbonara", "paella", "goulash"]  # placeholder until the dish list exists


SEED = 0
HELDOUT_FRAC = 0.05   # fraction of recipes reserved for the substitution benchmark
VAL_FRAC = 0.10       # fraction of the remaining edges used for validation
TEST_FRAC = 0.10      # fraction of the remaining edges used for testing

PRINT_DATA_NORMALISATION = True

FEATURE_MODEL = "all-mpnet-base-v2"   # small, fast text model, 384 numbers per text
COMPUTE_TITLE_FEATURES = True        # also embed recipe titles, as an optional extra
PROPERTIES = "ingredient_properties.csv"   # density (g/ml), piece and package weight per ingredient
SHARE_EXPONENT = 0.5   # edge weight = idf * share ** SHARE_EXPONENT; 1 = linear share, 0 = idf only


OUT = "data/processed"
# Load the dataset
slice_df = pd.read_csv(RAW, nrows=SLICE_SIZE, usecols=["title", "ingredients", "link", "source", "NER"])
ner = slice_df["NER"].apply(json.loads)
raw_counts = Counter(name for names in ner for name in names)
total_mentions = sum(raw_counts.values())
if PRINT_DATA_NORMALISATION:
    print("unique raw names:", len(raw_counts))


# Normalise the ingredient names and count the occurrences of each normalised name
norm_counts = Counter()
groups = defaultdict(list)
for raw, c in raw_counts.items():
    n = normalise(raw)
    if n:
        norm_counts[n] += c
        groups[n].append((raw, c))

if PRINT_DATA_NORMALISATION:
    print("unique after normalising:", len(norm_counts))


for k in [2, 5, 10]:
    covered = sum(c for c in norm_counts.values() if c >= k)
    kept = sum(1 for c in norm_counts.values() if c >= k)
    if PRINT_DATA_NORMALISATION:
        print(f"min count {k:2d}: keep {kept:5d} names, covering {covered / total_mentions:.1%}")

if PRINT_DATA_NORMALISATION:
    print("\ntop 50 after normalising:")
    for name, c in norm_counts.most_common(50):
        print(f"  {c:6d}  {name}")


# Print the biggest merges of raw names into normalised names
if PRINT_DATA_NORMALISATION:
    print("\nbiggest merges:")
    merged = sorted((g for g in groups.items() if len(g[1]) > 1),
                    key=lambda g: -sum(c for _, c in g[1]))
    for canon, forms in merged[:40]:
        top_forms = sorted(forms, key=lambda f: -f[1])[:5]
        print(f"  {canon:25s} <- {top_forms}")



# Build one row per recipe with its normalised ingredient set
recipes = pd.DataFrame({
    "row_id": slice_df.index,
    "title": slice_df["title"],
    "title_norm": slice_df["title"].apply(norm_title),
    "link": slice_df["link"],
    "source": slice_df["source"],
    "ings": ner.apply(lambda names: "|".join(sorted({n for n in map(normalise, names) if n}))),
})
recipes["n_ings"] = recipes["ings"].str.count(r"\|") + (recipes["ings"] != "")

# Filter and deduplicate
n_start = len(recipes)
recipes = recipes[recipes["n_ings"] >= MIN_INGS]
n_min = len(recipes)
recipes = recipes.drop_duplicates(subset="link")
n_link = len(recipes)
recipes = recipes.drop_duplicates(subset=["title_norm", "ings"])
n_dedup = len(recipes)

print(f"\nrecipes in slice: {n_start:,}")
print(f"dropped {n_start - n_min:,} with fewer than {MIN_INGS} ingredients")
print(f"dropped {n_min - n_link:,} duplicate links")
print(f"dropped {n_link - n_dedup:,} duplicate title + ingredient sets")
print(f"remaining: {n_dedup:,}")
print(recipes["source"].value_counts())

# Keyword coverage report
for kw in CHECK_KEYWORDS:
    hits = recipes["title_norm"].str.contains(rf"\b{re.escape(kw)}\b", regex=True).sum()
    print(f"keyword {kw!r}: {hits} recipes")


# Drop ingredients that are too rare or in the drop list, and recipes that are too small
ing_lists = recipes["ings"].str.split("|")
edges_before = ing_lists.str.len().sum()

round_ = 0
while True:
    round_ += 1
    counts = Counter(i for l in ing_lists for i in l)
    vocab = {i for i, c in counts.items() if c >= MIN_ING_COUNT and i not in DROP_INGREDIENTS}

    filtered = ing_lists.apply(lambda l: [i for i in l if i in vocab])
    keep_recipe = filtered.str.len() >= MIN_INGS

    changed = (filtered.str.len() != ing_lists.str.len()).any() or not keep_recipe.all()
    ing_lists = filtered[keep_recipe]
    print(f"round {round_}: {len(vocab):,} ingredients, {len(ing_lists):,} recipes")
    if not changed:
        break

recipes = recipes.loc[ing_lists.index].copy()
recipes["ings"] = ing_lists.str.join("|")
recipes["n_ings"] = ing_lists.str.len()

edges_after = recipes["n_ings"].sum()
print(f"\nfinal: {len(vocab):,} ingredients, {len(recipes):,} recipes, {edges_after:,} edges")
print(f"kept {edges_after / edges_before:.1%} of edges")
print(f"ingredients per recipe: median {recipes['n_ings'].median():.0f}, max {recipes['n_ings'].max()}")

# How dominant are the hub ingredients?
print("\nmost connected ingredients:")
for name, c in Counter(i for l in ing_lists for i in l).most_common(10):
    print(f"  {name:20s} in {c / len(recipes):.0%} of recipes")

#----------Graph Construction----------#

# Ingredient ids: alphabetical, so the numbering is stable across runs
ingredient_names = sorted(vocab)
ing2id = {name: i for i, name in enumerate(ingredient_names)}

# Recipe ids: 
recipes = recipes.reset_index(drop=True)
recipes["recipe_id"] = recipes.index
ing_lists = recipes["ings"].str.split("|")

# Edge list: one edge per (recipe, ingredient) pair.
# Example: Carbonara (0) has egg, guanciale, pecorino. Omelette (1) has butter, egg.

# Recipe side of each edge: repeat each recipe id once per ingredient it has.
# Recipe ids [0, 1] with ingredient counts [3, 2] become [0, 0, 0, 1, 1]
edge_recipe = np.repeat(recipes["recipe_id"].to_numpy(), recipes["n_ings"].to_numpy())

# Ingredient side of each edge: all ingredients of all recipes, in the same order, as ids.
# [egg, guanciale, pecorino, butter, egg] becomes [1, 3, 5, 0, 1]
edge_ing = np.array([ing2id[i] for l in ing_lists for i in l])

# Put the two sides together. Each column is one edge (recipe id, ingredient id).
# [[0, 0, 0, 1, 1],
#  [1, 3, 5, 0, 1]]   column 0 means "recipe 0 contains ingredient 1"
edge_index = np.stack([edge_recipe, edge_ing])

# Ingredient table: DataFrame describing every ingredient node: its id, its name, and how many recipes use it.
ing_counts = Counter(i for l in ing_lists for i in l)
ingredients = pd.DataFrame({
    "ingredient_id": range(len(ingredient_names)),
    "name": ingredient_names,
    "recipe_count": [ing_counts[n] for n in ingredient_names],
})

# Sanity checks: the script stops with an error if any of these is false
assert edge_index.shape[1] == recipes["n_ings"].sum()
assert edge_index[0].max() == len(recipes) - 1
assert edge_index[1].max() == len(ingredients) - 1
assert len(np.unique(edge_index[1])) == len(ingredients)   # every ingredient has an edge

print(f"\nedge_index shape: {edge_index.shape}")
print("first 5 edges:")
for r, i in edge_index[:, :5].T:
    print(f"  recipe {r} ({recipes.loc[r, 'title']!r}) -> ingredient {i} ({ingredient_names[i]!r})")

#----------Ingredient Quantities----------#

# Grams of every edge, parsed from the raw ingredient lines.
# Example: "1/2 c. brown sugar" -> 106 g (via density), "3 eggs" -> 150 g (via piece weight)
props = Properties(PROPERTIES)
raw_lines = slice_df["ingredients"].apply(json.loads)
parsed = [recipe_grams(raw_lines[row], ner[row], normalise, props) for row in recipes["row_id"]]
edge_grams = np.array([parsed[r].get(i, None) for r, l in enumerate(ing_lists) for i in l], dtype=object)

# Edges without a parseable quantity ("butter", "juice of 1/2 lemon") get the ingredient's
# median over the recipes where it was parsed, or the overall median if it never was
is_parsed = np.array([g is not None for g in edge_grams])
grams_df = pd.DataFrame({"ing": edge_index[1][is_parsed], "g": edge_grams[is_parsed].astype(float)})
median_grams = grams_df.groupby("ing")["g"].median().reindex(range(len(ingredient_names)))
median_grams = median_grams.fillna(grams_df["g"].median()).to_numpy()
edge_grams = np.where(is_parsed, edge_grams, median_grams[edge_index[1]]).astype(np.float64)

# Quantity share: fraction of the recipe's total grams. 500 g chicken in a 1 kg recipe gets 0.5,
# 200 g chicken in a 2 kg recipe gets 0.1. Shares are independent of how many servings a recipe makes.
recipe_total = np.bincount(edge_index[0], weights=edge_grams, minlength=len(recipes))
edge_share = edge_grams / recipe_total[edge_index[0]]
recipes["total_grams"] = recipe_total.round(1)

print(f"\nquantity parsed for {is_parsed.mean():.1%} of edges, the rest use the ingredient median")
print(f"recipe weight in grams: median {np.median(recipe_total):.0f}, max {recipe_total.max():.0f}")
example = recipes.index[recipes["n_ings"].between(4, 6)][0]
print(f"example: {recipes.loc[example, 'title']!r}")
for k in np.flatnonzero(edge_index[0] == example):
    print(f"  {ingredient_names[edge_index[1, k]]:20s} {edge_grams[k]:7.1f} g  share {edge_share[k]:.2f}")

#----------Edge Splits----------#

rng = np.random.default_rng(SEED)
n_recipes = len(recipes)
n_edges = edge_index.shape[1]

# Held-out recipes: all their edges go to the heldout split
heldout_recipes = rng.choice(n_recipes, size=int(HELDOUT_FRAC * n_recipes), replace=False)
recipes["heldout"] = recipes["recipe_id"].isin(heldout_recipes)
is_heldout = np.isin(edge_index[0], heldout_recipes)

heldout_idx = np.flatnonzero(is_heldout)      # positions of heldout edges
remaining = np.flatnonzero(~is_heldout)       # positions of all other edges

# Pick one edge per recipe and one per ingredient that must stay in train
shuffled = rng.permutation(remaining)
_, first_per_recipe = np.unique(edge_index[0, shuffled], return_index=True)
_, first_per_ing = np.unique(edge_index[1, shuffled], return_index=True)
must_train = np.union1d(shuffled[first_per_recipe], shuffled[first_per_ing])

# Split the other edges randomly into val, test and the rest of train
others = rng.permutation(np.setdiff1d(remaining, must_train))
n_val = int(VAL_FRAC * len(remaining))
n_test = int(TEST_FRAC * len(remaining))

val_idx = others[:n_val]
test_idx = others[n_val:n_val + n_test]
train_idx = np.concatenate([must_train, others[n_val + n_test:]])

splits = {
    "train": edge_index[:, train_idx],
    "val": edge_index[:, val_idx],
    "test": edge_index[:, test_idx],
    "heldout": edge_index[:, heldout_idx],
}

# Sanity checks
assert len(train_idx) + len(val_idx) + len(test_idx) + len(heldout_idx) == n_edges
assert len(np.unique(np.concatenate([train_idx, val_idx, test_idx, heldout_idx]))) == n_edges
non_heldout_recipes = recipes.loc[~recipes["heldout"], "recipe_id"].to_numpy()
assert np.isin(non_heldout_recipes, splits["train"][0]).all()

ings_outside_heldout = np.unique(edge_index[1, remaining])
assert np.isin(ings_outside_heldout, splits["train"][1]).all()
n_only_heldout = len(ingredients) - len(ings_outside_heldout)

print(f"\nheldout recipes: {len(heldout_recipes):,}")
for name, e in splits.items():
    print(f"{name:8s} {e.shape[1]:>9,} edges ({e.shape[1] / n_edges:.1%})")
print(f"ingredients appearing only in heldout recipes: {n_only_heldout}")


#----------Edge Weights----------#

# Weight = IDF of the ingredient * its quantity share in the recipe, so rare ingredients
# that make up much of a recipe count most, and a pinch of salt counts very little.
# The share is compressed (square root by default): with the linear share, heavy but generic
# bulk such as 2 qt. water or 3 c. flour dominates the recipe, which made similarity worse.
# IDF uses the training edges only, so val/test edges do not leak into it.
n_train_recipes = len(np.unique(splits["train"][0]))
train_df = np.bincount(splits["train"][1], minlength=len(ingredient_names))
idf = np.log((1 + n_train_recipes) / (1 + train_df)) + 1.0
edge_weight = (idf[edge_index[1]] * edge_share ** SHARE_EXPONENT).astype(np.float32)
weight_splits = {"train": edge_weight[train_idx], "val": edge_weight[val_idx],
                 "test": edge_weight[test_idx], "heldout": edge_weight[heldout_idx]}

ingredients["property_key"] = [props.match(n) for n in ingredient_names]
density, piece, package = zip(*(props.get(n) for n in ingredient_names))
ingredients["density_g_per_ml"], ingredients["piece_g"], ingredients["package_g"] = density, piece, package
ingredients["median_grams"] = median_grams.round(1)
ingredients["idf"] = idf.round(4)

print(f"\nedge weight: median {np.median(edge_weight):.3f}, max {edge_weight.max():.3f}")
for name in ["salt", "sugar", "flour", "chicken"]:
    if name in ing2id:
        w = edge_weight[edge_index[1] == ing2id[name]]
        print(f"  {name:10s} idf {idf[ing2id[name]]:.2f}, median weight {np.median(w):.3f}")


#----------Feature Construction----------#
model = SentenceTransformer(FEATURE_MODEL)   # downloads the model on first run

# Ingredient features: embed each ingredient name, scaled to length 1
ingredient_x = model.encode(ingredient_names, batch_size=256,
                            normalize_embeddings=True, show_progress_bar=True).astype(np.float32)


# Recipe features: mean of the ingredient features, using only the given edges.
# With weights, a weighted mean, so the main ingredients dominate the recipe feature.
def mean_ingredient_features(edges, weights=None):
    if weights is None:
        weights = np.ones(edges.shape[1], dtype=np.float32)
    # Adjacency matrix: row = recipe, column = ingredient, edge weight where an edge exists
    adj = csr_matrix((weights, (edges[0], edges[1])), shape=(n_recipes, len(ingredient_names)))
    sums = adj @ ingredient_x                              # weighted sum of each recipe's ingredient vectors
    totals = np.asarray(adj.sum(axis=1))                   # total weight per recipe
    return (sums / np.maximum(totals, 1e-12)).astype(np.float32)


train_edges = np.concatenate([splits["train"], splits["heldout"]], axis=1)
train_weights = np.concatenate([weight_splits["train"], weight_splits["heldout"]])
recipe_x_full = mean_ingredient_features(edge_index)
recipe_x_train = mean_ingredient_features(train_edges)
recipe_x_full_weighted = mean_ingredient_features(edge_index, edge_weight)
recipe_x_train_weighted = mean_ingredient_features(train_edges, train_weights)

if COMPUTE_TITLE_FEATURES:
    title_x = model.encode(recipes["title"].astype(str).tolist(), batch_size=256,
                           normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

# Sanity check: do the ingredient features make sense?
print("butter" in ing2id, len(ing2id), ingredient_names[:10])

print(f"\ningredient_x {ingredient_x.shape}, recipe_x {recipe_x_full.shape}")
for name in ["paprika", "butter", "cilantro", "beef"]:
    if name in ing2id:
        sims = ingredient_x @ ingredient_x[ing2id[name]]
        closest = np.argsort(-sims)[1:6]
        print(f"  {name:10s} -> {[ingredient_names[j] for j in closest]}")

#----------Save Outputs----------#

import os
os.makedirs(OUT, exist_ok=True)

# Tables for humans: recipes, ingredients, and how raw names were cleaned
recipes[["recipe_id", "row_id", "title", "link", "source", "ings", "n_ings", "total_grams", "heldout"]] \
    .to_csv(f"{OUT}/recipes.csv", index=False)
ingredients.to_csv(f"{OUT}/ingredients.csv", index=False)

ingredient_map = pd.DataFrame(
    [(raw, canon, c, canon in ing2id) for canon, forms in groups.items() for raw, c in forms],
    columns=["raw", "canonical", "count", "kept"],
).sort_values("count", ascending=False)
ingredient_map.to_csv(f"{OUT}/ingredient_map.csv", index=False)

# Edges: all splits in one compressed file, each array of shape [2, E]
np.savez_compressed(f"{OUT}/edges.npz", all=edge_index, **splits)

# Edge weights, aligned with the edges of the same name in edges.npz, plus the parts they are made of
np.savez_compressed(f"{OUT}/edge_weights.npz", all=edge_weight, **weight_splits,
                    grams=edge_grams.astype(np.float32), share=edge_share.astype(np.float32),
                    parsed=is_parsed)

# Features: one row per node id
np.save(f"{OUT}/ingredient_x.npy", ingredient_x)
np.save(f"{OUT}/recipe_x_full.npy", recipe_x_full)
np.save(f"{OUT}/recipe_x_train.npy", recipe_x_train)
np.save(f"{OUT}/recipe_x_full_weighted.npy", recipe_x_full_weighted)
np.save(f"{OUT}/recipe_x_train_weighted.npy", recipe_x_train_weighted)
if COMPUTE_TITLE_FEATURES:
    np.save(f"{OUT}/title_x.npy", title_x)

# Stats for the report, and the settings used, so the graph can be rebuilt exactly
stats = {
    "n_recipes": int(n_recipes),
    "n_ingredients": int(len(ingredients)),
    "n_edges": int(n_edges),
    "split_sizes": {k: int(v.shape[1]) for k, v in splits.items()},
    "dropped_recipes": {
        "too_few_ingredients": int(n_start - n_min),
        "duplicate_links": int(n_min - n_link),
        "duplicate_title_and_ingredients": int(n_link - n_dedup),
    },
    "recipe_degree": {"min": int(recipes["n_ings"].min()), "median": float(recipes["n_ings"].median()),
                      "mean": float(recipes["n_ings"].mean()), "max": int(recipes["n_ings"].max())},
    "quantity_parsed_frac": float(is_parsed.mean()),
    "recipe_grams": {"median": float(np.median(recipe_total)), "max": float(recipe_total.max())},
    "edge_weight": {"median": float(np.median(edge_weight)), "max": float(edge_weight.max())},
    "top_ingredients": [[n, int(c)] for n, c in
                        ingredients.nlargest(50, "recipe_count")[["name", "recipe_count"]].values],
}
config = {
    "slice_size": SLICE_SIZE, "min_ings": MIN_INGS, "min_ing_count": MIN_ING_COUNT,
    "drop_ingredients": sorted(DROP_INGREDIENTS), "seed": SEED, "heldout_frac": HELDOUT_FRAC,
    "val_frac": VAL_FRAC, "test_frac": TEST_FRAC, "feature_model": FEATURE_MODEL,
    "edge_weight": "idf(train edges) * quantity share ** share_exponent",
    "share_exponent": SHARE_EXPONENT, "properties": PROPERTIES,
}
with open(f"{OUT}/stats.json", "w") as f:
    json.dump(stats, f, indent=2)
with open(f"{OUT}/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\nsaved everything to {OUT}/")
