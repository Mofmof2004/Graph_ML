"""Data and tests shared by evaluate_weights.py and evaluate_models.py.

Substitution test (proposal 3.1): for each recipe with an ingredient that has a known substitute,
make two copies, one with the substitute and one with a random ingredient of similar frequency.
A representation wins a case if the substitute copy is closer to the original.

Same-dish retrieval: recipes with the same title ("Banana Bread") should be each other's
nearest neighbours. Titles are not used by any representation.
"""

import json

import numpy as np
import pandas as pd

from normalisation import normalise

DATA_DIR = "data/processed"
SUBSTITUTIONS = "data/eval/substitutions_all.csv"   # from prepare_substitutions.py
WINDOW = 50              # random ingredients are drawn from this many frequency ranks around the target
MAX_GROUP = 30           # same-dish groups larger than this ("Fudge") are too generic to count

edges = np.load(f"{DATA_DIR}/edges.npz")
weights = np.load(f"{DATA_DIR}/edge_weights.npz")
recipes = pd.read_csv(f"{DATA_DIR}/recipes.csv")
ingredients = pd.read_csv(f"{DATA_DIR}/ingredients.csv")
ingredient_x = np.load(f"{DATA_DIR}/ingredient_x.npy")
with open(f"{DATA_DIR}/config.json") as f:
    SHARE_EXPONENT = json.load(f).get("share_exponent", 1.0)

edge_index, share = edges["all"], weights["share"]
idf = ingredients["idf"].to_numpy()
n_recipes, n_ingredients = len(recipes), len(ingredients)
name2id = dict(zip(ingredients["name"], ingredients["ingredient_id"]))

# Frequency ranks, for drawing random ingredients as common as the true one
order = np.argsort(-ingredients["recipe_count"].to_numpy())
rank = np.empty(n_ingredients, dtype=int)
rank[order] = np.arange(n_ingredients)

edges_of_recipe = pd.Series(np.arange(edge_index.shape[1])).groupby(edge_index[0]).apply(np.array).to_dict()
ings_of_recipe = {r: set(edge_index[1, ks]) for r, ks in edges_of_recipe.items()}


def edge_weight(ids, shares, alpha=SHARE_EXPONENT):
    """Edge weight as in build_graph.py; alpha=None gives unit weights."""
    if alpha is None:
        return np.ones(len(ids))
    return idf[ids] * shares ** alpha


def matched_random(rng, target, exclude):
    lo, hi = max(rank[target] - WINDOW, 0), min(rank[target] + WINDOW, n_ingredients - 1)
    for _ in range(50):
        c = order[rng.integers(lo, hi + 1)]
        if c not in exclude and c != target:
            return c
    return None


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def by_tercile(values, mask_fn):
    """Apply mask_fn to the lower, middle and upper third of `values`."""
    cuts = np.quantile(values, [1 / 3, 2 / 3])
    bucket = np.digitize(values, cuts)
    return [mask_fn(bucket == b) for b in range(3)], cuts


#----------Substitution cases----------#

def load_pairs(path=SUBSTITUTIONS):
    subs = pd.read_csv(path)
    pairs = {}
    for a, b in zip(subs["ingredient"], subs["substitute"]):
        a, b = normalise(a), normalise(b)
        if a in name2id and b in name2id and a != b:
            pairs.setdefault(name2id[a], []).append(name2id[b])
    return pairs


class SubstitutionCases:
    """One case per recipe: (recipe, ingredient ids, shares, position swapped, substitute, random)."""

    def __init__(self, pairs, seed=0):
        rng = np.random.default_rng(seed)
        self.cases = []
        for r in rng.permutation(n_recipes):
            ks = edges_of_recipe[r]
            ids = edge_index[1, ks]
            cand = [p for p, i in enumerate(ids) if i in pairs]
            if not cand:
                continue
            p = cand[rng.integers(len(cand))]
            s = pairs[ids[p]][rng.integers(len(pairs[ids[p]]))]
            rnd = None if s in ids else matched_random(rng, s, set(ids))
            if rnd is not None:
                self.cases.append((r, ids, share[ks], p, s, rnd))
        self.n_sources = len(pairs)

        # Each case belongs to one (ingredient, substitute) pair. Frequent pairs such as
        # butter -> olive oil produce many cases, so macro accuracy averages per pair first,
        # and confidence intervals resample pairs, since cases of one pair are not independent.
        case_pair = np.array([f"{c[1][c[3]]}>{c[4]}" for c in self.cases])
        self.pair_ids, pair_of_case = np.unique(case_pair, return_inverse=True)
        self.cases_of_pair = [np.flatnonzero(pair_of_case == k) for k in range(len(self.pair_ids))]

        # Text similarity of each pair under the ingredient features, split at the median over pairs.
        # Below the median are substitutes whose names do not look alike (molasses -> honey),
        # where a language model has least to go on and graph structure should matter most.
        a, b = np.array([k.split(">") for k in self.pair_ids], dtype=int).T
        self.pair_sim = (ingredient_x[a] * ingredient_x[b]).sum(1)
        self.sim_median = float(np.median(self.pair_sim))
        self.similar_pair = self.pair_sim >= self.sim_median
        self.swapped_share = np.array([c[2][c[3]] for c in self.cases])
        self.heldout = recipes["heldout"].to_numpy()[[c[0] for c in self.cases]]

    def __len__(self):
        return len(self.cases)

    def copies(self):
        """Ingredient lists and shares of the original, substitute and random copy of every case.
        The new ingredient takes over the quantity share of the one it replaces."""
        orig, plausible, random_, shares = [], [], [], []
        for _, ids, sh, p, s, rnd in self.cases:
            a, b = ids.copy(), ids.copy()
            a[p], b[p] = s, rnd
            orig.append(ids)
            plausible.append(a)
            random_.append(b)
            shares.append(sh)
        return orig, plausible, random_, shares

    def score(self, z_orig, z_plausible, z_random):
        """Per-case win (1, or 0.5 for a tie) and margin, from embeddings of the three copies."""
        zo, za, zb = unit(z_orig), unit(z_plausible), unit(z_random)
        ca, cb = (zo * za).sum(1), (zo * zb).sum(1)
        # only exact ties count as half (e.g. Jaccard, where both copies differ by one ingredient);
        # tiny differences are real when the swapped ingredient has a small weight
        win = np.where(ca == cb, 0.5, (ca > cb).astype(float))
        return win, ca - cb

    def macro(self, win, pair_mask=None):
        per_pair = np.array([win[ix].mean() for ix in self.cases_of_pair])
        return float(per_pair[pair_mask].mean() if pair_mask is not None else per_pair.mean())

    def summary(self, win):
        return {"micro": float(win.mean()), "macro": self.macro(win),
                "macro_dissimilar": self.macro(win, ~self.similar_pair),
                "macro_similar": self.macro(win, self.similar_pair),
                "micro_heldout_recipes": float(win[self.heldout].mean())}

    def diff_ci(self, win, win_ref, rng, n_boot=2000):
        """Macro accuracy difference win - win_ref with a 95% CI from resampling pairs,
        for all pairs and the dissimilar and similar halves."""
        d = np.array([(win[ix] - win_ref[ix]).mean() for ix in self.cases_of_pair])
        out = {}
        for label, mask in [("all", np.ones(len(d), bool)), ("dissimilar", ~self.similar_pair),
                            ("similar", self.similar_pair)]:
            dm = d[mask]
            boot = [dm[rng.integers(0, len(dm), len(dm))].mean() for _ in range(n_boot)]
            out[label] = [float(dm.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        return out


#----------Same-dish retrieval----------#

titles = (recipes["title"].str.lower().str.replace(r"[^a-z ]", " ", regex=True)
          .str.split().str.join(" ").to_numpy())
group_size = pd.Series(titles).map(pd.Series(titles).value_counts()).to_numpy()
dish_queries = np.flatnonzero((group_size >= 2) & (group_size <= MAX_GROUP))


def same_dish(z):
    """MRR and hits@10 of finding another recipe with the same title. Recipes with an
    all-zero embedding (heldout recipes in Node2Vec) are left out."""
    z = unit(np.asarray(z, dtype=np.float32))
    valid = np.linalg.norm(z, axis=1) > 0
    q = [i for i in dish_queries if valid[i] and ((titles == titles[i]) & valid).sum() >= 2]
    sims = z[q] @ z.T
    sims[:, ~valid] = -np.inf
    sims[np.arange(len(q)), q] = -np.inf
    ranks = np.array([np.flatnonzero(titles[np.argsort(-sims[k])] == titles[i])[0] for k, i in enumerate(q)])
    return float(np.mean(1 / (ranks + 1))), float(np.mean(ranks < 10))
