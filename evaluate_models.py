"""Compare the GAT with the baselines and ablations from the proposal.

Methods
  jaccard          shared ingredients only (multi-hot vectors)
  mean features    mean of the ingredient text embeddings, i.e. the GAT's input, no graph
  sbert list       sentence transformer on the comma-joined ingredient list (language-only baseline)
  node2vec         structure only, weighted walks; same-dish test only, since it cannot embed new recipes
  gat_w            GAT + GraphSAGE with the edge weights as attention edge feature (our model)
  gat              same without the edge weights
  mean             mean aggregation instead of attention (ablation)
  mlp              no edges, node features only (ablation)

Tests (see eval_common.py): substitution, reported overall and for textually dissimilar pairs,
where a language model has least to go on, and same-dish retrieval.
Trained models are read from data/gat/{variant}_s{seed}.pt (train_gat.py).
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr

import train_gat
from eval_common import (SubstitutionCases, edge_index, edge_weight, edges_of_recipe, ingredient_x,
                         ingredients, load_pairs, n_ingredients, n_recipes, same_dish, share)

SEEDS = [0, 1, 2]
GAT_VARIANTS = ["gat_w", "gat", "mean", "mlp"]
NODE2VEC = "data/node2vec/eval_seed{seed}/node2vec_train_alpha05.npz"
OUTPUT = "data/eval/models.json"
with open(f"{train_gat.DATA_DIR}/config.json") as f:
    FEATURE_MODEL = json.load(f)["feature_model"]

names = ingredients["name"].to_numpy()


def multi_hot(lists):
    out = np.zeros((len(lists), n_ingredients), np.float32)
    for k, ids in enumerate(lists):
        out[k, ids] = 1.0
    return out


def mean_features(lists):
    return np.stack([ingredient_x[ids].mean(0) for ids in lists])


_sbert = None


def sbert_list(lists):
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer(FEATURE_MODEL)
    texts = [", ".join(sorted(names[ids])) for ids in lists]
    return _sbert.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)


graph = train_gat.RecipeGraph()


def gat_embedder(variant, seed):
    model, _ = train_gat.load(f"{variant}_s{seed}")
    return lambda lists, shares: train_gat.embed_recipes(
        model, graph, lists, [edge_weight(ids, sh) for ids, sh in zip(lists, shares)])


cases = SubstitutionCases(load_pairs())
orig, plausible, random_, shares = cases.copies()
all_lists = [edge_index[1, edges_of_recipe[r]] for r in range(n_recipes)]
all_shares = [share[edges_of_recipe[r]] for r in range(n_recipes)]

# name -> list of (embed function taking (lists, shares)); several entries = seeds
methods = {
    "jaccard": [lambda l, s: multi_hot(l)],
    "mean features": [lambda l, s: mean_features(l)],
    "sbert list": [lambda l, s: sbert_list(l)],
    **{v: [gat_embedder(v, seed) for seed in SEEDS] for v in GAT_VARIANTS},
}

results = {"substitution": {}, "same_dish": {}}
wins = {}
print(f"substitution: {len(cases)} cases, {len(cases.pair_ids)} pairs, median pair text similarity "
      f"{cases.sim_median:.2f}; same-dish: MRR / hits@10. GAT rows are mean ± std over seeds {SEEDS}")
print(f"{'':15s} {'micro':>6s} {'macro':>6s} {'dissim':>13s} {'similar':>6s} {'heldout':>7s}   {'same-dish':>13s}")
for name, embedders in methods.items():
    runs, dish = [], []
    for embed in embedders:
        win, _ = cases.score(embed(orig, shares), embed(plausible, shares), embed(random_, shares))
        runs.append(win)
        dish.append(same_dish(embed(all_lists, all_shares)))
    runs, dish = np.array(runs), np.array(dish)
    wins[name] = runs.mean(0)           # per-case win averaged over seeds, for the paired comparisons
    per_seed = [cases.summary(w) for w in runs]
    row = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
    row["macro_dissimilar_std"] = float(np.std([p["macro_dissimilar"] for p in per_seed]))
    results["substitution"][name] = row
    results["same_dish"][name] = {"mrr": float(dish[:, 0].mean()), "mrr_std": float(dish[:, 0].std()),
                                  "hits10": float(dish[:, 1].mean())}
    std = f" ± {row['macro_dissimilar_std']:.3f}" if len(runs) > 1 else " " * 8
    print(f"{name:15s} {row['micro']:6.3f} {row['macro']:6.3f} {row['macro_dissimilar']:6.3f}{std} "
          f"{row['macro_similar']:6.3f} {row['micro_heldout_recipes']:7.3f}   "
          f"{dish[:, 0].mean():.3f} / {dish[:, 1].mean():.3f}" + (f" (MRR ± {dish[:, 0].std():.3f})" if len(dish) > 1 else ""))

n2v = np.array([same_dish(np.load(NODE2VEC.format(seed=s))["recipe_embeddings"]) for s in SEEDS])
results["same_dish"]["node2vec"] = {"mrr": float(n2v[:, 0].mean()), "mrr_std": float(n2v[:, 0].std()),
                                    "hits10": float(n2v[:, 1].mean())}
print(f"{'node2vec':15s} {'':6s} {'':6s} {'':13s} {'':6s} {'':7s}   {n2v[:, 0].mean():.3f} / {n2v[:, 1].mean():.3f}")

#----------Paired comparisons on the substitution test----------#

print("\nmacro accuracy differences, 95% CI from resampling pairs:")
rng = np.random.default_rng(0)
# The mean-aggregation rows were added after the first results showed it doing best on dissimilar pairs
for a, b in [("gat_w", "sbert list"), ("gat_w", "mean features"), ("gat_w", "gat"),
             ("gat_w", "mean"), ("gat_w", "mlp"),
             ("mean", "mean features"), ("mean", "sbert list"), ("mean", "mlp")]:
    out = results.setdefault("comparisons", {})[f"{a} - {b}"] = cases.diff_ci(wins[a], wins[b], rng)
    print(f"  {a:6s} - {b:14s} " + "   ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in out.items()))

#----------Does attention learn the edge weights by itself?----------#

# Layer-1 attention of the GAT trained without edge weights, on ingredient -> recipe messages,
# compared with the IDF * share weight of the same edge (rank correlation within each recipe).
print("\nattention of the GAT without edge weights vs the prior edge weight:")
x, ei, w = graph.inputs(graph.split["train"], graph.split_w["train"])
for seed in SEEDS:
    model, _ = train_gat.load(f"gat_s{seed}")
    _, (att_ei, att) = model(x, ei, w, return_attention=True)
    att = att.mean(1).detach().cpu().numpy()
    src, dst = att_ei.cpu().numpy()
    to_recipe = dst < graph.R
    prior = w.cpu().numpy()[to_recipe]
    att, recipe = att[to_recipe], dst[to_recipe]
    rhos = []
    for r in np.unique(recipe):
        m = recipe == r
        if m.sum() >= 4:
            rho = spearmanr(att[m], prior[m]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
    results.setdefault("attention_vs_prior", []).append(float(np.mean(rhos)))
    print(f"  seed {seed}: mean within-recipe Spearman {np.mean(rhos):+.3f} over {len(rhos)} recipes")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved to {OUTPUT}")
