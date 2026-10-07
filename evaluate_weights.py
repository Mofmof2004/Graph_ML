"""Do the IDF * quantity-share edge weights help? Compares weightings on three tests.

weight = idf * share ** alpha, so alpha=0 is idf only and alpha=1 the linear share.

A. Substitution (proposal 3.1), see eval_common.py. Reported per case (micro), per pair (macro),
   and for pairs whose names are textually dissimilar vs similar (split at the median).
   Uses recipe features (mean of ingredient features), since Node2Vec cannot embed new recipes.
B. Node2Vec edge prediction: test edges vs frequency-matched fake edges (ROC AUC).
C. Same-dish retrieval, see eval_common.py.

Node2Vec embeddings are read from data/node2vec/eval_seed{seed}/node2vec_train{suffix}.npz,
one file per weighting and seed (see NODE2VEC below).
"""

import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

from eval_common import (SubstitutionCases, by_tercile, edge_index, edge_weight, edges,
                         ingredient_x, ings_of_recipe, load_pairs, matched_random, n_ingredients,
                         n_recipes, same_dish, share, unit)

NODE2VEC_DIR = "data/node2vec"
OUTPUT = "data/eval/weights.json"
SEEDS = [0, 1, 2]
ALPHAS = {"unweighted": None, "idf only (alpha 0)": 0.0, "idf x sqrt share (alpha 0.5)": 0.5,
          "idf x share (alpha 1)": 1.0}
NODE2VEC = {"unweighted": "", "idf only (alpha 0)": "_alpha0",
            "idf x sqrt share (alpha 0.5)": "_alpha05", "idf x share (alpha 1)": "_alpha1"}


def weighted_mean_features(lists, shares, alpha):
    return np.stack([(ingredient_x[ids] * w[:, None]).sum(0) / w.sum()
                     for ids, w in ((ids, edge_weight(ids, sh, alpha)) for ids, sh in zip(lists, shares))])


def load_node2vec(suffix, seed):
    return np.load(f"{NODE2VEC_DIR}/eval_seed{seed}/node2vec_train{suffix}.npz")


results = {}

#----------A. Substitution----------#

cases = SubstitutionCases(load_pairs())
orig, plausible, random_, shares = cases.copies()
print(f"A. substitution: {len(cases)} cases from {len(cases.pair_ids)} pairs ({cases.n_sources} source "
      f"ingredients); median pair text similarity {cases.sim_median:.2f}")
print(f"  {'':30s} {'micro':>6s} {'macro':>6s} {'dissim':>6s} {'similar':>7s}   micro by swapped share")
wins = {}
for name, alpha in ALPHAS.items():
    win, margin = cases.score(*(weighted_mean_features(l, shares, alpha) for l in (orig, plausible, random_)))
    wins[name] = win
    accs, cuts = by_tercile(cases.swapped_share, lambda m: float(win[m].mean()))
    row = results.setdefault("substitution", {})[name] = {
        **cases.summary(win), "margin": float(margin.mean()), "micro_by_swapped_share_tercile": accs}
    print(f"  {name:30s} {row['micro']:6.3f} {row['macro']:6.3f} {row['macro_dissimilar']:6.3f} "
          f"{row['macro_similar']:7.3f}   <{cuts[0]:.3f} / <{cuts[1]:.3f} / rest: "
          + " / ".join(f"{a:.3f}" for a in accs))

print("  macro accuracy vs unweighted, 95% CI from resampling pairs:")
rng = np.random.default_rng(0)
for name in list(ALPHAS)[1:]:
    out = results["substitution"][name]["macro_diff_vs_unweighted_ci95"] = \
        cases.diff_ci(wins[name], wins["unweighted"], rng)
    print(f"  {name:30s} " + "   ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in out.items()))

#----------B. Node2Vec edge prediction----------#

test = edges["test"]
share_of_edge = dict(zip(zip(edge_index[0], edge_index[1]), share))
test_share = np.array([share_of_edge[(r, i)] for r, i in test.T])
rng = np.random.default_rng(1)
fake = np.array([matched_random(rng, i, ings_of_recipe[r]) for r, i in test.T])
keep = fake != None   # noqa: E711, elementwise comparison on an object array
test, fake, test_share = test[:, keep], fake[keep].astype(int), test_share[keep]


def auc(pos, neg):
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


print(f"\nB. Node2Vec edge prediction: {test.shape[1]} test edges, mean ± std over seeds {SEEDS}")
for name, suffix in NODE2VEC.items():
    runs = []
    for seed in SEEDS:
        z = load_node2vec(suffix, seed)
        zr, zi = unit(z["recipe_embeddings"]), unit(z["ingredient_embeddings"])
        pos = (zr[test[0]] * zi[test[1]]).sum(1)
        neg = (zr[test[0]] * zi[fake]).sum(1)
        terc, cuts = by_tercile(test_share, lambda m: auc(pos[m], neg[m]))
        runs.append([auc(pos, neg)] + terc)
    runs = np.array(runs)
    results.setdefault("edge_auc", {})[name] = {
        "auc": runs[:, 0].mean(), "auc_std": runs[:, 0].std(),
        "auc_by_share_tercile": runs[:, 1:].mean(0).tolist()}
    print(f"  {name:30s} AUC {runs[:, 0].mean():.3f} ± {runs[:, 0].std():.3f}  "
          f"by share (<{cuts[0]:.3f} / <{cuts[1]:.3f} / rest): " + " / ".join(f"{a:.3f}" for a in runs[:, 1:].mean(0)))

#----------C. Same-dish retrieval----------#

print("\nC. same-dish retrieval (MRR / hits@10)")
for name, alpha in ALPHAS.items():
    w = edge_weight(edge_index[1], share, alpha)
    adj = csr_matrix((w, (edge_index[0], edge_index[1])), shape=(n_recipes, n_ingredients))
    row = {"features": same_dish(adj @ ingredient_x), "multi_hot": same_dish(adj.toarray())}
    n2v = np.array([same_dish(load_node2vec(NODE2VEC[name], seed)["recipe_embeddings"]) for seed in SEEDS])
    row["node2vec"] = n2v.mean(0).tolist()
    row["node2vec_mrr_std"] = float(n2v[:, 0].std())
    results.setdefault("same_dish", {})[name] = row
    print(f"  {name:30s} features {row['features'][0]:.3f} / {row['features'][1]:.3f}   "
          f"multi-hot {row['multi_hot'][0]:.3f} / {row['multi_hot'][1]:.3f}   "
          f"node2vec {row['node2vec'][0]:.3f} ± {row['node2vec_mrr_std']:.3f} / {row['node2vec'][1]:.3f}")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nsaved to {OUTPUT}")
