"""Do the IDF * quantity-share edge weights help? Compares weightings on three tests.

weight = idf * share ** alpha, so alpha=0 is idf only and alpha=1 the linear share.

A. Substitution (proposal 3.1), see eval_common.py. Reported per case (micro), per pair (macro),
   and for pairs whose names are textually dissimilar vs similar (split at the median).
   Uses recipe features (mean of ingredient features), since Node2Vec cannot embed new recipes.
B. Same-dish retrieval, see eval_common.py.

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


'''
THIS FILE WILL BE REMOVED, I AM KEEPING IT JUST FOR REFERENCE. DO NOT USE
'''