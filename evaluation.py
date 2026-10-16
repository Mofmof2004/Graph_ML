import json
import re
from collections.abc import Iterable
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from eval_common import calculate_edge_weights


def get_random_ingredients(
    k, ingredients=None, used_more_than=50, random_state=None
):
    """Return up to ``k`` random ingredient names above a usage threshold."""
    if ingredients is None:
        ingredients = pd.read_csv("data/processed/ingredients.csv")
    if not isinstance(ingredients, pd.DataFrame):
        raise TypeError("ingredients must be a pandas DataFrame.")
    if "name" not in ingredients or "recipe_count" not in ingredients:
        raise ValueError(
            "ingredients must contain 'name' and 'recipe_count' columns."
        )
    if not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer.")
    if k < 0:
        raise ValueError("k must be non-negative.")
    if not isinstance(used_more_than, (int, float, np.integer, np.floating)):
        raise TypeError("used_more_than must be numeric.")

    candidates = ingredients.loc[
        ingredients["recipe_count"] > used_more_than, "name"
    ].dropna().astype(str).str.strip().str.lower()
    candidates = candidates[candidates.ne("")].drop_duplicates().to_numpy()
    if k > len(candidates):
        raise ValueError(
            f"k={k} exceeds the {len(candidates)} ingredients used more than "
            f"{used_more_than} times."
        )

    generator = np.random.default_rng(random_state)
    return generator.choice(candidates, size=k, replace=False).tolist()


def plot_ingredient_auc(
    ingredient,
    embeddings_by_name,
    recipes_df=None,
    test_size=0.2,
    random_state=0,
    title=None,
    figsize=(8, 6),
    output_dir="evaluation_outputs",
):
    """Plot and save held-out ROC curves for one or more ingredients.

    A single ingredient returns ``(results, figure, axis)`` as before. For
    multiple ingredients, it returns a dictionary mapping each ingredient to
    that tuple. Plots are saved under ``output_dir/plots``; JSON results and
    an aggregate CSV are saved under     ``output_dir/results``. The aggregate statistics file reports the mean,
    standard deviation, minimum, and maximum for every numeric field in the
    evaluation results.
    """
    if recipes_df is None:
        recipes_df = pd.read_csv("data/processed/recipes.csv")

    if "ings" not in recipes_df:
        raise ValueError("recipes_df must contain an 'ings' column.")

    if isinstance(ingredient, str):
        ingredients = [ingredient]
        single_ingredient = True
    elif isinstance(ingredient, Iterable):
        ingredients = list(ingredient)
        single_ingredient = False


    output_path = Path(output_dir)
    plots_path = output_path / "plots"
    results_path = output_path / "results"
    plots_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    returned_results = {}
    for raw_ingredient in ingredients:
        if not isinstance(raw_ingredient, str):
            raise TypeError("every ingredient must be a string.")
        ingr = raw_ingredient.strip().lower()
        if not ingr:
            raise ValueError("ingredient must not be empty.")

        labels = recipes_df["ings"].fillna("").map(
            lambda value: ingr in {
                item for item in value.split("|") if item.strip()
            }
        ).to_numpy(dtype=np.int64)


        if labels.sum() == 0 or labels.sum() == len(labels):
            raise ValueError(
                f"'{ingr}' must occur in both positive and negative recipe classes."
            )

        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        figure, axis = plt.subplots(figsize=figsize)
        results = {}
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

        for (name, embedding_values), color in zip(
            embeddings_by_name.items(), colors
        ):
            embeddings = np.asarray(embedding_values)
            if len(embeddings) != len(recipes_df):
                raise ValueError(
                    f"'{name}' embeddings must match the number of recipe rows."
                )

            classifier = LogisticRegression(
                max_iter=1000, random_state=random_state
            )
            classifier.fit(embeddings[train_indices], labels[train_indices])
            probabilities = classifier.predict_proba(embeddings[test_indices])[:, 1]

            false_positive_rate, true_positive_rate, _ = roc_curve(
                labels[test_indices], probabilities
            )
            auc = roc_auc_score(labels[test_indices], probabilities)
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                label=f"{name} (AUC = {auc:.3f})",
            )
            results[name] = {"classifier": classifier, "auc": auc}

        axis.plot([0, 1], [0, 1], linestyle="--", color="lightgray", label="Random")
        axis.set_title(
            title
            or f"Predicting whether a recipe contains '{ingr}'"
        )
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.legend(loc="lower right")
        figure.tight_layout()
        plt.show()

        print(f"Ingredient: {ingr!r}")
        print(f"Positive recipes: {labels.sum():,}")
        print(f"Negative recipes: {(labels == 0).sum():,}")
        for name, result in results.items():
            print(f"{name} held-out ROC AUC: {result['auc']:.3f}")


        figure_path = plots_path / f"{ingr}_roc.png"
        figure.savefig(figure_path, dpi=150, bbox_inches="tight")

        report = {
            "ingredient": ingr,
            "positive_recipes": int(labels.sum()),
            "negative_recipes": int((labels == 0).sum()),
            "auc": {
                name: float(result["auc"]) for name, result in results.items()
            },
        }
        all_results[ingr] = report
        returned_results[ingr] = (results, figure, axis)
        print(f"Saved ROC plot to {figure_path}")

    report_path = results_path / "evaluation_results.json"
    aggregate_path = results_path / "aggregate_auc.csv"
    statistics_path = results_path / "aggregate_statistics.csv"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(all_results, report_file, indent=2)

    aggregate_rows = [
        {
            "ingredient": ingredient_name,
            "positive_recipes": report["positive_recipes"],
            "negative_recipes": report["negative_recipes"],
            **report["auc"],
        }
        for ingredient_name, report in all_results.items()
    ]
    aggregate_frame = pd.DataFrame(aggregate_rows)
    aggregate_frame.to_csv(aggregate_path, index=False)

    statistics = aggregate_frame.drop(columns="ingredient").agg(
        ["mean", "std", "min", "max"]
    ).T
    statistics.index.name = "metric"
    statistics.to_csv(statistics_path)
    print(f"Saved evaluation results to {report_path}")
    print(f"Saved aggregate AUC results to {aggregate_path}")
    print(f"Saved aggregate statistics to {statistics_path}")

    if single_ingredient:
        return returned_results[next(iter(all_results))]
    
    return returned_results

SUBSTITUTIONS = "data/eval/substitutions_all.csv"
def load_pairs(ingredients, path=SUBSTITUTIONS):
    name2id = dict(zip(ingredients["name"], ingredients["ingredient_id"]))
    subs = pd.read_csv(path)
    pairs = {}
    for a, b in zip(subs["ingredient"], subs["substitute"]):
        # a, b = normalise(a), normalise(b)
        if a in name2id and b in name2id and a != b:
            pairs.setdefault(name2id[a], []).append(name2id[b])
    return pairs

def random_substitution(edges, pairs, seed=None, keep_unused=False):
    """
    edges: array-like of shape (2, N); edges[0] = recipe IDs, edges[1] = ingredient IDs
    pairs: dict {ingredient: [candidate substitutes]}  (this is P)
    keep_unused: if True, substituted-away ingredients stay in the node set
                 even if no recipe uses them.

    Returns (graph, mapping):
        graph:   {"recipes": [...], "ingredients": [...], "edges": np.ndarray of shape (2, M)}
        mapping: {i: j} for every ingredient i with P(i) non-empty
    """
    rng = random.Random(seed)

    E = np.asarray(edges)
    if E.ndim != 2 or E.shape[0] != 2:
        raise ValueError(f"edges must have shape (2, N), got {E.shape}")

    recipes_in, ingr_in = E[0], E[1]

    # One choice per ingredient, made once, so every instance of i gets the same j
    mapping = {i: rng.choice(subs) for i, subs in pairs.items() if len(subs) > 0}

    # Single pass, no chaining; the set removes duplicate edges
    new_edges = sorted({(int(r), int(mapping.get(int(i), i)))
                        for r, i in zip(recipes_in, ingr_in)})

    new_E = (np.array(new_edges, dtype=int).T
             if new_edges else np.empty((2, 0), dtype=int))

    ingredients = {int(i) for i in new_E[1]}
    if keep_unused:
        ingredients |= {int(i) for i in ingr_in}

    graph = {
        "recipes": sorted({int(r) for r in recipes_in}),
        "ingredients": sorted(ingredients),
        "edges": new_E,
    }
    return graph, mapping

def build_substitution_graph(
    pairs,
    edge_list,
    num_ingredients=None,
    seed=None,
    keep_unused=False,
):
    graph, mapping = random_substitution(
        edge_list, pairs, seed=seed, keep_unused=keep_unused
    )
    if num_ingredients is None:
        num_ingredients = max(graph["ingredients"], default=-1) + 1

    edge_weights, idf = calculate_edge_weights(
        graph["edges"],
        num_ingredients,
        edge_share=1,
    )
    graph["idf"] = idf
    graph["edge_weights"] = edge_weights
    return graph, mapping


'''
TODO 
    - make a method to load a model, run it on the new graph with substitution
    - report AUC scores of the new embeddings, only on the nodes with substitutions
    - measure the distance of embeddings with a substitution and embeddings without
        Make a graph with completely random substitutions and see if the distances are 
        greater there
'''
if __name__ == "__main__":
    edges = np.load("data/processed/edges.npz")
    recipes = pd.read_csv("data/processed/recipes.csv")
    ingr_map = pd.read_csv("data/processed/ingredient_map.csv")
    ingr = pd.read_csv("data/processed/ingredients.csv")
    node2vec = np.load("data/node2vec/embeddings.npz")

    lightgcn_emb = np.load("data/similarity/lightgcn_inductive_s0_embeddings.npz")
    recipe_sentence_embeddings = np.load("data/processed/recipe_x_full.npy")

    plot_ingredient_auc(
        get_random_ingredients(100, ingr, used_more_than=10),
        embeddings_by_name={
            "Node2Vec": node2vec["recipe_embeddings"],
            "Sentence encoder": recipe_sentence_embeddings,
            "LightGCN": lightgcn_emb['embeddings']
        }
    )

