"""Run every evaluation on every embedding method and write one structured results folder.

    python run_experiments.py                       # all evaluations, GNN seeds 0 1 2
    python run_experiments.py --only substitution   # a subset
    python run_experiments.py --retrain             # retrain the GNN checkpoints first

Missing GNN checkpoints (and the Node2Vec embeddings) are trained before evaluating.

Output, under results/<timestamp>/ by default:
    config.json         arguments, git commit, method list
    summary.csv         one row per (evaluation, metric, method): mean and std over seeds
    metrics.csv         the same before aggregation, one row per seed
    summary.png         overview of the headline metrics
    <evaluation>/       per-item CSVs and plots of that evaluation

Evaluations
    separability    a logistic regression per ingredient predicts whether a recipe contains it,
                    on known recipes (in the training graph) and on unseen recipes
    substitution    a recipe copy with a real substitute should be closer to the original than a
                    copy with a random ingredient as frequent as the substitute
    same_dish       recipes with the same title should retrieve each other
    duplicates      checks that same-title recipes differ in ingredients (a data check, no methods)

Methods are defined in embedders.py.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import eval_common
from embedders import MultiHot, NODE2VEC_FILE, SIMILARITY_DIR, build_methods, load_data
from train_recipe_similarity import MODELS, train


SEED = 0
PROBE_INGREDIENTS = 100      
MIN_RECIPES = 10             
MIN_UNSEEN_POSITIVES = 5     
SUBSTITUTION_CASES = 3000
RETRIEVAL_K = 10


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def metric_rows(evaluation, metric, values):
    """values: {method: value} -> rows for metrics.csv."""
    return [{"evaluation": evaluation, "metric": metric, "method": m.name,
             "seed": getattr(m, "seed", None), "value": float(v)} for m, v in values.items()]


def bar_plot(ax, table, title, reference=None):
    ax.barh(table.index, table["mean"], xerr=table["std"], color="tab:blue")
    for row, value in enumerate(table["mean"]):
        ax.text(value, row, f" {value:.3f}", va="center", fontsize=8)
    if reference is not None:
        ax.axvline(reference, color="gray", linestyle="--")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)


def first_seed(methods):
    """One method per name, for plots that show a single run."""
    seen = {}
    for m in methods:
        seen.setdefault(m.name, m)
    return list(seen.values())

def separability(data, methods, embeddings, out):
    rng = np.random.default_rng(SEED)
    counts = data.ingredients["recipe_count"].to_numpy()
    probes = rng.choice(np.flatnonzero(counts > MIN_RECIPES), PROBE_INGREDIENTS, replace=False)
    known, unseen = np.flatnonzero(data.known), np.flatnonzero(~data.known)
    contains = [set(ids) for ids in data.recipe_ingredients]

    rows, curves = [], {}
    auc_known, auc_unseen = {m: [] for m in methods}, {m: [] for m in methods if m.inductive}
    for ingredient in probes:
        y = np.array([ingredient in s for s in contains], dtype=int)
        train_ids, test_ids = train_test_split(known, test_size=0.2, random_state=SEED,
                                               stratify=y[known])
        test_unseen = y[unseen].sum() >= MIN_UNSEEN_POSITIVES
        for m in methods:
            x = embeddings[m]
            classifier = LogisticRegression(max_iter=2000).fit(x[train_ids], y[train_ids])
            p_known = classifier.predict_proba(x[test_ids])[:, 1]
            row = {"ingredient": data.ingredients["name"][ingredient], "recipes": counts[ingredient],
                   "method": m.name, "seed": getattr(m, "seed", None),
                   "auc_known": roc_auc_score(y[test_ids], p_known), "auc_unseen": np.nan}
            auc_known[m].append(row["auc_known"])
            if m.inductive and test_unseen:
                row["auc_unseen"] = roc_auc_score(y[unseen], classifier.predict_proba(x[unseen])[:, 1])
                auc_unseen[m].append(row["auc_unseen"])
            rows.append(row)
            if len(curves) < 4 * len(methods) and ingredient in probes[:4]:
                curves[(ingredient, m)] = roc_curve(y[test_ids], p_known)[:2]
    table = pd.DataFrame(rows)
    table.to_csv(out / "per_ingredient.csv", index=False)

    metrics = metric_rows("separability", "auc_known", {m: np.mean(v) for m, v in auc_known.items()})
    metrics += metric_rows("separability", "auc_unseen", {m: np.mean(v) for m, v in auc_unseen.items()})

    # AUC against ingredient frequency
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, t in table.groupby("method", sort=False):
        mean = t.groupby("ingredient")[["recipes", "auc_known"]].mean()
        ax.scatter(mean.recipes, mean.auc_known, s=12, label=name, alpha=0.7)
    ax.set_xscale("log"); ax.set_xlabel("recipes containing the ingredient"); ax.set_ylabel("AUC (known recipes)")
    ax.legend(); ax.set_title("Separability per ingredient")
    fig.tight_layout(); fig.savefig(out / "auc_vs_frequency.png", dpi=130); plt.close(fig)

    # ROC curves of a few ingredients, first seed of every method
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, ingredient in zip(axes, probes[:4]):
        for m in first_seed(methods):
            if (ingredient, m) in curves:
                fpr, tpr = curves[(ingredient, m)]
                ax.plot(fpr, tpr, label=m.name)
        ax.plot([0, 1], [0, 1], "--", color="lightgray")
        ax.set_title(f"contains '{data.ingredients['name'][ingredient]}'", fontsize=10)
        ax.set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "roc_examples.png", dpi=130); plt.close(fig)
    return metrics

def substitution_cases(data, rng):
    """(recipe, position, ingredient, substitute, random) for recipes with a known substitute.
    The random ingredient is about as frequent as the substitute, so frequency (and with it
    the IDF edge weight) cannot decide the test."""
    pairs = eval_common.load_pairs()
    cases = []
    for recipe_id in rng.permutation(data.n_recipes):
        present = set(data.recipe_ingredients[recipe_id])
        options = [(p, a) for p, a in enumerate(data.recipe_ingredients[recipe_id]) if a in pairs]
        if not options:
            continue
        position, ingredient = options[rng.integers(len(options))]
        substitutes = [b for b in pairs[ingredient] if b not in present]
        if not substitutes:
            continue
        substitute = substitutes[rng.integers(len(substitutes))]
        random = eval_common.matched_random(rng, substitute, present | set(pairs[ingredient]) | {ingredient})
        if random is not None:
            cases.append((recipe_id, position, ingredient, substitute, random))
        if len(cases) == SUBSTITUTION_CASES:
            break
    return cases


def substitution(data, methods, embeddings, out):
    rng = np.random.default_rng(SEED)
    cases = substitution_cases(data, rng)
    original = [data.recipe(r) for r, *_ in cases]
    substituted = [data.recipe(r, (p, b)) for r, p, a, b, x in cases]
    randomised = [data.recipe(r, (p, x)) for r, p, a, b, x in cases]

    x = unit(data.ingredient_x)
    text_similarity = np.array([x[a] @ x[b] for _, _, a, b, _ in cases])
    groups = {"all": np.ones(len(cases), dtype=bool),
              "similar_names": text_similarity >= np.median(text_similarity),
              "dissimilar_names": text_similarity < np.median(text_similarity),
              "unseen_recipes": ~data.known[[r for r, *_ in cases]]}

    names = data.ingredients["name"].to_numpy()
    rows, metrics = [], []
    for m in [MultiHot(data)] + [m for m in methods if m.inductive]:
        o, s, r = (unit(m.embed(recipes)) for recipes in (original, substituted, randomised))
        to_substitute, to_random = (o * s).sum(1), (o * r).sum(1)
        win = (to_substitute > to_random) + 0.5 * (to_substitute == to_random)
        for group, mask in groups.items():
            metrics += metric_rows("substitution", f"win_rate_{group}", {m: win[mask].mean()})
        metrics += metric_rows("substitution", "cos_gap", {m: (to_substitute - to_random).mean()})
        rows.append(pd.DataFrame({
            "method": m.name, "seed": getattr(m, "seed", None),
            "recipe_id": [c[0] for c in cases], "ingredient": names[[c[2] for c in cases]],
            "substitute": names[[c[3] for c in cases]], "random": names[[c[4] for c in cases]],
            "name_similarity": text_similarity, "cos_substitute": to_substitute, "cos_random": to_random,
        }))
    table = pd.concat(rows)
    table.to_csv(out / "cases.csv", index=False)

    frame = pd.DataFrame(metrics)
    win = frame[frame.metric.str.startswith("win_rate")]
    pivot = win.groupby(["method", "metric"], sort=False)["value"].mean().unstack()
    pivot = pivot[[f"win_rate_{g}" for g in groups]]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot.bar(ax=ax, rot=0)
    ax.axhline(0.5, color="gray", linestyle="--"); ax.set_ylabel("win rate"); ax.set_ylim(0, 1)
    ax.set_title(f"Substitute closer than a frequency-matched random ingredient ({len(cases)} cases)")
    ax.legend([g.replace("_", " ") for g in groups], fontsize=8)
    fig.tight_layout(); fig.savefig(out / "win_rates.png", dpi=130); plt.close(fig)

    bins = np.quantile(text_similarity, np.linspace(0, 1, 6))
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, t in table.groupby("method", sort=False):
        t = t.assign(win=(t.cos_substitute > t.cos_random).astype(float),
                     bin=np.digitize(t.name_similarity, bins[1:-1]))
        curve = t.groupby("bin").win.mean()
        ax.plot(curve.index + 1, curve.values, marker="o", label=name)
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xlabel("name similarity of ingredient and substitute (quintile)"); ax.set_ylabel("win rate")
    ax.legend(fontsize=8); ax.set_title("Substitution test by name similarity")
    fig.tight_layout(); fig.savefig(out / "win_rate_by_name_similarity.png", dpi=130); plt.close(fig)
    return metrics

def retrieval_ranks(z, queries, candidates, titles):
    """Rank of the first other recipe with the same title among the candidates, per query."""
    z = unit(z.astype(np.float32))
    ranks = []
    for start in range(0, len(queries), 512):
        q = queries[start:start + 512]
        sims = z[q] @ z[candidates].T
        sims[candidates[None, :] == q[:, None]] = -np.inf
        order = np.argsort(-sims, axis=1)
        match = titles[candidates][order] == titles[q][:, None]
        ranks.append(match.argmax(1))
    return np.concatenate(ranks)


def same_dish(data, methods, embeddings, out):
    titles = eval_common.titles
    size = pd.Series(titles).map(pd.Series(titles).value_counts()).to_numpy()
    in_group = (size >= 2) & (size <= eval_common.MAX_GROUP)
    known = np.flatnonzero(data.known)
    everything = np.arange(data.n_recipes)
    # queries need another recipe with the same title among the candidates
    known_titles = pd.Series(titles[known]).value_counts()
    known_queries = known[in_group[known] & (pd.Series(titles[known]).map(known_titles).to_numpy() >= 2)]
    unseen_queries = np.flatnonzero(in_group & ~data.known)

    metrics, rank_table = [], []
    for m in methods:
        settings = [("known", known_queries, known)]
        if m.inductive:
            settings.append(("unseen", unseen_queries, everything))
        for setting, queries, candidates in settings:
            ranks = retrieval_ranks(embeddings[m], queries, candidates, titles)
            metrics += metric_rows("same_dish", f"mrr_{setting}", {m: np.mean(1 / (ranks + 1))})
            metrics += metric_rows("same_dish", f"hits@{RETRIEVAL_K}_{setting}", {m: np.mean(ranks < RETRIEVAL_K)})
            rank_table.append(pd.DataFrame({"method": m.name, "seed": getattr(m, "seed", None),
                                            "setting": setting, "recipe_id": queries, "rank": ranks}))
    table = pd.concat(rank_table)
    table.to_csv(out / "ranks.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, setting in zip(axes, ["known", "unseen"]):
        for name, t in table[table.setting == setting].groupby("method", sort=False):
            k = np.arange(1, 101)
            ax.plot(k, [(t["rank"] < i).mean() for i in k], label=name)
        ax.set_xscale("log"); ax.set_xlabel("k"); ax.set_ylabel("hits@k")
        ax.set_title(f"Same-dish retrieval, {setting} queries", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "hits_at_k.png", dpi=130); plt.close(fig)
    return metrics

def duplicates(data, methods, embeddings, out):
    rng = np.random.default_rng(SEED)
    sets = [set(ids) for ids in data.recipe_ingredients]
    titles = pd.Series(eval_common.titles)
    groups = [g for g in titles.groupby(titles).groups.values() if 2 <= len(g) <= eval_common.MAX_GROUP]

    def jaccard(a, b):
        return len(sets[a] & sets[b]) / len(sets[a] | sets[b])
    same = np.array([jaccard(g[i], g[j]) for g in groups for i in range(len(g)) for j in range(i + 1, len(g))])
    random = np.array([jaccard(*rng.choice(data.n_recipes, 2, replace=False)) for _ in range(len(same))])
    stats = {"title_groups": len(groups), "pairs": len(same),
             "mean_jaccard_same_title": same.mean(), "mean_jaccard_random": random.mean(),
             "identical_ingredient_sets": (same == 1).mean()}
    (out / "stats.json").write_text(json.dumps({k: float(v) for k, v in stats.items()}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist([same, random], bins=20, label=["same title", "random pair"], density=True)
    ax.set_xlabel("ingredient Jaccard"); ax.legend()
    ax.set_title("Same-title recipes share ingredients but are not copies")
    fig.tight_layout(); fig.savefig(out / "jaccard.png", dpi=130); plt.close(fig)
    return [{"evaluation": "duplicates", "metric": k, "method": "data", "seed": None, "value": float(v)}
            for k, v in stats.items()]


EVALUATIONS = {"separability": separability, "substitution": substitution,
               "same_dish": same_dish, "duplicates": duplicates}


def ensure_trained(seeds, steps, retrain):
    for model_name in MODELS:
        for seed in seeds:
            if retrain or not (SIMILARITY_DIR / f"{model_name}_inductive_s{seed}.pt").exists():
                print(f"training {model_name} seed {seed}")
                train(model_name, steps, seed)
    if not NODE2VEC_FILE.exists():
        print("training Node2Vec")
        import random_walk_embeddings
        random_walk_embeddings.main()


def summary_plot(summary, out):
    panels = [("separability", "auc_known", "Separability AUC, known recipes", None),
              ("separability", "auc_unseen", "Separability AUC, unseen recipes", None),
              ("substitution", "win_rate_all", "Substitution win rate", 0.5),
              ("substitution", "win_rate_dissimilar_names", "Substitution win rate, dissimilar names", 0.5),
              ("same_dish", "mrr_known", "Same-dish MRR, known recipes", None),
              ("same_dish", "mrr_unseen", "Same-dish MRR, unseen recipes", None)]
    panels = [p for p in panels if ((summary.evaluation == p[0]) & (summary.metric == p[1])).any()]
    if not panels:
        return
    fig, axes = plt.subplots((len(panels) + 1) // 2, 2, figsize=(13, 3.2 * ((len(panels) + 1) // 2)),
                             squeeze=False)
    for ax, (evaluation, metric, title, reference) in zip(axes.flat, panels):
        table = summary[(summary.evaluation == evaluation) & (summary.metric == metric)].set_index("method")
        bar_plot(ax, table, title, reference)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "summary.png", dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results") / time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--only", nargs="+", choices=list(EVALUATIONS), default=list(EVALUATIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1000, help="training steps for missing checkpoints")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    ensure_trained(args.seeds, args.steps, args.retrain)
    data = load_data()
    methods = build_methods(data, args.seeds)
    print("methods:", ", ".join(f"{m.name}{'' if getattr(m, 'seed', None) is None else f' s{m.seed}'}"
                                for m in methods))
    embeddings = {m: m.recipes() for m in methods}

    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (args.out / "config.json").write_text(json.dumps({
        "evaluations": args.only, "seeds": args.seeds, "git_commit": commit,
        "methods": [m.name for m in first_seed(methods)],
        "probe_ingredients": PROBE_INGREDIENTS, "substitution_cases": SUBSTITUTION_CASES,
    }, indent=2))

    metrics = []
    for name in args.only:
        print(f"\n== {name}")
        folder = args.out / name
        folder.mkdir(exist_ok=True)
        metrics += EVALUATIONS[name](data, methods, embeddings, folder)

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    summary = (metrics.groupby(["evaluation", "metric", "method"], sort=False)["value"]
               .agg(["mean", "std", "count"]).fillna({"std": 0.0}).reset_index())
    summary.to_csv(args.out / "summary.csv", index=False)
    summary_plot(summary, args.out)
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(summary.to_string(index=False, float_format="%.3f"))
    print(f"\nresults in {args.out}")


if __name__ == "__main__":
    main()
