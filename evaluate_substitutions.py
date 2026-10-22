"""Frozen, inductive substitution benchmark; no training or Node2Vec required.

The default methods use existing ingredient features and run without downloads.
Add --methods jaccard mean weighted_mean sbert for the sentence-list baseline.
Learned-model comparisons are a separate integration step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from normalisation import normalise


ROOT = Path(__file__).resolve().parent
BASELINES = ("jaccard", "mean", "weighted_mean", "weighted_multihot", "sbert")
FORMAT_VERSION = 2
DATA_FILES = ("recipes.csv", "ingredients.csv", "edges.npz", "edge_weights.npz",
              "ingredient_x.npy", "config.json")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass
class BenchmarkData:
    directory: Path
    recipes: pd.DataFrame
    ingredients: pd.DataFrame
    features: np.ndarray
    lists: list
    shares: list
    train_counts: np.ndarray
    config: dict
    fingerprints: dict
    train_edges: np.ndarray
    graph_splits: dict

    @property
    def idf(self):
        return np.log((1 + len(np.unique(self.train_edges[0]))) / (1 + self.train_counts)) + 1.0

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        fingerprints = {name: digest(directory / name) for name in DATA_FILES}
        recipes = pd.read_csv(directory / "recipes.csv")
        ingredients = pd.read_csv(directory / "ingredients.csv", keep_default_na=False)
        for table, column in [(recipes, "recipe_id"), (ingredients, "ingredient_id")]:
            if not np.array_equal(table[column].to_numpy(), np.arange(len(table))):
                raise ValueError(f"{column} must be contiguous and match table row order.")
        heldout = recipes["heldout"].astype(str).str.lower()
        if not heldout.isin(["true", "false"]).all():
            raise ValueError("recipes.csv heldout values must be True or False.")
        recipes["heldout"] = heldout.eq("true")
        features = np.load(directory / "ingredient_x.npy", allow_pickle=False)
        if (features.ndim != 2 or len(features) != len(ingredients)
                or not np.isfinite(features).all() or (np.linalg.norm(features, axis=1) == 0).any()):
            raise ValueError("Ingredient features must be finite, nonzero, and aligned with ingredients.csv.")
        idf = ingredients["idf"].to_numpy(dtype=float)
        if not np.isfinite(idf).all() or (idf <= 0).any():
            raise ValueError("Ingredient IDF values must be finite and positive.")
        with np.load(directory / "edges.npz", allow_pickle=False) as archive:
            edges, train = archive["all"], archive["train"]
            split_ids = {name: np.unique(archive[name][0])
                         for name in ["train", "val", "test", "heldout"] if name in archive}
            split_edges = {name: int(archive[name].shape[1]) for name in split_ids}
        with np.load(directory / "edge_weights.npz", allow_pickle=False) as archive:
            shares = archive["share"]
            train_weights = archive["train"]
        for array in [edges, train]:
            if (array.ndim != 2 or array.shape[0] != 2 or not np.issubdtype(array.dtype, np.integer)
                    or (array < 0).any() or (array[0] >= len(recipes)).any()
                    or (array[1] >= len(ingredients)).any()):
                raise ValueError("Invalid recipe/ingredient edge IDs.")
        if recipes["heldout"].to_numpy()[train[0]].any():
            raise ValueError("Training edges include held-out recipes.")
        all_keys = edges[0] * len(ingredients) + edges[1]
        train_keys = train[0] * len(ingredients) + train[1]
        if (not train.shape[1] or len(np.unique(train_keys)) != len(train_keys)
                or not np.isin(train_keys, all_keys).all()):
            raise ValueError("Training edges must be a nonempty, unique subset of all edges.")
        if shares.shape != (edges.shape[1],) or not np.isfinite(shares).all() or (shares <= 0).any():
            raise ValueError("Quantity shares must be positive and aligned with all edges.")
        order = np.argsort(edges[0], kind="stable")
        offsets = np.r_[0, np.cumsum(np.bincount(edges[0], minlength=len(recipes)))]
        lists, recipe_shares = [], []
        for recipe_id in range(len(recipes)):
            positions = order[offsets[recipe_id]:offsets[recipe_id + 1]]
            ids, quantities = edges[1, positions], shares[positions]
            if len(ids) < 2 or len(np.unique(ids)) != len(ids) or not np.isclose(quantities.sum(), 1):
                raise ValueError(f"Recipe {recipe_id} needs distinct ingredients and shares summing to one.")
            lists.append(ids)
            recipe_shares.append(quantities)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        alpha = float(config.get("share_exponent", 0.5))
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("share_exponent must be finite and nonnegative.")
        counts = np.bincount(train[1], minlength=len(ingredients))
        exact_idf = np.log((1 + len(np.unique(train[0]))) / (1 + counts)) + 1.0
        order_keys = np.argsort(all_keys)
        positions = order_keys[np.searchsorted(all_keys[order_keys], train_keys)]
        expected = exact_idf[train[1]] * shares[positions] ** alpha
        if train_weights.shape != expected.shape or not np.allclose(train_weights, expected, rtol=1e-5, atol=1e-7):
            raise ValueError("Training weights do not match training-edge IDF and quantity shares.")
        overlaps = {f"{a}/{b}": int(len(np.intersect1d(split_ids[a], split_ids[b])))
                    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
                    if a in split_ids and b in split_ids}
        graph_splits = {"recipes": {k: len(v) for k, v in split_ids.items()},
                        "edges": split_edges, "recipe_overlaps": overlaps,
                        "reserved_recipes_excluded_from_training": True}
        return cls(directory, recipes, ingredients, features, lists, recipe_shares,
                   counts, config, fingerprints, train, graph_splits)


def load_pairs(path, data):
    """Keep directional positives; exclude known substitutes in either direction as negatives."""
    table = pd.read_csv(path, keep_default_na=False)
    lookup = dict(zip(data.ingredients["name"], data.ingredients["ingredient_id"]))
    pairs, excluded, sources = {}, {}, {}
    for row in table.to_dict("records"):
        a, b = (normalise(str(row[column]).replace("_", " "))
                for column in ["ingredient", "substitute"])
        if a not in lookup or b not in lookup or a == b:
            continue
        a, b = int(lookup[a]), int(lookup[b])
        pairs.setdefault(a, set()).add(b)
        excluded.setdefault(a, set()).add(b)
        excluded.setdefault(b, set()).add(a)
        sources.setdefault((a, b), set()).add(str(row.get("source") or "manual"))
    if not pairs:
        raise ValueError("No substitution pairs match the processed ingredient vocabulary.")
    return {a: sorted(bs) for a, bs in pairs.items()}, excluded, sources


def select_recipes(data, split, split_seed, validation_fraction):
    ids = np.flatnonzero(data.recipes["heldout"].to_numpy())
    if split == "heldout":
        return ids
    if len(ids) < 2:
        raise ValueError("Validation/test partitioning needs at least two held-out recipes; use --split heldout.")
    ids = np.random.default_rng(split_seed).permutation(ids)
    count = max(1, min(len(ids) - 1, int(len(ids) * validation_fraction)))
    return np.sort(ids[:count] if split == "validation" else ids[count:])


def generate_cases(data, pairs, excluded, sources, recipe_ids, seed, window):
    rng = np.random.default_rng(seed)
    # Frequency matching uses training edges only, with stable ID ordering for ties.
    order = np.lexsort((np.arange(len(data.ingredients)), -data.train_counts))
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    cases = []
    coverage = {"selected_recipes": len(recipe_ids), "no_eligible_substitute": 0,
                "no_random_candidate": 0}
    names = data.ingredients["name"].tolist()
    for recipe_id in recipe_ids:
        recipe_id = int(recipe_id)
        ids, shares = data.lists[recipe_id], data.shares[recipe_id]
        present = set(ids)
        options = [(p, b) for p, a in enumerate(ids) for b in pairs.get(int(a), []) if b not in present]
        if not options:
            coverage["no_eligible_substitute"] += 1
            continue
        for option in rng.permutation(len(options)):
            position, substitute = options[option]
            removed = int(ids[position])
            forbidden = present | excluded.get(removed, set()) | {substitute}
            candidates = order[max(0, rank[substitute] - window):rank[substitute] + window + 1]
            candidates = [int(i) for i in candidates if i not in forbidden]
            if not candidates:
                continue
            random_id = int(rng.choice(candidates))
            a, b = data.features[[removed, substitute]].astype(np.float64)
            similarity = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            cases.append({
                "case_id": f"{recipe_id}:{removed}>{substitute}:{random_id}",
                "recipe_id": recipe_id, "title": str(data.recipes.iloc[recipe_id]["title"]),
                "ingredient_ids": ids.tolist(), "ingredients": [names[i] for i in ids],
                "quantity_shares": shares.tolist(),
                "position": int(position), "removed_id": removed, "removed": names[removed],
                "substitute_id": int(substitute), "substitute": names[substitute],
                "random_id": random_id, "random": names[random_id],
                "swapped_share": float(shares[position]), "pair_id": f"{removed}>{substitute}",
                "pair_text_similarity": similarity, "sources": sorted(sources[(removed, substitute)]),
            })
            break
        else:
            coverage["no_random_candidate"] += 1
    if not cases:
        raise ValueError("No eligible substitution cases. Check vocabulary, held-out coverage, and --window.")
    coverage.update(cases=len(cases), distinct_pairs=len({case["pair_id"] for case in cases}))
    return cases, coverage


def freeze_cases(path, data, substitutions, protocol):
    """Reuse an immutable case manifest; refuse stale data or changed generation settings."""
    path = Path(path)
    expected = {"format_version": FORMAT_VERSION, "protocol": protocol,
                "data_sha256": data.fingerprints, "substitutions_sha256": digest(substitutions)}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{path} belongs to different data/settings. Use a new --cases path.")
        if saved.get("cases_sha256") != hashlib.sha256(
                json.dumps(saved["cases"], sort_keys=True, allow_nan=False).encode()).hexdigest():
            raise ValueError(f"{path} cases have changed. Use a new manifest for edited cases.")
        return saved
    pairs, excluded, sources = load_pairs(substitutions, data)
    ids = select_recipes(data, protocol["split"], protocol["split_seed"], protocol["validation_fraction"])
    cases, coverage = generate_cases(data, pairs, excluded, sources, ids, protocol["seed"], protocol["window"])
    pair_similarities = {case["pair_id"]: case["pair_text_similarity"] for case in cases}
    manifest = {**expected, "coverage": coverage,
                "pair_similarity_median": float(np.median(list(pair_similarities.values()))),
                "cases": cases,
                "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return manifest


def variants(cases):
    original, plausible, random_, shares = [], [], [], []
    for case in cases:
        ids = np.asarray(case["ingredient_ids"], dtype=int)
        a, b = ids.copy(), ids.copy()
        a[case["position"]], b[case["position"]] = case["substitute_id"], case["random_id"]
        original.append(ids)
        plausible.append(a)
        random_.append(b)
        shares.append(np.asarray(case["quantity_shares"], dtype=float))
    return original, plausible, random_, shares


def ingredient_weights(data, lists, shares):
    idf = data.idf
    alpha = float(data.config.get("share_exponent", 1.0))
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("share_exponent must be finite and nonnegative.")
    return [idf[ids] * share ** alpha for ids, share in zip(lists, shares)]


def mean_embeddings(data, lists, shares, weighted=False):
    weights = ingredient_weights(data, lists, shares) if weighted else [np.ones(len(ids)) for ids in lists]
    return np.stack([np.average(data.features[ids], axis=0, weights=w) for ids, w in zip(lists, weights)])


def cosine_scores(original, other):
    original, other = np.asarray(original, dtype=np.float64), np.asarray(other, dtype=np.float64)
    if original.ndim != 2 or original.shape != other.shape:
        raise ValueError("Embedding arrays must have matching (cases, dimensions) shapes.")
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(other, axis=1)
    if not np.isfinite(original).all() or not np.isfinite(other).all() or (denominator <= 0).any():
        raise ValueError("Models must produce finite, nonzero embeddings for every case.")
    return np.clip(np.einsum("ij,ij->i", original, other) / denominator, -1, 1)


def jaccard_scores(original, other):
    return np.asarray([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(original, other)])


def wins_from_scores(plausible, random_, tolerance):
    margin = plausible - random_
    return np.where(np.abs(margin) <= tolerance, 0.5, (margin > 0).astype(float)), margin


def bootstrap_ci(values, seed, samples):
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 128):
        draws = rng.integers(0, len(values), (min(128, samples - start), len(values)))
        means.extend(values[draws].mean(axis=1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(cases, wins, margins, median, bootstrap_seed, samples):
    pair_ids = np.asarray([case["pair_id"] for case in cases])
    similarities = np.asarray([case["pair_text_similarity"] for case in cases])
    subsets = {"all": np.ones(len(cases), bool), "lower_text_similarity": similarities < median,
               "higher_text_similarity": similarities >= median}
    result = {}
    for name, mask in subsets.items():
        pairs = sorted(set(pair_ids[mask]))
        per_pair = np.asarray([wins[mask & (pair_ids == pair)].mean() for pair in pairs])
        result[name] = {"cases": int(mask.sum()), "pairs": len(pairs),
                        "micro_accuracy": float(wins[mask].mean()) if mask.any() else None,
                        "macro_accuracy": float(per_pair.mean()) if len(pairs) else None,
                        "macro_ci95": bootstrap_ci(per_pair, bootstrap_seed, samples),
                        "mean_margin": float(margins[mask].mean()) if mask.any() else None}
    return result


def paired_comparison(cases, wins, reference, median, seed, samples):
    # Summary's pair mean and CI applied to paired win differences give a paired,
    # pair-clustered comparison, using exactly the same cases for both methods.
    summary = summarize(cases, wins - reference, np.zeros(len(cases)), median, seed, samples)
    return {name: {"pairs": row["pairs"], "macro_difference": row["macro_accuracy"],
                   "ci95": row["macro_ci95"]} for name, row in summary.items()}


def baseline_runs(data, methods, copies, batch_size, device):
    original, plausible, random_, shares = copies
    for method in methods:
        if method == "jaccard":
            a, b = jaccard_scores(original, plausible), jaccard_scores(original, random_)
        else:
            if method == "weighted_multihot":
                def embed(lists):
                    z = np.zeros((len(lists), len(data.ingredients)), dtype=np.float64)
                    for row, (ids, weights) in enumerate(zip(lists, ingredient_weights(data, lists, shares))):
                        z[row, ids] = weights
                    return z
            elif method == "sbert":
                from sentence_transformers import SentenceTransformer
                sentence_model = SentenceTransformer(data.config["feature_model"], device=device)
                names = data.ingredients["name"].to_numpy()

                def embed(lists):
                    texts = [", ".join(sorted(names[ids])) for ids in lists]
                    return sentence_model.encode(texts, batch_size=batch_size,
                                                 normalize_embeddings=True, show_progress_bar=True)
            else:
                def embed(lists):
                    return mean_embeddings(data, lists, shares, weighted=method == "weighted_mean")
            z_original = embed(original)
            a, b = cosine_scores(z_original, embed(plausible)), cosine_scores(z_original, embed(random_))
        yield {"method": method, "seed": None, "checkpoint": None}, a, b


def write_comparison_outputs(report, rows, output):
    """Render the same saved metrics into a compact report and optional plot."""
    headline = [row for row in rows if row["stratum"] == "all"]
    lines = ["# Substitution baseline comparison", "",
             f"Split: **{report['protocol']['split']}**. "
             f"{report['coverage']['cases']} cases, {report['coverage']['distinct_pairs']} directed pairs.", "",
             "Each case replaces one ingredient, retaining its mass share. A plausible replacement "
             "must keep the recipe closer to the original than a frequency-matched random replacement. "
             "Ties receive half credit. All methods use identical frozen cases.", "",
             "| Method | Case accuracy | Pair macro accuracy | Pair-bootstrap 95% CI | Runs |",
             "| --- | ---: | ---: | --- | ---: |"]
    for row in headline:
        lo, hi = row["macro_ci95"]
        lines.append(f"| {row['method']} | {row['micro_accuracy']:.4f} | "
                     f"{row['macro_accuracy']:.4f} | [{lo:.4f}, {hi:.4f}] | {row['runs']} |")
    lines += ["", f"Paired differences against **{report['scoring']['reference']}**:", ""]
    for name, comparison in report["paired_comparisons"].items():
        row = comparison["all"]
        lo, hi = row["ci95"]
        lines.append(f"- {name}: {row['macro_difference']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}].")
    lines += ["", "## Interpretation limits", "",
              "- Validation results support development; reserve the test split for a fixed final comparison.",
              "- A plausible pair is a benchmark label, not a guarantee that the swap works in every recipe.",
              "- Deterministic baselines and single-seed runs have no estimated seed variability. "
              "The confidence interval measures variability across substitution pairs, not across training runs.",
              "- Binary Jaccard should tie at 0.5: both variants change exactly one ingredient.",
              "- Node2Vec is omitted because it cannot embed these new recipe variants without an additional method.",
              "- Dataset, implementation and case hashes are recorded in summary.json. Cached graph split "
              "overlaps are recorded separately; benchmark recipes must always be absent from training edges.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(headline) + 1.5)))
    for index, row in enumerate(headline):
        lo, hi = row["macro_ci95"]
        ax.plot([lo, hi], [index, index], color="tab:blue", linewidth=2)
        ax.scatter([row["macro_accuracy"]], [index], color="tab:blue", zorder=3)
    ax.set_yticks(range(len(headline)), [row["method"] for row in headline])
    ax.invert_yaxis()
    ax.axvline(0.5, color="gray", linestyle="--")
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Pair macro accuracy; pair-bootstrap 95% confidence interval")
    ax.set_title(f"Substitution comparison ({report['protocol']['split']})")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)


def evaluate(args):
    data = BenchmarkData.load(args.data_dir)
    substitutions = args.substitutions
    if substitutions is None:
        expanded = ROOT / "data/eval/substitutions_all.csv"
        substitutions = expanded if expanded.exists() else ROOT / "data/substitutions.csv"
    output = args.output_dir or ROOT / "data/eval/substitution_v2" / args.split
    protocol = {"split": args.split, "seed": args.seed, "split_seed": args.split_seed,
                "validation_fraction": args.validation_fraction, "window": args.window}
    cases_path = args.cases or output / "cases.json"
    manifest = freeze_cases(cases_path, data, substitutions, protocol)
    cases, median = manifest["cases"], manifest["pair_similarity_median"]
    copies = variants(cases)
    groups, records, run_reports = {}, [], []
    seen = set()
    runs = baseline_runs(data, args.methods, copies, args.batch_size, args.device)
    for info, plausible, random_ in runs:
        identity = (info["method"], info["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate method/seed: {identity}. Specify each baseline only once.")
        seen.add(identity)
        wins, margins = wins_from_scores(plausible, random_, args.tie_tolerance)
        groups.setdefault(info["method"], []).append((wins, margins))
        run_reports.append({**info, "metrics": summarize(cases, wins, margins, median,
                                                         args.bootstrap_seed, args.bootstrap_samples)})
        for case, a, b, win, margin in zip(cases, plausible, random_, wins, margins):
            records.append({"case_id": case["case_id"], "recipe_id": case["recipe_id"],
                            "pair_id": case["pair_id"], **{k: v for k, v in info.items() if k != "training"},
                            "plausible_score": float(a),
                            "random_score": float(b), "margin": float(margin), "win": float(win)})
        print(f"{info['method']} seed={info['seed']}: macro={run_reports[-1]['metrics']['all']['macro_accuracy']:.4f}")
    averaged = {name: (np.mean([run[0] for run in runs], axis=0),
                       np.mean([run[1] for run in runs], axis=0)) for name, runs in groups.items()}
    summaries, rows = {}, []
    for name, (wins, margins) in averaged.items():
        metrics = summarize(cases, wins, margins, median, args.bootstrap_seed, args.bootstrap_samples)
        for stratum, row in metrics.items():
            seed_macros = [run["metrics"][stratum]["macro_accuracy"] for run in run_reports
                           if run["method"] == name]
            row["seed_macro_std"] = (float(np.std(seed_macros)) if len(seed_macros) > 1
                                     and seed_macros[0] is not None else None)
            row["runs"] = len(seed_macros)
            rows.append({"method": name, "stratum": stratum, **row})
        summaries[name] = metrics
    comparisons = {name: paired_comparison(cases, wins, averaged[args.reference][0], median,
                                           args.bootstrap_seed, args.bootstrap_samples)
                   for name, (wins, _) in averaged.items() if name != args.reference}
    report = {"format_version": FORMAT_VERSION, "cases_file": str(cases_path.resolve()),
              "implementation_sha256": {name: digest(ROOT / name) for name in
                                         ["evaluate_substitutions.py", "normalisation.py"]},
              "cases_sha256": manifest["cases_sha256"], "data_sha256": data.fingerprints,
              "substitutions_file": str(substitutions.resolve()),
              "substitutions_sha256": manifest["substitutions_sha256"], "protocol": protocol,
              "coverage": manifest["coverage"], "pair_similarity_median": median,
              "graph_splits": data.graph_splits,
              "scoring": {"tie_tolerance": args.tie_tolerance, "bootstrap_seed": args.bootstrap_seed,
                          "bootstrap_samples": args.bootstrap_samples, "reference": args.reference,
                          "ci_unit": "substitution pair; seed-averaged per-case wins",
                          "replacement_quantity": "same mass share; replacement IDF recomputed",
                          "random_frequency": "training-edge frequency rank",
                          "idf": "unrounded; recomputed from training edges",
                          "evaluation_population": "reserved recipes, partitioned into validation and test",
                          "seed_std_ddof": 0},
              "methods": summaries, "runs": run_reports, "paired_comparisons": comparisons}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "scores.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    write_json(output / "summary.json", report)
    write_comparison_outputs(report, rows, output)
    print(f"{len(cases)} {args.split} cases, {manifest['coverage']['distinct_pairs']} pairs; saved to {output}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--substitutions", type=Path, help="Default: prepared Recipe1MSubs CSV, else manual CSV.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", type=Path, help="Frozen manifest to create or reuse; mismatches are rejected.")
    parser.add_argument("--split", choices=["validation", "test", "heldout"], default="validation")
    parser.add_argument("--seed", type=int, default=0, help="Case-generation seed.")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.5,
                        help="Fraction of reserved held-out recipes assigned to validation.")
    parser.add_argument("--window", type=int, default=50, help="Random replacement frequency-rank window.")
    parser.add_argument("--methods", nargs="+", choices=BASELINES, default=list(BASELINES[:4]))
    parser.add_argument("--reference", choices=BASELINES, default="weighted_mean",
                        help="Select on validation data before running the final test.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tie-tolerance", type=float, default=1e-7)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one.")
    if args.window < 0 or args.batch_size < 1 or args.bootstrap_samples < 1:
        parser.error("Window must be nonnegative; batch size and bootstrap samples must be positive.")
    if not np.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        parser.error("--tie-tolerance must be finite and nonnegative.")
    if args.reference not in args.methods:
        parser.error("--reference must be included in --methods.")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must not contain duplicates.")
    return args


def main():
    args = parse_args()
    try:
        evaluate(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"Substitution evaluation: {error}") from error


if __name__ == "__main__":
    main()
