# Graph_ML

## Workflow

The commands below are intended to be run from the repository root in
PowerShell. They use the active Python interpreter, so activate your chosen
environment first if you are using one.

### 1. Create the environment and install dependencies

Create a virtual environment once (optional but recommended):

```powershell
python -m venv .venv
```

Activate it if you created one:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include graph construction, Sentence Transformer features,
Node2Vec, PyTorch Geometric, the GAT/GraphSAGE model, and evaluation tools.

### 2. Download RecipeNLG

Configure Kaggle credentials for `kagglehub`, then run:

```powershell
python download_recipe_nlg.py
```

The input file must be available as:

```text
data/raw/RecipeNLG_dataset.csv
```

If the downloaded archive places the file in a nested directory, copy or move
that file to the path above before continuing.

### 3. Build the processed graph

Run this before any embedding, GNN, or evaluation command:

```powershell
python build_graph.py
```

This creates `data/processed/`, including the graph splits, node features,
quantity-aware `edge_weights.npz`, and configuration/statistics JSON files.

### 4. Train the Node2Vec baseline

```powershell
python random_walk_embeddings.py
```

The result is written to `data/node2vec/`.

### 5. Train the GNN

The default model is the weighted GAT + GraphSAGE encoder:

```powershell
python train_gat.py
```

To run named ablations, use:

```powershell
python train_gat.py gat_w_s0 gat 1 0
python train_gat.py gat_s0 gat 0 0
python train_gat.py mean_s0 mean 0 0
python train_gat.py mlp_s0 none 0 0
```

Models and training metadata are written to `data/gat/`.
For a quick CPU smoke run, append a step count, for example:

```powershell
python train_gat.py gat_w_smoke gat 1 0 20
```

### 6. Train recipe-similarity models

For the inductive recipe-similarity task, similarity is defined from ingredient
sets rather than recipe names. The model trains on non-held-out recipes, then
embeds held-out recipes from their ingredient lists and retrieves similar
training recipes. A pair is relevant when its weighted ingredient Jaccard similarity is at least 0.30, or
when it has at least 0.15 weighted overlap and at least 0.88 cosine similarity
between weighted ingredient-text vectors. Negative training pairs must have at
most 0.05 weighted Jaccard and at most 0.80 semantic cosine similarity. The
weights are the same IDF and quantity-share weights used by graph construction,
so common ingredients such as salt contribute less. They are normalized per
recipe when constructing feature means; edge values themselves are not
globally normalized. The evaluation reports MRR, Hits@10, and NDCG@10.

LightGCN-style propagation uses the recipe/ingredient graph and projected
ingredient features, so it can embed a new recipe:

```powershell
python train_recipe_similarity.py lightgcn --steps 1000 --seed 0
```

GraphSAGE uses text-derived node features, graph message passing, and an MLP
pair decoder:

```powershell
python train_recipe_similarity.py graphsage --steps 1000 --seed 0
```

Similarity checkpoints and metrics are written to `data/similarity/`.

### 7. Prepare substitution-evaluation data (optional)

The substitution benchmark uses Recipe1MSubs files in addition to the
repository's manual pairs. Download them with:

```powershell
python download_recipe1m_substitutions.py
python prepare_substitutions.py
```

The first command creates:

```text
data/eval/recipe1msubs/train_comments_subs.pkl
data/eval/recipe1msubs/val_comments_subs.pkl
data/eval/recipe1msubs/test_comments_subs.pkl
```

The second creates `data/eval/substitutions_all.csv`.

### 8. Evaluate models and edge-weight choices

Run these only after the required baselines/models have been trained:

```powershell
python evaluate_models.py
python evaluate_weights.py
```

The evaluation scripts expect the corresponding multi-seed GAT and Node2Vec
files described in their module docstrings. The interactive embedding
comparisons are in `evaluate_embeddings.ipynb`.

## Data and generated files

Raw and generated data are ignored by Git because they can be large. The
repository contains the source code and small manual metadata files; rerun
the commands above to recreate generated artifacts.

The manually maintained metadata files are stored in `data/`:

```text
data/ingredient_properties.csv
data/substitutions.csv
```

## Frozen substitution baseline comparison

This optional evaluation compares plausible ingredient substitutions with random
replacements of similar training frequency. Both copies preserve the replaced
ingredient's mass share and use identical frozen cases across methods.

```powershell
python evaluate_substitutions.py --output-dir results/substitution_validation
```

The defaults compare binary Jaccard, the mean of ingredient text embeddings,
the IDF/quantity-weighted mean, and weighted ingredient multi-hot vectors.
They reuse processed data without training or downloads. Add
`--methods jaccard mean weighted_mean weighted_multihot sbert` to also encode
whole ingredient lists with the configured sentence transformer; this may download
the encoder. Recipe titles are excluded from that baseline.

The new benchmark is also available in the experiment runner:

```powershell
python run_experiments.py --only substitution_benchmark --out results/substitution_validation_runner
```

The original runner functions and default evaluation selection are preserved.
The standalone script exposes protocol options and avoids importing the model stack.
Learned-model checkpoint integration is deferred from this initial evaluation change.

Reserved recipes are excluded from training edges, then deterministically divided
into validation and test populations. Develop on the default validation partition;
use `--split test` only after fixing the comparison. The script prefers prepared
`data/eval/substitutions_all.csv` when available, otherwise the manual
`data/substitutions.csv`; `--substitutions` selects an explicit CSV.

Outputs include `cases.json`, `scores.csv`, `summary.csv`, `summary.json`,
`comparison.md`, and (with matplotlib installed) `comparison.png`. Cases are
fingerprinted against the data, substitution labels, and protocol; incompatible
reuse is rejected. Summaries include case accuracy, substitution-pair macro
accuracy, pair-bootstrap confidence intervals, text-similarity strata, and paired
differences against the weighted mean. Single-run seed variability is unavailable.
Jaccard should tie at 0.5 because each variant changes exactly one ingredient.
These labels do not guarantee that a substitute works in every recipe context.

Run the local protocol and baseline checks with:

```powershell
python -m unittest discover -s tests -v
```
