"""Train Node2Vec embeddings for the recipe-ingredient graph.

With p = q = 1, Node2Vec walks are plain weighted random walks, so they are drawn with
WeightedWalker and fed straight to word2vec. The node2vec package precomputes
second-order transition tables instead, which took 9 hours and ran out of memory on the
100k graph, because their size grows with the square of ingredient degree.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec

from train_recipe_similarity import WeightedWalker, bidirectional_edges


DATA_DIR = Path("data/processed")
SPLIT = "train"
WEIGHTED = True      # walk along edges with probability proportional to the edge weight from build_graph.py
OUTPUT = Path("data/node2vec")
DIMENSIONS = 32
WALK_LENGTH = 10     # nodes per walk, start included
NUM_WALKS = 5        # walks started from every node with edges
WINDOW = 10
WORKERS = max(1, (os.cpu_count() or 2) - 1)
EPOCHS = 5
SEED = 0


def recipe_node(recipe_id):
	return f"r{recipe_id}"


def ingredient_node(ingredient_id):
	return f"i{ingredient_id}"


def load_graph(data_dir, split, weighted):
	edges = np.load(data_dir / "edges.npz")
	recipes = pd.read_csv(data_dir / "recipes.csv")
	ingredients = pd.read_csv(data_dir / "ingredients.csv")

	edge_index = torch.from_numpy(edges[split].astype(np.int64))
	if weighted:
		weights = torch.from_numpy(np.load(data_dir / "edge_weights.npz")[split]).float()
	else:
		weights = torch.ones(edge_index.shape[1])
	return edge_index, weights, recipes, ingredients


def generate_walks(edge_index, weights, n_recipes, n_ingredients):
	"""NUM_WALKS walks from every node with edges, in a fresh random order each round.
	Nodes are numbered as in training: recipes first, then ingredients."""
	walker = WeightedWalker(bidirectional_edges(edge_index, n_recipes), weights.repeat(2),
							n_recipes + n_ingredients)
	start_nodes = torch.nonzero(walker.degree > 0).squeeze(1)
	generator = torch.Generator().manual_seed(SEED)
	walks = [walker.walk(start_nodes[torch.randperm(len(start_nodes), generator=generator)],
						 WALK_LENGTH - 1, generator)
			 for _ in range(NUM_WALKS)]
	return torch.cat(walks).numpy(), len(start_nodes)


def train_embeddings(sentences):
	return Word2Vec(
		sentences,
		vector_size=DIMENSIONS,
		window=WINDOW,
		min_count=1,
		sg=1,
		workers=WORKERS,
		batch_words=256,
		epochs=EPOCHS,
		seed=SEED,
	)


def collect_embeddings(model, labels, dimensions):
	embeddings = np.zeros((len(labels), dimensions), dtype=np.float32)
	present = np.zeros(len(labels), dtype=bool)
	for index, label in enumerate(labels):
		if label in model.wv:
			embeddings[index] = model.wv.get_vector(label)
			present[index] = True
	return embeddings, present


def main(data_dir=DATA_DIR, output_dir=OUTPUT):
	edge_index, weights, recipes, ingredients = load_graph(Path(data_dir), SPLIT, WEIGHTED)
	n_recipes, n_ingredients = len(recipes), len(ingredients)
	walks, graph_nodes = generate_walks(edge_index, weights, n_recipes, n_ingredients)
	print(f"{len(walks)} walks of {WALK_LENGTH} nodes over {graph_nodes} nodes")

	# Nodes are named by id, not title: many recipes share a title, and a recipe
	# called "Butter" would otherwise merge with the ingredient butter
	tokens = np.array([recipe_node(i) for i in range(n_recipes)]
					  + [ingredient_node(i) for i in range(n_ingredients)], dtype=object)
	model = train_embeddings(tokens[walks].tolist())

	recipe_labels = [recipe_node(i) for i in recipes['recipe_id']]
	ingredient_labels = [ingredient_node(i) for i in ingredients['ingredient_id']]
	recipe_embeddings, recipe_present = collect_embeddings(
		model, recipe_labels, DIMENSIONS
	)
	ingredient_embeddings, ingredient_present = collect_embeddings(
		model, ingredient_labels, DIMENSIONS
	)

	output = Path(output_dir) / f"node2vec_{SPLIT}{'_weighted' if WEIGHTED else ''}.npz"
	output.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(
		output,
		recipe_embeddings=recipe_embeddings,
		ingredient_embeddings=ingredient_embeddings,
		recipe_present=recipe_present,
		ingredient_present=ingredient_present,
	)

	metadata = {
		"split": SPLIT,
		"weighted": WEIGHTED,
		"walks": "first-order weighted (node2vec with p = q = 1)",
		"dimensions": DIMENSIONS,
		"walk_length": WALK_LENGTH,
		"num_walks": NUM_WALKS,
		"window": WINDOW,
		"workers": WORKERS,
		"epochs": EPOCHS,
		"seed": SEED,
		"graph_nodes": graph_nodes,
		"graph_edges": int(edge_index.shape[1]),
		"recipe_embeddings_present": int(recipe_present.sum()),
		"ingredient_embeddings_present": int(ingredient_present.sum()),
		"output": str(output),
	}
	with output.with_suffix(".json").open("w") as file:
		json.dump(metadata, file, indent=2)

	print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
	main()
