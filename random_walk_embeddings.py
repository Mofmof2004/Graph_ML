"""Train Node2Vec embeddings for the recipe-ingredient graph."""

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec


DATA_DIR = Path("data/processed")
SPLIT = "train"
OUTPUT = Path("data/node2vec")
DIMENSIONS = 32
WALK_LENGTH = 10
NUM_WALKS = 5
WINDOW = 10
P = 1.0
Q = 1.0
WORKERS = max(1, (os.cpu_count() or 2) - 1)
EPOCHS = 5
SEED = 0


def load_graph(data_dir, split):
	edges = np.load(data_dir / "edges.npz")
	recipes = pd.read_csv(data_dir / "recipes.csv")
	ingredients = pd.read_csv(data_dir / "ingredients.csv")

	edge_index = edges[split]
	graph = nx.Graph()
	graph.add_edges_from(
		(recipes.iloc[recipe_id]['title'], ingredients.iloc[ingredient_id]['name'])
		for recipe_id, ingredient_id in edge_index.T
	)
	return graph, recipes, ingredients


def train_embeddings(graph):
	node2vec = Node2Vec(
		graph,
		dimensions=DIMENSIONS,
		walk_length=WALK_LENGTH,
		num_walks=NUM_WALKS,
		p=P,
		q=Q,
		workers=WORKERS,
		seed=SEED,
		quiet=False,
	)
	return node2vec.fit(
		window=WINDOW,
		min_count=1,
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


def main():
	np.random.seed(SEED)

	graph, recipes, ingredients = load_graph(DATA_DIR, SPLIT)
	model = train_embeddings(graph)

	recipe_labels = recipes['title']
	ingredient_labels = ingredients['name']
	recipe_embeddings, recipe_present = collect_embeddings(
		model, recipe_labels, DIMENSIONS
	)
	ingredient_embeddings, ingredient_present = collect_embeddings(
		model, ingredient_labels, DIMENSIONS
	)

	output = OUTPUT / f"node2vec_{SPLIT}.npz"
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
		"dimensions": DIMENSIONS,
		"walk_length": WALK_LENGTH,
		"num_walks": NUM_WALKS,
		"window": WINDOW,
		"p": P,
		"q": Q,
		"workers": WORKERS,
		"epochs": EPOCHS,
		"seed": SEED,
		"graph_nodes": graph.number_of_nodes(),
		"graph_edges": graph.number_of_edges(),
		"recipe_embeddings_present": int(recipe_present.sum()),
		"ingredient_embeddings_present": int(ingredient_present.sum()),
		"output": str(output),
	}
	with output.with_suffix(".json").open("w") as file:
		json.dump(metadata, file, indent=2)

	print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
	main()
