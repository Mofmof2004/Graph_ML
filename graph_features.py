"""Graph-building utilities that can be imported before processed data exists.

The sentence encoder is imported only when encoding is requested.
"""

import numpy as np


FEATURE_MODEL = "all-mpnet-base-v2"


def calculate_edge_weights(graph, num_ingredients, edge_share=1, idf_graph=None,
                           share_exponent=0.5):
    """Fit IDF on the supplied reference graph, then weight the requested edges."""
    reference = graph if idf_graph is None else idf_graph
    counts = np.bincount(reference[1], minlength=num_ingredients)
    idf = np.log((1 + len(np.unique(reference[0]))) / (1 + counts)) + 1.0
    values = idf[graph[1]] * np.asarray(edge_share) ** share_exponent
    return values.astype(np.float32), idf


def encode_words(words, model_name=FEATURE_MODEL):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(words, batch_size=256, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
