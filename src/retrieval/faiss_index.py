"""
FAISS (CPU) retrieval over two-tower or SASRec item embeddings.

Index is built once from a cached item-embeddings parquet and written to
local disk; querying at serving time only ever reads that local index file,
never recomputes embeddings live -- matches the "no live external calls at
serving time" requirement.
"""

from pathlib import Path

import faiss
import numpy as np
import polars as pl


class FaissRetriever:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.idx_to_item_id: dict[int, int] = {}

    def build(self, item_embeddings: pl.DataFrame, embedding_col: str = "embedding", id_col: str = "movieId") -> None:
        """item_embeddings: one row per item, an `embedding` list column and
        an id column. Inner-product index over L2-normalized vectors, i.e.
        cosine similarity ranking."""
        ids = item_embeddings[id_col].to_list()
        vectors = np.array(item_embeddings[embedding_col].to_list(), dtype=np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.idx_to_item_id = {i: item_id for i, item_id in enumerate(ids)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        id_map_path = path.with_suffix(".idmap.parquet")
        pl.DataFrame(
            {"faiss_idx": list(self.idx_to_item_id.keys()), "movieId": list(self.idx_to_item_id.values())}
        ).write_parquet(id_map_path)

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
        id_map_path = path.with_suffix(".idmap.parquet")
        id_map = pl.read_parquet(id_map_path)
        self.idx_to_item_id = dict(zip(id_map["faiss_idx"].to_list(), id_map["movieId"].to_list()))

    def query(self, user_embedding: np.ndarray, top_n: int = 100) -> list[tuple[int, float]]:
        """Returns [(movieId, similarity_score), ...] ranked descending."""
        vec = np.asarray(user_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, indices = self.index.search(vec, top_n)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.idx_to_item_id[int(idx)], float(score)))
        return results


def build_and_save_index(item_embeddings_path: Path, index_output_path: Path) -> None:
    item_embeddings = pl.read_parquet(item_embeddings_path)
    retriever = FaissRetriever()
    retriever.build(item_embeddings)
    retriever.save(index_output_path)
    print(f"FAISS index with {retriever.index.ntotal} items written to {index_output_path}")
