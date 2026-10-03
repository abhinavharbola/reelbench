import json
from pathlib import Path

import faiss
import numpy as np
import polars as pl


class FaissRetriever:
    def __init__(self, normalize: bool = True):
        self.index: faiss.Index | None = None
        self.idx_to_item_id: dict[int, int] = {}
        self.item_id_to_idx: dict[int, int] = {}
        self.normalize = normalize

    def _set_id_map(self, idx_to_item_id: dict[int, int]) -> None:
        self.idx_to_item_id = idx_to_item_id
        self.item_id_to_idx = {item_id: idx for idx, item_id in idx_to_item_id.items()}

    def _prepare(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def build(self, item_embeddings: pl.DataFrame, embedding_col: str = "embedding", id_col: str = "movieId") -> None:
        ids = item_embeddings[id_col].to_list()
        vectors = np.array(item_embeddings[embedding_col].to_list(), dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("item embeddings contain NaN or infinite values; refusing to build an index")

        vectors = self._prepare(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self._set_id_map({i: item_id for i, item_id in enumerate(ids)})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        pl.DataFrame(
            {"faiss_idx": list(self.idx_to_item_id.keys()), "movieId": list(self.idx_to_item_id.values())}
        ).write_parquet(path.with_suffix(".idmap.parquet"))
        path.with_suffix(".meta.json").write_text(json.dumps({"normalize": self.normalize}))

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
        id_map = pl.read_parquet(path.with_suffix(".idmap.parquet"))
        self._set_id_map(dict(zip(id_map["faiss_idx"].to_list(), id_map["movieId"].to_list())))
        meta_path = path.with_suffix(".meta.json")
        self.normalize = json.loads(meta_path.read_text())["normalize"] if meta_path.exists() else True

    def item_ids(self) -> set[int]:
        return set(self.idx_to_item_id.values())

    def has_item(self, item_id: int) -> bool:
        return item_id in self.item_id_to_idx

    def vector_for(self, item_id: int) -> np.ndarray:
        return self.index.reconstruct(self.item_id_to_idx[item_id])

    def query(
        self,
        user_embedding: np.ndarray,
        top_n: int = 100,
        exclude: set | None = None,
    ) -> list[tuple[int, float]]:
        vec = np.asarray(user_embedding, dtype=np.float32).reshape(1, -1)
        if not np.isfinite(vec).all():
            return []
        if vec.shape[1] != self.index.d:
            raise ValueError(f"query dimension {vec.shape[1]} does not match index dimension {self.index.d}")
        vec = self._prepare(vec)

        exclude = exclude or set()
        search_k = min(self.index.ntotal, top_n + len(exclude))
        if search_k <= 0:
            return []

        scores, indices = self.index.search(vec, search_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            item_id = self.idx_to_item_id[int(idx)]
            if item_id in exclude:
                continue
            results.append((item_id, float(score)))
            if len(results) == top_n:
                break
        return results


def build_and_save_index(item_embeddings_path: Path, index_output_path: Path, normalize: bool = True) -> None:
    item_embeddings = pl.read_parquet(item_embeddings_path)
    retriever = FaissRetriever(normalize=normalize)
    retriever.build(item_embeddings)
    retriever.save(index_output_path)
    print(f"FAISS index with {retriever.index.ntotal} items written to {index_output_path}")
