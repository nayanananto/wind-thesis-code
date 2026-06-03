from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class SimilarRegimeSearcher:
    metric: str = "cosine"
    n_neighbors: int = 6
    model: NearestNeighbors | None = None
    embedding_columns: list[str] = field(default_factory=list)
    state_frame: pd.DataFrame | None = None

    def fit(self, state_frame: pd.DataFrame) -> "SimilarRegimeSearcher":
        embedding_columns = [
            column for column in state_frame.columns if column.startswith("embedding_")
        ]
        if not embedding_columns:
            raise ValueError("Semantic state frame must include embedding columns.")

        matrix = state_frame[embedding_columns].to_numpy(dtype=float)
        neighbor_count = max(1, min(int(self.n_neighbors), len(state_frame)))
        self.model = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric=self.metric,
        )
        self.model.fit(matrix)
        self.embedding_columns = embedding_columns
        self.state_frame = state_frame.reset_index(drop=True).copy()
        return self

    def query_by_window_id(self, window_id: str, top_k: int = 5) -> pd.DataFrame:
        if self.model is None or self.state_frame is None:
            raise ValueError("Searcher must be fitted or loaded before querying.")

        matches = self.state_frame[self.state_frame["window_id"] == window_id]
        if matches.empty:
            raise ValueError(f"Window id '{window_id}' not found in semantic state frame.")

        query_vector = matches.iloc[0][self.embedding_columns].to_numpy(dtype=float).reshape(1, -1)
        distances, indices = self.model.kneighbors(
            query_vector,
            n_neighbors=max(1, min(int(top_k) + 1, len(self.state_frame))),
        )

        neighbors = self.state_frame.iloc[indices[0]].copy()
        neighbors["retrieval_distance"] = distances[0]
        neighbors = neighbors[neighbors["window_id"] != window_id]
        return neighbors.head(top_k).reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "metric": self.metric,
                "n_neighbors": self.n_neighbors,
                "model": self.model,
                "embedding_columns": self.embedding_columns,
                "state_frame": self.state_frame,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SimilarRegimeSearcher":
        payload = joblib.load(path)
        instance = cls(
            metric=payload["metric"],
            n_neighbors=payload["n_neighbors"],
        )
        instance.model = payload["model"]
        instance.embedding_columns = payload["embedding_columns"]
        instance.state_frame = payload["state_frame"]
        return instance
