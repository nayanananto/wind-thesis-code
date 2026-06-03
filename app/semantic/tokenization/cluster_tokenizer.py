from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


@dataclass
class ClusterTokenizer:
    n_clusters: int = 32
    batch_size: int = 256
    random_state: int = 42
    model: MiniBatchKMeans | None = None

    def fit(self, embeddings: np.ndarray) -> "ClusterTokenizer":
        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")

        cluster_count = max(1, min(int(self.n_clusters), len(matrix)))
        self.model = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=min(self.batch_size, max(cluster_count, len(matrix))),
            random_state=self.random_state,
            n_init=10,
        )
        self.model.fit(matrix)
        return self

    def transform(self, embeddings: np.ndarray) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Tokenizer has not been fitted yet.")

        matrix = np.asarray(embeddings, dtype=float)
        labels = self.model.predict(matrix)
        centers = self.model.cluster_centers_[labels]
        distances = np.linalg.norm(matrix - centers, axis=1)
        return pd.DataFrame(
            {
                "token_id": labels.astype(int),
                "token_distance": distances.astype(float),
            }
        )

    def fit_transform(self, embeddings: np.ndarray) -> pd.DataFrame:
        self.fit(embeddings)
        return self.transform(embeddings)

    def token_summary(self, token_frame: pd.DataFrame) -> pd.DataFrame:
        summary = (
            token_frame.groupby("token_id")
            .agg(
                count=("token_id", "size"),
                avg_distance=("token_distance", "mean"),
                max_distance=("token_distance", "max"),
            )
            .reset_index()
            .sort_values("token_id")
        )
        return summary

    def save(self, path: str) -> str:
        joblib.dump(
            {
                "n_clusters": self.n_clusters,
                "batch_size": self.batch_size,
                "random_state": self.random_state,
                "model": self.model,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "ClusterTokenizer":
        payload = joblib.load(path)
        instance = cls(
            n_clusters=payload["n_clusters"],
            batch_size=payload["batch_size"],
            random_state=payload["random_state"],
        )
        instance.model = payload["model"]
        return instance

