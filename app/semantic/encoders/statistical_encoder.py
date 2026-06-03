from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class StatisticalWindowEncoder:
    n_components: int | None = 8
    whiten: bool = False
    scaler: StandardScaler = field(default_factory=StandardScaler)
    reducer: PCA | None = None
    feature_columns: list[str] = field(default_factory=list)
    fill_values: dict[str, float] = field(default_factory=dict)

    def _numeric_frame(self, feature_frame: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        numeric = feature_frame.copy()
        for column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

        if fitting:
            self.feature_columns = list(numeric.columns)
            medians = numeric.median(axis=0, skipna=True).fillna(0.0)
            self.fill_values = medians.to_dict()
        else:
            numeric = numeric.reindex(columns=self.feature_columns)

        fill_values = pd.Series(self.fill_values)
        numeric = numeric.fillna(fill_values).fillna(0.0)
        return numeric

    def fit(self, feature_frame: pd.DataFrame) -> "StatisticalWindowEncoder":
        numeric = self._numeric_frame(feature_frame, fitting=True)
        scaled = self.scaler.fit_transform(numeric)

        if self.n_components:
            component_count = max(
                1,
                min(int(self.n_components), scaled.shape[0], scaled.shape[1]),
            )
            self.reducer = PCA(
                n_components=component_count,
                whiten=self.whiten,
                random_state=42,
            )
            self.reducer.fit(scaled)
        else:
            self.reducer = None

        return self

    def transform(self, feature_frame: pd.DataFrame) -> np.ndarray:
        numeric = self._numeric_frame(feature_frame, fitting=False)
        scaled = self.scaler.transform(numeric)
        if self.reducer is None:
            return scaled
        return self.reducer.transform(scaled)

    def fit_transform(self, feature_frame: pd.DataFrame) -> np.ndarray:
        self.fit(feature_frame)
        return self.transform(feature_frame)

    def save(self, path: str) -> str:
        joblib.dump(
            {
                "n_components": self.n_components,
                "whiten": self.whiten,
                "scaler": self.scaler,
                "reducer": self.reducer,
                "feature_columns": self.feature_columns,
                "fill_values": self.fill_values,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "StatisticalWindowEncoder":
        payload = joblib.load(path)
        instance = cls(
            n_components=payload["n_components"],
            whiten=payload["whiten"],
        )
        instance.scaler = payload["scaler"]
        instance.reducer = payload["reducer"]
        instance.feature_columns = payload["feature_columns"]
        instance.fill_values = payload["fill_values"]
        return instance

