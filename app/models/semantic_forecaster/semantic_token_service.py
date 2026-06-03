from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.models.semantic_forecaster.semantic_input_builder import (
    SEMANTIC_META_COLUMNS,
    SemanticSampleBundle,
    build_semantic_training_samples,
)
from app.semantic.encoders.statistical_encoder import StatisticalWindowEncoder
from app.semantic.tokenization.cluster_tokenizer import ClusterTokenizer


def _prepare_semantic_model_frame(
    feature_frame: pd.DataFrame,
    encoder: StatisticalWindowEncoder,
    tokenizer: ClusterTokenizer,
    fit: bool,
) -> pd.DataFrame:
    numeric_frame = feature_frame.drop(columns=SEMANTIC_META_COLUMNS, errors="ignore")

    if fit:
        embeddings = encoder.fit_transform(numeric_frame)
        tokens = tokenizer.fit_transform(embeddings)
    else:
        embeddings = encoder.transform(numeric_frame)
        tokens = tokenizer.transform(embeddings)

    embedding_columns = [f"embedding_{idx:02d}" for idx in range(embeddings.shape[1])]
    model_frame = pd.concat(
        [
            numeric_frame.reset_index(drop=True),
            pd.DataFrame(embeddings, columns=embedding_columns),
            tokens.reset_index(drop=True),
        ],
        axis=1,
    )
    return model_frame


@dataclass
class SemanticTokenForecaster:
    window_size: int = 48
    step_size: int = 1
    n_components: int = 8
    n_clusters: int = 16
    n_estimators: int = 200
    min_samples_leaf: int = 2
    random_state: int = 42
    encoder: StatisticalWindowEncoder = field(init=False)
    tokenizer: ClusterTokenizer = field(init=False)
    regressor: RandomForestRegressor = field(init=False)
    feature_columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.encoder = StatisticalWindowEncoder(n_components=self.n_components)
        self.tokenizer = ClusterTokenizer(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
        )
        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )

    def fit(self, bundle: SemanticSampleBundle) -> "SemanticTokenForecaster":
        model_frame = _prepare_semantic_model_frame(
            bundle.feature_frame,
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            fit=True,
        )
        self.feature_columns = list(model_frame.columns)
        self.regressor.fit(model_frame, bundle.targets)
        return self

    def predict(self, bundle: SemanticSampleBundle) -> pd.DataFrame:
        if not self.feature_columns:
            raise ValueError("SemanticTokenForecaster must be fitted before prediction.")

        model_frame = _prepare_semantic_model_frame(
            bundle.latest_feature_row,
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            fit=False,
        )
        model_frame = model_frame.reindex(columns=self.feature_columns, fill_value=0.0)

        prediction = self.regressor.predict(model_frame)[0]
        prediction = np.asarray(prediction, dtype=float).clip(min=0.0)

        return pd.DataFrame(
            {
                "datetime": bundle.future_index,
                "wind_speed": prediction,
            }
        )

    def fit_predict(self, train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        bundle = build_semantic_training_samples(
            train_df,
            horizon=horizon,
            window_size=self.window_size,
            step_size=self.step_size,
        )
        self.fit(bundle)
        return self.predict(bundle)


def semantic_token_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    window_size: int = 48,
    step_size: int = 1,
    n_components: int = 8,
    n_clusters: int = 16,
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    forecaster = SemanticTokenForecaster(
        window_size=window_size,
        step_size=step_size,
        n_components=n_components,
        n_clusters=n_clusters,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    return forecaster.fit_predict(train_df=train_df, horizon=horizon)
