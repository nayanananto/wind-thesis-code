from datetime import datetime

import pandas as pd

from app.config.paths import (
    METADATA_DIR,
    SEMANTIC_DATA_DIR,
    SEMANTIC_ENCODER_DIR,
    SEMANTIC_INDEX_DIR,
    TOKENIZER_ARTIFACT_DIR,
    ensure_project_dirs,
)
from app.preprocessing.feature_engineering import prepare_semantic_frame
from app.semantic.labeling.regime_labeler import (
    attach_cluster_labels,
    build_cluster_profiles,
    label_cluster_profiles,
)
from app.semantic.encoders.statistical_encoder import StatisticalWindowEncoder
from app.semantic.retrieval.similar_regime_search import SimilarRegimeSearcher
from app.semantic.summarizers.physics_summary import summarize_windows
from app.semantic.tokenization.cluster_tokenizer import ClusterTokenizer
from app.semantic.windowing.window_builder import WindowConfig, build_windows
from app.storage.local.json_store import save_json


META_COLUMNS = [
    "window_id",
    "window_start",
    "window_end",
    "window_rows",
    "time_span_hours",
]


def _default_run_name() -> str:
    return datetime.utcnow().strftime("semantic_%Y%m%d_%H%M%S")


def run_semantic_build(
    raw_df: pd.DataFrame,
    window_size: int = 24,
    step_size: int = 1,
    n_components: int = 8,
    n_clusters: int = 32,
    enable_llm_labels: bool = False,
    run_name: str | None = None,
) -> dict:
    ensure_project_dirs()
    run_name = run_name or _default_run_name()

    prepared = prepare_semantic_frame(raw_df)
    windows = build_windows(
        prepared,
        WindowConfig(window_size=window_size, step_size=step_size),
    )
    if not windows:
        raise ValueError("No semantic windows could be built from the supplied data.")

    features_df = summarize_windows(windows)
    feature_matrix = features_df.drop(columns=META_COLUMNS, errors="ignore")

    encoder = StatisticalWindowEncoder(n_components=n_components)
    embeddings = encoder.fit_transform(feature_matrix)
    embedding_columns = [f"embedding_{idx:02d}" for idx in range(embeddings.shape[1])]

    tokenizer = ClusterTokenizer(n_clusters=n_clusters)
    token_frame = tokenizer.fit_transform(embeddings)
    token_summary = tokenizer.token_summary(token_frame)

    semantic_states_df = pd.concat(
        [
            features_df[META_COLUMNS].reset_index(drop=True),
            pd.DataFrame(embeddings, columns=embedding_columns),
            token_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    merged_states_df, cluster_profiles_df, representative_examples_df = build_cluster_profiles(
        feature_frame=features_df,
        embeddings_frame=semantic_states_df,
    )
    labeled_cluster_df = label_cluster_profiles(
        profile_frame=cluster_profiles_df,
        example_frame=representative_examples_df,
        enable_llm=enable_llm_labels,
    )
    semantic_states_df = attach_cluster_labels(
        semantic_state_frame=merged_states_df,
        labeled_profile_frame=labeled_cluster_df,
    )

    searcher = SimilarRegimeSearcher().fit(semantic_states_df)

    features_path = SEMANTIC_DATA_DIR / f"{run_name}_window_features.csv"
    embeddings_path = SEMANTIC_DATA_DIR / f"{run_name}_embeddings.csv"
    token_summary_path = SEMANTIC_DATA_DIR / f"{run_name}_token_summary.csv"
    semantic_states_path = SEMANTIC_DATA_DIR / f"{run_name}_semantic_states.csv"
    cluster_profiles_path = SEMANTIC_DATA_DIR / f"{run_name}_cluster_profiles.csv"
    representative_examples_path = SEMANTIC_DATA_DIR / f"{run_name}_representative_examples.csv"
    encoder_path = SEMANTIC_ENCODER_DIR / f"{run_name}_encoder.joblib"
    tokenizer_path = TOKENIZER_ARTIFACT_DIR / f"{run_name}_tokenizer.joblib"
    search_index_path = SEMANTIC_INDEX_DIR / f"{run_name}_search_index.joblib"
    metadata_path = METADATA_DIR / f"{run_name}_semantic_build.json"

    features_df.to_csv(features_path, index=False)
    pd.concat(
        [
            semantic_states_df[META_COLUMNS].reset_index(drop=True),
            semantic_states_df[embedding_columns + ["token_id", "token_distance"]].reset_index(drop=True),
        ],
        axis=1,
    ).to_csv(embeddings_path, index=False)
    token_summary.to_csv(token_summary_path, index=False)
    semantic_states_df.to_csv(semantic_states_path, index=False)
    labeled_cluster_df.to_csv(cluster_profiles_path, index=False)
    representative_examples_df.to_csv(representative_examples_path, index=False)
    encoder.save(str(encoder_path))
    tokenizer.save(str(tokenizer_path))
    searcher.save(search_index_path)

    metadata = {
        "run_name": run_name,
        "window_size": int(window_size),
        "step_size": int(step_size),
        "n_windows": int(len(features_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_tokens": int(token_frame["token_id"].nunique()),
        "feature_columns": list(feature_matrix.columns),
        "llm_labels_enabled": bool(enable_llm_labels),
        "output_paths": {
            "features": str(features_path),
            "embeddings": str(embeddings_path),
            "token_summary": str(token_summary_path),
            "semantic_states": str(semantic_states_path),
            "cluster_profiles": str(cluster_profiles_path),
            "representative_examples": str(representative_examples_path),
            "encoder": str(encoder_path),
            "tokenizer": str(tokenizer_path),
            "search_index": str(search_index_path),
        },
    }
    save_json(metadata_path, metadata)
    metadata["output_paths"]["metadata"] = str(metadata_path)
    return metadata
