import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

logger = logging.getLogger(__name__)

# ─── Feature Schema ───────────────────────────────────────────────────────────

# The 12 behavioural features our model is trained on.
# These are derived from session aggregation of raw HTTP/WAF logs.
# NOTE: 'is_anomaly' is the Ground Truth label — stripped before training,
#       used only for evaluation (Precision / Recall / F1).
FEATURE_COLUMNS = [
    "request_count",
    "error_rate",
    "auth_failure_count",
    "avg_response_time_ms",
    "p95_response_time_ms",
    "unique_endpoints",
    "unique_ips",
    "anomalous_path_count",
    "post_ratio",
    "js_error_count",
    "request_rate",
    "session_duration_s",
]

LABEL_COLUMN = "is_anomaly"          # Ground Truth — eval only, never trained on
DEFAULT_SCALER_PATH = "models/scaler.pkl"


# ─── Functions ────────────────────────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the aggregated session CSV produced by dataset_aggregator.py.

    Args:
        filepath: Path to risk_trace_training_data.csv

    Returns:
        Raw DataFrame with all columns including is_anomaly.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    try:
        df = pd.read_csv(filepath, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, low_memory=False, encoding="latin-1")

    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")

    logger.info(f"Loaded dataset: {filepath} — shape: {df.shape}")
    logger.info(f"Normal sessions : {(df[LABEL_COLUMN] == 0).sum()}")
    logger.info(f"Attack sessions : {(df[LABEL_COLUMN] == 1).sum()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, coerce, and sanitize the session feature DataFrame.

    Our dataset is fully numerical (no categorical columns), so this step:
      - Drops any unexpected non-feature columns
      - Fills rare NaN gaps with column medians
      - Clips negative values to 0 (rates/counts are always >= 0)

    Args:
        df: Raw DataFrame from load_data()

    Returns:
        Clean DataFrame ready for scaling.
    """
    df = df.copy()

    # Keep only the known feature columns + ground truth label
    cols_to_keep = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c in df.columns]
    df = df[cols_to_keep]

    # Coerce everything to numeric (silently converts bad strings to NaN)
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN with column median (safe for skewed distributions)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Rates and counts cannot be negative — clip as safety net
    rate_cols = ["error_rate", "post_ratio", "request_rate"]
    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0, upper=1.0)

    count_cols = [
        "request_count", "auth_failure_count", "unique_endpoints",
        "unique_ips", "anomalous_path_count", "js_error_count",
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    logger.info(f"Data cleaned — final shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    No-op for the RiskTrace dataset — all features are already numerical.

    This function is kept for API compatibility with train.py and predict.py.
    If future categorical features are added (e.g. device_type), this is where
    their LabelEncoder logic would live.

    Args:
        df:       Clean DataFrame.
        training: Unused — kept for interface consistency.

    Returns:
        Unchanged DataFrame.
    """
    logger.info("encode_categoricals: all features are numerical, no encoding needed.")
    return df


def scale_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Apply StandardScaler to all feature columns (excluding is_anomaly label).

    Scaling ensures high-range features like request_count (0–10,000) do not
    dominate low-range features like error_rate (0.0–1.0) when the Isolation
    Forest builds its binary split trees.

    During training  → fits scaler on X and saves it to models/scaler.pkl
    During inference → loads saved scaler and transforms incoming features

    Args:
        df:       Clean DataFrame (contains FEATURE_COLUMNS + optionally is_anomaly).
        training: True during train.py; False during predict.py inference.

    Returns:
        DataFrame with scaled feature columns.
    """
    df = df.copy()
    os.makedirs(os.path.dirname(DEFAULT_SCALER_PATH) or ".", exist_ok=True)

    # Only scale the actual feature columns — never touch the label
    cols_to_scale = [c for c in FEATURE_COLUMNS if c in df.columns]

    if not cols_to_scale:
        logger.warning("No feature columns found to scale.")
        return df

    if training:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, DEFAULT_SCALER_PATH)
        logger.info(f"Scaler fitted and saved to {DEFAULT_SCALER_PATH}")
    else:
        if not os.path.exists(DEFAULT_SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler not found at {DEFAULT_SCALER_PATH}. Run train.py first."
            )
        scaler = joblib.load(DEFAULT_SCALER_PATH)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        logger.info("Scaler loaded and applied.")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Split the DataFrame into the feature matrix X and the label vector y.

    The label (is_anomaly) is NEVER passed to the Isolation Forest during
    training — it is an unsupervised model. The label is only used afterward
    for evaluation (Confusion Matrix, F1-Score, Precision, Recall).

    Args:
        df: Scaled DataFrame (output of scale_features()).

    Returns:
        (X, y) where:
            X → pd.DataFrame of shape (n_sessions, len(FEATURE_COLUMNS))
            y → np.ndarray of 0/1 labels, or None if label column is absent
    """
    df = df.copy()

    if LABEL_COLUMN in df.columns:
        y = df[LABEL_COLUMN].values.astype(int)
        X = df.drop(columns=[LABEL_COLUMN])
    else:
        y = None
        X = df

    # Ensure column order always matches FEATURE_COLUMNS
    X = X[[c for c in FEATURE_COLUMNS if c in X.columns]]

    logger.info(f"Features ready — X: {X.shape}, y: {'present' if y is not None else 'absent'}")
    return X, y



