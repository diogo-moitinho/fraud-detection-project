"""Artifact persistence and the scoring entry point.

Import as ``from fraud_detection import deployment as dep``.

The artifact bundles the threshold with the pipeline deliberately. A model saved
without its operating point is not usable: whoever loads it will call
``predict`` at 0.5 and get a recall unrelated to the one reported.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import polars as pl

from . import config as cfg
from . import model as mdl
from . import preprocessing as prep


def build_artifact(
    result: mdl.EvalResult,
    splits: prep.Splits,
    study=None,
) -> dict:
    """Assemble the serialisable bundle: pipeline, threshold, contract, provenance."""
    import sklearn
    import xgboost

    return {
        'pipeline': result.pipeline,
        'threshold': result.threshold,
        'features': result.features,
        'categorical': cfg.CATEGORICAL,
        'excluded_post_settlement': cfg.POST_SETTLEMENT_FEATURES,
        'test_pr_auc': result.pr_auc,
        'test_recall': result.recall,
        'test_precision': result.precision,
        'train_steps': (int(splits.train['step'].min()), int(splits.train['step'].max())),
        'test_steps': (int(splits.test['step'].min()), int(splits.test['step'].max())),
        'best_params': study.best_params if study is not None else None,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'sklearn_version': sklearn.__version__,
        'xgboost_version': xgboost.__version__,
    }


def save_artifact(
    result: mdl.EvalResult,
    splits: prep.Splits,
    study=None,
    directory: str | None = None,
    filename: str | None = None,
    verbose: bool = True,
) -> str:
    """Write the artifact to disk, creating the directory if needed."""
    directory = cfg.MODEL_DIR if directory is None else directory
    filename = cfg.ARTIFACT_NAME if filename is None else filename

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    artifact = build_artifact(result, splits, study)
    joblib.dump(artifact, path)

    if verbose:
        print(json.dumps(
            {k: v for k, v in artifact.items() if k != 'pipeline'},
            indent=2, default=str,
        ))

    return path


def load_artifact(path: str | None = None) -> dict:
    """Read a saved artifact."""
    if path is None:
        path = os.path.join(cfg.MODEL_DIR, cfg.ARTIFACT_NAME)
    return joblib.load(path)


def score_transactions(frame: pl.DataFrame, artifact: dict) -> pl.DataFrame:
    """Score transactions and apply the stored decision threshold.

    ``frame`` must contain every column in ``artifact['features']``. Three are
    derived rather than raw — ``is_merchant_dest``, ``hour_of_day`` and
    ``day_of_week`` — and are the caller's responsibility; all three depend only
    on fields available at authorisation time.

    Post-settlement columns are rejected rather than ignored. A caller supplying
    them is describing a transaction that has already completed, at which point
    prevention is no longer possible, and silently accepting them is how a
    forensic model ends up deployed as a real-time one.
    """
    leaked = [c for c in cfg.POST_SETTLEMENT_FEATURES if c in artifact['features']]
    if leaked:
        raise ValueError(
            f"artifact declares post-settlement features {leaked}, which are not "
            "available at authorisation time; this artifact is forensic-only"
        )

    missing = [c for c in artifact['features'] if c not in frame.columns]
    if missing:
        raise ValueError(f'frame is missing required feature columns: {missing}')

    X = frame.select(artifact['features']).to_pandas()
    probabilities = artifact['pipeline'].predict_proba(X)[:, 1]

    return frame.with_columns([
        pl.Series('fraud_probability', probabilities),
        pl.Series('flagged', (probabilities >= artifact['threshold']).astype(int)),
    ])
