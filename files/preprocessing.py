"""Loading, feature engineering, chronological splitting and pipeline assembly.

Import as ``from fraud_detection import preprocessing as prep``.

Two invariants this module exists to protect:

1. Only row-wise arithmetic runs before the split. Anything that aggregates
   across rows would let the test partition inform the training data.
2. Every fitted transformation (encoder, scaler, resampler) is returned inside a
   pipeline, never applied to a dataset in advance. A resampler fitted outside
   cross-validation leaks synthetic points into the validation folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import polars as pl
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


# --------------------------------------------------------------------------- #
# Loading and feature engineering
# --------------------------------------------------------------------------- #
def load_transactions(path: str | None = None) -> pl.DataFrame:
    """Read the raw CSV and derive every engineered column.

    All derivations are row-wise, so running this before the split is safe.
    ``isFlaggedFraud`` is dropped: it is the output of the incumbent rules engine
    and would be direct label leakage. The frame is returned sorted by ``step``,
    which every downstream function assumes.
    """
    path = path or cfg.DATA_PATH

    return (
        pl.read_csv(path)
        .with_columns([
            # Accounting reconciliation (post-settlement, forensic use only)
            (pl.col('oldbalanceOrg') - pl.col('amount') - pl.col('newbalanceOrig'))
            .alias('errorBalanceOrig'),
            (pl.col('oldbalanceDest') + pl.col('amount') - pl.col('newbalanceDest'))
            .alias('errorBalanceDest'),
            # Structural
            pl.col('nameDest').str.starts_with('M').cast(pl.Int8).alias('is_merchant_dest'),
            # Temporal: cyclical features are model-safe
            (pl.col('step') % 24).alias('hour_of_day'),
            ((pl.col('step') // 24) % 7).alias('day_of_week'),
            # Temporal: monotonic indices, exploration only
            (pl.col('step') // 24 + 1).alias('day_of_month'),
        ])
        .with_columns(
            pl.when(pl.col('day_of_month') <= 10).then(pl.lit('Start'))
            .when(pl.col('day_of_month') <= 20).then(pl.lit('Middle'))
            .otherwise(pl.lit('End'))
            .alias('month_period')
        )
        .drop(['nameOrig', 'nameDest', 'isFlaggedFraud'])
        .sort('step')
    )


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Splits:
    """Chronologically ordered partitions. ``test`` is read exactly once."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame

    def __iter__(self):
        yield from (('TRAIN', self.train), ('VAL', self.val), ('TEST', self.test))


def chronological_split(
    df: pl.DataFrame,
    test_quantile: float | None = None,
    val_quantile: float | None = None,
) -> Splits:
    """Partition by ``step`` so that every fold is strictly later than its history.

    A random split would place future transactions in training. It would also
    separate PaySim's paired fraud rows — a TRANSFER followed by a CASH_OUT of
    the same amount — across partitions, letting the model match an amount it has
    already seen. ``stratify`` does not prevent that; it balances the label, not
    the entity.
    """
    test_quantile = cfg.TEST_QUANTILE if test_quantile is None else test_quantile
    val_quantile = cfg.VAL_QUANTILE if val_quantile is None else val_quantile

    cut_test = df.select(pl.col('step').quantile(test_quantile)).item()
    dev = df.filter(pl.col('step') <= cut_test)
    test = df.filter(pl.col('step') > cut_test)

    cut_val = dev.select(pl.col('step').quantile(val_quantile)).item()
    train = dev.filter(pl.col('step') <= cut_val)
    val = dev.filter(pl.col('step') > cut_val)

    return Splits(train=train, val=val, test=test)


def describe_splits(splits: Splits) -> pd.DataFrame:
    """Row counts, step and day ranges, and fraud prevalence per partition.

    Prevalence is reported per partition because PR-AUC is a function of base
    rate: a test score is not comparable to a validation score without both.
    """
    rows = []
    for name, frame in splits:
        n = frame.height
        k = frame.select(pl.col(cfg.TARGET).sum()).item()
        rows.append({
            'partition': name,
            'step_from': frame['step'].min(),
            'step_to': frame['step'].max(),
            'day_from': frame['day_of_month'].min(),
            'day_to': frame['day_of_month'].max(),
            'rows': n,
            'frauds': k,
            'fraud_rate': k / n,
        })
    return pd.DataFrame(rows).set_index('partition')


def stratified_search_sample(
    frame: pl.DataFrame,
    fraction: float | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """Subsample for screening and tuning, preserving prevalence and ordering.

    Stratifying on the target keeps the base rate identical to the full frame, so
    PR-AUC stays on the same scale. Re-sorting by ``step`` preserves the ordering
    that forward-chaining cross-validation depends on.
    """
    fraction = cfg.SEARCH_FRACTION if fraction is None else fraction
    seed = cfg.RANDOM_STATE if seed is None else seed

    return (
        frame
        .with_columns(
            pl.int_range(pl.len()).shuffle(seed=seed).over(cfg.TARGET).alias('_rank'),
            pl.len().over(cfg.TARGET).alias('_group_size'),
        )
        .filter(pl.col('_rank') < (pl.col('_group_size') * fraction).ceil())
        .drop(['_rank', '_group_size'])
        .sort('step')
    )


# --------------------------------------------------------------------------- #
# Frame conversion
# --------------------------------------------------------------------------- #
def to_xy(frame: pl.DataFrame, features: list[str]) -> tuple[pd.DataFrame, 'pd.Series']:
    """polars -> pandas feature matrix and 1-D target array.

    The conversion is deliberate: imblearn's resamplers do not accept polars
    output, and a (n, 1) target shape triggers a DataConversionWarning downstream.
    """
    X = frame.select(features).to_pandas()
    y = frame.select(cfg.TARGET).to_pandas()[cfg.TARGET].to_numpy().ravel()
    return X, y


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #
def build_preprocessor(features: list[str], scale: bool = False) -> ColumnTransformer:
    """One-hot the categorical column, optionally standardise the numeric block.

    Scaling is only needed by distance- and gradient-sensitive models; tree
    ensembles are invariant to it.
    """
    categorical = [c for c in features if c in cfg.CATEGORICAL]
    numeric = [c for c in features if c not in cfg.CATEGORICAL]

    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
            ('num', StandardScaler() if scale else 'passthrough', numeric),
        ],
        remainder='drop',
    )


def build_pipeline(
    model,
    features: list[str],
    balance: str = 'none',
    scale: bool = False,
    smote_k: int = 5,
) -> ImbPipeline:
    """Assemble preprocessing, optional resampling and the estimator.

    ``balance='smote'`` inserts the resampler as a pipeline step so it is
    re-fitted on each training fold and never sees the validation fold. Applying
    SMOTE to a dataset in advance is the leak this signature exists to prevent.
    ``sklearn.pipeline.Pipeline`` rejects resamplers, hence the imblearn variant.
    """
    if balance not in {'none', 'smote'}:
        raise ValueError(f"balance must be 'none' or 'smote', got {balance!r}")

    steps = [('preprocessor', build_preprocessor(features, scale=scale))]
    if balance == 'smote':
        steps.append(('smote', SMOTE(random_state=cfg.RANDOM_STATE, k_neighbors=smote_k)))
    steps.append(('model', model))

    return ImbPipeline(steps)


def feature_names(pipeline: ImbPipeline) -> list[str]:
    """Post-encoding column names, with the ColumnTransformer prefix stripped."""
    raw = pipeline.named_steps['preprocessor'].get_feature_names_out()
    return [name.split('__', 1)[-1] for name in raw]
