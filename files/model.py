"""Cross-validation, model screening, hyperparameter search and evaluation.

Import as ``from fraud_detection import model as mdl``.

Metric choice runs through this whole module: ``average_precision`` (area under
the Precision-Recall curve) rather than accuracy or ROC-AUC. At a 0.13% base
rate, predicting "not fraud" for everything scores 99.87% accuracy, and ROC-AUC
stays flattering because the false-positive rate barely moves when negatives
outnumber positives by three orders of magnitude.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
import polars as pl
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as cfg
from . import preprocessing as prep

SCORING = 'average_precision'

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def pr_auc_cv(pipeline, X, y, n_splits: int | None = None) -> tuple[float, float]:
    """Mean and standard deviation of PR-AUC under forward-chaining CV.

    ``TimeSeriesSplit`` guarantees every validation fold is strictly later than
    its training folds, which is what makes the score an estimate of forecasting
    rather than of interpolation.
    """
    n_splits = cfg.CV_SPLITS if n_splits is None else n_splits
    scores = cross_val_score(
        pipeline, X, y,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring=SCORING,
        n_jobs=1,
    )
    return float(scores.mean()), float(scores.std())


def positive_weight(y) -> float:
    """Negative-to-positive ratio, the natural ``scale_pos_weight`` reference."""
    y = np.asarray(y)
    positives = max(int((y == 1).sum()), 1)
    return float((y == 0).sum() / positives)


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #
def candidate_models() -> dict[str, tuple[object, bool]]:
    """Screening candidates mapped to ``(estimator, needs_scaling)``."""
    rs = cfg.RANDOM_STATE
    return {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=rs), True),
        'Decision Tree': (DecisionTreeClassifier(random_state=rs), False),
        'Random Forest': (RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1), False),
        'XGBoost': (XGBClassifier(tree_method='hist', eval_metric='aucpr',
                                  random_state=rs, n_jobs=-1), False),
    }


def screen_models(
    search_frame: pl.DataFrame,
    features: list[str],
    balance: str = 'smote',
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every candidate under identical CV and return a comparison table.

    Fit time is included as a first-class column, not a footnote: a model that
    ties on PR-AUC at a tenth of the cost is the better choice, because the
    tuning stage multiplies that cost by trials times folds.
    """
    X, y = prep.to_xy(search_frame, features)

    rows = {}
    for name, (estimator, needs_scaling) in candidate_models().items():
        started = time.time()
        pipeline = prep.build_pipeline(estimator, features, balance=balance, scale=needs_scaling)
        mean_ap, std_ap = pr_auc_cv(pipeline, X, y)
        elapsed = time.time() - started
        rows[name] = {'PR-AUC (CV)': mean_ap, 'Std': std_ap, 'Fit time (s)': elapsed}
        if verbose:
            print(f"{name:<22} PR-AUC {mean_ap:.4f} (+/- {std_ap:.4f})  [{elapsed:.1f}s]")

    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Hyperparameter search
# --------------------------------------------------------------------------- #
def _suggest_params(trial: optuna.Trial, pos_weight: float) -> tuple[dict, str, int]:
    """Sample an XGBoost configuration plus a class-balancing strategy.

    The balancing strategy is itself a hyperparameter. For gradient boosting,
    ``scale_pos_weight`` is usually as effective as SMOTE and far cheaper on
    millions of rows, where SMOTE's nearest-neighbour step dominates fit time.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 900, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 20.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
    }

    balance = trial.suggest_categorical('balance', ['smote', 'scale_pos_weight'])
    smote_k = 5
    if balance == 'scale_pos_weight':
        # Wide floor so the search can express a genuine preference rather than
        # settling against a binding bound.
        params['scale_pos_weight'] = trial.suggest_float(
            'scale_pos_weight', 0.01 * pos_weight, 2.0 * pos_weight, log=True
        )
    else:
        smote_k = trial.suggest_int('smote_k_neighbors', 3, 10)

    return params, balance, smote_k


def build_from_params(params: dict, features: list[str]):
    """Reconstruct a pipeline from an Optuna ``best_params`` dict."""
    p = dict(params)
    balance = p.pop('balance')
    smote_k = p.pop('smote_k_neighbors', 5)
    estimator = XGBClassifier(
        **p, tree_method='hist', eval_metric='aucpr',
        random_state=cfg.RANDOM_STATE, n_jobs=-1,
    )
    return prep.build_pipeline(
        estimator, features,
        balance='smote' if balance == 'smote' else 'none',
        smote_k=smote_k,
    )


def tune(
    search_frame: pl.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
    n_trials: int | None = None,
    verbose: bool = True,
) -> dict[str, optuna.Study]:
    """Run one independent study per feature set.

    Separate studies matter: hyperparameters selected on the 11-column forensic
    matrix do not transfer cleanly to the 7-column real-time matrix. Optimal
    depth and reweighting both shift when four highly separable columns are
    removed, so tuning once and reusing the result would quietly handicap
    whichever set was not tuned.
    """
    feature_sets = cfg.FEATURE_SETS if feature_sets is None else feature_sets
    n_trials = cfg.N_TRIALS if n_trials is None else n_trials

    studies: dict[str, optuna.Study] = {}

    for label, features in feature_sets.items():
        X, y = prep.to_xy(search_frame, features)
        pos_weight = positive_weight(y)

        def objective(trial: optuna.Trial, _X=X, _y=y, _f=features, _w=pos_weight) -> float:
            params, balance, smote_k = _suggest_params(trial, _w)
            estimator = XGBClassifier(
                **params, tree_method='hist', eval_metric='aucpr',
                random_state=cfg.RANDOM_STATE, n_jobs=-1,
            )
            pipeline = prep.build_pipeline(
                estimator, _f,
                balance='smote' if balance == 'smote' else 'none',
                smote_k=smote_k,
            )
            mean_ap, _ = pr_auc_cv(pipeline, _X, _y)
            return mean_ap

        def log_trial(study, trial, _v=verbose):
            if _v and (trial.number % 5 == 0 or trial.value == study.best_value):
                print(f"  trial {trial.number:>3} | PR-AUC {trial.value:.4f} "
                      f"| best {study.best_value:.4f}")

        if verbose:
            print(f"\n--- tuning: {label} ({len(features)} features) ---")

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=cfg.RANDOM_STATE),
            study_name=f'xgb_{label}',
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                       callbacks=[log_trial])
        studies[label] = study

        if verbose:
            print(f"  best CV PR-AUC: {study.best_value:.4f} "
                  f"| balance: {study.best_params['balance']}")

    return studies


def study_duration_minutes(study: optuna.Study) -> float:
    """Wall-clock minutes spent across all completed trials."""
    return sum(t.duration.total_seconds() for t in study.trials if t.duration) / 60


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    """Test-set outcome for one feature set at one decision threshold."""

    label: str
    features: list[str]
    threshold: float
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_test: int
    pipeline: object = field(repr=False, default=None)

    @property
    def flag_rate(self) -> float:
        """Share of test transactions the model would send for review."""
        return (self.tp + self.fp) / self.n_test

    @property
    def uplift_vs_incumbent(self) -> float:
        """Recall as a multiple of the incumbent rule's ceiling."""
        return self.recall / cfg.INCUMBENT_RECALL_CEILING

    def as_row(self) -> dict:
        return {
            'threshold': self.threshold, 'PR-AUC': self.pr_auc, 'ROC-AUC': self.roc_auc,
            'precision': self.precision, 'recall': self.recall, 'flag rate': self.flag_rate,
            'TP': self.tp, 'FP': self.fp, 'FN': self.fn, 'TN': self.tn,
            'test rows': self.n_test,
        }


def select_threshold(y_true, probabilities) -> float:
    """Threshold maximising F1 on the supplied set.

    F1 weights a false positive and a missed fraud equally. They are not equal —
    a blocked legitimate transaction costs customer trust, a missed fraud costs
    the transaction value — so this is a defensible default and not a final
    answer. Pass an explicit cost ratio to ``threshold_for_cost_ratio`` once the
    business supplies one.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def threshold_for_cost_ratio(y_true, probabilities, fn_over_fp: float) -> float:
    """Threshold minimising expected cost when a missed fraud costs ``fn_over_fp``
    times a false positive."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    positives = int(np.asarray(y_true).sum())

    best_threshold, best_cost = 0.5, float('inf')
    for i, threshold in enumerate(thresholds):
        tp = recall[i] * positives
        fp = (tp / precision[i] - tp) if precision[i] > 0 else 0.0
        cost = fp + fn_over_fp * (positives - tp)
        if cost < best_cost:
            best_threshold, best_cost = float(threshold), cost
    return best_threshold


def evaluate(
    splits: prep.Splits,
    features: list[str],
    params: dict,
    label: str,
    verbose: bool = True,
) -> EvalResult:
    """Fit on TRAIN, pick the threshold on VAL, report once on TEST."""
    X_train, y_train = prep.to_xy(splits.train, features)
    X_val, y_val = prep.to_xy(splits.val, features)
    X_test, y_test = prep.to_xy(splits.test, features)

    pipeline = build_from_params(params, features)
    pipeline.fit(X_train, y_train)

    threshold = select_threshold(y_val, pipeline.predict_proba(X_val)[:, 1])

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    result = EvalResult(
        label=label, features=features, threshold=threshold,
        pr_auc=float(average_precision_score(y_test, probabilities)),
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        precision=float(tp / (tp + fp)) if tp + fp else 0.0,
        recall=float(tp / (tp + fn)) if tp + fn else 0.0,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        n_test=int(len(y_test)), pipeline=pipeline,
    )

    if verbose:
        print(f"\n{'=' * 64}\n{label}  |  threshold = {threshold:.4f}\n{'=' * 64}")
        print(f"PR-AUC {result.pr_auc:.4f} | ROC-AUC {result.roc_auc:.4f}")
        print(classification_report(y_test, predictions, digits=4,
                                    target_names=['Legit', 'Fraud'], zero_division=0))
        print(f"TP {tp:,} | FP {fp:,} | FN {fn:,} | TN {tn:,}")

    return result


def evaluate_feature_sets(
    splits: prep.Splits,
    studies: dict[str, optuna.Study],
    feature_sets: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> dict[str, EvalResult]:
    """Evaluate each feature set with its own tuned hyperparameters."""
    feature_sets = cfg.FEATURE_SETS if feature_sets is None else feature_sets
    return {
        label: evaluate(splits, features, studies[label].best_params, label, verbose=verbose)
        for label, features in feature_sets.items()
    }


def summarize(results: dict[str, EvalResult]) -> pd.DataFrame:
    """Side-by-side comparison table of evaluated feature sets."""
    return pd.DataFrame({label: r.as_row() for label, r in results.items()})


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
def gain_importance(pipeline) -> pd.Series:
    """XGBoost gain per encoded feature, descending.

    Gain counts how often and how usefully a feature is chosen for splits. It is
    biased toward high-cardinality features, so read it alongside
    ``permutation_table``, which measures the metric directly.
    """
    gains = pipeline.named_steps['model'].feature_importances_
    names = prep.feature_names(pipeline)
    return pd.Series(gains, index=names).sort_values(ascending=False)


def permutation_table(
    pipeline,
    frame: pl.DataFrame,
    features: list[str],
    n_repeats: int = 5,
    sample: int | None = None,
) -> pd.DataFrame:
    """PR-AUC degradation when each feature is shuffled.

    Runs on a subsample of the supplied frame — usually validation, never test —
    because permutation refits nothing but rescores once per feature per repeat.
    """
    sample = cfg.PERM_SAMPLE if sample is None else sample
    subset = frame.sample(n=min(sample, frame.height), seed=cfg.RANDOM_STATE)
    X, y = prep.to_xy(subset, features)

    result = permutation_importance(
        pipeline, X, y, scoring=SCORING,
        n_repeats=n_repeats, random_state=cfg.RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame({'mean PR-AUC drop': result.importances_mean,
                      'std': result.importances_std}, index=features)
        .sort_values('mean PR-AUC drop', ascending=False)
    )
