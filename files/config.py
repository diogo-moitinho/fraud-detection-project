"""Project-wide constants.

Everything tunable lives here so the notebooks contain narrative and calls only.
Import as ``from fraud_detection import config as cfg``.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Paths and reproducibility
# --------------------------------------------------------------------------- #
DATA_PATH = '/data/PROJETOS/ALURA-FRAUDE/data/Fraud.csv'
MODEL_DIR = 'models'
ARTIFACT_NAME = 'fraud_realtime_v1.joblib'

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Split geometry
# --------------------------------------------------------------------------- #
# Quantiles of `step` weighted by row count, not by calendar time. Volume is
# front-loaded (half of all records fall in the first ten days), so 0.80 lands
# near day 13 rather than day 25.
TEST_QUANTILE = 0.80   # last 20% of TRAIN+VAL+TEST -> test
VAL_QUANTILE = 0.80    # last 20% of TRAIN+VAL      -> validation

# --------------------------------------------------------------------------- #
# Search budget
# --------------------------------------------------------------------------- #
SEARCH_FRACTION = 0.15   # stratified subsample of TRAIN used for screening/tuning
N_TRIALS = 25            # Optuna trials per feature set
CV_SPLITS = 3            # forward-chaining folds
PERM_SAMPLE = 200_000    # rows used for permutation importance

# --------------------------------------------------------------------------- #
# Feature contracts
# --------------------------------------------------------------------------- #
TARGET = 'isFraud'
CATEGORICAL = ['type']

# Available at authorisation time. This is the deployable contract.
REALTIME_FEATURES = [
    'type',
    'amount',
    'oldbalanceOrg',
    'oldbalanceDest',
    'is_merchant_dest',
    'hour_of_day',
    'day_of_week',
]

# Adds post-settlement balances. Valid for forensic review, never for real-time
# scoring: these columns do not exist when the authorisation decision is made,
# and in PaySim they encode the label almost tautologically.
POST_SETTLEMENT_FEATURES = [
    'newbalanceOrig',
    'newbalanceDest',
    'errorBalanceOrig',
    'errorBalanceDest',
]

FORENSIC_FEATURES = REALTIME_FEATURES + POST_SETTLEMENT_FEATURES

FEATURE_SETS = {
    'forensic': FORENSIC_FEATURES,
    'realtime': REALTIME_FEATURES,
}

# Exploration-only. Both are monotonic indices whose test-set values were never
# observed in training, so a model keyed on them extrapolates rather than
# generalises. hour_of_day and day_of_week cycle and are safe.
EXPLORATION_ONLY = ['day_of_month', 'month_period', 'step']

# --------------------------------------------------------------------------- #
# Reference figures from the incumbent rules engine (measured in 01_eda)
# --------------------------------------------------------------------------- #
INCUMBENT_FLAG_RATE = 0.000003   # share of transactions flagged by isFlaggedFraud
DATASET_FRAUD_RATE = 0.001291    # share of transactions that are actually fraud

# Upper bound on the incumbent's recall: even if every flag were correct, it
# could not exceed this share of frauds.
INCUMBENT_RECALL_CEILING = INCUMBENT_FLAG_RATE / DATASET_FRAUD_RATE

# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
COLOR_PRIMARY = '#00334e'
COLOR_ACCENT = '#ff5500'
FRAUD_PALETTE = {0: '#2ecc71', 1: '#e74c3c'}
