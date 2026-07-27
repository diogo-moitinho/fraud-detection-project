"""Fraud detection pipeline for the PaySim transactional dataset.

Modules
-------
config          project constants: paths, seeds, split geometry, feature contracts
preprocessing   loading, row-wise feature engineering, chronological splits, pipelines
model           cross-validation, screening, Optuna tuning, evaluation, importances
reporting       modeling-stage plots and generated insight prose
deployment      artifact persistence and the scoring entry point
visualizer      EDA panels (existing)
statistical_test  hypothesis tests (existing)
"""

__version__ = '1.0.0'
