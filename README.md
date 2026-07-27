# Fraud Detection on Transactional Data

Detecting fraudulent mobile-money transactions in the PaySim dataset — and measuring how
much of the accuracy usually reported on this dataset comes from data leakage rather than
from signal.

---

## The problem

A financial institution needs to flag fraudulent transactions **at authorisation time**,
catching as much fraud as possible while keeping false positives low enough that
legitimate customers are not blocked.

The incumbent control is a single rule: flag transfers above 200,000. It fires on
`0.000003` of transactions against a true fraud rate of `0.001291`, so even if every one
of its flags were correct it could not catch more than **0.23%** of fraud. That is the bar
to beat, and it is the number the business case should be framed against — not zero.

## Why this repository is not another PaySim notebook

Most published work on this dataset reports ROC-AUC above 0.99. This project reproduces
that figure, then shows it is largely an artefact of three modeling errors, and reports
what remains once they are removed.

**1. Resampling fitted before cross-validation.** SMOTE interpolates between a real
record and its nearest neighbours. Applied to a dataset before the folds are drawn, a
synthetic point lands in the validation fold while the record it came from stays in
training — the model is asked to recognise a linear combination of rows it has memorised.
Here, resampling is a step inside an `imblearn` pipeline, re-fitted on every fold.

**2. Random splits on sequential data.** `step` maps 743 consecutive hours. A random
split puts the future in training. It also separates PaySim's paired fraud rows — a
TRANSFER out of a compromised account followed by a CASH_OUT of the same amount — across
partitions, leaving the model to match an amount it has already seen. `stratify` does not
prevent this: it balances the label, not the entity. Here the split is chronological.

**3. Post-settlement features in a real-time problem.** `newbalanceOrig`,
`newbalanceDest` and the balance-error columns derived from them do not exist when the
authorisation decision is made. In PaySim a fraudulent agent drains the account, so
`amount` equals `oldbalanceOrg` and `newbalanceOrig` collapses to zero — meaning
`errorBalanceOrig` resolves to exactly 0.00 for a large share of frauds. It is not a
predictor, it is the label wearing a disguise.

The project therefore trains **two models**: a *real-time* model restricted to fields
available at authorisation, and a *forensic* model that keeps the post-settlement columns
and is valid only for a post-settlement review queue. The gap between them is reported as
the measured cost of the leakage.

## Data

| | |
|---|---|
| Source | [PaySim synthetic mobile money dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| Rows | 6,362,620 |
| Fraud rate | 0.1291% (8,213 cases) |
| Period | 743 simulated hours (~31 days) |
| Fraud channels | `TRANSFER` and `CASH_OUT` only — `PAYMENT`, `CASH_IN` and `DEBIT` contain zero recorded fraud |

The CSV is not versioned. Place it at the path configured in
`fraud_detection/config.py`.

## Method

**Partitioning.** Chronological, by `step`, at the 0.80 and 0.80 quantiles weighted by
row count. Because volume is front-loaded — half of all transactions fall in the first ten
days — the cut lands near day 13, not day 25.

| Partition | Steps | Days | Rows | Frauds | Rate |
|---|---|---|---|---|---|
| TRAIN | 1–301 | 1–13 | 4,104,531 | 3,405 | 0.0830% |
| VAL | 302–355 | 13–15 | 1,009,353 | 558 | 0.0553% |
| TEST | 356–743 | 15–31 | 1,248,736 | 4,250 | 0.3403% |

The test period is **4.1× more hostile** than the training period, and holds 51.7% of all
fraud in 19.6% of the rows. This is a property of the data, not a sampling defect: the
model is fitted on the calm part of the month and evaluated on the busy one. Prevalence is
reported per partition because PR-AUC is a function of base rate.

**Metric.** `average_precision` — area under the Precision-Recall curve. At a 0.13% base
rate, predicting "not fraud" for everything scores 99.87% accuracy, and ROC-AUC stays
flattering because the false-positive rate barely moves when negatives outnumber positives
by three orders of magnitude.

**Validation.** `TimeSeriesSplit` forward chaining, so every validation fold is strictly
later than its training folds. Screening and tuning run on a stratified subsample of
TRAIN that preserves both prevalence and ordering.

**Screening.** Four candidates under identical CV, on the forensic feature set:

| Model | PR-AUC (CV) | Std | Fit time |
|---|---|---|---|
| Logistic Regression | 0.6404 | 0.0434 | 8.1s |
| Decision Tree | 0.6697 | 0.2649 | 5.9s |
| Random Forest | **0.9884** | 0.0102 | 79.7s |
| XGBoost | 0.9693 | 0.0233 | **6.6s** |

Random Forest's lead sits inside XGBoost's own standard deviation, so the two are a
statistical tie — and XGBoost fits 12× faster, which matters once the number is multiplied
by trials times folds. The Decision Tree's 0.2649 standard deviation is the more
interesting result: a single tree memorises whichever fraud burst falls in its training
window, a failure mode a random split would have hidden entirely.

**Tuning.** Optuna TPE, one independent study per feature set, because hyperparameters
selected on the 11-column forensic matrix do not transfer cleanly to the 7-column
real-time one. The class-balancing strategy — SMOTE versus `scale_pos_weight` — is itself
a tuned parameter rather than an assumption.

**Threshold.** Selected on the validation partition, never on test, and serialised
alongside the model. A model saved without its operating point is not usable: whoever
loads it will call `predict` at 0.5 and get a recall unrelated to the one reported.

## Results

> Populate this section from the output of `02_model.ipynb` after a full
> `Restart & Run All`. The numbers are generated by `reporting.evaluation_insights`, so
> they can be copied directly rather than transcribed.

| | Real-time (deployable) | Forensic (post-settlement) |
|---|---|---|
| PR-AUC | _TBD_ | _TBD_ |
| Recall | _TBD_ | _TBD_ |
| Precision | _TBD_ | _TBD_ |
| Flag rate | _TBD_ | _TBD_ |
| Uplift vs. incumbent | _TBD_ | — |

**Cost of leakage:** _TBD_ PR-AUC — the difference between the two columns, attributable
entirely to four columns that do not exist at decision time.

## Repository structure

```
.
├── 01_eda.ipynb                 phases 1–4: business framing, data understanding, EDA
├── 02_model.ipynb               phases 5–9: preprocessing, tuning, evaluation, deploy
├── fraud_detection/
│   ├── config.py                paths, seed, split geometry, feature contracts
│   ├── preprocessing.py         loading, row-wise engineering, splits, pipelines
│   ├── model.py                 CV, screening, Optuna, evaluation, importances
│   ├── reporting.py             modeling-stage plots and generated narrative
│   ├── deployment.py            artifact persistence and the scoring entry point
│   ├── visualizer.py            EDA panels
│   └── statistical_test.py      hypothesis tests
├── models/                      serialised artifacts (gitignored)
└── data/                        raw CSV (gitignored)
```

The notebooks contain **no function definitions**. Every reusable behaviour lives in the
package, so the notebooks read as narrative and the logic is testable, diffable and
importable outside Jupyter.

Insight prose in `02_model.ipynb` is generated from live objects via
`reporting.show(...)` rather than typed by hand, and renders as markdown in the cell
output. Change a search bound and the narrative changes with it — hardcoded figures in a
notebook silently become false the moment a parameter moves.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# place Fraud.csv where config.DATA_PATH points, then:
jupyter lab
```

Run `01_eda.ipynb` end to end, then `02_model.ipynb`. Budget roughly 20 minutes for the
modeling notebook; the two Optuna studies account for most of it. Both notebooks are
committed without outputs — `Restart & Run All` is the intended entry point.

Core dependencies: `polars`, `pandas`, `scikit-learn`, `imbalanced-learn`, `xgboost`,
`optuna`, `matplotlib`, `seaborn`, `plotly`, `joblib`, `ipywidgets`. Python 3.10.

## Limitations

- **Synthetic data.** PaySim's fraud agents follow programmed rules, learnable in a way
  real adversaries are not. Every figure here is an upper bound on real-world performance.
- **No adversarial drift.** Validated on one month of a static simulation. Real fraud
  patterns shift in response to detection, so production performance decays without
  retraining.
- **Single-transaction scope.** Each transaction is scored in isolation. The model cannot
  see that an account received three transfers in the previous ten minutes, which is how
  fraud rings actually appear. Per-account velocity features over a trailing `step` window
  are the highest-value extension available and the reason the hyperparameter search
  plateaus early — the constraint is the feature set, not the model.
- **Symmetric cost assumption.** The default threshold maximises F1, which prices a
  blocked legitimate customer and a missed fraud identically. They are not identical.
  `model.threshold_for_cost_ratio` accepts an explicit ratio once the business supplies
  one.

## Next steps

1. Per-account velocity aggregates over a trailing `step` window, computed strictly from
   past rows to preserve causal ordering.
2. Re-derive the operating point against explicit false-positive and false-negative costs.
3. Monitoring baseline on prediction distribution and flag rate, so drift is caught before
   recall degrades silently.
4. Validation against real transactional data before any production consideration.