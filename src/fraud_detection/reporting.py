"""Model-stage plots and generated insight prose.

Import as ``from fraud_detection import reporting as rpt``.

The ``*_insights`` functions return markdown built from live objects rather than
hardcoded figures. Every number in a notebook narrative is otherwise a hostage:
change ``N_TRIALS`` or a search bound and the prose silently becomes false.
Generating it means the text cannot drift from the run that produced it.

Companion to the existing ``visualizer`` module, which covers the EDA panels;
this one covers the modeling stage.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from . import config as cfg
from . import model as mdl


def show(markdown: str) -> None:
    """Render generated prose as real markdown in the notebook output.

    Avoids the copy-paste step: the narrative renders where it is computed, so it
    cannot fall out of sync with the run. Falls back to ``print`` outside IPython.
    """
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(markdown)
        return
    display(Markdown(markdown))


def _despine(ax, sides=('top', 'right')) -> None:
    ax.spines[list(sides)].set_visible(False)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_screening(df_results: pd.DataFrame, ax=None):
    """Horizontal PR-AUC comparison with fold-variance error bars.

    The error bars are not decoration. At this prevalence the spread across folds
    is often wider than the gap between models, and overlapping intervals mean a
    tie rather than a ranking.
    """
    data = df_results.sort_values('PR-AUC (CV)')

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.barh(data.index, data['PR-AUC (CV)'],
            xerr=data['Std'], color=cfg.COLOR_PRIMARY,
            error_kw={'ecolor': cfg.COLOR_ACCENT, 'capsize': 4})
    ax.set_title('Model screening — forward-chaining CV', fontsize=15, pad=15)
    ax.set_xlabel('PR-AUC (average precision)', fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)
    _despine(ax, ('top', 'right', 'left'))

    for y, (value, std) in enumerate(zip(data['PR-AUC (CV)'], data['Std'])):
        ax.text(value + std + 0.015, y, f'{value:.3f}', va='center',
                fontsize=10, color='#333333')

    plt.tight_layout()
    return ax


def plot_tuning(studies: dict[str, optuna.Study], print_importance: bool = True):
    """Convergence trace and hyperparameter importance, one row per study.

    Importances are also printed so the narrative can cite verified values
    instead of numbers read off a chart by eye.
    """
    n = len(studies)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n), squeeze=False)

    for row, (label, study) in enumerate(studies.items()):
        history = pd.DataFrame(
            [{'trial': t.number, 'pr_auc': t.value}
             for t in study.trials if t.value is not None]
        )
        history['best_so_far'] = history['pr_auc'].cummax()

        ax = axes[row, 0]
        ax.scatter(history['trial'], history['pr_auc'], s=28,
                   color=cfg.COLOR_PRIMARY, alpha=0.7, label='Trial')
        ax.plot(history['trial'], history['best_so_far'],
                color=cfg.COLOR_ACCENT, lw=2, label='Best so far')
        ax.set(title=f'Convergence — {label}', xlabel='Trial', ylabel='CV PR-AUC')
        ax.legend(frameon=False)

        ax = axes[row, 1]
        try:
            importance = optuna.importance.get_param_importances(study)
            ax.barh(list(importance)[::-1], list(importance.values())[::-1],
                    color=cfg.COLOR_PRIMARY)
            ax.set(title=f'Hyperparameter importance — {label}',
                   xlabel='Relative importance')
            if print_importance:
                print(f"\n{label} — parameter importance")
                for key, value in importance.items():
                    print(f"  {key:<20} {value:.4f}")
        except RuntimeError:
            # Every trial returned the same score. Usually means the objective is
            # saturated, which is worth noticing rather than hiding.
            ax.text(0.5, 0.5, 'Zero variance across trials', ha='center', va='center')
            ax.set_axis_off()

    for ax in axes.ravel():
        _despine(ax)

    plt.tight_layout()
    return fig


def plot_importance(importance: pd.Series, title: str = 'Feature importance (gain)', ax=None):
    """Horizontal gain chart for one fitted pipeline."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    data = importance.sort_values()
    ax.barh(data.index, data.values, color=cfg.COLOR_PRIMARY)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel('Relative gain')
    ax.grid(axis='x', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)
    _despine(ax, ('top', 'right', 'left'))

    plt.tight_layout()
    return ax


# --------------------------------------------------------------------------- #
# Generated narrative
# --------------------------------------------------------------------------- #
def split_insights(summary: pd.DataFrame) -> str:
    """Markdown describing the chronological partition and prevalence drift."""
    train, val, test = summary.loc['TRAIN'], summary.loc['VAL'], summary.loc['TEST']
    total_frauds = int(summary['frauds'].sum())
    total_rows = int(summary['rows'].sum())
    drift = test['fraud_rate'] / train['fraud_rate'] if train['fraud_rate'] else float('nan')

    return f"""##### 1. Why the Split Is Chronological
* `step` maps {int(summary['step_to'].max())} consecutive hours of simulation, so the dataset is a time
  series wearing a tabular disguise. A random split would place future
  transactions in training and past transactions in testing.
* PaySim implements each fraud as two linked rows — a TRANSFER out of the
  compromised account, then a CASH_OUT of the same amount. A random split
  separates those twins across partitions, leaving the model only to match an
  amount it has already seen. `stratify` does not prevent this: it balances the
  label, not the entity.
* **Modeling Decision:** Train on steps {int(train['step_from'])}–{int(train['step_to'])} (days {int(train['day_from'])} to {int(train['day_to'])}), select the
  threshold on steps {int(val['step_from'])}–{int(val['step_to'])} (days {int(val['day_from'])} to {int(val['day_to'])}), and test on steps {int(test['step_from'])}–{int(test['step_to'])}
  (days {int(test['day_from'])} to {int(test['day_to'])}). The test partition is read exactly once, after every
  modeling decision is locked.

##### 2. Prevalence Drifts Sharply Between Partitions
* Volume is front-loaded, so the {cfg.TEST_QUANTILE:.0%} quantile of `step` weighted by row count
  lands at step {int(train['step_to'])}, day {int(train['day_to'])} — not near day 25 as a naive reading of
  {cfg.TEST_QUANTILE} × {int(summary['step_to'].max())} would suggest.
* Fraud prevalence is **{train['fraud_rate']:.4%}** in TRAIN, **{val['fraud_rate']:.4%}** in VAL and **{test['fraud_rate']:.4%}** in
  TEST. The test period is **{drift:.1f}x more hostile** than the period the model learns from.
* TEST holds {int(test['frauds']):,} of the {total_frauds:,} frauds in the dataset — **{test['frauds'] / total_frauds:.1%} of all fraud
  in {test['rows'] / total_rows:.1%} of the rows**. The chronological cut did not merely reorder the
  data, it concentrated the positive class in the held-out period.
* **Modeling Decision:** Report prevalence per partition alongside every metric.
  PR-AUC is a function of base rate, so a test score is not comparable to a
  validation score without both. A drop from validation to test is expected here
  and is not by itself evidence of overfitting."""


def screening_insights(df_results: pd.DataFrame, search_frame, n_folds: int | None = None) -> str:
    """Markdown interpreting the screening table, including ties and runtime."""
    n_folds = cfg.CV_SPLITS if n_folds is None else n_folds
    ranked = df_results.sort_values('PR-AUC (CV)', ascending=False)
    best, second = ranked.index[0], ranked.index[1]
    worst = ranked.index[-1]

    rows = search_frame.height
    frauds = int(search_frame[cfg.TARGET].sum())
    blocks = n_folds + 1
    gap = ranked.loc[best, 'PR-AUC (CV)'] - ranked.loc[second, 'PR-AUC (CV)']
    tied = gap < ranked.loc[second, 'Std']
    speedup = ranked.loc[best, 'Fit time (s)'] / ranked.loc[second, 'Fit time (s)']

    verdict = (
        f"* **The Top Two Are a Statistical Tie:** {best} leads at "
        f"**{ranked.loc[best, 'PR-AUC (CV)']:.4f} ± {ranked.loc[best, 'Std']:.4f}** against {second} at "
        f"**{ranked.loc[second, 'PR-AUC (CV)']:.4f} ± {ranked.loc[second, 'Std']:.4f}**. The {gap:.4f} gap sits inside "
        f"{second}'s own standard deviation, so this is not a ranking."
        if tied else
        f"* **A Clear Leader:** {best} reaches **{ranked.loc[best, 'PR-AUC (CV)']:.4f} ± {ranked.loc[best, 'Std']:.4f}**, "
        f"separating from {second} (**{ranked.loc[second, 'PR-AUC (CV)']:.4f}**) by more than the fold variance."
    )

    faster = second if speedup > 1 else best
    slower = best if speedup > 1 else second
    ratio = max(speedup, 1 / speedup)

    return f"""* **A Leak-Free Baseline Produces Spread:** PR-AUC ranges from **{ranked.loc[worst, 'PR-AUC (CV)']:.4f}**
  ({worst}) to **{ranked.loc[best, 'PR-AUC (CV)']:.4f}** ({best}). The spread is itself evidence that the
  evaluation works — near-identical perfect scores across algorithms of very
  different capacity is the signature of a leaked target, not of an easy problem.
{verdict}
* **Fold Variance Is Load-Bearing:** the search sample holds {frauds:,} frauds across
  {rows:,} rows, split into {blocks} chronological blocks of roughly {rows // blocks:,} rows —
  about {frauds // blocks} frauds each. No ranking at that resolution should be treated as
  decisive, and models with a standard deviation above 0.10 are memorising
  whichever fraud burst happens to sit in their training window.
* **Modeling Decision:** Carry **{faster}** into the tuning stage. It fits in
  **{ranked.loc[faster, 'Fit time (s)']:.1f}s against {ranked.loc[slower, 'Fit time (s)']:.1f}s** — {ratio:.0f}x faster — and runtime is a
  selection criterion, not a footnote: the tuning stage multiplies it by trials
  times folds."""


def tuning_insights(
    studies: dict[str, optuna.Study],
    df_results: pd.DataFrame,
    search_frame,
    screening_label: str = 'XGBoost',
    screening_feature_set: str = 'forensic',
) -> str:
    """Markdown interpreting the Optuna studies.

    Interpretation follows the numbers rather than assuming them: the reweighting
    verdict is derived from the sampled values, so it cannot contradict the
    figures printed beside it.
    """
    lines = [
        '##### 1. Class Balancing Treated as a Hyperparameter',
        '* Rather than assuming SMOTE, the search chooses between synthetic oversampling',
        '  and `scale_pos_weight`, letting the objective settle the question empirically.',
    ]

    natural = mdl.positive_weight(search_frame[cfg.TARGET].to_numpy())

    ratios: dict[str, float] = {}
    for label, study in studies.items():
        chosen = study.best_params['balance']
        line = f'* On the **{label}** feature set it converged on **`{chosen}`**'
        if chosen == 'scale_pos_weight':
            weight = study.best_params['scale_pos_weight']
            ratios[label] = weight / natural
            line += (f' at a weight of **{weight:,.1f}** — **{weight / natural:.0%}** of the '
                     f'natural imbalance ratio of {natural:,.1f}')
        else:
            line += f", with k_neighbors={study.best_params.get('smote_k_neighbors')}"
        lines.append(line + '.')

    # The verdict must describe only the sets that actually reweighted, and only
    # generalise when every set did. Otherwise the prose overstates the evidence.
    scope = 'Every set' if len(ratios) == len(studies) else 'Where reweighting was chosen, it'
    if ratios and all(r < 0.9 for r in ratios.values()):
        lines.append(f'* {scope} lands **below** the natural ratio, so full inverse-frequency '
                     'weighting overshoots on this data — a modeling finding, not a default.')
    elif ratios and all(r > 1.1 for r in ratios.values()):
        lines.append(f'* {scope} lands **above** the natural ratio, meaning the objective '
                     'rewards over-correcting the minority class here.')
    elif ratios:
        lines.append('* The feature sets disagree on how hard to reweight, which is itself '
                     'evidence that balancing strength is feature-set dependent and should '
                     'not be fixed by convention.')
    if not ratios:
        lines.append('* No set chose `scale_pos_weight`; synthetic oversampling won on every '
                     'feature set, which is worth re-checking at full data volume where '
                     "SMOTE's cost grows fastest.")

    lines += [
        f'* On a {search_frame.height:,}-row search sample, SMOTE\'s nearest-neighbour step also',
        '  dominates fit time, while gradient boosting absorbs the imbalance through the loss.',
        '* **Modeling Decision:** Record the chosen strategy explicitly. A reviewer will',
        '  otherwise assume it was left at its default.',
        '',
        '##### 2. What the Search Actually Bought',
    ]

    for label, study in studies.items():
        minutes = mdl.study_duration_minutes(study)
        last = study.best_trial.number == len(study.trials) - 1
        tail = (' — the best trial was the last one, so the search had not yet plateaued.'
                if last else '.')
        lines.append(f'* **{label}**: best CV PR-AUC **{study.best_value:.4f}**, at trial '
                     f'**{study.best_trial.number} of {len(study.trials)}**, '
                     f'{minutes:.1f} min total{tail}')

    if screening_feature_set in studies and screening_label in df_results.index:
        baseline = df_results.loc[screening_label, 'PR-AUC (CV)']
        delta = studies[screening_feature_set].best_value - baseline
        lines += [
            f'* Against the untuned {screening_label} screening baseline of **{baseline:.4f}** (same',
            f'  {screening_feature_set} feature set), tuning moved the objective by **{delta:+.4f}**.',
            '* The other feature set has no screening counterpart — the comparison in section 6',
            f'  ran on the {screening_feature_set} matrix — so its score is reported on its own',
            '  terms rather than as a delta.',
        ]

    lines += [
        '* **Modeling Decision:** An early plateau means hyperparameters are no longer the',
        '  binding constraint. Further effort belongs in feature engineering — per-account',
        '  velocity counters over a trailing `step` window, the one signal class this matrix',
        '  lacks entirely — rather than in a wider or longer search.',
    ]

    return '\n'.join(lines)


def evaluation_insights(
    results: dict[str, mdl.EvalResult],
    deployable: str = 'realtime',
    reference: str = 'forensic',
) -> str:
    """Markdown for the final test-set evaluation, including the leakage cost."""
    r = results[deployable]
    f = results.get(reference)

    leakage = ''
    if f is not None:
        leakage = f"""##### 1. The Measured Cost of Leakage
* The {reference} feature set reaches **{f.pr_auc:.4f}** PR-AUC on the held-out period; the
  {deployable} set reaches **{r.pr_auc:.4f}**. The gap of **{f.pr_auc - r.pr_auc:+.4f}** is attributable to
  {len(cfg.POST_SETTLEMENT_FEATURES)} columns that do not exist at the moment an authorisation decision must
  be made.
* **Modeling Decision:** The {deployable} figure belongs in the project summary. The
  {reference} figure describes a different product — a post-settlement review queue —
  and is labelled as such wherever it appears.

"""

    return f"""{leakage}##### 2. Precision, Recall, and the Cost Asymmetry
* At the validation-selected threshold of **{r.threshold:.4f}**, the deployable model catches
  **{r.recall:.2%}** of frauds at **{r.precision:.2%}** precision: {r.tp:,} true positives against {r.fp:,}
  false positives and {r.fn:,} missed frauds, over {r.n_test:,} test transactions.
* It sends **{r.flag_rate:.3%}** of all traffic for review. For every 1,000 flags, roughly
  {1000 * r.precision:.0f} are genuine fraud.
* Section 1 defined success as a high detection rate while strictly controlling false
  positives. Those objectives trade continuously along the PR curve, and the threshold
  is where the trade is priced.
* The threshold maximises F1 on validation, which assumes a false positive and a missed
  fraud cost the same. They do not.
* **Modeling Decision:** Re-derive the operating point with
  `model.threshold_for_cost_ratio` once the business supplies a ratio. The model does
  not change; only the cut does.

##### 3. Benchmark Against the Incumbent Rule
* The existing engine flags transfers above 200,000, firing on {cfg.INCUMBENT_FLAG_RATE:.6f} of
  transactions against a true fraud rate of {cfg.DATASET_FRAUD_RATE:.6f} — an upper bound of
  **{cfg.INCUMBENT_RECALL_CEILING:.2%}** on the share of frauds it could possibly catch.
* The deployable model detects **{r.recall:.2%}**, roughly **{r.uplift_vs_incumbent:.0f}x** that ceiling.
* **Modeling Decision:** Frame the business case against the current rule, not against
  zero. The question is whether the model beats what is already running, by a margin
  that justifies the false positives it introduces."""


def conclusion_insights(
    results: dict[str, mdl.EvalResult],
    importance: pd.Series,
    deployable: str = 'realtime',
) -> str:
    """Markdown for the interpretation section, driven by the importance table."""
    r = results[deployable]
    top = importance.sort_values(ascending=False)
    minor = int((top < 0.01).sum())

    hour = top.get('hour_of_day')
    hour_line = (
        f"* `hour_of_day` contributes {hour:.1%}, consistent with the observation that "
        "fraudulent\n  volume stays flat overnight while legitimate volume collapses. The model is "
        "not\n  learning that 4:00 AM is dangerous; it is learning that 4:00 AM is *empty*, and "
        "that\n  anything moving through an empty network deserves scrutiny."
        if hour is not None else
        "* `hour_of_day` did not survive encoding into the top features; read the "
        "permutation\n  table before concluding the hour carries no signal."
    )

    return f"""##### 1. What the Model Learned
* Gain concentrates in `{top.index[0]}` ({top.iloc[0]:.1%}) and `{top.index[1]}` ({top.iloc[1]:.1%}),
  confirming Section 4's finding that transaction type alone separates safe channels
  from risky ones — PAYMENT, CASH_IN and DEBIT contain zero recorded frauds, so the
  model spends no capacity on them.
{hour_line}
* {minor} of {len(top)} encoded features carry under 1% of gain each. Read this beside the
  permutation table: gain rewards features used often in splits, permutation rewards
  features the metric actually depends on, and where the two disagree the permutation
  result is the one tied to PR-AUC.

##### 2. Performance Against the Incumbent
* The deployable model achieves **{r.recall:.2%}** recall at **{r.precision:.2%}** precision on a test
  period it never saw, roughly **{r.uplift_vs_incumbent:.0f}x** the incumbent rule's ceiling of
  {cfg.INCUMBENT_RECALL_CEILING:.2%}.
* **Business Reading:** For every 1,000 transactions flagged, approximately
  {1000 * r.precision:.0f} are genuine frauds, and {r.fn:,} frauds still pass unflagged. Whether that
  trade is acceptable is a business decision, not a modeling one, and it moves
  continuously along the threshold."""
