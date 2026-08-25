from __future__ import annotations

from typing import Iterable, Literal, Optional, Union

import numpy as np
import pandas as pd


def estimate_pass_at_k(
    num_samples: Union[int, np.ndarray],
    num_correct: np.ndarray,
    k: int,
) -> np.ndarray:
    """Per-task pass@k estimator matching the HumanEval formula."""
    def estimator(n: int, c: int, k_value: int) -> float:
        if n - c < k_value:
            return 1.0
        return 1.0 - np.prod(1.0 - k_value / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples = np.full(shape=len(num_correct), fill_value=num_samples, dtype=int)
    else:
        num_samples = np.asarray(num_samples, dtype=int)
    num_correct = np.asarray(num_correct, dtype=int)
    return np.array([estimator(int(n), int(c), int(k)) for n, c in zip(num_samples, num_correct)])


def summarize_task_outcomes(
    df: pd.DataFrame,
    *,
    task_col: str = "task_id",
    passed_col: str = "passed",
) -> pd.DataFrame:
    """Collapse row-level outcomes into one row per task with sample/correct counts."""
    if task_col not in df.columns:
        raise KeyError(f"Missing task column: {task_col}")
    if passed_col not in df.columns:
        raise KeyError(f"Missing passed column: {passed_col}")

    summary = (
        df.assign(**{passed_col: df[passed_col].astype(int)})
        .groupby(task_col, sort=True)[passed_col]
        .agg(num_samples="size", num_correct="sum")
        .reset_index()
    )
    return summary


def summarize_task_answer_probabilities(
    df: pd.DataFrame,
    *,
    task_col: str = "task_id",
    answer_prob_col: str = "answer_prob",
) -> pd.DataFrame:
    """Collapse row-level answer probabilities into one row per task."""
    if task_col not in df.columns:
        raise KeyError(f"Missing task column: {task_col}")
    if answer_prob_col not in df.columns:
        raise KeyError(f"Missing answer-probability column: {answer_prob_col}")

    answer_prob_df = df[[task_col, answer_prob_col]].dropna().copy()
    if answer_prob_df.empty:
        raise ValueError("No non-null answer probabilities are available.")

    summary = (
        answer_prob_df.groupby(task_col, sort=True)[answer_prob_col]
        .agg(answer_prob="first", num_unique="nunique")
        .reset_index()
    )
    if not (summary["num_unique"] <= 1).all():
        raise ValueError("Found multiple different answer probabilities for the same task.")
    return summary[[task_col, "answer_prob"]]


def pass_curve_dataframe(
    df: pd.DataFrame,
    *,
    ks: Optional[Iterable[int]] = None,
    max_k: Optional[int] = None,
    step: int = 1,
    task_col: str = "task_id",
    passed_col: str = "passed",
    method: Literal["auto", "empirical", "answer_prob"] = "auto",
    answer_prob_col: str = "answer_prob",
) -> pd.DataFrame:
    """
    Return a dataframe with `k` and mean pass@ rate.

    `method="auto"` uses answer probabilities for MMLU temperature curves and
    empirical pass@k elsewhere.
    """
    dataset = df.attrs.get("dataset")
    variant_type = df.attrs.get("variant_type")
    if method == "auto":
        use_answer_prob = variant_type == "temperature" and dataset == "mmlu"
    elif method == "answer_prob":
        use_answer_prob = True
    else:
        use_answer_prob = False

    if use_answer_prob:
        summary = summarize_task_answer_probabilities(df, task_col=task_col, answer_prob_col=answer_prob_col)
        max_available_k = int(df.attrs.get("requested_numvariants", 1))
        if dataset == "mmlu":
            max_available_k *= 5
        if max_available_k < 1:
            raise ValueError("Answer-probability pass@ curves require a positive requested_numvariants value.")

        if ks is None:
            upper = max_available_k if max_k is None else min(max_available_k, max_k)
            ks_array = np.arange(1, upper + 1, step, dtype=int)
        else:
            ks_array = np.array(sorted(int(k) for k in ks), dtype=int)
            ks_array = ks_array[(ks_array >= 1) & (ks_array <= max_available_k)]

        if ks_array.size == 0:
            raise ValueError("No valid k values remain after clipping to the requested number of variants.")

        answer_probs = summary["answer_prob"].to_numpy(dtype=float)
        pass_rates = np.array([np.mean(1.0 - np.power(1.0 - answer_probs, int(k))) for k in ks_array])

        pass_rates_std = np.array([np.sqrt(np.sum((1.0 - np.power(1.0 - answer_probs, int(k)) - pass_mean)**2)/(len(answer_probs)-1))/np.sqrt(len(answer_probs))
                                 for k,pass_mean in zip(ks_array, pass_rates)])

        return pd.DataFrame({"k": ks_array, "pass_rate": pass_rates, 'pass_rate_std': pass_rates_std})

    summary = summarize_task_outcomes(df, task_col=task_col, passed_col=passed_col)
    if summary.empty:
        raise ValueError("Cannot compute a pass@ curve from an empty dataframe.")

    min_num_samples = int(summary["num_samples"].min())
    if min_num_samples < 1:
        raise ValueError("Each task needs at least one sample.")

    if ks is None:
        upper = min_num_samples if max_k is None else min(min_num_samples, max_k)
        ks_array = np.arange(1, upper + 1, step, dtype=int)
    else:
        ks_array = np.array(sorted(int(k) for k in ks), dtype=int)
        ks_array = ks_array[(ks_array >= 1) & (ks_array <= min_num_samples)]

    if ks_array.size == 0:
        raise ValueError("No valid k values remain after clipping to the available number of samples per task.")

    num_samples = summary["num_samples"].to_numpy(dtype=int)
    num_correct = summary["num_correct"].to_numpy(dtype=int)

    pass_rates = np.array(
        [estimate_pass_at_k(num_samples=num_samples, num_correct=num_correct, k=int(k)).mean() for k in ks_array]
    )

    pass_rates_std = np.array(
        [estimate_pass_at_k(num_samples=num_samples, num_correct=num_correct, k=int(k)).std()/np.sqrt(len(num_samples)) for k in ks_array]
    )

    return pd.DataFrame(
        {
            "k": ks_array,
            "pass_rate": pass_rates,
            "pass_rate_std":pass_rates_std

        }
    )


def pass_curve_points(
    df: pd.DataFrame,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(x, y)` arrays ready for plotting pass@ curves."""
    curve_df = pass_curve_dataframe(df, **kwargs)
    return curve_df["k"].to_numpy(dtype=int), curve_df["pass_rate"].to_numpy(dtype=float), curve_df["pass_rate_std"].to_numpy(dtype=float)
