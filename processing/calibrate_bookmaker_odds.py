#!/usr/bin/env python3
"""Calibrate EPL opening-odds consensus on chronological holdouts."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from train_bookmaker_model import OUTCOMES, SIDES, load_archive, make_features, prepare_matches


def fit_adjustment(log_probs, targets, method):
    def objective(value):
        scores = adjust(log_probs, method, value)
        return -scores[np.arange(len(targets)), targets].mean()

    bounds = (0.0, 10.0) if method == "confidence" else (-5.0, 5.0)
    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success:
        raise RuntimeError(f"Could not fit the {method} adjustment: {result.message}")
    return float(result.x)


def adjust(log_probs, method, value):
    if method == "confidence":
        scores = value * log_probs
    else:
        scores = log_probs.copy()
        scores[:, 1] += value
    return log_softmax(scores, axis=1)


def evaluate(matches, consensus, holdout_fraction, windows, first_test_date, include_draw):
    dates = matches["match_date"].drop_duplicates().to_numpy()
    if first_test_date is None:
        start = min(max(int(len(dates) * (1 - holdout_fraction)), 1), len(dates) - 1)
        test_dates = dates[start:]
    else:
        test_dates = dates[dates >= np.datetime64(first_test_date)]
    if len(dates) < 2 or len(test_dates) < windows:
        raise ValueError("Need at least one distinct test date per window and earlier training dates.")

    targets = matches["outcome"].map(dict(zip(OUTCOMES, range(len(OUTCOMES))))).to_numpy()
    log_probs = np.log(consensus)
    frames = []
    fits = []
    methods = ["confidence", "draw_adjusted"] if include_draw else ["confidence"]
    for fold, window in enumerate(np.array_split(test_dates, windows), start=1):
        training = matches["match_date"].lt(window[0]).to_numpy()
        testing = matches["match_date"].between(window[0], window[-1]).to_numpy()
        if not training.any():
            raise ValueError("The first test window has no earlier training matches.")
        if set(targets[training]) != {0, 1, 2}:
            raise ValueError("Each training window must include home wins, draws, and away wins.")
        predictions = {"raw": log_probs[testing]}
        for method in methods:
            value = fit_adjustment(log_probs[training], targets[training], method)
            predictions[method] = adjust(log_probs[testing], method, value)
            fits.append({
                "window": fold, "method": method, "value": value,
                "train_matches": int(training.sum()), "test_matches": int(testing.sum()),
                "first_test_date": pd.Timestamp(window[0]).date().isoformat(),
                "last_test_date": pd.Timestamp(window[-1]).date().isoformat(),
            })
        actual = targets[testing]
        observed = np.eye(len(OUTCOMES))[actual]
        for method, log_prediction in predictions.items():
            probabilities = np.exp(log_prediction)
            frame = matches.loc[testing, ["match_date", "outcome"]].reset_index()
            frame["window"] = fold
            frame["method"] = method
            for i, side in enumerate(SIDES):
                frame[f"{side}_probability"] = probabilities[:, i]
                frame[f"{side}_fair_odds"] = 1 / probabilities[:, i]
            frame["log_loss"] = -log_prediction[np.arange(len(actual)), actual]
            frame["brier"] = np.square(probabilities - observed).sum(axis=1)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(fits)


def reliability_table(predictions, bins):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for method, frame in predictions.groupby("method", sort=False):
        for side, outcome in zip(SIDES, OUTCOMES):
            probs = frame[f"{side}_probability"]
            indices = np.minimum(np.searchsorted(edges, probs, side="right") - 1, bins - 1)
            won = frame["outcome"].eq(outcome)
            for i in range(bins):
                group = indices == i
                rows.append({
                    "method": method, "outcome": outcome, "bin_low": edges[i],
                    "bin_high": edges[i + 1], "matches": int(group.sum()),
                    "mean_probability": probs[group].mean(), "observed_frequency": won[group].mean(),
                })
    return pd.DataFrame(rows)


def save_plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for axis, outcome in zip(axes, OUTCOMES):
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        subset = table.loc[table["outcome"].eq(outcome) & table["matches"].gt(0)]
        for method, data in subset.groupby("method", sort=False):
            axis.plot(data["mean_probability"], data["observed_frequency"], "o-", label=method)
            if method == "raw":
                for row in data.itertuples():
                    axis.annotate(str(row.matches), (row.mean_probability, row.observed_frequency),
                                  xytext=(4, -10), textcoords="offset points", fontsize=8)
        axis.set(xlim=(0, 1), ylim=(0, 1), title=outcome.replace("_", " ").title(),
                 xlabel="Mean predicted probability")
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out bookmaker calibration · labels show raw bin counts")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def score_table(predictions):
    metrics = ["brier", "log_loss"]
    raw = predictions.loc[predictions["method"].eq("raw"), ["window", "season", "match_id", *metrics]]
    paired = predictions.merge(raw, on=["window", "season", "match_id"], suffixes=("", "_raw"),
                               validate="many_to_one")
    for metric in metrics:
        paired[f"{metric}_delta"] = paired[metric] - paired[f"{metric}_raw"]
    return paired.groupby("method", sort=False).agg(
        matches=("match_id", "size"), brier=("brier", "mean"), log_loss=("log_loss", "mean"),
        brier_delta=("brier_delta", "mean"), log_loss_delta=("log_loss_delta", "mean"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=Path("premier_league_opening_odds"))
    parser.add_argument("--output-dir", type=Path, default=Path("bookmaker_calibration"))
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--test-fraction", type=float, default=0.2, help="fraction of distinct dates held out")
    split.add_argument("--first-test-date", type=pd.Timestamp, help="first eligible test date, YYYY-MM-DD")
    parser.add_argument("--windows", type=int, default=1, help="number of expanding chronological test windows")
    parser.add_argument("--bins", type=int, default=10, help="number of equal-width reliability bins")
    parser.add_argument("--include-draw-adjustment", action="store_true",
                        help="also fit a separate one-parameter draw adjustment")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or args.windows < 1 or args.bins < 2:
        parser.error("require 0 < test fraction < 1, windows >= 1, and bins >= 2")
    try:
        matches, quotes = prepare_matches(load_archive(args.archive))
        consensus, _, _ = make_features(matches, quotes, [])
        predictions, fits = evaluate(matches, consensus, args.test_fraction, args.windows,
                                     args.first_test_date, args.include_draw_adjustment)
        scores = score_table(predictions)
        by_window = pd.concat({window: score_table(group) for window, group in
                               predictions.groupby("window", sort=True)}, names=["window"])
        reliability = reliability_table(predictions, args.bins)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame, index in (
            ("predictions", predictions, False), ("fits", fits, False), ("scores", scores, True),
            ("scores_by_window", by_window, True), ("reliability", reliability, False),
        ):
            frame.to_csv(args.output_dir / f"{name}.csv", index=index)
        save_plot(reliability, args.output_dir / "reliability.png")
        print(fits.to_string(index=False))
        print(scores.to_string(float_format=lambda value: f"{value:.5f}"))
        print(f"Wrote diagnostics to {args.output_dir}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
