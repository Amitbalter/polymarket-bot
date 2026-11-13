#!/usr/bin/env python3
"""Train an EPL outcome model from the OddsPortal opening-odds Parquet archive.

Install pandas, pyarrow and scikit-learn. Run this script with the archive folder
as its positional argument, or omit it to use premier_league_opening_odds.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTCOMES = ("home_win", "draw", "away_win")
SIDES = ("home", "draw", "away")
ODDS = [f"{side}_opening_odds" for side in SIDES]
MATCH = ["season", "match_id"]
REQUIRED = [*MATCH, "match_date", "outcome", "bookmaker", *ODDS]


def load_archive(path):
    paths = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No Parquet files found in {path}")
    return pd.concat(
        (pd.read_parquet(file, columns=REQUIRED) for file in paths),
        ignore_index=True,
    )


def prepare_matches(rows, audit_path=None):
    rows = rows.copy()
    rows["outcome"] = rows["outcome"].replace({"1": "home_win", "X": "draw", "2": "away_win"})
    dates = rows["match_date"].astype("string").str.strip()
    rows["match_date"] = pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce")
    missing_dates = rows["match_date"].isna()
    rows.loc[missing_dates, "match_date"] = pd.to_datetime(
        dates[missing_dates], format="%d %b %Y", errors="coerce"
    )
    rows["bookmaker"] = rows["bookmaker"].astype("string").str.strip().str.casefold()
    rows.loc[rows["bookmaker"].eq(""), "bookmaker"] = pd.NA
    for column in MATCH:
        rows[column] = rows[column].astype("string").str.strip().replace("", pd.NA)
    identified = rows.dropna(subset=MATCH)
    if len(identified) != len(rows):
        print(f"Input rows without a match ID or season: {len(rows) - len(identified)}")
    rows = identified

    labeled = rows.dropna(subset=["outcome"])
    invalid = set(labeled["outcome"]) - set(OUTCOMES)
    if invalid:
        raise ValueError(f"Unknown outcome labels: {sorted(invalid)}")
    metadata = labeled.dropna(subset=["match_date"])[
        [*MATCH, "match_date", "outcome"]
    ].drop_duplicates()
    if metadata.duplicated(MATCH).any():
        raise ValueError("At least one match has conflicting dates or outcomes.")

    quotes = rows.dropna(subset=["bookmaker"])[[*MATCH, "bookmaker", *ODDS]].copy()
    quotes[ODDS] = quotes[ODDS].apply(pd.to_numeric, errors="coerce").astype(float)
    prices = quotes[ODDS]
    valid = (prices.gt(1) & np.isfinite(prices)).all(axis=1)
    quotes = quotes.loc[valid].drop_duplicates()
    if quotes.duplicated([*MATCH, "bookmaker"]).any():
        raise ValueError("At least one match has conflicting opening lines from the same bookmaker.")

    implied = 1.0 / quotes[ODDS]
    probabilities = implied.div(implied.sum(axis=1), axis=0)
    quotes[ODDS] = probabilities
    quotes = quotes.set_index(MATCH).sort_index()
    matches = metadata.set_index(MATCH).sort_values("match_date")
    matches = matches.loc[matches.index.isin(quotes.index)]
    audit = rows.groupby(MATCH).size().to_frame("source_rows")
    audit["has_date"] = audit.index.isin(rows.dropna(subset=["match_date"]).set_index(MATCH).index)
    audit["has_outcome"] = audit.index.isin(labeled.set_index(MATCH).index)
    audit["complete_bookmaker_lines"] = quotes.groupby(level=MATCH).size().reindex(audit.index, fill_value=0)
    audit["eligible"] = audit.index.isin(matches.index)
    print(f"Loaded {len(audit)} distinct matches from Parquet. Eligible: {len(matches)}.")
    print(f"Missing usable date: {(~audit['has_date']).sum()}. "
          f"Missing outcome: {(~audit['has_outcome']).sum()}. "
          f"No complete bookmaker line: {audit['complete_bookmaker_lines'].eq(0).sum()}. "
          "Reasons can overlap.")
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path)
        print(f"Saved per-match eligibility to {audit_path}")
    if matches.empty:
        raise ValueError("No dated, settled matches with complete opening odds were found.")
    return matches, quotes


def make_features(matches, quotes, bookmakers):
    quotes = quotes.loc[quotes.index.isin(matches.index)]
    grouped = quotes.groupby(level=MATCH)[ODDS]
    consensus = grouped.mean().reindex(matches.index)
    spread = grouped.std(ddof=0).reindex(matches.index)
    pooled = consensus.set_axis([f"consensus_{side}" for side in SIDES], axis=1)
    pooled = pooled.join(spread.set_axis([f"disagreement_{side}" for side in SIDES], axis=1))
    pooled["bookmaker_count"] = grouped.size()
    features = [pooled]
    for index, bookmaker in enumerate(bookmakers):
        line = quotes.loc[quotes["bookmaker"].eq(bookmaker), ODDS].reindex(matches.index)
        available = line[ODDS[0]].notna().astype(int)
        line = line.fillna(consensus).set_axis(
            [f"book_{index}_{side}" for side in SIDES], axis=1
        )
        line[f"book_{index}_available"] = available
        features.append(line)
    return consensus.to_numpy(), pooled, pd.concat(features, axis=1)


def split_by_date(matches, test_fraction):
    dates = matches["match_date"].drop_duplicates()
    if len(dates) < 2:
        raise ValueError("At least two distinct match dates are required.")
    index = min(max(int(len(dates) * (1 - test_fraction)), 1), len(dates) - 1)
    cutoff = dates.iloc[index]
    training = matches["match_date"].lt(cutoff)
    if set(matches.loc[training, "outcome"]) != set(OUTCOMES):
        raise ValueError("Training data must contain home wins, draws, and away wins. Add more seasons.")
    return training, cutoff


def report(name, actual, probabilities):
    observed = (actual.to_numpy()[:, None] == np.array(OUTCOMES)).astype(int)
    brier = np.square(probabilities - observed).sum(axis=1).mean()
    loss = log_loss(observed, probabilities)
    accuracy = accuracy_score(actual, np.array(OUTCOMES)[probabilities.argmax(axis=1)])
    print(f"{name:25s} {brier:9.4f} {loss:10.4f} {accuracy:10.2%}")


def train(path, test_fraction, min_bookmaker_coverage, output, audit_path=None):
    matches, quotes = prepare_matches(load_archive(path), audit_path)
    training, cutoff = split_by_date(matches, test_fraction)
    train_index = matches.index[training]
    coverage = quotes.loc[quotes.index.isin(train_index), "bookmaker"].value_counts()
    bookmakers = sorted(coverage[coverage / len(train_index) >= min_bookmaker_coverage].index)
    consensus, pooled, individual = make_features(matches, quotes, bookmakers)
    target = matches["outcome"]
    actual = target.loc[~training]
    print(f"Train: {training.sum()} matches. Test: {(~training).sum()} matches. "
          f"First test date: {cutoff.date()}.")
    print(f"Bookmaker-specific inputs ({len(bookmakers)}): {', '.join(bookmakers) or 'none'}")
    print(f"{'Model':25s} {'Brier':>9s} {'Log loss':>10s} {'Accuracy':>10s}")
    report("Mean bookmaker consensus", actual, consensus[~training.to_numpy()])

    models = {}
    for name, features in (("Consensus + disagreement", pooled), ("Bookmaker-specific", individual)):
        if name == "Bookmaker-specific" and not bookmakers:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(features.loc[training], target.loc[training])
        order = [list(model.classes_).index(outcome) for outcome in OUTCOMES]
        report(name, actual, model.predict_proba(features.loc[~training])[:, order])
        models[name] = model

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": models, "bookmakers": bookmakers, "outcome_order": OUTCOMES,
                     "first_test_date": cutoff}, output)
        print(f"Saved fitted models to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=Path("premier_league_opening_odds"))
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="fraction of distinct match dates held out (default: 0.2)")
    parser.add_argument("--min-bookmaker-coverage", type=float, default=0.2,
                        help="minimum training-match coverage for a bookmaker (default: 0.2)")
    parser.add_argument("--model-out", type=Path, help="optional output path for the fitted models")
    parser.add_argument("--audit-out", type=Path, help="optional CSV listing why matches are ineligible")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if not 0 < args.min_bookmaker_coverage <= 1:
        parser.error("--min-bookmaker-coverage must be greater than 0 and at most 1")
    try:
        train(args.archive, args.test_fraction, args.min_bookmaker_coverage, args.model_out, args.audit_out)
    except (OSError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
