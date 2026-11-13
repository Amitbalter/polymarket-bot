"""Fit and test a soccer contract predictor using export_soccer_sdk.py's CSV."""

import argparse
import csv
from datetime import datetime
from math import isfinite, log

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def implied_probability(price_1, price_2):
    prices = float(price_1), float(price_2)
    if not all(isfinite(p) and 0 <= p <= 1 for p in prices) or sum(prices) == 0:
        raise ValueError("Provide two prices from 0 to 1 with a positive sum")
    return prices[0] / sum(prices)


def feature(probability):
    p = min(max(probability, 1e-6), 1 - 1e-6)
    return [log(p / (1 - p))]


def load_rows(path, moneyline_only):
    with open(path, newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if moneyline_only and row.get("market_type") != "moneyline":
                continue
            labels = row["outcome_1"], row["outcome_2"]
            if not row["event_id"] or not all(labels) or labels[0] == labels[1]:
                continue
            try:
                probability = implied_probability(row["initial_price_1"], row["initial_price_2"])
                kickoff = datetime.fromisoformat(row["kickoff_at"])
                resolved = datetime.fromisoformat(row["resolved_at"]) if row.get("resolved_at") else None
                target = int(labels.index(row["winning_outcome"]) == 0)
            except ValueError:
                continue
            if kickoff.tzinfo and (resolved is None or resolved.tzinfo):
                yield row, probability, kickoff, resolved, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file")
    parser.add_argument("--moneyline-only", action="store_true")
    parser.add_argument("--prices", nargs=2, metavar=("PRICE_1", "PRICE_2"))
    parser.add_argument("--output", default="soccer_predictions.csv")
    args = parser.parse_args()

    rows = list(load_rows(args.csv_file, args.moneyline_only))
    event_times = {}
    for row, _, kickoff, _, _ in rows:
        event_id = row["event_id"]
        event_times[event_id] = min(event_times.get(event_id, kickoff), kickoff)
    dates = sorted(set(event_times.values()))
    if len(dates) < 2:
        parser.error("Need usable opening prices from matches at two or more kickoff times.")
    cutoff = dates[max(1, int(len(dates) * 0.8))]
    train, test = [], []
    for row, p, kickoff, resolved, target in rows:
        if event_times[row["event_id"]] >= cutoff:
            test.append((row, p, target))
        elif kickoff < cutoff and (resolved is None or resolved < cutoff):
            train.append((p, target, resolved))
    if {target for _, target, _ in train} != {0, 1}:
        parser.error("Training needs wins and losses. Export more resolved markets with opening prices.")

    model = LogisticRegression().fit([feature(p) for p, _, _ in train],
                                     [target for _, target, _ in train])
    raw = [p for _, p, _ in test]
    actual = [target for _, _, target in test]
    calibrated = model.predict_proba([feature(p) for p in raw])[:, 1]
    print(f"Train: {len(train)} contracts. Test: {len(test)}. Cutoff: {cutoff.isoformat()}")
    print(f"Training contracts with unknown settlement time: {sum(r is None for _, _, r in train)}")
    print(f"Brier score (lower is better): opening prices {brier_score_loss(actual, raw):.4f}, "
          f"model {brier_score_loss(actual, calibrated):.4f}")

    if args.prices:
        try:
            p = implied_probability(*args.prices)
        except ValueError as error:
            parser.error(str(error))
        prediction = model.predict_proba([feature(p)])[0, 1]
        print(f"Outcome 1: {prediction:.1%}. Outcome 2: {1 - prediction:.1%}.")

    with open(args.output, "w", newline="", encoding="utf-8") as output:
        fields = ["event_id", "condition_id", "question", "outcome_1", "outcome_2", "winning_outcome"]
        writer = csv.writer(output)
        writer.writerow(fields + ["opening_probability_1", "predicted_probability_1",
                                  "predicted_probability_2", "predicted_outcome"])
        for (row, p, _), prediction in zip(test, calibrated):
            winner = row["outcome_1"] if prediction >= 0.5 else row["outcome_2"]
            writer.writerow([row[field] for field in fields] + [p, prediction, 1 - prediction, winner])


if __name__ == "__main__":
    main()
