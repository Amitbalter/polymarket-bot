"""Export closed soccer markets using polymarket-client==0.18.0 (Python 3.11+).

Opening odds come from the earliest pre-kickoff taker trade returned by the
SDK. The other side's price is implied as 1 minus the executed price.
"""

import argparse
import csv
from collections import Counter
from itertools import islice

from polymarket import PublicClient

COLUMNS = """
event_id event_title market_id condition_id question kickoff_at resolved_at
outcome_1 outcome_2 winning_outcome initial_price_1 initial_price_2
first_trade_at traded_outcome trade_price trade_tx opening_status market_type outcome_source
""".split()


def opening_trade(client, market, cutoff):
    outcomes = market.outcomes.yes, market.outcomes.no
    tokens = {outcome.token_id: index for index, outcome in enumerate(outcomes) if outcome.token_id}
    trades = client.list_trades(
        condition_id=market.condition_id, taker_only=True, full_history=True, page_size=1000,
    )
    first = None
    prices = set()
    for trade in trades.iter_items():
        side = tokens.get(trade.asset_id)
        if side is None or trade.timestamp >= cutoff or not 0 <= trade.price <= 1:
            continue
        if first is None or trade.timestamp < first.timestamp:
            first, prices = trade, set()
        if trade.timestamp == first.timestamp:
            prices.add(trade.price if side == 0 else 1 - trade.price)
    if first is None:
        return ("", "", "", "", "", "", "no_pre_kickoff_trade")
    if len(prices) != 1:
        return ("", "", first.timestamp, "", "", "", "ambiguous_first_second")
    price_1 = prices.pop()
    traded_outcome = outcomes[tokens[first.asset_id]].label
    return (price_1, 1 - price_1, first.timestamp, traded_outcome,
            first.price, first.transaction_hash, "available")


def winning_side(payouts, outcomes, references):
    if payouts in ((1, 0), (0, 1)):
        return payouts.index(1), "resolution_payouts"
    if payouts and all(value > 0 for value in payouts):
        return None, "split payout"
    tokens = [references.get(outcome.token_id) for outcome in outcomes]
    if all(token and token.resolved for token in tokens):
        prices = tuple(token.final_price for token in tokens)
        if prices in ((1, 0), (0, 1)):
            return prices.index(1), "token_settlement"
        return None, "token settlement has no single winner"
    # The caller has already verified that this market is resolved.
    prices = tuple(outcome.price for outcome in outcomes)
    if prices in ((1, 0), (0, 1)):
        return prices.index(1), "resolved_gamma_prices"
    return None, "missing payout" if payouts is None else f"unrecognized payout {payouts}"


def export_rows(client, event_limit, counts, moneyline_only=False, skip_trades=False):
    events = client.list_events(tag_slug="soccer", closed=True).iter_items()
    seen = set()
    get_token_references = getattr(client, "get_token_references", None)
    for event in islice(events, event_limit):
        counts["events scanned"] += 1
        if (len(event.markets) != 3):
            counts["skipped: not 3 markets"] += 1
            continue
        markets = {}
        for market in event.markets:
            counts["markets scanned"] += 1
            condition = market.condition_id
            if not condition:
                counts["skipped: missing condition ID"] += 1
            elif condition in seen:
                counts["skipped: duplicate condition ID"] += 1
            elif market.state.closed is False:
                counts["skipped: market still open"] += 1
            elif moneyline_only and market.sports.sports_market_type != "moneyline":
                counts["skipped: missing or different moneyline type"] += 1
            else:
                markets[condition] = market
                seen.add(condition)
        ids = list(markets)
        for start in range(0, len(ids), 20):
            batch = ids[start:start + 20]
            results = {r.condition_id: r for r in client.get_resolutions(condition_ids=batch)}
            # Some finalized resolutions omit payouts. Fetch token settlement in one batch.
            assets = [
                outcome.token_id
                for condition, result in results.items()
                if condition in batch and result.status == "resolved"
                and result.payouts not in ((1, 0), (0, 1))
                and not (result.payouts and all(value > 0 for value in result.payouts))
                for outcome in (markets[condition].outcomes.yes, markets[condition].outcomes.no)
                if outcome.token_id
            ]
            references = {}
            if get_token_references and assets:
                references = {token.asset_id: token for token in get_token_references(asset_ids=assets)}
            for condition in batch:
                result = results.get(condition)
                if result is None:
                    counts["skipped: no resolution returned"] += 1
                    continue
                if result.status != "resolved":
                    counts[f"skipped: resolution status {result.status}"] += 1
                    continue
                market = markets[condition]
                outcomes = market.outcomes.yes, market.outcomes.no
                side, source = winning_side(result.payouts, outcomes, references)
                if side is None:
                    counts[f"skipped: {source}"] += 1
                    continue
                counts[f"winner source: {source}"] += 1
                first, second = (outcome.label for outcome in outcomes)
                winner = outcomes[side].label
                kickoff = market.sports.game_start_time or event.schedule.start_time
                odds = ("", "", "", "", "", "", "missing_kickoff")
                if skip_trades:
                    odds = ("", "", "", "", "", "", "not_requested")
                elif kickoff:
                    cutoff = min(kickoff, result.resolved_at) if result.resolved_at else kickoff
                    odds = opening_trade(client, market, cutoff)
                counts[f"exported: {odds[-1]}"] += 1
                yield (event.id, event.title, market.id, market.condition_id, market.question,
                       kickoff, result.resolved_at, first, second, winner, *odds,
                       market.sports.sports_market_type, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=5, help="Maximum closed soccer events to scan")
    parser.add_argument("--moneyline-only", action="store_true")
    parser.add_argument("--skip-trades", action="store_true", help="Check filters without downloading trades")
    parser.add_argument("--output", default="soccer.csv")
    args = parser.parse_args()
    if args.events < 1:
        parser.error("--events must be at least 1")
    counts = Counter({"events scanned": 0, "markets scanned": 0})
    with PublicClient() as client:
        rows = list(export_rows(client, args.events, counts, args.moneyline_only, args.skip_trades))
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} contracts to {args.output}.")
    for label, count in counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
