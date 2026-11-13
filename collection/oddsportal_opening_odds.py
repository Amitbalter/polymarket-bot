#!/usr/bin/env python3
"""Export OddsPortal's full-time 1X2 opening odds for a match or the Premier League archive."""

import argparse
import csv
import json
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.parse import urljoin, urlsplit

import pandas as pd
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright


RESULTS_URL = "https://www.oddsportal.com/football/england/premier-league/results/"
SEASON_LINKS = 'a[href*="/football/england/premier-league"][href$="/results/"]'
FIXTURE_LINKS = 'a[href^="/football/h2h/"][href*="#"]'
MATCH_LINKS = f"{FIXTURE_LINKS}:visible"
PAGINATION = 'nav[aria-label="Pagination"]'
EMPTY_RESULTS = re.compile(r"no matches can be displayed because there are no odds available", re.I)
ACCESS_MESSAGE = re.compile(r"verify (that )?you are human|checking your browser|unusual traffic", re.I)
OUTCOMES = ("home", "draw", "away")
LEGACY_OUTCOMES = {"1": "home_win", "X": "draw", "2": "away_win"}
ODDS_NUMBER = re.compile(r"\d+[.,]\d+")
OPENING_LABEL = re.compile(r"^opening odds:$", re.IGNORECASE)
FINAL_SCORE = re.compile(r"\bFinal result\s*:?\s*(\d+)\s*[:–-]\s*(\d+)\b", re.I)
ARCHIVE_SCORE = re.compile(r"^(?:Finished|FIN)\s*(\d+)\s+.+\s+(\d+)$", re.I)
BOOKMAKER_LINK = 'a[href^="/proxy/bookmakers/"][href$="/link/"]'
MATCH_PAGE_DATE = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*"
    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4}),\s*\d{1,2}:\d{2}\b"
)
FIELDS = ["match_url", "match_date", "bookmaker", "outcome"]
for outcome in OUTCOMES:
    FIELDS.extend((f"{outcome}_opening_odds", f"{outcome}_opening_time_display"))


def reject_cookies(page):
    button = page.get_by_role("button", name="Reject All", exact=True).first
    if button.is_visible():
        button.click()


def opening_from_cell(page, cell):
    price = cell.locator("p.cursor-pointer").filter(has_text=ODDS_NUMBER).first
    if not price.count():
        return "", "", "no decimal price in the table"

    label = cell.get_by_role("tooltip").get_by_text(OPENING_LABEL)
    for _ in range(2):
        reject_cookies(page)
        page.mouse.move(0, 0)
        try:
            price.hover(timeout=4000)
            label.wait_for(state="visible", timeout=5000)
            timestamp, opening = read_opening(label)
            return timestamp, opening, ""
        except PlaywrightTimeout:
            continue
        except ValueError as exc:
            return "", "", str(exc)
    return "", "", "opening tooltip did not appear"


def read_opening(label):
    # The next element after "Opening odds:" contains the date and opening price.
    fields = label.locator("xpath=following-sibling::*[1]/*").all_inner_texts()
    if len(fields) != 2:
        raise ValueError("opening tooltip has an unexpected layout")
    timestamp, price = (field.strip() for field in fields)
    if not timestamp or not ODDS_NUMBER.fullmatch(price):
        raise ValueError("opening tooltip has an invalid date or decimal price")
    odds = Decimal(price.replace(",", "."))
    if odds <= 1:
        raise ValueError("opening decimal odds must be greater than 1")
    return timestamp, str(odds)


class SiteAccessError(RuntimeError):
    """Stop the crawl when OddsPortal denies or rate-limits access."""


def check_access(page):
    if page.get_by_text(ACCESS_MESSAGE).first.is_visible():
        raise SiteAccessError("OddsPortal is asking this browser to verify access")


def navigate(page, url):
    response = page.goto(url, wait_until="domcontentloaded", timeout=45000)
    if response is not None and not response.ok:
        error = SiteAccessError if response.status in (403, 429) else RuntimeError
        raise error(f"OddsPortal returned HTTP {response.status} for {url}")
    check_access(page)


def match_outcome(text, score_pattern):
    score = score_pattern.search(" ".join(text.split()))
    if score is None:
        return None
    home, away = map(int, score.groups())
    if home == away:
        return "draw"
    return "home_win" if home > away else "away_win"


def normalize_match_date(value):
    for date_format in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(" ".join((value or "").split()), date_format).date().isoformat()
        except ValueError:
            continue
    return None


def read_match_date(page):
    heading = page.get_by_text(MATCH_PAGE_DATE).filter(visible=True).first
    try:
        heading.wait_for(state="visible", timeout=12000)
    except PlaywrightTimeout:
        check_access(page)
        return None
    match = MATCH_PAGE_DATE.search(heading.inner_text())
    return normalize_match_date(match[1]) if match else None


def scrape(page, url):
    navigate(page, url)
    match_date = read_match_date(page)
    reject_cookies(page)
    full_time = page.get_by_role("button", name="Full Time", exact=True)
    full_time.wait_for(state="visible", timeout=25000)
    selected = full_time.evaluate(
        """button => button.getAttribute('aria-pressed') === 'true'
            || button.getAttribute('aria-selected') === 'true'
            || button.classList.contains('bg-black-main')"""
    )
    if not selected:
        full_time.click()
    all_bookies = page.get_by_text("All Bookies", exact=True).first
    if all_bookies.is_visible():
        all_bookies.click()

    header = page.get_by_role("columnheader", name="Bookmakers", exact=True)
    table = page.get_by_role("table").filter(has=header)
    table.wait_for(state="visible", timeout=20000)
    headers = table.locator("thead th").all_inner_texts()
    if [text.strip() for text in headers[1:4]] != ["1", "X", "2"]:
        raise RuntimeError(f"Expected a 1/X/2 table. Found headers {headers!r}")

    rows = table.locator("tbody tr:visible").filter(has=page.locator(BOOKMAKER_LINK))
    rows.first.wait_for(state="visible", timeout=15000)
    result = match_outcome(page.locator("body").inner_text(), FINAL_SCORE)
    records = []
    missing = []
    for row in rows.all():
        cells = row.locator("td")
        bookmaker = cells.first.locator(f"{BOOKMAKER_LINK} p").first.inner_text().strip()
        record = {"match_url": url, "match_date": match_date, "bookmaker": bookmaker, "outcome": result}
        for column, outcome in enumerate(OUTCOMES, start=1):
            timestamp, opening, problem = opening_from_cell(page, cells.nth(column))
            record[f"{outcome}_opening_odds"] = opening
            record[f"{outcome}_opening_time_display"] = timestamp
            if problem:
                missing.append(f"{bookmaker}, {outcome}: {problem}")
        records.append(record)
    return records, missing, match_date


MATCH_METADATA = ("season", "match_id", "match_date", "home_team", "away_team")
ARCHIVE_FIELDS = ["season", "page_number", *MATCH_METADATA[1:], *[field for field in FIELDS if field != "match_date"]]
ISSUE_FIELDS = ["season", "match_url", "status", "detail"]


def discover_seasons(page):
    navigate(page, RESULTS_URL)
    links = page.locator(SEASON_LINKS).filter(has_text=re.compile(r"^\d{4}/\d{4}$"))
    links.first.wait_for(state="visible", timeout=25000)
    return {
        label.replace("/", "-"): urljoin(RESULTS_URL, href)
        for label, href in links.evaluate_all(
            "(links) => links.map(a => [a.textContent.trim(), a.getAttribute('href')])"
        )
    }


def visible_matches(page, season):
    items = page.locator(MATCH_LINKS).evaluate_all(
        """(links) => links.map(link => ({
            href: link.getAttribute('href'),
            teams: Array.from(link.querySelectorAll('img[alt]'), img => img.alt),
            scoreText: link.innerText,
        }))"""
    )
    matches = {}
    for item in items:
        url = urljoin(RESULTS_URL, item["href"])
        match_id = urlsplit(url).fragment
        if not re.fullmatch(r"[A-Za-z0-9]+", match_id):
            print(f"Skipping link without a fixture ID: {url}", file=sys.stderr)
            continue
        if len(item["teams"]) != 2:
            raise RuntimeError(f"Could not identify home and away teams: {url}")
        home, away = item["teams"]
        matches[match_id] = dict(zip(MATCH_METADATA, (season, match_id, None, home, away)))
        matches[match_id]["match_url"] = url
        matches[match_id]["outcome"] = match_outcome(item["scoreText"], ARCHIVE_SCORE)
    return matches


def season_matches(page, season, url, delay):
    navigate(page, url)
    reject_cookies(page)
    first_match = page.locator(MATCH_LINKS).first
    empty = page.get_by_text(EMPTY_RESULTS).first
    first_match.or_(empty).first.wait_for(state="visible", timeout=25000)
    if empty.is_visible():
        raise RuntimeError(empty.inner_text().strip())

    pagination = page.locator(PAGINATION)
    current = pagination.locator('[aria-current="page"]')
    next_page = pagination.locator("button:visible").filter(has_text=re.compile(r"^Next$"))
    seen = set()
    page_number = 1
    while True:
        matches = visible_matches(page, season)
        new_ids = matches.keys() - seen
        if not new_ids:
            raise RuntimeError(f"No new matches on results page {page_number}")
        print(f"{season} page {page_number}: {len(new_ids)} matches", flush=True)
        for position, (match_id, match) in enumerate(matches.items(), start=1):
            if match_id in new_ids:
                yield {**match, "page_number": page_number, "page_match_order": position}
        seen.update(new_ids)

        if not next_page.is_visible():
            return
        sleep(delay)
        reject_cookies(page)
        next_page.click()
        page_number += 1
        current.filter(has_text=re.compile(rf"^{page_number}$")).wait_for(
            state="visible", timeout=25000
        )
        try:
            page.wait_for_function(
                """({seen, selector}) => {
                    const previous = new Set(seen);
                    const ids = Array.from(document.querySelectorAll(selector))
                        .filter(a => a.getClientRects().length
                            && getComputedStyle(a).visibility !== 'hidden')
                        .map(a => a.hash.slice(1))
                        .filter(id => /^[A-Za-z0-9]+$/.test(id));
                    return ids.some(id => !previous.has(id));
                }""",
                arg={"seen": sorted(seen), "selector": FIXTURE_LINKS},
                timeout=25000,
            )
        except PlaywrightTimeout as exc:
            raise RuntimeError(
                f"Paginator reached page {page_number}, but no unseen fixture IDs "
                "became visible within 25 seconds."
            ) from exc


def open_checkpoint(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            season TEXT NOT NULL,
            match_id TEXT NOT NULL,
            url TEXT NOT NULL,
            metadata TEXT NOT NULL,
            status TEXT NOT NULL,
            odds TEXT NOT NULL,
            issues TEXT NOT NULL,
            PRIMARY KEY (season, match_id)
        );
        CREATE TABLE IF NOT EXISTS seasons (
            season TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            issue TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS archive_pages (
            season TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            PRIMARY KEY (season, page_number)
        );
    """)
    return db


def save_match(db, match, status, odds, issues):
    with db:
        db.execute(
            """INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(season, match_id) DO UPDATE SET
                   url=excluded.url, metadata=excluded.metadata, status=excluded.status,
                   odds=CASE WHEN excluded.status='error' THEN matches.odds ELSE excluded.odds END,
                   issues=excluded.issues
               WHERE matches.status != 'complete'""",
            (match["season"], match["match_id"], match["match_url"], json.dumps(match),
             status, json.dumps(odds), json.dumps(issues)),
        )


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_parquet(path, fields, rows):
    frame = pd.DataFrame.from_records(rows, columns=fields)
    for field in fields:
        if field.endswith("_opening_odds"):
            frame[field] = pd.to_numeric(frame[field].replace("", None)).astype("Float64")
        elif field == "page_number":
            frame[field] = frame[field].astype("Int64")
        else:
            frame[field] = frame[field].astype("string")

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as file:
        temporary = Path(file.name)
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def archived_odds(db, season, page_number):
    for metadata, odds in db.execute(
        """SELECT metadata, odds FROM matches
           WHERE season=? AND json_extract(metadata, '$.page_number')=?
           ORDER BY json_extract(metadata, '$.page_match_order'), match_id""",
        (season, page_number),
    ):
        match = json.loads(metadata)
        for row in json.loads(odds):
            outcome = row.get("outcome") or match.get("outcome")
            yield {**match, **row, "match_date": match.get("match_date"),
                   "outcome": LEGACY_OUTCOMES.get(outcome, outcome)}


def write_archive_pages(db, output):
    with db:
        # Remember previously exported pages so a match moving pages cannot leave a stale copy.
        db.execute(
            """INSERT OR IGNORE INTO archive_pages
               SELECT DISTINCT season, json_extract(metadata, '$.page_number')
               FROM matches WHERE json_type(metadata, '$.page_number')='integer'"""
        )
    pages = db.execute("SELECT season, page_number FROM archive_pages ORDER BY season, page_number")
    for season, page_number in pages:
        path = output / season / f"page_{page_number:03d}.parquet"
        write_parquet(path, ARCHIVE_FIELDS, archived_odds(db, season, page_number))


def archive_issues(db):
    for season, url in db.execute(
        """SELECT season, url FROM matches
           WHERE status != 'error' AND COALESCE(json_extract(metadata, '$.match_date'), '') = ''"""
    ):
        yield dict(zip(ISSUE_FIELDS, (season, url, "missing_date", "Match page date unavailable")))
    for season, url, detail in db.execute("SELECT * FROM seasons ORDER BY season"):
        yield dict(zip(ISSUE_FIELDS, (season, url, "archive_issue", detail)))
    for season, url, status, issues in db.execute(
        "SELECT season, url, status, issues FROM matches WHERE status != 'complete' "
        "ORDER BY season, match_id"
    ):
        for detail in json.loads(issues):
            yield dict(zip(ISSUE_FIELDS, (season, url, status, detail)))


def crawl(context, args, db):
    archive_page = context.new_page()
    match_page = context.new_page()
    seasons = discover_seasons(archive_page)
    selected = {season.replace("/", "-") for season in args.season}
    unknown = selected - seasons.keys()
    if unknown:
        raise ValueError(f"Seasons absent from the archive: {', '.join(sorted(unknown))}")

    attempts = 0
    for season, url in seasons.items():
        if selected and season not in selected:
            continue
        try:
            for match in season_matches(archive_page, season, url, args.delay):
                old = db.execute(
                    "SELECT status, metadata FROM matches WHERE season=? AND match_id=?",
                    (season, match["match_id"]),
                ).fetchone()
                if old and old[0] == "complete":
                    previous = json.loads(old[1])
                    match["match_date"] = normalize_match_date(previous.get("match_date"))
                    if not match["match_date"]:
                        if args.limit is not None and attempts >= args.limit:
                            return
                        attempts += 1
                        sleep(args.delay)
                        try:
                            navigate(match_page, match["match_url"])
                            match["match_date"] = read_match_date(match_page)
                        except SiteAccessError:
                            raise
                        except (PlaywrightError, RuntimeError) as exc:
                            check_access(match_page)
                            print(f"    Date backfill failed: {str(exc).splitlines()[0]}", file=sys.stderr)
                    outcome = match.get("outcome") or previous.get("outcome")
                    match["outcome"] = LEGACY_OUTCOMES.get(outcome, outcome)
                    if match != previous:
                        with db:
                            db.execute(
                                "UPDATE matches SET metadata=? WHERE season=? AND match_id=?",
                                (json.dumps(match), season, match["match_id"]),
                            )
                    continue
                if args.limit is not None and attempts >= args.limit:
                    return
                attempts += 1
                print(f"  {match['match_date']} {match['home_team']} vs {match['away_team']}", flush=True)
                sleep(args.delay)
                try:
                    rows, missing, date = scrape(match_page, match["match_url"])
                    match["match_date"] = date or normalize_match_date(
                        json.loads(old[1]).get("match_date") if old else None
                    )
                except (PlaywrightError, RuntimeError) as exc:
                    if isinstance(exc, SiteAccessError):
                        raise
                    check_access(match_page)
                    detail = str(exc)
                    save_match(db, match, "error", [], [detail])
                    print(f"    {detail}", file=sys.stderr)
                else:
                    save_match(db, match, "partial" if missing else "complete", rows, missing)
                    if missing:
                        print(f"    {len(missing)} opening prices missing", file=sys.stderr)
        except (PlaywrightError, RuntimeError) as exc:
            if isinstance(exc, SiteAccessError):
                raise
            check_access(archive_page)
            detail = str(exc)
            with db:
                db.execute("INSERT OR REPLACE INTO seasons VALUES (?, ?, ?)", (season, url, detail))
            print(f"{season}: {detail}", file=sys.stderr)
        else:
            with db:
                db.execute("DELETE FROM seasons WHERE season=?", (season,))


def run_archive(context, args):
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "_checkpoint.sqlite3"
    if not checkpoint.exists():
        for suffix in (".parquet.sqlite3", ".csv.sqlite3"):
            previous = args.output.with_name(args.output.name + suffix)
            if previous.is_file():
                checkpoint = previous
                break
    print(f"Checkpoint: {checkpoint}", flush=True)
    issues = args.output / "_issues.csv"
    with closing(open_checkpoint(checkpoint)) as db:
        try:
            crawl(context, args, db)
        finally:
            write_archive_pages(db, args.output)
            write_csv(issues, ISSUE_FIELDS, archive_issues(db))
            total, complete = db.execute(
                "SELECT COUNT(*), COUNT(CASE WHEN status='complete' THEN 1 END) FROM matches"
            ).fetchone()
            print(f"{complete}/{total} stored matches complete. Odds: {args.output}. Issues: {issues}.")
            unassigned = db.execute(
                "SELECT COUNT(*) FROM matches WHERE json_extract(metadata, '$.page_number') IS NULL"
            ).fetchone()[0]
            if unassigned:
                print(f"{unassigned} checkpoint matches are waiting for an archive page.", file=sys.stderr)
        return 2 if any(archive_issues(db)) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="scrape one match URL instead of the archive")
    parser.add_argument("--season", action="append", default=[], help="restrict to a season, e.g. 2025-2026; repeatable")
    parser.add_argument("--limit", type=int, help="maximum matches to attempt in this archive run")
    parser.add_argument("--delay", type=float, default=0.75, help="seconds between archive requests (default: 0.75)")
    parser.add_argument("-o", "--output", type=Path, help="archive directory or single-match Parquet path")
    parser.add_argument("--headed", action="store_true", help="show the Chromium window")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if not 0 <= args.delay < float("inf"):
        parser.error("--delay must be a finite non-negative number")
    if args.url and (args.season or args.limit is not None):
        parser.error("--season and --limit apply only to archive mode")
    if args.output is None:
        args.output = Path("opening_odds.parquet" if args.url else "premier_league_opening_odds")
    if args.url and args.output.suffix.lower() != ".parquet":
        parser.error("single-match --output must have a .parquet extension")
    if not args.url and args.output.suffix.lower() == ".parquet":
        parser.error("archive --output must be a directory")

    try:
        import_module("pyarrow")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not args.headed)
            context = browser.new_context(locale="en-GB", viewport={"width": 1440, "height": 1000})
            if not args.url:
                return run_archive(context, args)
            records, missing, date = scrape(context.new_page(), args.url)
            if date is None:
                missing.append("match page date unavailable")
            write_parquet(args.output, FIELDS, records)
    except KeyboardInterrupt:
        print("Interrupted. Rerun with the same output path to resume archive mode.", file=sys.stderr)
        return 130
    except (ImportError, OSError, PlaywrightError, RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"Scrape failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(records)} bookmaker rows to {args.output}")
    for problem in missing:
        print(f"Missing opening odds: {problem}", file=sys.stderr)
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
