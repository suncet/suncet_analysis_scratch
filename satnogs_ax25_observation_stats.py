#!/usr/bin/env python3
"""
AX.25 usage proxy from SatNOGS public APIs.

Two data sources (--source):

  network (default)
    Walks https://network.satnogs.org/api/observations/ (cursor pages).
    Counts distinct NORAD IDs with observation *end* in the last N days (UTC),
    excluding passes that have not ended yet. Labels AX.25 from
    `transmitter_description` + `transmitter_mode` matching `AX.25` or a
    whole-word `AX25` (no dot). `AX.100` is still not counted as AX.25.

  db-transmitters
    Walks https://db.satnogs.org/api/transmitters/ — this is the SatNOGS DB
    you linked. It does **not** use observation history; it answers: among
    satellites that have at least one transmitter row in the DB, what share
    have no AX.25/AX25-labeled transmitter entry.

Important:
  - True "AX.25 header present in demodulated frames" is not exposed in bulk;
    this script uses catalog/transmitter text fields only.
  - The Network API is aggressively throttled. Use a generous --sleep (e.g. 3–6s)
    for long runs; the script backs off on HTTP 429 when possible.

Usage examples:
  python3 satnogs_ax25_observation_stats.py --source db-transmitters --sleep 0.4
  python3 satnogs_ax25_observation_stats.py --source network --days 365 --sleep 4 --max-pages 2000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

NETWORK_OBS_URL = "https://network.satnogs.org/api/observations/"
DB_TX_URL = "https://db.satnogs.org/api/transmitters/"

# Dot form, or "AX25" as a token (not a substring of e.g. AX255).
AX25_RE = re.compile(r"AX\.25|\bAX25\b", re.IGNORECASE)

THROTTLE_DETAIL_RE = re.compile(
    r"available in\s+(\d+)\s+seconds?", re.IGNORECASE
)


def parse_link_next(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section and "rel='next'" not in section:
            continue
        if section.startswith("<"):
            url = section[1 : section.index(">")]
            return url
    return None


def parse_obs_time(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    s = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _sleep_from_429(headers: dict[str, str], err_body: str) -> float:
    ra = headers.get("Retry-After") or headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    m = THROTTLE_DETAIL_RE.search(err_body)
    if m:
        return float(m.group(1))
    return 60.0


def fetch_json(url: str, *, max_retries: int = 8) -> tuple[object, dict[str, str]]:
    last_err: BaseException | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "satnogs_ax25_observation_stats/1.1 (research; polite pacing)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                headers = {k: v for k, v in resp.headers.items()}
                return json.loads(body), headers
        except urllib.error.HTTPError as e:
            err_body = ""
            if e.fp:
                err_body = e.fp.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt + 1 < max_retries:
                wait = _sleep_from_429(dict(e.headers.items()), err_body) + 2.0
                print(
                    f"HTTP 429 (throttled); sleeping {wait:.0f}s before retry "
                    f"({attempt + 1}/{max_retries})…",
                    file=sys.stderr,
                )
                time.sleep(wait)
                last_err = e
                continue
            if e.fp:
                e.fp.close()
            raise
        except urllib.error.URLError as e:
            last_err = e
            if attempt + 1 < max_retries:
                w = 10.0 * (attempt + 1)
                print(f"Network error; retry in {w:.0f}s: {e}", file=sys.stderr)
                time.sleep(w)
                continue
            raise
    assert last_err is not None
    raise last_err


def run_db_transmitters(*, sleep: float, max_pages: int) -> int:
    params: list[tuple[str, str]] = [("limit", "100")]
    url = DB_TX_URL + "?" + urllib.parse.urlencode(params)

    all_norad: set[int] = set()
    norad_with_ax25: set[int] = set()
    pages = 0
    total_rows = 0

    while url:
        pages += 1
        if max_pages and pages > max_pages:
            print(
                f"Stopped at --max-pages={max_pages} (partial DB sample).",
                file=sys.stderr,
            )
            break
        try:
            data, headers = fetch_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"Request failed: {e}", file=sys.stderr)
            return 1

        if not isinstance(data, list):
            print(f"Unexpected JSON (expected list): {type(data)}", file=sys.stderr)
            return 1
        if not data:
            break

        total_rows += len(data)

        for row in data:
            norad = row.get("norad_cat_id")
            if norad is None:
                continue
            nid = int(norad)
            all_norad.add(nid)
            desc = str(row.get("description") or "")
            mode = str(row.get("mode") or "")
            if AX25_RE.search(f"{desc} {mode}"):
                norad_with_ax25.add(nid)

        url = parse_link_next(headers.get("Link"))
        if sleep:
            time.sleep(sleep)

    n_sats = len(all_norad)
    n_with = len(norad_with_ax25)
    n_without = n_sats - n_with
    pct_without = (100.0 * n_without / n_sats) if n_sats else 0.0
    pct_with = (100.0 * n_with / n_sats) if n_sats else 0.0

    print("SatNOGS DB transmitters → AX.25 / AX25 (description/mode text) summary")
    print("=" * 60)
    print(f"API pages fetched: {pages}")
    print(f"Transmitter rows processed: {total_rows}")
    print(f"Distinct NORAD IDs (from transmitter rows): {n_sats}")
    print(f"  At least one AX.25 / AX25-labeled transmitter: {n_with} ({pct_with:.2f}%)")
    print(f"  No AX.25 / AX25-labeled transmitter: {n_without} ({pct_without:.2f}%)")
    print()
    print(
        "This is DB catalog metadata, not a one-year observation sample. "
        'Use --source network for observation-window stats (slow; use --sleep).'
    )
    if max_pages and pages >= max_pages:
        print()
        print("Caveat: pagination was capped; increase --max-pages for complete DB pass.")
    return 0


def run_network(
    *,
    days: int,
    sleep: float,
    max_pages: int,
    status: str,
) -> int:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=days)
    params: list[tuple[str, str]] = [("limit", "100")]
    if status:
        params.append(("status", status))
    url = NETWORK_OBS_URL + "?" + urllib.parse.urlencode(params)

    all_norad: set[int] = set()
    norad_with_ax25: set[int] = set()
    pages = 0
    total_obs = 0
    skipped_future = 0
    skipped_no_norad = 0
    skipped_outside_window = 0

    while url:
        pages += 1
        if max_pages and pages > max_pages:
            print(
                f"Stopped at --max-pages={max_pages} (partial sample).",
                file=sys.stderr,
            )
            break
        try:
            data, headers = fetch_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"Request failed: {e}", file=sys.stderr)
            return 1

        if not isinstance(data, list):
            print(f"Unexpected JSON (expected list): {type(data)}", file=sys.stderr)
            return 1

        if not data:
            break

        entire_page_before_window = True

        for obs in data:
            total_obs += 1
            end_dt = parse_obs_time(obs.get("end"))
            if end_dt is None:
                skipped_outside_window += 1
                continue
            if end_dt > now:
                skipped_future += 1
                entire_page_before_window = False
                continue
            if end_dt < window_start:
                skipped_outside_window += 1
                continue
            entire_page_before_window = False

            norad = obs.get("norad_cat_id")
            if norad is None:
                skipped_no_norad += 1
                continue
            desc = str(obs.get("transmitter_description") or "")
            mode = str(obs.get("transmitter_mode") or "")
            blob = f"{desc} {mode}"
            nid = int(norad)
            all_norad.add(nid)
            if AX25_RE.search(blob):
                norad_with_ax25.add(nid)

        if entire_page_before_window:
            break

        url = parse_link_next(headers.get("Link"))
        if sleep:
            time.sleep(sleep)

    n_sats = len(all_norad)
    n_with = len(norad_with_ax25)
    n_without = n_sats - n_with
    pct_without = (100.0 * n_without / n_sats) if n_sats else 0.0
    pct_with = (100.0 * n_with / n_sats) if n_sats else 0.0

    print("SatNOGS Network observation → AX.25 / AX25 (transmitter metadata) summary")
    print("=" * 60)
    print(f"Window (UTC): {window_start.isoformat()} → {now.isoformat()} ({days} days)")
    if status:
        print(f"Status filter: {status}")
    print(f"API pages fetched: {pages}")
    print(f"Observations seen (all pages): {total_obs}")
    print(f"  skipped (end in future): {skipped_future}")
    print(f"  skipped (outside window / bad end): {skipped_outside_window}")
    print(f"  skipped (missing norad_cat_id): {skipped_no_norad}")
    print(f"Distinct NORAD IDs in window: {n_sats}")
    print(f"  At least one AX.25 / AX25-labeled obs: {n_with} ({pct_with:.2f}%)")
    print(f"  No AX.25 / AX25-labeled obs: {n_without} ({pct_without:.2f}%)")
    print()
    print(
        "Note: Satellites with multiple transmitters count as AX.25 if any in-window "
        "observation used an AX.25- or AX25-labeled transmitter entry."
    )
    if max_pages and pages >= max_pages:
        print()
        print(
            "Caveat: --max-pages capped pagination; increase for fuller coverage "
            "(expect long runtimes and strict rate limits)."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=("network", "db-transmitters"),
        default="network",
        help="network = observation history; db-transmitters = DB catalog (default: network).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Rolling window for --source network (default: 365).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max API pages (0 = no cap). Strongly recommended for network runs.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=-1.0,
        help="Seconds between requests. Default: 3.0 for network, 0.4 for db-transmitters.",
    )
    parser.add_argument(
        "--status",
        default="",
        help='Optional network-only filter, e.g. "good".',
    )
    args = parser.parse_args()

    if args.sleep < 0:
        sleep = 3.0 if args.source == "network" else 0.4
    else:
        sleep = args.sleep

    if args.source == "db-transmitters":
        return run_db_transmitters(sleep=sleep, max_pages=args.max_pages)
    return run_network(
        days=args.days,
        sleep=sleep,
        max_pages=args.max_pages,
        status=args.status,
    )


if __name__ == "__main__":
    raise SystemExit(main())
