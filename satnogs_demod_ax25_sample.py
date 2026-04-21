#!/usr/bin/env python3
"""
Sample SatNOGS *demodulated* payloads (one per satellite) and heuristically
detect raw AX.25 UI-style framing (address fields + control/PID).

Strategy (keeps Network API traffic small):
  1. Load distinct NORAD IDs from SatNOGS DB `/api/transmitters/` (one request).
  2. Shuffle the full list (`--seed`), then walk NORADs in order.
  3. For each NORAD, call Network `/api/observations/?norad_cat_id=N&status=good`
     with cursor pagination until we find an observation with non-empty
     `demoddata`, or exhaust `--max-pages-per-sat` pages.
  4. Download only the first `payload_demod` URL, capped at `--max-bytes`.
  5. **Stop once `--min-with-demod` satellites** have a **successful download**
     we could classify (or hit `--max-attempts` / run out of NORADs).

**Summary percentages** are computed **only** over satellites where demod data
was downloaded (not over `no_demod` skips). Skips are still counted separately.

This is **not** a full protocol verifier: it looks for AX.25-like headers
(shifted 6+1 callsign fields, UI control, common PID). KISS-wrapped frames
are unwrapped once.

Respect `--sleep-network` between Network requests and `--sleep-download`
between object-store GETs to reduce throttling.

Many `good` observations have **no** `demoddata` (no decoder output for that
mode/client). Those satellites are reported as `no_demod` unless you raise
`--max-pages-per-sat` to search deeper.

Examples:
  python3 satnogs_demod_ax25_sample.py --min-with-demod 50 --sleep-network 7
  python3 satnogs_demod_ax25_sample.py --min-with-demod 20 --max-attempts 400 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

NETWORK_OBS = "https://network.satnogs.org/api/observations/"
DB_TX = "https://db.satnogs.org/api/transmitters/"

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
            return section[1 : section.index(">")]
    return None


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


def fetch_json(url: str, *, max_retries: int = 6) -> tuple[object, dict[str, str]]:
    last_err: BaseException | None = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "satnogs_demod_ax25_sample/1.0 (research)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body), {k: v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            err_body = e.fp.read().decode("utf-8", errors="replace") if e.fp else ""
            if e.code == 429 and attempt + 1 < max_retries:
                wait = _sleep_from_429(dict(e.headers.items()), err_body) + 2.0
                print(f"HTTP 429; sleeping {wait:.0f}s…", file=sys.stderr)
                time.sleep(wait)
                last_err = e
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            if attempt + 1 < max_retries:
                time.sleep(10.0 * (attempt + 1))
                continue
            raise
    assert last_err is not None
    raise last_err


def fetch_bytes(url: str, max_bytes: int, *, max_retries: int = 4) -> bytes:
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "satnogs_demod_ax25_sample/1.0 (research)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read(max_bytes + 1)[:max_bytes]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt + 1 < max_retries:
                time.sleep(30.0)
                continue
            raise
        except urllib.error.URLError:
            if attempt + 1 < max_retries:
                time.sleep(15.0)
                continue
            raise
    return b""


def _callsign_from_6(b6: bytes) -> str | None:
    if len(b6) < 6:
        return None
    chars = []
    for i in range(6):
        c = (b6[i] >> 1) & 0x7F
        if c == 0x20:
            chars.append(" ")
        elif 0x30 <= c <= 0x39 or 0x41 <= c <= 0x5A:
            chars.append(chr(c))
        else:
            return None
    return "".join(chars).strip()


def _unwrap_kiss_once(buf: bytes) -> bytes:
    if not buf or buf[0] != 0xC0:
        return buf
    end = buf.find(0xC0, 1)
    if end <= 1:
        return buf
    inner = buf[1:end]
    if not inner:
        return buf
    # Port byte + AX.25 frame
    return inner[1:] if len(inner) > 1 else inner


def looks_like_ax25_ui(buf: bytes) -> bool:
    """Heuristic: two 7-byte address fields + UI control + plausible PID."""
    if not buf:
        return False
    b = buf
    if b[0] == 0x7E and len(b) > 1:
        b = b[1:]
    b = _unwrap_kiss_once(b)
    if b[0] == 0x7E and len(b) > 1:
        b = b[1:]
    if len(b) < 16:
        return False
    dest = _callsign_from_6(b[0:6])
    src = _callsign_from_6(b[7:13])
    if not dest or not src:
        return False
    ctrl = b[14]
    pid = b[15]
    # UI frames; PID often 0xF0 (no layer 3), 0xCF, 0x06 (NET/ROM), etc.
    if ctrl not in (0x03, 0x13, 0x73):
        return False
    if pid not in (0xF0, 0xCF, 0x06, 0x07, 0x08):
        return False
    return True


def classify_demod(buf: bytes) -> str:
    if not buf:
        return "empty"
    if buf[:4] == b"\x89PNG":
        return "png"
    if buf[:2] in (b"\xff\xd8", b"BM"):
        return "jpeg_or_bmp"
    if buf[:1] in (b"{", b"["):
        try:
            json.loads(buf.decode("utf-8"))
            return "json"
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    if looks_like_ax25_ui(buf):
        return "ax25_like"
    if 0xC0 in buf[:64]:
        return "kiss_or_binary"
    return "other_binary"


@dataclass
class SatResult:
    norad: int
    outcome: str
    detail: str = ""
    obs_id: Optional[int] = None
    url: Optional[str] = None


def load_norads() -> list[int]:
    data, _ = fetch_json(DB_TX)
    if not isinstance(data, list):
        raise RuntimeError("DB transmitters: expected JSON list")
    out: set[int] = set()
    for row in data:
        n = row.get("norad_cat_id")
        if n is not None:
            out.add(int(n))
    return sorted(out)


def find_first_demod_url(
    norad: int, max_pages: int, sleep_net: float
) -> tuple[Optional[int], Optional[str], str]:
    """Returns (obs_id, payload_url, error_or_empty)."""
    params = [
        ("norad_cat_id", str(norad)),
        ("status", "good"),
        ("limit", "25"),
    ]
    url = NETWORK_OBS + "?" + urllib.parse.urlencode(params)
    pages = 0
    while url and pages < max_pages:
        pages += 1
        try:
            obs_list, headers = fetch_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            return None, None, str(e)
        if not isinstance(obs_list, list):
            return None, None, "bad_json"
        for obs in obs_list:
            dd = obs.get("demoddata") or []
            if not dd:
                continue
            p0 = dd[0].get("payload_demod") if isinstance(dd[0], dict) else None
            if not p0:
                continue
            return int(obs["id"]), str(p0), ""
        url = parse_link_next(headers.get("Link"))
        if sleep_net:
            time.sleep(sleep_net)
    return None, None, "no_demod_in_pages"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-with-demod",
        type=int,
        default=50,
        help="Stop after this many satellites with a downloaded demod payload (default: 50).",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=3000,
        help="Max NORADs to try before giving up (default: 3000).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pages-per-sat", type=int, default=4)
    ap.add_argument("--max-bytes", type=int, default=65536)
    ap.add_argument("--sleep-network", type=float, default=4.0)
    ap.add_argument("--sleep-download", type=float, default=0.25)
    ap.add_argument(
        "--list-skips",
        action="store_true",
        help="Print every no_demod / download_fail NORAD (can be long).",
    )
    args = ap.parse_args()

    print("Loading NORAD list from SatNOGS DB…", file=sys.stderr)
    norads = load_norads()
    rng = random.Random(args.seed)
    rng.shuffle(norads)
    print(
        f"Walking {len(norads)} NORADs (shuffled, seed={args.seed}) until "
        f"{args.min_with_demod} with demod or {args.max_attempts} attempts.",
        file=sys.stderr,
    )

    inspectable: list[SatResult] = []
    outcome_counts: dict[str, int] = {}
    skipped_no_demod = 0
    skipped_download_fail: list[SatResult] = []
    attempts = 0

    for norad in norads:
        if len(inspectable) >= args.min_with_demod:
            break
        if attempts >= args.max_attempts:
            print(
                f"Stopped: reached --max-attempts={args.max_attempts} "
                f"with only {len(inspectable)} inspectable demods.",
                file=sys.stderr,
            )
            break
        attempts += 1
        if args.sleep_network and attempts > 1:
            time.sleep(args.sleep_network)

        oid, demod_url, err = find_first_demod_url(
            norad, args.max_pages_per_sat, sleep_net=args.sleep_network
        )
        if err:
            skipped_no_demod += 1
            if args.list_skips:
                print(
                    f"  skip no_demod norad={norad} ({err})",
                    file=sys.stderr,
                )
            continue
        assert demod_url is not None
        try:
            if args.sleep_download:
                time.sleep(args.sleep_download)
            blob = fetch_bytes(demod_url, args.max_bytes)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            fail = SatResult(
                norad, "download_fail", str(e)[:80], oid, demod_url
            )
            skipped_download_fail.append(fail)
            if args.list_skips:
                print(f"  skip download_fail norad={norad} ({e})", file=sys.stderr)
            continue

        label = classify_demod(blob)
        r = SatResult(norad, label, f"{len(blob)} bytes", oid, demod_url)
        inspectable.append(r)
        outcome_counts[label] = outcome_counts.get(label, 0) + 1
        print(
            f"  inspectable {len(inspectable)}/{args.min_with_demod}  "
            f"norad={norad}  {label}",
            file=sys.stderr,
        )

    n_in = len(inspectable)
    print()
    print("SatNOGS demod sample (one payload per satellite, inspectable set only)")
    print("=" * 60)
    print(f"NORAD attempts: {attempts}")
    print(f"Skipped (no demod in search): {skipped_no_demod}")
    print(f"Skipped (download failed): {len(skipped_download_fail)}")
    print(f"Inspectable demods (downloaded): {n_in}")
    if n_in < args.min_with_demod:
        print(
            f"  (wanted {args.min_with_demod}; raise --max-attempts or "
            f"--max-pages-per-sat, or retry later.)"
        )
    print(f"Pages per sat (max): {args.max_pages_per_sat}")
    print(f"Download cap: {args.max_bytes} bytes")
    print()
    print("Classification counts (percent of inspectable demods only)")
    print("-" * 60)
    if n_in == 0:
        print("  (none)")
    else:
        for k in sorted(outcome_counts.keys(), key=lambda x: -outcome_counts[x]):
            c = outcome_counts[k]
            print(f"  {k}: {c} ({100 * c / n_in:.1f}%)")
    print()
    print("Heuristic `ax25_like` = shifted callsign fields + UI ctrl + common PID.")
    print("Not counted as AX.25: png/json/APRS-only quirks/other protocols.")
    print()
    print("Per-satellite (inspectable only): norad, outcome, bytes, obs_id")
    print("-" * 60)
    for r in inspectable:
        print(
            f"  {r.norad:6d}  {r.outcome:16s}  {r.detail:26s}  "
            f"obs={r.obs_id!s}"
        )
    if args.list_skips and skipped_download_fail:
        print()
        print("Download failures (--list-skips)")
        print("-" * 60)
        for r in skipped_download_fail:
            print(
                f"  {r.norad:6d}  {r.outcome:16s}  {r.detail!s}  obs={r.obs_id!s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
