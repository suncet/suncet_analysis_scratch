#!/usr/bin/env python
"""Look up HK, ADCS and DSPS playback start addresses and page counts by UTC.

Examples (run in the suncet-current environment):
    python find_nand_pages.py "2026-10-11 18:01:29" "2026-10-12 11:45:07"
    python find_nand_pages.py "2026-10-11T18:01:29Z" --match before
    python find_nand_pages.py "2026-10-11 18:01:29" --database /path/to/data.duckdb

Values are the raw beacon write pointers. They are sampled storage boundaries;
this script does not interpolate pointers or convert them to physical addresses.
For an interval, page count = end write pointer - start write pointer: the end
write pointer is an exclusive boundary, so there is no extra +1 page.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import duckdb

from suncet_processing_pipeline.data_paths import data_path, SuncetDataPathError
from suncet_processing_pipeline.spacecraft_time import (
    J2000_UTC_EPOCH,
    POST_J2000_LEAP_SECOND_EFFECTIVE_UTC,
    j2000_seconds_to_iso8601,
)


DEFAULT_DATABASE = (
    "test_data", "2026-10-07_tvac2", "2026-10-07_tvac2_time_corrected.duckdb"
)
POINTER_FIELDS = {
    "HK": "sw_store_partition_write_hk_beac",
    "ADCS": "sw_store_partition_write_adcs_beac",
    "DSPS": "sw_store_partition_write_dsps_beac",
}


def parse_utc(value: str) -> datetime:
    """Accept ISO strings with a space or T; timezone-free strings mean UTC."""
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid UTC time {value!r}; use YYYY-MM-DD HH:MM:SS") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _spacecraft_milliseconds(utc: datetime) -> float:
    # Invert the pipeline's shared UTC convention, including its leap offset.
    leaps = sum(utc >= effective for effective in POST_J2000_LEAP_SECOND_EFFECTIVE_UTC)
    return ((utc - J2000_UTC_EPOCH).total_seconds() - leaps) * 1000


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def lookup_write_pointers(
    start_utc: str,
    end_utc: str | None = None,
    *,
    database: str | Path | None = None,
    match: str = "nearest",
    max_offset: float | None = None,
) -> dict:
    """Return sampled pointers and signed time offsets; open DuckDB read-only.

    nearest chooses the earlier sample on a tie. before/after constrain the
    match direction. A two-time lookup returns endpoint snapshots and playback
    start_address/page_count pairs. Counts are unavailable for pointer decreases.
    """
    if match not in {"nearest", "before", "after"}:
        raise ValueError("match must be nearest, before, or after")
    if max_offset is not None and (not math.isfinite(max_offset) or max_offset < 0):
        raise ValueError("max_offset must be a finite, nonnegative number of seconds")
    times = [parse_utc(start_utc)]
    if end_utc is not None:
        times.append(parse_utc(end_utc))
        if times[1] < times[0]:
            raise ValueError("End UTC must be at or after start UTC")
    path = Path(database).expanduser().resolve() if database is not None else data_path(*DEFAULT_DATABASE)
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry database not found: {path}")
    result = {"database": str(path), "match": match, "samples": [], "warnings": []}
    with duckdb.connect(str(path), read_only=True) as connection:
        catalog = connection.execute(
            "SELECT table_name FROM _apid_catalog WHERE apid = 1"
        ).fetchone()
        if catalog is None:
            raise ValueError("Database has no APID 1 beacon table")
        table = _quote(catalog[0])
        columns = {row[0] for row in connection.execute(f"DESCRIBE {table}").fetchall()}
        required = {"combined_time_coarse", "combined_time_fine", *POINTER_FIELDS.values()}
        if missing := required - columns:
            raise ValueError("Beacon table is missing fields: " + ", ".join(sorted(missing)))
        # Integer milliseconds avoid rounding the beacon's millisecond timestamps.
        timestamp = '("combined_time_coarse" * 1000 + "combined_time_fine")'
        valid = (
            '"combined_time_coarse" >= 0 AND "combined_time_fine" BETWEEN 0 AND 999'
        )
        lower, upper = connection.execute(
            f"SELECT min({timestamp}), max({timestamp}) FROM {table} WHERE {valid}"
        ).fetchone()
        if lower is None:
            raise ValueError("Beacon table has no valid spacecraft timestamps")
        fields = ", ".join(_quote(field) for field in POINTER_FIELDS.values())
        for utc in times:
            target = _spacecraft_milliseconds(utc)
            requested = utc.isoformat().replace("+00:00", "Z")
            if not lower <= target <= upper:
                raise ValueError(
                    f"{requested} is outside beacon coverage "
                    f"{j2000_seconds_to_iso8601(lower / 1000)} to "
                    f"{j2000_seconds_to_iso8601(upper / 1000)}"
                )
            direction = {"nearest": "", "before": f" AND {timestamp} <= ?", "after": f" AND {timestamp} >= ?"}[match]
            parameters = ([target] if direction else []) + [target]
            row = connection.execute(
                f"SELECT {timestamp}, {fields} FROM {table} WHERE {valid}{direction} "
                f"ORDER BY abs({timestamp} - ?), {timestamp} LIMIT 1", parameters,
            ).fetchone()
            if row is None:
                raise ValueError(f"No {match} beacon sample for {requested}")
            # Repeated imports are harmless only if the pointers agree.
            variants = connection.execute(
                f"SELECT DISTINCT {fields} FROM {table} WHERE {valid} AND {timestamp} = ?",
                [row[0]],
            ).fetchall()
            if len(variants) > 1:
                raise ValueError(f"Conflicting write pointers at beacon timestamp {row[0]} ms")
            if any(value is None or int(value) != value or value < 0 for value in row[1:]):
                raise ValueError(f"Missing or invalid write pointers for {requested}")
            offset = (row[0] - target) / 1000
            if max_offset is not None and abs(offset) > max_offset:
                raise ValueError(f"Sample for {requested} is {offset:+.3f} s away; exceeds --max-offset {max_offset}")
            if abs(offset) > 60:
                result["warnings"].append(f"Sample for {requested} is {offset:+.3f} s away (telemetry gap).")
            result["samples"].append({
                "requested_utc": requested,
                "sample_utc": j2000_seconds_to_iso8601(row[0] / 1000),
                "offset_seconds": offset,
                "write_pointers": dict(zip(POINTER_FIELDS, map(int, row[1:]))),
            })
        if len(times) == 2:
            decreases = ", ".join(
                f"count(*) FILTER (WHERE {_quote(name)} < lag_{name})" for name in POINTER_FIELDS
            )
            lag_fields = ", ".join(
                f"{_quote(field)} AS {_quote(name)}, lag({_quote(field)}) OVER (ORDER BY sample_ms) AS lag_{name}"
                for name, field in POINTER_FIELDS.items()
            )
            bounds = [
                _spacecraft_milliseconds(parse_utc(sample["sample_utc"]))
                for sample in result["samples"]
            ]
            counts = connection.execute(
                f"WITH samples AS (SELECT DISTINCT {timestamp} AS sample_ms, {fields} "
                f"FROM {table} WHERE {valid} AND {timestamp} BETWEEN ? AND ?), "
                f"ordered_samples AS (SELECT {lag_fields} FROM samples) "
                f"SELECT {decreases} FROM ordered_samples", bounds,
            ).fetchone()
            result["playback"] = {}
            for name, count in zip(POINTER_FIELDS, counts):
                start_address = result["samples"][0]["write_pointers"][name]
                end_address = result["samples"][1]["write_pointers"][name]
                result["playback"][name] = {
                    "start_address": start_address,
                    "page_count": None if count else end_address - start_address,
                }
                if count:
                    result["warnings"].append(
                        f"{name} write pointer decreases {count} time(s) within the sampled interval; "
                        "a wrap, reset, or pointer change prevents treating the endpoints as one increasing range."
                    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("start", help="UTC string, e.g. '2026-10-11 18:01:29'")
    parser.add_argument("end", nargs="?", help="Optional end UTC string")
    parser.add_argument("--database", "--duckdb", type=Path, help="DuckDB path; defaults to the corrected TVAC2 database under $suncet_data")
    parser.add_argument("--match", choices=("nearest", "before", "after"), default="nearest")
    parser.add_argument("--max-offset", type=float, help="Fail if a matched beacon is more than this many seconds away")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        result = lookup_write_pointers(args.start, args.end, database=args.database, match=args.match, max_offset=args.max_offset)
    except (ValueError, FileNotFoundError, SuncetDataPathError, duckdb.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Database: {result['database']}\nMatch: {result['match']} (raw write-pointer values)")
        for sample in result["samples"]:
            print(f"\nRequested UTC: {sample['requested_utc']}")
            print(f"Beacon UTC:    {sample['sample_utc']} ({sample['offset_seconds']:+.3f} s)")
            if "playback" not in result:
                for partition, pointer in sample["write_pointers"].items():
                    print(f"  {partition:4s} {pointer:10d}  (0x{pointer:08X})")
        if "playback" in result:
            print("\nPlayback (page count = end write pointer - start write pointer):")
            print(f"  {'Partition':9s} {'Start address':>13s} {'Start hex':>12s} {'Number of pages':>16s}")
            for partition, playback in result["playback"].items():
                address = playback["start_address"]
                count = playback["page_count"]
                display_count = str(count) if count is not None else "unavailable"
                print(f"  {partition:9s} {address:13d}   0x{address:08X} {display_count:>16s}")
        for warning in result["warnings"]:
            print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
