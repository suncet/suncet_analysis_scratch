"""Exercise UTC alignment and pointer lookup against a small real DuckDB."""

import duckdb
import pytest

from find_nand_pages import DEFAULT_DATABASE, lookup_write_pointers, parse_utc


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "telemetry.duckdb"
    # 2026-10-11 18:01:29 UTC is coarse 845056884 in the pipeline convention.
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE _apid_catalog (apid INTEGER, table_name VARCHAR)")
        connection.execute("INSERT INTO _apid_catalog VALUES (1, 'apid_0001')")
        connection.execute("""
            CREATE TABLE apid_0001 (
                combined_time_coarse BIGINT, combined_time_fine BIGINT,
                sw_store_partition_write_hk_beac BIGINT,
                sw_store_partition_write_adcs_beac BIGINT,
                sw_store_partition_write_dsps_beac BIGINT
            )
        """)
        connection.execute("""
            INSERT INTO apid_0001 VALUES
                (845056883, 970, 45520, 20831, 6826),
                (845056884, 970, 45520, 20832, 6826),
                (845056885, 970, 45521, 20833, 6827)
        """)
    return path


def test_nearest_utc_and_timezone_alignment(database):
    before = database.stat().st_mtime_ns
    result = lookup_write_pointers("2026-10-11 18:01:29", database=database)
    sample = result["samples"][0]
    assert sample["sample_utc"] == "2026-10-11T18:01:28.970Z"
    assert sample["offset_seconds"] == pytest.approx(-0.030)
    assert sample["write_pointers"] == {"HK": 45520, "ADCS": 20831, "DSPS": 6826}
    assert lookup_write_pointers("2026-10-11T12:01:29-06:00", database=database) == result
    assert database.stat().st_mtime_ns == before


def test_direction_tie_and_exact_match(database):
    after = lookup_write_pointers("2026-10-11 18:01:29", database=database, match="after")
    assert after["samples"][0]["write_pointers"]["ADCS"] == 20832
    earlier = lookup_write_pointers("2026-10-11 18:01:29.470", database=database)
    assert earlier["samples"][0]["write_pointers"]["ADCS"] == 20831
    exact = lookup_write_pointers("2026-10-11 18:01:29.970Z", database=database)
    assert exact["samples"][0]["offset_seconds"] == 0


def test_default_database_and_pointer_decrease(database, tmp_path, monkeypatch):
    monkeypatch.setenv("suncet_data", str(tmp_path))
    default = tmp_path.joinpath(*DEFAULT_DATABASE)
    default.parent.mkdir(parents=True)
    database.rename(default)
    with duckdb.connect(str(default)) as connection:
        connection.execute("UPDATE apid_0001 SET sw_store_partition_write_hk_beac = 1 WHERE combined_time_coarse = 845056885")
    result = lookup_write_pointers("2026-10-11 18:01:29", "2026-10-11 18:01:30.970")
    assert len(result["samples"]) == 2
    assert len(result["warnings"]) == 1
    assert "HK write pointer decreases" in result["warnings"][0]
    assert result["playback"]["HK"] == {"start_address": 45520, "page_count": None}
    assert result["playback"]["ADCS"] == {"start_address": 20831, "page_count": 2}


def test_playback_counts_use_exclusive_end_pointer(database):
    result = lookup_write_pointers(
        "2026-10-11 18:01:29", "2026-10-11 18:01:30.970", database=database
    )
    assert result["playback"] == {
        "HK": {"start_address": 45520, "page_count": 1},
        "ADCS": {"start_address": 20831, "page_count": 2},
        "DSPS": {"start_address": 6826, "page_count": 1},
    }
    same_time = lookup_write_pointers(
        "2026-10-11 18:01:29", "2026-10-11 18:01:29", database=database
    )
    assert all(playback["page_count"] == 0 for playback in same_time["playback"].values())


def test_errors_for_coverage_tolerance_and_reversed_interval(database):
    with pytest.raises(ValueError, match="outside beacon coverage"):
        lookup_write_pointers("2026-10-12 00:00:00", database=database)
    with pytest.raises(ValueError, match="exceeds --max-offset"):
        lookup_write_pointers("2026-10-11 18:01:29", database=database, max_offset=0.01)
    with pytest.raises(ValueError, match="End UTC"):
        lookup_write_pointers("2026-10-12 00:00:00", "2026-10-11 00:00:00", database=database)
    with pytest.raises(ValueError, match="Invalid UTC time"):
        parse_utc("bad time")


def test_conflicting_duplicate_timestamp_is_rejected(database):
    with duckdb.connect(str(database)) as connection:
        connection.execute("INSERT INTO apid_0001 VALUES (845056883, 970, 99999, 20831, 6826)")
    with pytest.raises(ValueError, match="Conflicting write pointers"):
        lookup_write_pointers("2026-10-11 18:01:29", database=database)
