"""Tools are deterministic, confined, and refuse anything they shouldn't touch."""

from __future__ import annotations

import pytest

from cagematch.tools import TOOL_NAMES, TOOLS_SCHEMA, ToolError, make_tool_registry


def test_schema_names_unique(data_dir):
    assert sorted(TOOL_NAMES) == sorted(t["function"]["name"] for t in TOOLS_SCHEMA)
    assert "finish" in TOOL_NAMES


def test_series_stats_matches_scoring(data_dir, tmp_path):
    reg = make_tool_registry(data_dir, tmp_path)
    out = reg["series_stats"](file="data/spy_daily.csv", column="close",
                              stat="max_drawdown")
    assert -1.0 < out["value"] < 0.0


def test_missing_value_sentinel_skipped(data_dir, tmp_path):
    # FRED calendar series contain '.' for non-trading days; stats must skip
    # them, not treat them as zero (which would corrupt min/drawdown).
    reg = make_tool_registry(data_dir, tmp_path)
    out = reg["series_stats"](file="data/sp500_daily.csv", column="SP500", stat="min")
    assert out["value"] > 100.0  # a zero-contaminated min would be <= 0


def test_path_escape_refused(data_dir, tmp_path):
    reg = make_tool_registry(data_dir, tmp_path)
    with pytest.raises(ToolError):
        reg["read_data_csv"](file="../../etc/passwd")
    with pytest.raises(ToolError):
        reg["read_data_csv"](file="/etc/passwd")
    with pytest.raises(ToolError):
        reg["write_data_csv"](file="../outside.csv", rows=[{"a": 1}])


def test_calc_rejects_non_arithmetic(data_dir, tmp_path):
    reg = make_tool_registry(data_dir, tmp_path)
    assert reg["calc"](expr="2*(3+4)//2")["value"] == 7.0
    for evil in ("__import__('os')", "open('/etc/passwd')", "1 if 1 else 2", "a+b"):
        with pytest.raises(ToolError):
            reg["calc"](expr=evil)


def test_calc_division_by_zero_is_finding_not_crash(data_dir, tmp_path):
    reg = make_tool_registry(data_dir, tmp_path)
    with pytest.raises(ToolError):
        reg["calc"](expr="1/0")


def test_write_then_read_roundtrip(data_dir, tmp_path):
    ws = tmp_path / "ws"
    reg = make_tool_registry(data_dir, ws)
    rows = [{"date": "2026-01-01", "x": 1}, {"date": "2026-01-02", "x": 2}]
    out = reg["write_data_csv"](file="out.csv", rows=rows)
    assert out["written_rows"] == 2
    stats = reg["series_stats"](file="ws:out.csv", column="x", stat="sum")
    assert stats["value"] == 3.0


def test_write_requires_csv(data_dir, tmp_path):
    reg = make_tool_registry(data_dir, tmp_path / "ws")
    with pytest.raises(ToolError):
        reg["write_data_csv"](file="evil.sh", rows=[{"a": 1}])
