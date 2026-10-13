"""Sandboxed tool implementations shared by every harness cell.

Tools are deterministic and filesystem-confined: a cell may only touch files
under the data/ whitelist or its per-run workspace directory. Same tool surface
for every cell so the comparison measures harnesses, not tool access.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_data_csv",
            "description": "Read a whitelisted CSV file. Returns up to `limit` rows "
                           "starting at `offset`, as a list of dicts (string values).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "offset": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_data_csv",
            "description": "Write rows (list of dicts) as CSV into the run workspace. "
                           "file is relative to the workspace. Returns the row count written.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "rows": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["file", "rows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "series_stats",
            "description": "Compute stats over a numeric CSV column. stat is one of: "
                           "count, sum, mean, min, max, first, last, max_drawdown. "
                           "max_drawdown = largest peak-to-trough fractional fall "
                           "(negative number) over rows in file order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "column": {"type": "string"},
                    "stat": {"type": "string"},
                },
                "required": ["file", "column", "stat"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a pure arithmetic expression (+ - * / // % ** and "
                           "parentheses only) and return the number.",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final structured answer and end the run. "
                           "answer must be a JSON object exactly as the task requires.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "object"}},
                "required": ["answer"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS_SCHEMA]

_ALLOWED_ARITH_TOKENS = set("0123456789.+-*/%() \t")


class ToolError(Exception):
    """A tool refused or failed; the text goes back to the model as the result."""


def _resolve_data_path(name: str, data_dir: Path) -> Path:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ToolError(f"path not allowed: {name}")
    full = (data_dir / p).resolve()
    if not str(full).startswith(str(data_dir.resolve()) + "/") and full != data_dir.resolve():
        raise ToolError(f"path escapes data directory: {name}")
    if not full.is_file() and p.parts and p.parts[0] == "data":
        # prompts and transcripts name files repo-relative ('data/x.csv');
        # data_dir may itself BE the data directory — collapse the prefix
        full = (data_dir / Path(*p.parts[1:])).resolve()
        if not str(full).startswith(str(data_dir.resolve()) + "/"):
            raise ToolError(f"path escapes data directory: {name}")
    if not full.is_file():
        raise ToolError(f"file not found: {name}")
    return full


def _resolve_ws_path(name: str, ws_dir: Path) -> Path:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ToolError(f"path not allowed: {name}")
    full = (ws_dir / p).resolve()
    ws_root = ws_dir.resolve()
    if not str(full).startswith(str(ws_root) + "/"):
        raise ToolError(f"path escapes workspace: {name}")
    return full


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    return worst


def _column(path: Path, column: str) -> list[float]:
    rows = _read_rows(path)
    if rows and column not in rows[0]:
        raise ToolError(f"column '{column}' not in file; columns: {sorted(rows[0].keys())}")
    out = []
    for row in rows:
        raw = row[column]
        if raw is None or str(raw).strip() == "" or str(raw).strip().upper() == ".":
            continue  # FRED missing-value sentinel
        try:
            out.append(float(raw))
        except ValueError:
            raise ToolError(f"non-numeric value in column '{column}': {raw!r}")
    if not out:
        raise ToolError(f"column '{column}' has no numeric rows")
    return out


def _safe_calc(expr: str) -> float:
    if not expr or any(ch not in _ALLOWED_ARITH_TOKENS for ch in expr):
        raise ToolError("expression may contain only numbers and + - * / // % ** ( )")
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        raise ToolError("division by zero")
    except SyntaxError:
        raise ToolError("not a valid arithmetic expression")


def make_tool_registry(data_dir: Path, ws_dir: Path):
    """Return {name: callable} bound to this run's confined directories."""
    ws_dir.mkdir(parents=True, exist_ok=True)

    def read_data_csv(file: str, offset: int = 0, limit: int = 50) -> dict:
        rows = _read_rows(_resolve_data_path(file, data_dir))
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 500))
        return {"total_rows": len(rows), "rows": rows[offset:offset + limit]}

    def write_data_csv(file: str, rows: list) -> dict:
        full = _resolve_ws_path(file, ws_dir)
        if not str(full).endswith(".csv"):
            raise ToolError("workspace output must be a .csv file")
        full.parent.mkdir(parents=True, exist_ok=True)
        fields: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ToolError("rows must be objects")
            for key in row:
                if key not in fields:
                    fields.append(key)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        full.write_text(buf.getvalue(), encoding="utf-8")
        return {"written_rows": len(rows), "path": str(full.relative_to(ws_dir))}

    def series_stats(file: str, column: str, stat: str) -> dict:
        if file.startswith("ws:"):
            path = _resolve_ws_path(file[3:], ws_dir)
        else:
            path = _resolve_data_path(file, data_dir)
        values = _column(path, column)
        stats = {
            "count": lambda: float(len(values)),
            "sum": lambda: sum(values),
            "mean": lambda: sum(values) / len(values),
            "min": lambda: min(values),
            "max": lambda: max(values),
            "first": lambda: values[0],
            "last": lambda: values[-1],
            "max_drawdown": lambda: _max_drawdown(values),
        }
        if stat not in stats:
            raise ToolError(f"unknown stat '{stat}'; choose from {sorted(stats)}")
        return {"stat": stat, "value": stats[stat]()}

    def calc(expr: str) -> dict:
        return {"value": _safe_calc(str(expr))}

    def finish(answer: dict) -> dict:
        if not isinstance(answer, dict):
            raise ToolError("answer must be a JSON object")
        return {"accepted": True, "answer": answer}

    return {
        "read_data_csv": read_data_csv,
        "write_data_csv": write_data_csv,
        "series_stats": series_stats,
        "calc": calc,
        "finish": finish,
    }
