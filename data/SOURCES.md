# Data sources

All files in this directory are real, downloaded observations. No synthetic or
hand-entered values.

| file | source | fetched | rows |
|---|---|---|---|
| spy_daily.csv | Yahoo Finance chart API `v8/finance/chart/SPY?range=5y&interval=1d`, adjusted close | 2026-10-13 | 1255 |
| qqq_daily.csv | Yahoo Finance chart API `v8/finance/chart/QQQ?range=5y&interval=1d` | 2026-10-13 | 1255 |
| sp500_daily.csv | FRED series SP500 (`fredgraph.csv?id=SP500`, cosd=2024-01-01) | 2026-10-13 | 696 |
| dgs10_daily.csv | FRED series DGS10 (`fredgraph.csv?id=DGS10`, cosd=2024-01-01) | 2026-10-13 | ~695 |

Reproduce:

```bash
curl -H 'User-Agent: Mozilla/5.0' \
  'https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5y&interval=1d' \
  > /tmp/spy.json    # then convert timestamp/close columns to CSV
curl 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2024-01-01' \
  > data/sp500_daily.csv
```

Note: FRED calendar-day series omit or blank non-trading days. The scoring
layer treats blank/'.' observations as missing, never zero — treating them as
zeros silently corrupts min and drawdown statistics, which is itself one of
the failure modes this harness is built to catch.

Ground-truth reference values recomputed at score time (see
`src/cagematch/scoring.py`), captured 2026-10-13:

- SPY max drawdown over the 5y sample: -0.253606
- QQQ max drawdown over the 5y sample: -0.356172
- SPY mean close, most recent 252 trading rows: 713.9996
- Last SP500 observation: 7811.54
