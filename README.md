# rsi-alert

Weekly first-pass watchlist filter. Scans Binance spot for USDT pairs whose RSI
touched an extreme (>= 80 or <= 20) on the 4H or 1D chart in the last 30 days.

Two files, no install. Both are [uv](https://docs.astral.sh/uv/) scripts that
declare their own dependencies (PEP 723), so there is no virtualenv to manage.

```bash
./rsi_scanner.py                    # top 200 USDT pairs, 4H + 1D
./test_rsi_scanner.py               # the test suite, same deal
```

(uv is the only prerequisite: `brew install uv`. No API keys — public data only.)

```
+-------+------+----------------+---------------+
| TOKEN | SIDE | 4H             | 1D            |
+-------+------+----------------+---------------+
| LSK   | OB   | 98.2  3b ->71  | 96.1  NOW     |
| KAVA  | OB   | 90.1  33b ->43 | 94.3  NOW     |
| CVC   | OB   | 93.1  NOW      | -             |
| STEEM | OB   | 94.5  1b ->62  | -             |
| ZEC   | OB   | 82.0  40b ->40 | 86.5  6b ->63 |
| BCH   | OS   | 19.8  18b ->32 | -             |
+-------+------+----------------+---------------+
  OB/OS = overbought / oversold   "92.2  3b ->53" = hit 92.2 three bars ago, now 53
  "-" = scanned, no extreme   "?" = scan failed (see log)
  closed candles through: 4H 2026-09-13 12:00Z   1D 2026-09-12 00:00Z

RSI scan 2026-09-13 17:09 UTC | tf=4h,1d rsi=14 thresholds=20/80 lookback=7d | symbols=50 ok=98 failed=2 coverage=98% | flagged=24
```

One row per token per side, one column per timeframe, so cross-timeframe
confluence reads at a glance: LSK is extreme on both, CVC only on 4H. A token
gets a second row when it hit *both* extremes in the window (BCH ran to 93 and
then dumped to 20 last month) — that is information, not a duplicate.

Cells are `extreme, how long ago, where it sits now`. `NOW` means the extreme
*is* the current reading. `--format grouped` gives the older per-timeframe view
with price and 24h volume.

## Flags

```bash
./rsi_scanner.py --top-n 50 --lookback-days 14
./rsi_scanner.py --symbols BTC/USDT ETH/USDT --timeframes 1d
./rsi_scanner.py --offline --overbought 88          # retune, no network
./rsi_scanner.py --help
```

| Flag | Default | |
|------|---------|--|
| `--timeframes` | `4h 1d` | charts to scan |
| `--rsi-period` | `14` | Wilder period |
| `--overbought` / `--oversold` | `80` / `20` | thresholds |
| `--lookback-days` | `30` | how far back an extreme still counts |
| `--top-n` / `--quote` | `200` / `USDT` | auto selection by 24h quote volume |
| `--symbols` | — | explicit list, overrides `--top-n` |
| `--include-leveraged` | off | keep `UP`/`DOWN`/`BULL`/`BEAR` tokens |
| `--format` | `matrix` | or `grouped` |
| `--csv-dir` / `--no-csv` | `.` | `rsi_extremes_<ts>.csv`, written even when empty |
| `--min-market-cap` | off | skip coins below `50M` / `1.5B` / `5e7` |
| `--show-mcap` | off | MCAP + FLOAT columns without filtering |
| `--keep-unknown-mcap` | off | keep coins CoinGecko has no data for |
| `--cache-path` / `--no-cache` | `rsi_cache.sqlite` | the candle store |
| `--offline` / `--refresh` | off | replay from cache / force a full refetch |
| `--telegram` | off | needs `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` |
| `--min-success-rate` | `0.80` | coverage floor before the run is called broken |

Defaults also live in the config block at the top of `rsi_scanner.py`.

## The candle store

Every run writes closed candles to `rsi_cache.sqlite` (stdlib `sqlite3`, no new
dependency). Add it to `.gitignore`.

It does **not** save you requests — Binance charges per request, not per candle,
so a weekly scan makes the same ~400 calls either way. Measured on a top-50 run:

| | time |
|---|---|
| cold, online | 18.2s |
| warm, online (fetches only the gap) | 16.0s |
| `--offline` replay | **0.45s** |

The win is `--offline`: retuning `--overbought`, `--lookback-days` or `--top-n`
against data you already have is instant instead of a minute per attempt. The
second win is time — history accumulates past Binance's 1000-candle ceiling, so
after a few months there is enough data to actually backtest a threshold.

Three rules keep the cache honest:

* **Only closed candles are ever stored**, enforced in `upsert_candles` rather
  than trusted from the caller. A persisted half-formed bar never gets corrected,
  and every RSI computed from that series afterwards would be quietly wrong. The
  scan is deliberately up to one bar behind live price, and the output prints
  exactly which closed candle it is standing on.
* **Gaps are never spliced.** RSI over a series with a hole is not approximately
  right, it is wrong and plausible. Reads are trimmed to the contiguous tail and
  the trim is logged; the incremental fetch always overlaps what it already holds.
* **An offline replay refuses a cache that is too shallow.** Wilder's smoothing
  is seeded from its first observation and that seed decays exponentially without
  ever reaching zero — at 60 stored bars the residue is still ~0.2 of an RSI
  point, enough to flip a threshold. Below `window + period + 100` bars the
  symbol is failed and counted, not computed. With a full cache, `--offline`
  reproduces a live scan exactly, column for column.

## Market cap

Off by default. `--min-market-cap 50M` filters the universe *before* scanning, so
an excluded coin costs no requests; `--show-mcap` adds the columns without
filtering.

```
+---------+------+--------+-------+----------------+---------------+
| TOKEN   | SIDE |   MCAP | FLOAT | 4H             | 1D            |
+---------+------+--------+-------+----------------+---------------+
| PHA     | OB   |   $72M |   86% | 95.1  28b ->76 | 93.3  NOW     |
| SEI     | OB   |  $498M |   67% | 87.0  28b ->76 | 82.0  NOW     |
| SUI     | OB   | $4.76B |   41% | 82.6  25b ->70 | -             |
| ONDO    | OB   | $2.65B |   49% | 80.7  5b ->67  | -             |
+---------+------+--------+-------+----------------+---------------+
```

**What it is for:** market cap is a decent proxy for how hard a token is to push
around. A $20M float moves on one actor's size. That is a liquidity judgement and
it is the honest use of this filter.

**What it is not for:** spotting junk. A new listing's market cap is chosen, not
earned — pick the supply, let the listing pop, and you have a "$100M coin" on day
four. In a real top-50 run the two most obviously sketchy pairs (listed 4 and 9
days earlier) carried $101M and $117M caps and sailed through a $50M floor, while
Civic (2017) and Travala (2017) sat at $31M and $19M and were cut. It filters on
size, and size is not age or legitimacy. If a coin being *new* is what bothers
you, a listing-age filter is the tool — and that data is already in the cache.

**FLOAT is the column worth reading.** Circulating over total supply: SUI at 41%
and ONDO at 49% mean most of the supply has yet to unlock. No amount of price
history shows you that.

Mechanics: an exchange has no concept of market cap — it needs circulating
supply, so this is a second data source (CoinGecko, keyless). Two facts shaped
the implementation:

* **The base→id map comes from CoinGecko's own Binance ticker listing**, not from
  matching ticker strings. Tickers collide: `ONE` is both Harmony and a
  larger-cap unrelated coin, so "take the bigger one" picks wrong.
* **`market_cap_rank` is unusable.** CoinGecko leaves it empty for wrapped
  assets, so WBTC reads as unranked despite a ~$10B cap. The filter uses the cap
  value.

Both live in the same SQLite store — the mapping for 30 days, the caps for 24
hours — so a repeat run makes zero CoinGecko calls. The free tier throttles hard
(429 after a handful of calls), so the client self-paces at 8s between requests
and backs off on 429. Set `COINGECKO_API_KEY` (a free demo key) to drop that to
2s. `--offline` uses only what is cached and calls nothing.

Coins CoinGecko cannot resolve are skipped by default and listed under
`--verbose`. That is a real choice: in a top-200 sample about 12% did not map,
and several were legitimate projects that had simply fallen below rank 1000.
`--keep-unknown-mcap` inverts it. If CoinGecko is down, a `--show-mcap` run
carries on without the columns; a `--min-market-cap` run exits `2`, because
scanning the unfiltered universe when you asked for a gate would be a lie.

## Web dashboard

```bash
./rsi_scanner.py --top-n 200 --show-mcap --json site/data.json --html site/index.html
open site/index.html
```

`site/index.html` is one self-contained file - logic and data inlined, no fetch,
no CORS, opens from disk. Every scanned symbol ships, not just the flagged ones,
so the threshold / lookback / market-cap controls recompute **in the browser**.
That is `--offline` for anyone with the link: retuning is instant and the server
is not involved. 200 symbols x 2 timeframes is ~140KB gzipped.

RSI period is fixed at 14 in the page (changing it needs raw candles, ~20x the
payload). The lookback slider can narrow below the shipped window but not widen
past it - bars outside it simply are not in the file.

### Deploying it free

GitHub Actions runs the scan every 4 hours and publishes to GitHub Pages. No
server, no cold starts: 6 runs/day x ~2 min is about 360 minutes a month against
2000 free for private repos, unlimited for public.

**Run `.github/workflows/probe.yml` first** (Actions -> Binance reachability
probe -> Run workflow). Binance answers US IPs with HTTP 451 and GitHub's runners
are US-based, so this is the question that decides whether any of it works. The
probe tries `api.binance.com`, then the public data mirror, and tells you which
one answered. If only the mirror works, uncomment `BINANCE_PUBLIC_URL` in
`scan.yml` - the mirror serves the same klines and tickers with no account.

Then enable Pages (Settings -> Pages -> Source: GitHub Actions) and optionally
add `COINGECKO_API_KEY` as a repo secret. `scan.yml` handles the rest: it caches
the candle store between runs, publishes even a degraded scan (the page shows a
coverage banner) and goes red in the Actions list so you notice.

Three things that will bite:

* **Scheduled workflows are disabled after 60 days of repo inactivity.** Nothing
  runs, so nothing complains - which is why the page carries its own dead-man
  switch and turns red when the data is older than six bars.
* **No `--offline` in CI.** A cold Actions cache plus `--offline` exits fatal and
  takes the deploy with it. A test asserts the flag never appears in the workflow.
* **Scheduled runs drift** by minutes under load. Harmless at a 4H cadence.

If you would rather have a real server, **Oracle Cloud Always Free** is the only
genuinely perpetual free VM worth pointing at, and you choose the region, which
sidesteps the geoblock entirely. Avoid Vercel Hobby (cron capped at 2 runs/day,
functions time out before a 90s scan) and PythonAnywhere free (one daily task,
whitelist-only outbound HTTP).

### Keeping the CLI and the page honest

The page reimplements the flagging logic in JavaScript, which is the one place
this tool can disagree with itself: the CLI says a coin tagged 82 nine bars ago,
the page says something else, and neither looks wrong. So `hits_from_rsi()` in
Python and `hitsFromRsi()` in `dashboard.js` are kept deliberately parallel and
tested against each other over ~70 generated series plus the shapes that actually
break ports - ties, exact-threshold values, nulls, both-sides hits, windows wider
than the data.

Two measured facts came out of that:

* **Tie-breaking has to match.** `pandas.idxmax` returns the *first* occurrence,
  so the JS uses strict `>` rather than `>=`. Using `>=` reports a different
  "bars ago" than the CLI whenever a level is tagged twice.
* **Payload precision is a correctness knob, not a size knob.** At 2dp, rounding
  collapsed two distinct bars onto the same value on 1 hit in 693 real ones, and
  the first-occurrence rule then picked the earlier bar - same number, different
  age. 4dp costs 48KB gzipped and removed it.

```bash
./test_rsi_scanner.py          # parity tests run automatically if node is present
npm install jsdom              # optional: also renders the page and drives its controls
```

## Cron

```cron
PATH=/opt/homebrew/bin:/usr/bin:/bin
0 8 * * 1 cd ~/code/tu.rsi-alert && ./rsi_scanner.py --csv-dir ./reports >> ./reports/scan.log 2>&1
```

`crontab -e`, Mondays 08:00. The `PATH` line matters: cron's default `PATH` will
not find `uv`, and the shebang needs it. `mkdir -p reports` first, `cd` so the
cache is found, and on macOS give cron Full Disk Access.

Exit codes: `0` healthy, `1` ran but coverage below threshold (results
incomplete), `2` could not reach the exchange.

## Not silently broken

Every run ends with a heartbeat line — symbols, ok, failed, coverage, flagged —
even when nothing is flagged, so *no output* always means broken rather than
"quiet market". Below `--min-success-rate` it prints a loud `WARNING` and exits
non-zero. Individual symbol failures (delisted, listed four days ago, thin cache)
are logged, skipped, and counted against coverage; in the matrix they show as
`?` rather than an innocent-looking `-`.

The RSI is pinned by tests: it matches `pandas-ta` on a fixed synthetic series to
~1.4e-14, and matches a hand-written Wilder loop so it stays pinned even without
pandas-ta. Monotonic up ends > 95, monotonic down < 5, flat series is NaN. The
single network test is skipped unless `RUN_NETWORK_TESTS=1`; it pulls real
BTC/USDT candles and checks shape, strictly increasing timestamps with no gaps,
and recency.

```bash
./test_rsi_scanner.py                      # 93 tests, offline, ~1.5s
RUN_NETWORK_TESTS=1 ./test_rsi_scanner.py  # + the live pull
```

## Notes

* RSI uses Wilder's smoothing (`ewm(alpha=1/period, adjust=False)`) to match
  TradingView. A plain `ewm(span=14)` is `alpha=2/15` and gives different
  numbers. One deliberate difference from pandas-ta: the first 14 bars stay NaN
  instead of being seeded off a single observation.
* One request per symbol per timeframe, capped at Binance's 1000, with
  `enableRateLimit=True`.
* Leveraged tokens are excluded by default; their RSI echoes the underlying.
* 80/20 over 30 days is loose on 4H — a top-50 run flags ~40 pairs, most of them
  one market-wide rally showing up 25 times. Tighten with `--lookback-days 7` or
  `--overbought 88` for that timeframe.
