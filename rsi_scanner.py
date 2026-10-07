#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ccxt>=4.2", "pandas>=2.0"]
# ///
"""
rsi_scanner.py — scan Binance spot for tokens that hit RSI extremes.

Intended use: run once a week (cron) as a *first-pass watchlist filter*. It is
deliberately dumb — it does not decide anything, it just tells you which liquid
USDT pairs printed an RSI >= 80 or <= 20 on the 4H or 1D chart at any point in
the last ~30 days, and how stale that reading is.

Why "how many bars ago" matters: a pair that tagged RSI 85 three days ago and a
pair that tagged it 29 days ago would otherwise look identical in the output.
Reporting the extreme value, when it happened, *and* the current RSI keeps stale
spikes visible instead of silently repeating them week after week.

Design notes
------------
* RSI uses Wilder's smoothing (RMA) so values line up with TradingView's
  `ta.rsi`. See `wilder_rsi()` — do not swap in a plain EMA/SMA.
* Only CLOSED candles are used. The in-progress candle is dropped, otherwise the
  "current RSI" would flicker intraday and the scan wouldn't be reproducible.
* Every symbol is wrapped in try/except. A delisted or data-less pair is logged
  and skipped; it must never take down a weekly cron run.
* The run always emits a summary line (a heartbeat) plus scan-coverage stats,
  and exits non-zero if coverage drops below a threshold. That way "no output"
  from cron unambiguously means "the job is broken", never "quiet market".

This file is a self-contained uv script: the PEP 723 header above declares its
own dependencies, so there is nothing to install and no virtualenv to manage.

    ./rsi_scanner.py                         # top 200 USDT pairs, 4h + 1d
    ./rsi_scanner.py --top-n 50 --lookback-days 14
    ./rsi_scanner.py --symbols BTC/USDT ETH/USDT --timeframes 1d
    ./rsi_scanner.py --telegram              # also push the summary to Telegram
    ./rsi_scanner.py --help                  # every flag

(`uv run rsi_scanner.py ...` is equivalent if you would rather not chmod +x.)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Sequence

import ccxt
import pandas as pd

# ---------------------------------------------------------------------------
# Config block — every tunable lives here and is overridable from the CLI.
# ---------------------------------------------------------------------------

TIMEFRAMES = ["4h", "1d"]        # charts to scan
RSI_PERIOD = 14                  # Wilder period
OVERBOUGHT = 80.0                # flag when RSI >= this
OVERSOLD = 20.0                  # flag when RSI <= this
LOOKBACK_DAYS = 30               # how far back an extreme still counts
TOP_N = 200                      # number of USDT pairs by 24h quote volume
QUOTE = "USDT"                   # quote currency for auto symbol selection
SYMBOLS: list[str] = []          # non-empty overrides the auto top-N selection
WARMUP_BARS = 250                # extra bars so Wilder's RMA has converged
MIN_SUCCESS_RATE = 0.80          # below this => loud warning + non-zero exit
# Bars after a signal at which to measure what happened. Raw returns are stored
# at each; the verdict is applied at report time (see signal_stats).
SIGNAL_HORIZONS = (1, 3, 5, 10, 20)
ALERT_MAX_AGE_BARS = 2           # older crossings are recorded but not notified
EXCHANGE_ID = "binance"
OUTPUT_FORMAT = "matrix"         # "matrix" (token x timeframe) or "grouped"
CACHE_PATH = "rsi_cache.sqlite"  # closed candles, accumulated across runs
CSV_DIR = "."
MAX_CANDLES = 1000               # Binance per-request cap

# Leveraged tokens (BTCUP/BTCDOWN/ETHBULL/...) are derivative products whose RSI
# is a distorted echo of the underlying. They are noise on a spot watchlist.
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# Exit codes, so cron / a wrapper script can tell the failure modes apart.
EXIT_OK = 0
EXIT_LOW_COVERAGE = 1
EXIT_FATAL = 2

log = logging.getLogger("rsi_scanner")


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (RMA).

    This is the formulation that matches TradingView's `ta.rsi` and pandas-ta's
    `ta.rsi` (verified to ~1e-14 in the test suite). The key detail is
    `ewm(alpha=1/period, adjust=False)`: a recursive smoother where each new
    observation gets weight 1/period. A plain `ewm(span=period)` uses
    alpha=2/(period+1) and produces visibly different numbers — do not
    substitute one for the other.

    Returns a Series aligned to `close`, with NaN for the first `period` rows
    (not enough data) and NaN wherever the series is flat (0/0 -> undefined).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)          # upward moves, else 0
    loss = -delta.clip(upper=0)         # downward moves as positive numbers
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss            # 0/0 -> NaN (flat), x/0 -> inf (only gains)
    rsi = 100 - 100 / (1 + rs)
    return rsi


# ---------------------------------------------------------------------------
# Exchange plumbing
# ---------------------------------------------------------------------------

def build_exchange(exchange_id: str = EXCHANGE_ID,
                   public_url: str | None = None) -> ccxt.Exchange:
    """Public (keyless) exchange client with ccxt's built-in throttle enabled.

    `public_url` (or $BINANCE_PUBLIC_URL) overrides the public REST base. This
    exists for one reason: Binance answers US IPs with HTTP 451, and most free
    CI and hosting runs on US addresses. Pointing at the public data mirror,
    https://data-api.binance.vision/api/v3, serves the same klines and tickers
    without an account. Run the probe workflow to find out which you need.
    """
    klass = getattr(ccxt, exchange_id)
    exchange = klass({
        "enableRateLimit": True,   # ccxt sleeps between calls so we don't get banned
        "options": {"defaultType": "spot"},
    })
    public_url = public_url or os.getenv("BINANCE_PUBLIC_URL")
    if public_url:
        exchange.urls["api"]["public"] = public_url
        log.info("using public endpoint %s", public_url)
    return exchange


def timeframe_seconds(exchange: ccxt.Exchange, timeframe: str) -> int:
    """Bar length in seconds ('4h' -> 14400). ccxt already knows how to parse this."""
    return int(exchange.parse_timeframe(timeframe))


def lookback_bars(exchange: ccxt.Exchange, timeframe: str, lookback_days: int) -> int:
    """How many closed bars cover `lookback_days` on this timeframe."""
    bars = math.ceil(lookback_days * 86_400 / timeframe_seconds(exchange, timeframe))
    return max(1, bars)


def candles_needed(exchange: ccxt.Exchange, timeframe: str, lookback_days: int,
                   period: int = RSI_PERIOD, warmup: int = WARMUP_BARS) -> int:
    """Bars to request: lookback window + RSI warmup + a buffer for the dropped
    in-progress candle. Capped at Binance's 1000-per-request limit so a symbol is
    always a single request."""
    needed = lookback_bars(exchange, timeframe, lookback_days) + period + warmup + 5
    return min(MAX_CANDLES, needed)


def fetch_top_symbols(exchange: ccxt.Exchange, top_n: int = TOP_N, quote: str = QUOTE,
                      include_leveraged: bool = False) -> dict[str, float]:
    """Top `top_n` active spot pairs for `quote`, ranked by 24h quote volume.

    Returns an ordered {symbol: 24h quote volume} mapping — the caller needs both
    the ranking and the volumes, and ccxt does not reliably cache the tickers it
    just fetched, so we keep them here.

    Quote volume (not base volume) is the right ranking key: it is already
    denominated in USDT, so it is comparable across pairs.
    """
    exchange.load_markets()
    tickers = exchange.fetch_tickers()      # one call for the whole board

    ranked: list[tuple[float, str]] = []
    for symbol, ticker in tickers.items():
        market = exchange.markets.get(symbol)
        if not market or not market.get("spot") or not market.get("active"):
            continue
        if market.get("quote") != quote:
            continue
        base = market.get("base", "")
        if not include_leveraged and base.endswith(LEVERAGED_SUFFIXES):
            continue
        volume = ticker.get("quoteVolume")
        if volume is None:
            continue
        ranked.append((float(volume), symbol))

    ranked.sort(reverse=True)
    return {symbol: volume for volume, symbol in ranked[:top_n]}


def fetch_volumes(exchange: ccxt.Exchange, symbols: list[str]) -> dict[str, float]:
    """24h quote volumes for an explicit symbol list. Best-effort: the volume
    column is context, not a result, so a failure here must not stop the scan."""
    try:
        tickers = exchange.fetch_tickers(symbols)
    except Exception as exc:                              # noqa: BLE001
        log.warning("could not fetch 24h volumes: %s: %s", type(exc).__name__, exc)
        return {}
    return {s: float(t.get("quoteVolume") or 0.0) for s, t in tickers.items()}


def fetch_closed_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str,
                       limit: int) -> pd.DataFrame:
    """Fetch OHLCV and return only *closed* candles, oldest first.

    Binance includes the in-progress candle as the last row. Its close moves with
    every tick, so including it would make the scan non-reproducible and would
    fire alerts on prices that can still be taken back before the bar closes.
    """
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return drop_forming_candle(df, timeframe_seconds(exchange, timeframe) * 1000,
                               exchange.milliseconds())


def drop_forming_candle(df: pd.DataFrame, timeframe_ms: int, now_ms: int) -> pd.DataFrame:
    """Drop a trailing candle whose close time is still in the future.

    Split out from the fetch so it is testable without touching the network.
    """
    if df.empty:
        return df
    last_open = int(df["timestamp"].iloc[-1])
    if last_open + timeframe_ms > now_ms:
        return df.iloc[:-1].reset_index(drop=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Candle store (SQLite)
# ---------------------------------------------------------------------------
#
# Why a store at all: Binance charges per *request*, not per candle, so caching
# saves roughly zero requests on a weekly cron. It buys two other things —
#
#   1. `--offline` replays a scan with no network at all, which is what makes
#      threshold tuning (`--overbought 88`, `--lookback-days 7`, ...) instant
#      instead of a 60-second refetch every time.
#   2. History accumulates past Binance's 1000-candle wall, so the data to
#      backtest a threshold actually exists after a few months of running.
#
# The invariant that makes the store trustworthy: **only closed candles are ever
# written**, enforced in `upsert_candles` rather than trusted from the caller.
# A persisted half-formed bar would never be corrected, and every RSI computed
# from that series afterwards would be quietly wrong.

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    ts        INTEGER NOT NULL,          -- bar OPEN time, epoch ms, UTC
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, timeframe, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS symbols (
    symbol       TEXT PRIMARY KEY,
    quote_volume REAL,                   -- last seen 24h quote volume
    updated_ms   INTEGER
);

-- Binance base asset -> CoinGecko coin id. Authoritative (it comes from
-- CoinGecko's own Binance ticker listing), so collisions resolve correctly:
-- "ONE" is harmony here, not the larger-cap coin that shares the ticker.
-- A NULL coin_id records "looked, found nothing" so we stop re-crawling.
CREATE TABLE IF NOT EXISTS coin_map (
    base       TEXT PRIMARY KEY,
    coin_id    TEXT,
    updated_ms INTEGER
);

-- One row per crossing INTO extreme territory: the bar where RSI first went
-- >= overbought (or <= oversold) having not been there the bar before. The
-- UNIQUE constraint is the whole deduplication mechanism -- re-seeing the same
-- crossing on the next run is a no-op, so a signal alerts exactly once even if
-- a run crashes halfway through or fires twice.
CREATE TABLE IF NOT EXISTS signals (
    id         INTEGER PRIMARY KEY,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    side       TEXT NOT NULL,
    signal_ts  INTEGER NOT NULL,      -- open time of the crossing bar
    rsi        REAL,
    price      REAL,                  -- close of the crossing bar
    detected_ms INTEGER,              -- when this run first saw it
    alerted_ms  INTEGER,              -- NULL = never notified
    UNIQUE (symbol, timeframe, side, signal_ts)
);

-- What happened after. Raw returns only: whether a move counts as vindication
-- depends on a horizon and a threshold that belong to a trading view, not to a
-- schema, so the verdict is computed at report time and can be rescored later
-- without re-collecting anything.
CREATE TABLE IF NOT EXISTS outcomes (
    signal_id    INTEGER NOT NULL,
    horizon_bars INTEGER NOT NULL,
    price        REAL,
    return_pct   REAL,
    PRIMARY KEY (signal_id, horizon_bars)
);

CREATE TABLE IF NOT EXISTS marketcaps (
    coin_id       TEXT PRIMARY KEY,
    symbol TEXT, name TEXT,
    market_cap REAL, fully_diluted REAL,
    circulating REAL, total_supply REAL,
    updated_ms    INTEGER
);
"""

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
OVERLAP_BARS = 5          # bars refetched on top of the gap, to verify the splice
# Wilder's RMA is seeded from its first observation and that seed decays by
# (1 - 1/period) per bar. At ~100 bars past the window the residue is ~1e-3 of an
# RSI point, i.e. invisible at the precision we print. Below that a shallow cache
# genuinely disagrees with a live scan — measured at 0.18 RSI on 60 bars — so an
# offline replay refuses to compute rather than quietly report a different number.
MIN_WARMUP_BARS = 100


class CandleStore:
    """Closed OHLCV bars in SQLite. Deliberately boring: stdlib only."""

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")   # survives a killed cron job
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- candles ----------------------------------------------------------

    def last_ts(self, symbol: str, timeframe: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(ts) FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe)).fetchone()
        return row[0] if row and row[0] is not None else None

    def depth(self, symbol: str, timeframe: str) -> int:
        """How many bars we hold. The incremental fetch below needs this: a store
        that is merely *current* can still be too shallow for the window asked
        for, and must be backfilled rather than topped up."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe)).fetchone()[0]

    def upsert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame,
                       timeframe_ms: int, now_ms: int) -> int:
        """Write bars, replacing any the exchange has since revised.

        Refuses to store a bar that has not closed yet. This is the one rule the
        whole cache rests on, so it is checked here at the boundary instead of
        relying on every caller having remembered to trim.
        """
        if df.empty:
            return 0
        closed = df[df["timestamp"] + timeframe_ms <= now_ms]
        if len(closed) != len(df):
            log.debug("%s %s: refused to store %d unclosed bar(s)",
                      symbol, timeframe, len(df) - len(closed))
        rows = [(symbol, timeframe, int(r.timestamp), float(r.open), float(r.high),
                 float(r.low), float(r.close), float(r.volume))
                for r in closed.itertuples()]
        self.conn.executemany(
            """INSERT INTO candles (symbol, timeframe, ts, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
                   open=excluded.open, high=excluded.high, low=excluded.low,
                   close=excluded.close, volume=excluded.volume""", rows)
        self.conn.commit()
        return len(rows)

    def load(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """The most recent `limit` stored bars, oldest first."""
        rows = self.conn.execute(
            """SELECT ts, open, high, low, close, volume FROM candles
               WHERE symbol = ? AND timeframe = ?
               ORDER BY ts DESC LIMIT ?""", (symbol, timeframe, limit)).fetchall()
        df = pd.DataFrame(rows[::-1], columns=OHLCV_COLUMNS)
        if not df.empty:
            df["timestamp"] = df["timestamp"].astype("int64")
        return df

    # -- symbol metadata --------------------------------------------------

    def save_volumes(self, volumes: dict[str, float], now_ms: int) -> None:
        """Remember 24h volumes so `--offline` can still rank the top N."""
        self.conn.executemany(
            """INSERT INTO symbols (symbol, quote_volume, updated_ms) VALUES (?, ?, ?)
               ON CONFLICT(symbol) DO UPDATE SET
                   quote_volume=excluded.quote_volume, updated_ms=excluded.updated_ms""",
            [(s, float(v), now_ms) for s, v in volumes.items()])
        self.conn.commit()

    def top_symbols(self, top_n: int, quote: str,
                    include_leveraged: bool = False) -> dict[str, float]:
        """Offline equivalent of `fetch_top_symbols`, from the last online run."""
        rows = self.conn.execute(
            """SELECT symbol, quote_volume FROM symbols
               WHERE symbol LIKE ? ORDER BY quote_volume DESC""",
            (f"%/{quote}",)).fetchall()
        ranked = {}
        for symbol, volume in rows:
            base = symbol.split("/")[0]
            if not include_leveraged and base.endswith(LEVERAGED_SUFFIXES):
                continue
            ranked[symbol] = volume or 0.0
            if len(ranked) == top_n:
                break
        return ranked

    def volumes_for(self, symbols: list[str]) -> dict[str, float]:
        """Last known volumes for specific symbols (offline explicit-symbol runs)."""
        if not symbols:
            return {}
        holes = ",".join("?" * len(symbols))
        rows = self.conn.execute(
            f"SELECT symbol, quote_volume FROM symbols WHERE symbol IN ({holes})",
            symbols).fetchall()
        return {symbol: volume or 0.0 for symbol, volume in rows}

    # -- signals and outcomes ---------------------------------------------

    def record_signals(self, signals: list["Signal"], now_ms: int,
                       alert_max_age_bars: int) -> int:
        """Store crossings, suppressing alerts for ones that are already old.

        A cold database scanning 30 days finds hundreds of crossings. They are
        all worth recording -- they seed the outcome history -- but notifying
        about a signal from three weeks ago is noise, so anything older than
        `alert_max_age_bars` is written pre-marked as alerted. Returns how many
        rows were new; re-seeing a crossing is a no-op.
        """
        inserted = 0
        for sig in signals:
            backfill = sig.bars_ago > alert_max_age_bars
            cursor = self.conn.execute(
                """INSERT OR IGNORE INTO signals
                   (symbol, timeframe, side, signal_ts, rsi, price,
                    detected_ms, alerted_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (sig.symbol, sig.timeframe, sig.side, sig.signal_ts, sig.rsi,
                 sig.price, now_ms, now_ms if backfill else None))
            inserted += cursor.rowcount
        self.conn.commit()
        return inserted

    def unalerted_signals(self) -> list[tuple[int, "Signal"]]:
        """Signals recorded but never notified.

        Separate from recording so a failed send is retried next run rather than
        lost: nothing is marked alerted until it actually went out.
        """
        rows = self.conn.execute(
            """SELECT id, symbol, timeframe, side, signal_ts, rsi, price
               FROM signals WHERE alerted_ms IS NULL
               ORDER BY signal_ts""").fetchall()
        return [(r[0], Signal(symbol=r[1], timeframe=r[2], side=r[3],
                              signal_ts=r[4], rsi=r[5], price=r[6])) for r in rows]

    def mark_alerted(self, ids: list[int], now_ms: int) -> None:
        self.conn.executemany("UPDATE signals SET alerted_ms = ? WHERE id = ?",
                              [(now_ms, i) for i in ids])
        self.conn.commit()

    def evaluate_outcomes(self, steps: dict[str, int],
                          horizons: "Sequence[int]") -> int:
        """Score past signals against candles already in the store.

        Costs nothing extra: the bars needed to judge a signal from last week are
        the same bars this week's scan just fetched. Only pairs that have a bar
        at exactly the target offset are scored, so a signal still inside its
        horizon is simply skipped and picked up on a later run.
        """
        written = 0
        for timeframe, step_ms in steps.items():
            for horizon in horizons:
                rows = self.conn.execute(
                    """SELECT s.id, s.price, c.close
                       FROM signals s
                       JOIN candles c
                         ON c.symbol = s.symbol AND c.timeframe = s.timeframe
                        AND c.ts = s.signal_ts + ? * ?
                      WHERE s.timeframe = ?
                        AND s.price > 0
                        AND NOT EXISTS (SELECT 1 FROM outcomes o
                                        WHERE o.signal_id = s.id
                                          AND o.horizon_bars = ?)""",
                    (horizon, step_ms, timeframe, horizon)).fetchall()
                self.conn.executemany(
                    """INSERT OR IGNORE INTO outcomes
                       (signal_id, horizon_bars, price, return_pct)
                       VALUES (?, ?, ?, ?)""",
                    [(sid, horizon, close, 100.0 * (close - price) / price)
                     for sid, price, close in rows])
                written += len(rows)
        self.conn.commit()
        return written

    def signal_stats(self, favorable_move: float = 0.0) -> list[dict]:
        """Hit rate per (timeframe, side, horizon), scored at report time.

        `favorable_move` is the move required to call a signal vindicated, in
        percent: overbought wants price *down* by at least that much, oversold
        wants it up. Scoring here rather than at collection time means changing
        your mind rescores history instead of needing to re-collect it.
        """
        rows = self.conn.execute(
            """SELECT s.timeframe, s.side, o.horizon_bars,
                      COUNT(*) AS n,
                      SUM(CASE WHEN (s.side = 'OVERBOUGHT' AND o.return_pct <= -?)
                                 OR (s.side = 'OVERSOLD'   AND o.return_pct >=  ?)
                               THEN 1 ELSE 0 END) AS favorable,
                      AVG(o.return_pct) AS avg_return
                 FROM outcomes o JOIN signals s ON s.id = o.signal_id
                GROUP BY s.timeframe, s.side, o.horizon_bars
                ORDER BY s.timeframe, s.side, o.horizon_bars""",
            (favorable_move, favorable_move)).fetchall()
        return [{"timeframe": r[0], "side": r[1], "horizon_bars": r[2],
                 "n": r[3], "favorable": r[4], "avg_return": r[5],
                 "hit_rate": r[4] / r[3] if r[3] else 0.0} for r in rows]

    def signal_counts(self) -> tuple[int, int]:
        total = self.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        scored = self.conn.execute(
            "SELECT COUNT(DISTINCT signal_id) FROM outcomes").fetchone()[0]
        return total, scored

    # -- market cap -------------------------------------------------------

    def load_coin_map(self, bases: list[str], max_age_ms: int,
                      now_ms: int) -> dict[str, str | None]:
        """Cached base -> coin id, dropping entries older than `max_age_ms`."""
        if not bases:
            return {}
        holes = ",".join("?" * len(bases))
        rows = self.conn.execute(
            f"""SELECT base, coin_id FROM coin_map
                WHERE base IN ({holes}) AND updated_ms >= ?""",
            [*bases, now_ms - max_age_ms]).fetchall()
        return {base: coin_id for base, coin_id in rows}

    def save_coin_map(self, mapping: dict[str, str | None], now_ms: int) -> None:
        self.conn.executemany(
            """INSERT INTO coin_map (base, coin_id, updated_ms) VALUES (?, ?, ?)
               ON CONFLICT(base) DO UPDATE SET
                   coin_id=excluded.coin_id, updated_ms=excluded.updated_ms""",
            [(base, coin_id, now_ms) for base, coin_id in mapping.items()])
        self.conn.commit()

    def load_caps(self, coin_ids: list[str], max_age_ms: int,
                  now_ms: int) -> dict[str, "CoinInfo"]:
        if not coin_ids:
            return {}
        holes = ",".join("?" * len(coin_ids))
        rows = self.conn.execute(
            f"""SELECT coin_id, symbol, name, market_cap, fully_diluted,
                       circulating, total_supply FROM marketcaps
                WHERE coin_id IN ({holes}) AND updated_ms >= ?""",
            [*coin_ids, now_ms - max_age_ms]).fetchall()
        return {r[0]: CoinInfo(*r) for r in rows}

    def save_caps(self, infos: dict[str, "CoinInfo"], now_ms: int) -> None:
        self.conn.executemany(
            """INSERT INTO marketcaps (coin_id, symbol, name, market_cap,
                   fully_diluted, circulating, total_supply, updated_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(coin_id) DO UPDATE SET
                   symbol=excluded.symbol, name=excluded.name,
                   market_cap=excluded.market_cap,
                   fully_diluted=excluded.fully_diluted,
                   circulating=excluded.circulating,
                   total_supply=excluded.total_supply,
                   updated_ms=excluded.updated_ms""",
            [(i.coin_id, i.symbol, i.name, i.market_cap, i.fully_diluted,
              i.circulating, i.total_supply, now_ms) for i in infos.values()])
        self.conn.commit()

    def stats(self) -> str:
        candles, symbols = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM candles").fetchone()
        size = os.path.getsize(self.path) / 1e6 if os.path.exists(self.path) else 0
        return f"{self.path}: {candles:,} candles, {symbols} symbols, {size:.1f} MB"


def contiguous_tail(df: pd.DataFrame, timeframe_ms: int, symbol: str = "") -> pd.DataFrame:
    """Longest run of evenly spaced bars at the end of `df`.

    RSI over a series with a hole in it is not "slightly off", it is wrong — and
    plausibly wrong, which is worse. Rather than trusting that the store is
    gap-free, compute on the contiguous tail and say so when trimming happened.
    Gaps should be rare (a trading halt, or a store written by an older version);
    the incremental fetch below always overlaps what it already has.
    """
    if len(df) < 2:
        return df
    steps = df["timestamp"].diff().to_numpy()
    start = 0
    for i in range(len(df) - 1, 0, -1):
        if steps[i] != timeframe_ms:
            start = i
            break
    if start:
        log.warning("%s: gap in history, using the last %d of %d bars",
                    symbol, len(df) - start, len(df))
    return df.iloc[start:].reset_index(drop=True)


def get_candles(exchange: ccxt.Exchange, store: CandleStore | None, symbol: str,
                timeframe: str, limit: int, offline: bool = False,
                refresh: bool = False, min_bars: int = 0) -> pd.DataFrame:
    """Closed, contiguous candles for one symbol — from cache, network, or both.

    `min_bars` applies to offline replays only. Online, however much history the
    exchange has is by definition all there is, so a young listing is scanned
    with what exists. Offline, a short series means the *cache* is thin rather
    than the market being young, and computing anyway would quietly report
    numbers that a live run would not reproduce.
    """
    step_ms = timeframe_seconds(exchange, timeframe) * 1000

    if store is None:                       # --no-cache: the original stateless path
        return fetch_closed_ohlcv(exchange, symbol, timeframe, limit)

    if offline:
        df = store.load(symbol, timeframe, limit)
        if df.empty:
            raise ValueError("nothing cached (run once without --offline first)")
        df = contiguous_tail(df, step_ms, symbol)
        if len(df) < min_bars:
            raise ValueError(
                f"cached history too shallow ({len(df)} bars, need {min_bars}); "
                f"run once without --offline to fill it")
        return df

    # Ask only for what we are missing, plus an overlap. Same one request either
    # way, but a much smaller response — and the overlap lets us notice if the
    # exchange has revised bars we already stored.
    fetch_limit = limit
    last = store.last_ts(symbol, timeframe)
    deep_enough = store.depth(symbol, timeframe) >= limit
    if last is not None and deep_enough and not refresh:
        gap_bars = (exchange.milliseconds() - last) // step_ms
        if 0 <= gap_bars <= limit - OVERLAP_BARS:
            fetch_limit = int(gap_bars) + OVERLAP_BARS

    fresh = fetch_closed_ohlcv(exchange, symbol, timeframe, fetch_limit)
    store.upsert_candles(symbol, timeframe, fresh, step_ms, exchange.milliseconds())

    df = store.load(symbol, timeframe, limit)
    return contiguous_tail(df, step_ms, symbol)


def min_offline_bars(window_bars: int, period: int, limit: int) -> int:
    """Shortest cached series an offline replay will accept for one symbol."""
    return min(limit, window_bars + period + MIN_WARMUP_BARS)


# ---------------------------------------------------------------------------
# Market cap (CoinGecko)
# ---------------------------------------------------------------------------
#
# An exchange has no idea what a market cap is — it knows price and volume, and
# market cap needs circulating supply, which only an aggregator tracks. So this
# is a second data source, and it is opt-in: nothing here runs unless you pass
# --min-market-cap or --show-mcap, which keeps a plain scan free of a second
# point of failure.
#
# What the number is good for: market cap is a reasonable proxy for how hard a
# token is to push around. A $20M float moves on one actor's size. That is a
# liquidity judgement and it is the honest use of this filter.
#
# What it is NOT good for: spotting junk. A new listing's market cap is chosen,
# not earned — pick the supply, let the listing pop, and you have a "$100M coin"
# on day four with no history behind the number. Both of the obviously-sketchy
# pairs in a recent top-50 run (4 and 9 days old) had caps above $100M, while
# Civic (2017) and Travala (2017) sat near $20-30M. Size, not legitimacy.

COINGECKO_API = "https://api.coingecko.com/api/v3"
COINGECKO_PAUSE = 8.0        # seconds between keyless calls; the free tier 429s fast
COINGECKO_PAUSE_KEYED = 2.0  # with COINGECKO_API_KEY set (demo tier)
COINGECKO_MAX_PAGES = 15     # Binance is ~1400 pairs at 100 per page
COIN_MAP_TTL_MS = 30 * 86_400_000    # tickers rarely change hands
MARKET_CAP_TTL_MS = 24 * 3_600_000   # caps move daily, and we scan weekly
MARKETS_BATCH = 250          # ids per /coins/markets call


@dataclass
class CoinInfo:
    """One coin's size, as reported by CoinGecko."""
    coin_id: str
    symbol: str
    name: str
    market_cap: float
    fully_diluted: float
    circulating: float
    total_supply: float

    @property
    def float_pct(self) -> float:
        """Circulating as a share of total supply.

        The genuinely useful column here: a low float means most of the supply
        has yet to unlock, which price history cannot show you.
        """
        return 100.0 * self.circulating / self.total_supply if self.total_supply else 0.0


class CoinGecko:
    """Minimal CoinGecko client. Stdlib only, and patient by default."""

    def __init__(self, api_key: str | None = None, pause: float | None = None):
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.pause = pause if pause is not None else (
            COINGECKO_PAUSE_KEYED if self.api_key else COINGECKO_PAUSE)
        self._last_call = 0.0

    def _get(self, path: str, params: dict[str, str], tries: int = 3):
        """GET with self-throttling and backoff. Raises after `tries` failures."""
        url = f"{COINGECKO_API}{path}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "rsi-alert/0.1", "Accept": "application/json"}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        for attempt in range(tries):
            wait = self.pause - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == tries - 1:
                    raise
                # The free tier throttles aggressively; back off hard.
                backoff = 30 * (attempt + 1)
                log.warning("coingecko rate limited, retrying in %ds", backoff)
                time.sleep(backoff)
        raise RuntimeError("unreachable")

    def binance_map(self, needed: set[str]) -> dict[str, str | None]:
        """Resolve Binance base assets to CoinGecko ids, authoritatively.

        Uses CoinGecko's own Binance ticker listing rather than matching on the
        ticker string, because tickers collide: "ONE" is both Harmony and a
        larger-cap unrelated coin, and guessing by size picks the wrong one.
        Stops as soon as everything asked for is resolved.
        """
        resolved: dict[str, str | None] = {}
        outstanding = set(needed)
        for page in range(1, COINGECKO_MAX_PAGES + 1):
            if not outstanding:
                break
            payload = self._get("/exchanges/binance/tickers", {"page": str(page)})
            tickers = payload.get("tickers") or []
            if not tickers:
                break
            for ticker in tickers:
                base = ticker.get("base")
                if base in outstanding and ticker.get("coin_id"):
                    resolved[base] = ticker["coin_id"]
                    outstanding.discard(base)
        # Remember the misses too, so the next run does not crawl for them again.
        for base in outstanding:
            resolved[base] = None
        if outstanding:
            log.info("coingecko: %d base(s) unmapped: %s", len(outstanding),
                     ", ".join(sorted(outstanding)[:10]))
        return resolved

    def markets(self, coin_ids: list[str]) -> dict[str, CoinInfo]:
        """Market caps for specific coin ids (not a top-N list).

        Fetching by id rather than by rank matters: CoinGecko leaves
        `market_cap_rank` empty for wrapped assets, so WBTC looks unranked
        despite a ~$10B cap. Rank is not a usable filter; the cap value is.
        """
        out: dict[str, CoinInfo] = {}
        for start in range(0, len(coin_ids), MARKETS_BATCH):
            batch = coin_ids[start:start + MARKETS_BATCH]
            rows = self._get("/coins/markets", {
                "vs_currency": "usd", "ids": ",".join(batch),
                "per_page": str(len(batch)), "page": "1"})
            for row in rows:
                out[row["id"]] = CoinInfo(
                    coin_id=row["id"],
                    symbol=(row.get("symbol") or "").upper(),
                    name=row.get("name") or "",
                    market_cap=float(row.get("market_cap") or 0.0),
                    fully_diluted=float(row.get("fully_diluted_valuation") or 0.0),
                    circulating=float(row.get("circulating_supply") or 0.0),
                    total_supply=float(row.get("total_supply") or 0.0))
        return out


def market_info(store: CandleStore | None, symbols: list[str], offline: bool = False,
                refresh: bool = False, client: CoinGecko | None = None,
                now_ms: int | None = None) -> dict[str, CoinInfo]:
    """{symbol: CoinInfo} for as many symbols as can be resolved.

    Cached in the same SQLite store as the candles: the base -> id mapping for a
    month, the caps for a day. An offline run uses whatever is cached and asks
    for nothing.
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    bases = {symbol.split("/")[0]: symbol for symbol in symbols}

    # Offline means "do not call the network", not "pretend you know nothing".
    # Expiring cached caps here while happily using cached candles would leave an
    # offline run with --min-market-cap excluding every symbol, which looks
    # exactly like a market with no large caps in it.
    cap_ttl = now_ms if offline else MARKET_CAP_TTL_MS
    map_ttl = now_ms if offline else COIN_MAP_TTL_MS

    if store is None:
        mapping, caps = {}, {}
    else:
        mapping = {} if refresh else store.load_coin_map(list(bases), map_ttl, now_ms)
        known = [cid for cid in mapping.values() if cid]
        caps = {} if refresh else store.load_caps(known, cap_ttl, now_ms)
        if offline and caps:
            age = store.conn.execute(
                "SELECT MIN(updated_ms) FROM marketcaps").fetchone()[0]
            if age and now_ms - age > MARKET_CAP_TTL_MS:
                log.warning("offline: market caps are %.1f days old",
                            (now_ms - age) / 86_400_000)

    if not offline:
        client = client or CoinGecko()
        missing_bases = set(bases) - set(mapping)
        if missing_bases:
            log.info("coingecko: resolving %d base asset(s)", len(missing_bases))
            found = client.binance_map(missing_bases)
            mapping.update(found)
            if store:
                store.save_coin_map(found, now_ms)

        wanted = sorted({cid for cid in mapping.values() if cid} - set(caps))
        if wanted:
            log.info("coingecko: fetching market caps for %d coin(s)", len(wanted))
            fetched = client.markets(wanted)
            caps.update(fetched)
            if store:
                store.save_caps(fetched, now_ms)

    return {symbol: caps[mapping[base]]
            for base, symbol in bases.items()
            if mapping.get(base) and mapping[base] in caps}


def parse_money(text: str) -> float:
    """Accept 50M / 1.5B / 2.5e7 / 50000000 so the flag reads like a number."""
    cleaned = str(text).strip().replace(",", "").replace("$", "").upper()
    scale = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    if cleaned and cleaned[-1] in scale:
        return float(cleaned[:-1]) * scale[cleaned[-1]]
    return float(cleaned)


def filter_by_market_cap(symbols: list[str], info: dict[str, CoinInfo],
                         min_cap: float, keep_unknown: bool = False
                         ) -> tuple[list[str], dict[str, str]]:
    """Split the universe on market cap. Returns (kept, {symbol: why dropped}).

    Applied before scanning, so an excluded symbol costs no requests at all.
    """
    kept, dropped = [], {}
    for symbol in symbols:
        coin = info.get(symbol)
        if coin is None:
            if keep_unknown:
                kept.append(symbol)
            else:
                dropped[symbol] = "no market cap data"
            continue
        if coin.market_cap < min_cap:
            dropped[symbol] = f"mcap ${coin.market_cap:,.0f}"
        else:
            kept.append(symbol)
    return kept, dropped


def format_money(value: float) -> str:
    """Compact money for a table cell: 1.7T, 32.1B, 860M, 19M, -."""
    if not value:
        return "-"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= scale:
            return f"${value / scale:.3g}{suffix}"
    return f"${value:.0f}"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    """One flagged (symbol, timeframe, side) combination."""
    timeframe: str
    symbol: str
    side: str              # "OVERBOUGHT" or "OVERSOLD"
    rsi_now: float         # RSI of the latest CLOSED candle
    rsi_extreme: float     # most extreme RSI reached inside the lookback window
    bars_ago: int          # 0 = the latest closed candle
    extreme_time: str      # UTC timestamp of the extreme bar
    price: float           # close of the latest closed candle
    quote_volume_24h: float
    market_cap: float = 0.0        # 0 = not looked up, or no data
    float_pct: float = 0.0         # circulating / total supply

    def sort_key(self) -> tuple[float, int]:
        """Most extreme first; ties broken by most recent."""
        return (-abs(self.rsi_extreme - 50.0), self.bars_ago)


def hits_from_rsi(rsi: "Sequence[float | None]", symbol: str, timeframe: str,
                  window_bars: int, last_ts: int, step_ms: int,
                  overbought: float = OVERBOUGHT, oversold: float = OVERSOLD,
                  price: float = 0.0, quote_volume: float = 0.0) -> list[Hit]:
    """Find extremes in an RSI series. The single source of truth for flagging.

    Deliberately takes a plain list rather than a DataFrame: the browser
    dashboard reimplements exactly this function in JavaScript, and a pure
    list-in/rows-out shape is what makes the two testable against each other.
    Everything with a chance of silently diverging lives here — tie-breaking,
    whether the thresholds are inclusive, how `bars_ago` is counted, and the
    fact that one symbol can produce two rows.

    `rsi[-1]` is the latest CLOSED bar, which opened at `last_ts`. Entries may be
    None (RSI is undefined for a series that has never moved); they are skipped
    without shifting anyone's position, so `bars_ago` stays honest.
    """
    if not rsi:
        raise ValueError("empty RSI series")

    last_index = len(rsi) - 1
    start = max(0, len(rsi) - window_bars)
    # (absolute index, value) for every defined point inside the window.
    window = [(i, v) for i, v in enumerate(rsi) if i >= start and v is not None]
    if not window:
        raise ValueError("no RSI values in the window")

    rsi_now = rsi[last_index]
    if rsi_now is None:
        raise ValueError("latest RSI is undefined (flat or incomplete series)")

    hits: list[Hit] = []
    for side in ("OVERBOUGHT", "OVERSOLD"):
        # `>` and `<` (never `>=`/`<=`) so ties keep the FIRST occurrence, which
        # is what pandas' idxmax/idxmin do. Taking the last would silently report
        # a different bars_ago whenever a level is tagged twice.
        best_index, best = window[0]
        for index, value in window[1:]:
            if (value > best) if side == "OVERBOUGHT" else (value < best):
                best_index, best = index, value

        triggered = best >= overbought if side == "OVERBOUGHT" else best <= oversold
        if not triggered:
            continue

        bars_ago = last_index - best_index
        hits.append(Hit(
            timeframe=timeframe,
            symbol=symbol,
            side=side,
            rsi_now=float(rsi_now),
            rsi_extreme=float(best),
            bars_ago=bars_ago,
            extreme_time=datetime.fromtimestamp(
                (last_ts - bars_ago * step_ms) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            price=price,
            quote_volume_24h=quote_volume,
        ))
    return hits


@dataclass
class Signal:
    """A crossing into extreme territory. One episode, one signal, one alert."""
    symbol: str
    timeframe: str
    side: str              # "OVERBOUGHT" or "OVERSOLD"
    signal_ts: int         # open time of the crossing bar
    rsi: float
    price: float
    bars_ago: int = 0      # relative to the latest closed bar, at detection time


def detect_crossings(rsi: "Sequence[float | None]", symbol: str, timeframe: str,
                     last_ts: int, step_ms: int, closes: "Sequence[float]",
                     window_bars: int, overbought: float = OVERBOUGHT,
                     oversold: float = OVERSOLD) -> list[Signal]:
    """Bars where RSI *entered* extreme territory.

    Different question from `hits_from_rsi`, and the difference is the whole
    point of alerting. "Most extreme RSI in the last 30 days" is the right answer
    for a weekly watchlist and the wrong one for notifications: a single spike
    would re-fire every four hours for a month. A crossing -- at or beyond the
    threshold when the previous defined bar was not -- fires once per episode.

    A bar with no previous defined value (start of the series) cannot be a
    crossing: we have no idea whether it had just arrived or had been sitting
    there for a week.
    """
    signals: list[Signal] = []
    last_index = len(rsi) - 1
    start = max(1, len(rsi) - window_bars)

    for i in range(start, len(rsi)):
        value = rsi[i]
        if value is None:
            continue
        # Nearest earlier defined value, so a gap of undefined bars does not
        # manufacture a crossing out of nothing.
        previous = next((rsi[j] for j in range(i - 1, -1, -1) if rsi[j] is not None),
                        None)
        if previous is None:
            continue

        for side, threshold, entered in (
            ("OVERBOUGHT", overbought, value >= overbought and previous < overbought),
            ("OVERSOLD", oversold, value <= oversold and previous > oversold),
        ):
            if not entered:
                continue
            bars_ago = last_index - i
            signals.append(Signal(
                symbol=symbol, timeframe=timeframe, side=side,
                signal_ts=last_ts - bars_ago * step_ms,
                rsi=float(value),
                price=float(closes[i]) if i < len(closes) else 0.0,
                bars_ago=bars_ago))
    return signals


def rsi_series(df: pd.DataFrame, period: int = RSI_PERIOD) -> list[float | None]:
    """RSI over a candle frame as a plain list, NaN rendered as None."""
    values = wilder_rsi(df["close"].reset_index(drop=True), period)
    return [None if pd.isna(v) else float(v) for v in values]


def find_hits(df: pd.DataFrame, symbol: str, timeframe: str, window_bars: int,
              period: int = RSI_PERIOD, overbought: float = OVERBOUGHT,
              oversold: float = OVERSOLD, quote_volume: float = 0.0) -> list[Hit]:
    """Compute RSI over `df` and return a Hit per extreme touched in the window.

    A thin wrapper over `hits_from_rsi`: candles in, flags out. Timestamps are
    derived arithmetically from the last bar and the bar length, which is exact
    because every frame reaching here has been through `contiguous_tail`.
    """
    if len(df) < 2:
        raise ValueError(f"only {len(df)} candle(s)")
    series = rsi_series(df, period)
    step_ms = int(df["timestamp"].iloc[-1]) - int(df["timestamp"].iloc[-2])
    return hits_from_rsi(series, symbol, timeframe, window_bars,
                         last_ts=int(df["timestamp"].iloc[-1]), step_ms=step_ms,
                         overbought=overbought, oversold=oversold,
                         price=float(df["close"].iloc[-1]),
                         quote_volume=quote_volume)


@dataclass
class ScanHealth:
    """Coverage bookkeeping for one run. A (symbol, timeframe) pair is one task."""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    flagged: int = 0
    # Which tasks failed, so the matrix can distinguish "scanned, stayed calm"
    # from "never got a look at it".
    failed_tasks: set[tuple[str, str]] = field(default_factory=set)
    # Newest closed bar seen per timeframe. This is the "good late" marker: the
    # scan is deliberately behind live price by up to one bar.
    data_through: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


def scan(exchange: ccxt.Exchange, symbols: list[str], timeframes: list[str],
         args: argparse.Namespace, volumes: dict[str, float] | None = None,
         store: "CandleStore | None" = None, offline: bool = False,
         refresh: bool = False,
         series_out: list[dict] | None = None,
         signals_out: list[Signal] | None = None) -> tuple[list[Hit], ScanHealth]:
    """Scan every (symbol, timeframe) pair. Never raises for a single symbol.

    Pass `series_out` to also collect the RSI series of every symbol that
    scanned cleanly, flagged or not. The dashboard needs all of them: its
    threshold and window controls have to be able to widen into symbols this
    run did not flag, which a list of hits cannot provide.
    """
    volumes = volumes or {}
    hits: list[Hit] = []
    health = ScanHealth(total=len(symbols) * len(timeframes))

    for timeframe in timeframes:
        limit = candles_needed(exchange, timeframe, args.lookback_days, args.rsi_period)
        window = lookback_bars(exchange, timeframe, args.lookback_days)
        min_bars = min_offline_bars(window, args.rsi_period, limit) if offline else 0
        log.info("scanning %d symbols on %s (%d candles/symbol, %d-bar window)",
                 len(symbols), timeframe, limit, window)

        for symbol in symbols:
            try:
                df = get_candles(exchange, store, symbol, timeframe, limit,
                                 offline=offline, refresh=refresh,
                                 min_bars=min_bars)
                if len(df) < args.rsi_period + 1:
                    raise ValueError(f"only {len(df)} closed candles")
                found = find_hits(df, symbol, timeframe, window, args.rsi_period,
                                  args.overbought, args.oversold,
                                  volumes.get(symbol, 0.0))
                step_ms = int(df["timestamp"].iloc[-1]) - int(df["timestamp"].iloc[-2])
                if signals_out is not None:
                    # Crossings, not window extremes: see detect_crossings().
                    signals_out.extend(detect_crossings(
                        rsi_series(df, args.rsi_period), symbol, timeframe,
                        last_ts=int(df["timestamp"].iloc[-1]), step_ms=step_ms,
                        closes=df["close"].tolist(), window_bars=window,
                        overbought=args.overbought, oversold=args.oversold))
                if series_out is not None:
                    series_out.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "last_ts": int(df["timestamp"].iloc[-1]),
                        "step_ms": step_ms,
                        # Only the window travels: warmup bars exist to make RSI
                        # converge, and shipping them would triple the payload.
                        "rsi": [None if v is None else round(v, JSON_RSI_DECIMALS)
                                for v in rsi_series(df, args.rsi_period)[-window:]],
                        "price": float(df["close"].iloc[-1]),
                        "quote_volume_24h": float(volumes.get(symbol, 0.0)),
                    })
                health.succeeded += 1
                health.flagged += len(found)
                hits.extend(found)
                last_ts = int(df["timestamp"].iloc[-1])
                health.data_through[timeframe] = max(
                    health.data_through.get(timeframe, 0), last_ts)
            except Exception as exc:                    # noqa: BLE001 - by design
                # Delisted pair, brand-new listing, transient network blip: log it
                # and keep going. One bad symbol must not abort the weekly run.
                health.failed += 1
                health.failed_tasks.add((symbol, timeframe))
                log.warning("skip %s %s: %s: %s", symbol, timeframe,
                            type(exc).__name__, exc)

    hits.sort(key=lambda hit: (timeframes.index(hit.timeframe),) + hit.sort_key())
    return hits, health


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

COLUMNS = [
    ("symbol", "SYMBOL", "<", "{}"),
    ("side", "SIDE", "<", "{}"),
    ("rsi_now", "RSI NOW", ">", "{:.1f}"),
    ("rsi_extreme", "EXTREME", ">", "{:.1f}"),
    ("bars_ago", "BARS AGO", ">", "{}"),
    ("extreme_time", "WHEN (UTC)", "<", "{}"),
    ("price", "PRICE", ">", "{:.8g}"),
    ("quote_volume_24h", "24H VOL", ">", "{:,.0f}"),
]


def format_table(hits: list[Hit], timeframes: list[str]) -> str:
    """Aligned plain-text table, grouped by timeframe, most extreme first."""
    out: list[str] = []
    for timeframe in timeframes:
        group = [hit for hit in hits if hit.timeframe == timeframe]
        out.append("")
        out.append(f"=== {timeframe.upper()} — {len(group)} flagged ===")
        if not group:
            out.append("  (nothing at the extremes)")
            continue

        rows = [[fmt.format(getattr(hit, attr)) for attr, _h, _a, fmt in COLUMNS]
                for hit in group]
        headers = [header for _attr, header, _a, _f in COLUMNS]
        widths = [max(len(headers[i]), *(len(row[i]) for row in rows))
                  for i in range(len(COLUMNS))]

        def line(cells: list[str]) -> str:
            return "  " + "  ".join(
                cell.ljust(widths[i]) if COLUMNS[i][2] == "<" else cell.rjust(widths[i])
                for i, cell in enumerate(cells)
            ).rstrip()

        out.append(line(headers))
        out.append("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
        out.extend(line(row) for row in rows)
    return "\n".join(out)


# --- matrix view -----------------------------------------------------------
#
# One row per (token, side), one column per timeframe. This is the view you
# actually read: confluence across timeframes is the signal, and a token that is
# extreme on both 4H and 1D should not need a mental join across two tables.
#
# A (token, side) row rather than a (token) row because a pair can legitimately
# hit both extremes inside one window — BCH tagged 93 and then 20 last month —
# and a single cell cannot honestly show both.

CELL_NONE = "-"      # scanned, nothing at the extremes
CELL_FAIL = "?"      # never scanned: delisted, too few candles, network blip
SIDE_SHORT = {"OVERBOUGHT": "OB", "OVERSOLD": "OS"}


def _cell(hit: Hit | None, failed: bool) -> str:
    """Render one timeframe cell for one token."""
    if hit is None:
        return CELL_FAIL if failed else CELL_NONE
    if hit.bars_ago == 0:
        # The extreme *is* the current reading; printing both would be redundant.
        return f"{hit.rsi_extreme:.1f}  NOW"
    return f"{hit.rsi_extreme:.1f}  {hit.bars_ago}b ->{hit.rsi_now:.0f}"


def _ascii_table(headers: list[str], rows: list[list[str]],
                 aligns: list[str]) -> list[str]:
    """Bordered ASCII table. Plain +-| so it survives cron mail, logs and pipes."""
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows), 1)
              if rows else len(headers[i]) for i in range(len(headers))]
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(cells: list[str]) -> str:
        padded = [c.ljust(widths[i]) if aligns[i] == "<" else c.rjust(widths[i])
                  for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    out = [rule, line(headers), rule]
    out.extend(line(row) for row in rows)
    out.append(rule)
    return out


def format_matrix(hits: list[Hit], timeframes: list[str],
                  health: ScanHealth | None = None,
                  info: dict[str, "CoinInfo"] | None = None) -> str:
    """Token x timeframe matrix, freshest and most extreme at the top.

    `info` adds MCAP and FLOAT columns. FLOAT is the one worth reading: a low
    circulating share means most of the supply has yet to unlock, which nothing
    in the price history can tell you.
    """
    failed = health.failed_tasks if health else set()
    through = health.data_through if health else {}

    # Group hits into {(symbol, side): {timeframe: hit}}.
    grouped: dict[tuple[str, str], dict[str, Hit]] = {}
    for hit in hits:
        grouped.setdefault((hit.symbol, hit.side), {})[hit.timeframe] = hit

    # Drop the quote suffix (BTC/USDT -> BTC) only when every symbol shares one
    # quote, otherwise BTC/USDT and BTC/EUR would collapse into the same label.
    quotes = {symbol.split("/")[-1] for symbol, _side in grouped}
    strip_quote = len(quotes) == 1

    def row_key(item: tuple[tuple[str, str], dict[str, Hit]]) -> tuple:
        _key, by_timeframe = item
        fresh = any(hit.bars_ago == 0 for hit in by_timeframe.values())
        peak = max(abs(hit.rsi_extreme - 50) for hit in by_timeframe.values())
        return (0 if fresh else 1, -peak, _key[0])

    show_size = bool(info)
    rows = []
    for (symbol, side), by_timeframe in sorted(grouped.items(), key=row_key):
        label = symbol.split("/")[0] if strip_quote else symbol
        size: list[str] = []
        if show_size:
            coin = info.get(symbol)
            size = [format_money(coin.market_cap) if coin else "?",
                    f"{coin.float_pct:.0f}%" if coin and coin.total_supply else "-"]
        cells = [_cell(by_timeframe.get(tf), (symbol, tf) in failed)
                 for tf in timeframes]
        rows.append([label, SIDE_SHORT.get(side, side)] + size + cells)

    out: list[str] = [""]
    if not rows:
        out.append("No tokens at the extremes.")
    else:
        headers = (["TOKEN", "SIDE"] + (["MCAP", "FLOAT"] if show_size else [])
                   + [tf.upper() for tf in timeframes])
        aligns = (["<", "<"] + ([">", ">"] if show_size else [])
                  + ["<"] * len(timeframes))
        out.extend(_ascii_table(headers, rows, aligns))

    # Legend, because the cells are dense on purpose.
    out.append(f'  OB/OS = overbought / oversold   '
               f'"92.2  3b ->53" = hit 92.2 three bars ago, now 53')
    out.append(f'  "{CELL_NONE}" = scanned, no extreme   '
               f'"{CELL_FAIL}" = scan failed (see log)')
    if show_size:
        out.append('  FLOAT = circulating / total supply; a low share means '
                   'supply still to unlock')
    if through:
        # Say out loud how late the data is. Late is the point: these are closed
        # candles only, so nothing here can be walked back by an unfinished bar.
        stamps = "   ".join(
            f"{tf.upper()} {datetime.fromtimestamp(ts / 1000, tz=timezone.utc):%Y-%m-%d %H:%MZ}"
            for tf, ts in through.items())
        out.append(f"  closed candles through: {stamps}")
    return "\n".join(out)


def write_csv(hits: list[Hit], csv_dir: str, run_time: datetime) -> str:
    """Write every hit to a timestamped CSV. Always written, even when empty, so
    the file itself is evidence the run happened."""
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, f"rsi_extremes_{run_time.strftime('%Y%m%d_%H%M%S')}.csv")
    names = [f.name for f in fields(Hit)]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for hit in hits:
            writer.writerow([getattr(hit, name) for name in names])
    return path


def summary_line(health: ScanHealth, args: argparse.Namespace) -> str:
    """The heartbeat. Printed on every run, flagged or not."""
    return (
        f"RSI scan {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} | "
        f"tf={','.join(args.timeframes)} rsi={args.rsi_period} "
        f"thresholds={args.oversold:g}/{args.overbought:g} "
        f"lookback={args.lookback_days}d | "
        f"symbols={health.total // max(1, len(args.timeframes))} "
        f"ok={health.succeeded} failed={health.failed} "
        f"coverage={health.success_rate:.0%} | flagged={health.flagged}"
    )


# ---------------------------------------------------------------------------
# Telegram notifier (OPTIONAL — off by default)
# ---------------------------------------------------------------------------
#
# Enable with --telegram and export:
#     TELEGRAM_BOT_TOKEN=123456:ABC...      (from @BotFather)
#     TELEGRAM_CHAT_ID=-1001234567890       (your chat / channel id)
#
# Deliberately stdlib-only (urllib) so the notifier adds no dependency, and
# deliberately non-fatal: a Telegram outage must not fail the scan.

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_CHARS = 4000          # API hard limit is 4096


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None,
                  timeout: int = 15) -> bool:
    """Send `text` to Telegram. Returns True on success, False on any failure."""
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.error("telegram enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID unset")
        return False

    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text[:TELEGRAM_MAX_CHARS],
        "disable_web_page_preview": "true",
    }).encode()

    try:
        request = urllib.request.Request(TELEGRAM_API.format(token=token), data=payload)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
        if not body.get("ok"):
            log.error("telegram rejected the message: %s", body)
            return False
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.error("telegram send failed: %s: %s", type(exc).__name__, exc)
        return False


def telegram_message(hits: list[Hit], health: ScanHealth, args: argparse.Namespace,
                     max_rows: int = 25) -> str:
    """Compact summary for chat. Always includes the heartbeat line, so a healthy
    but quiet week still produces a message."""
    lines = [summary_line(health, args)]
    if health.success_rate < args.min_success_rate:
        lines.append(f"WARNING: coverage below {args.min_success_rate:.0%} — check the job")
    if not hits:
        lines.append("No RSI extremes in the window.")
    else:
        for hit in hits[:max_rows]:
            lines.append(
                f"{hit.timeframe} {hit.symbol} {hit.side} "
                f"extreme={hit.rsi_extreme:.1f} ({hit.bars_ago} bars ago) "
                f"now={hit.rsi_now:.1f}"
            )
        if len(hits) > max_rows:
            lines.append(f"... and {len(hits) - max_rows} more (see CSV)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal alerts and scoring
# ---------------------------------------------------------------------------

def signal_alert_message(signals: list[Signal]) -> str:
    """Telegram text for newly crossed signals."""
    lines = [f"RSI alert: {len(signals)} new signal(s)"]
    for sig in signals:
        when = datetime.fromtimestamp(sig.signal_ts / 1000, tz=timezone.utc)
        arrow = "^" if sig.side == "OVERBOUGHT" else "v"
        lines.append(f"{arrow} {sig.symbol} {sig.timeframe} {sig.side} "
                     f"RSI {sig.rsi:.1f} @ {sig.price:.8g} "
                     f"({when:%Y-%m-%d %H:%M}Z)")
    return "\n".join(lines)


def format_signal_report(stats: list[dict], total: int, scored: int,
                         favorable_move: float) -> str:
    """How the alerts have actually done.

    Read this as a sanity check, not a backtest -- the caveats printed under the
    table are not boilerplate, they are the reason a good-looking number here
    does not mean a good strategy.
    """
    out = ["", f"=== Signal outcomes ({scored} of {total} signals scored) ==="]
    if not stats:
        out.append("  No scored signals yet. Outcomes appear once enough bars")
        out.append("  have closed after a signal -- 20 bars on 1D is 20 days.")
        return "\n".join(out)

    rule = (f"price moves at least {favorable_move:g}% the right way"
            if favorable_move else "price moves the right way at all")
    headers = ["TF", "SIDE", "BARS", "N", "HIT RATE", "AVG RET"]
    rows = [[s["timeframe"], "OB" if s["side"] == "OVERBOUGHT" else "OS",
             str(s["horizon_bars"]), str(s["n"]),
             f"{s['hit_rate']:.0%}", f"{s['avg_return']:+.2f}%"] for s in stats]
    out.extend(_ascii_table(headers, rows, ["<", "<", ">", ">", ">", ">"]))
    out.append(f"  Vindicated = {rule}; overbought wants price down, oversold up.")
    out.append("  Caveats that matter: only symbols that were in the scanned")
    out.append("  universe at the time appear here, there are no fees or")
    out.append("  slippage, and a hit rate over a few dozen samples is noise.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# JSON document (input to the dashboard)
# ---------------------------------------------------------------------------

DOC_VERSION = 1

# Decimal places for RSI in the payload. Measured on a real 200-symbol scan:
# 2dp moved `bars_ago` on 1 hit in 693 (rounding created a tie that did not
# exist, and the first-occurrence rule then picked a different bar); 3dp and 4dp
# moved none. Full precision costs 3.9x the bytes for a guarantee that 4dp gets
# within a rounding error of -- a collision there needs two bars agreeing to
# 1e-4, at which point which bar "won" is arbitrary anyway.
JSON_RSI_DECIMALS = 4


def build_document(series: list[dict], health: ScanHealth, args: argparse.Namespace,
                   info: dict[str, CoinInfo] | None = None,
                   generated_ms: int | None = None) -> dict:
    """Everything the dashboard needs, in one JSON-safe dict.

    RSI arrives already rounded to `JSON_RSI_DECIMALS` (see the note there for
    why 4). The residual divergence from the CLI is bounded and tested by
    `test_json_rounding_stays_within_its_bound`.
    """
    info = info or {}
    rows = []
    for row in series:
        coin = info.get(row["symbol"])
        rows.append({**row,
                     "market_cap": float(coin.market_cap) if coin else 0.0,
                     "float_pct": float(coin.float_pct) if coin else 0.0})

    return {
        "doc_version": DOC_VERSION,
        "generated_ms": generated_ms if generated_ms is not None
                        else int(time.time() * 1000),
        "rsi_period": args.rsi_period,
        "timeframes": list(args.timeframes),
        # The slider can narrow below this but never widen past it: bars outside
        # the shipped window simply are not in the payload.
        "max_lookback_days": args.lookback_days,
        "defaults": {
            "overbought": args.overbought,
            "oversold": args.oversold,
            "lookback_days": args.lookback_days,
            "min_market_cap": args.min_market_cap or 0.0,
        },
        "health": {
            "total": health.total,
            "succeeded": health.succeeded,
            "failed": health.failed,
            "flagged": health.flagged,
            "coverage": round(health.success_rate, 4),
        },
        "rows": rows,
    }


HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_HTML = os.path.join(HERE, "dashboard.html")
TEMPLATE_JS = os.path.join(HERE, "dashboard.js")


def build_page(document: dict, path: str, template: str = TEMPLATE_HTML,
               logic: str = TEMPLATE_JS) -> str:
    """Inline the logic and the data into one self-contained HTML file.

    Self-contained on purpose: no fetch means no CORS, no second request, no
    half-loaded page, and it opens straight from disk for inspection before you
    ever deploy it.
    """
    html = open(template, encoding="utf-8").read()
    script = open(logic, encoding="utf-8").read()

    # The data rides inside a <script> tag, so "<" must not survive literally:
    # a "</script>" anywhere in the JSON would end the tag early and break the
    # page. \u003c parses back to "<" in JSON, so no information is lost.
    blob = json.dumps(document, allow_nan=False, separators=(",", ":")) \
        .replace("<", "\\u003c")

    for marker, value in (("__RSI_LOGIC__", script), ("__RSI_DATA__", blob)):
        if marker not in html:
            raise ValueError(f"{template}: missing {marker} placeholder")
        html = html.replace(marker, value)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path


def dump_json(document: dict, path: str) -> str:
    """Write the document as strictly valid JSON.

    `allow_nan=False` is the point: Python's default emits bare `NaN`, which is
    not JSON, and `JSON.parse` throws on it — a blank dashboard with an error
    only visible in the browser console. Fail here instead, loudly.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, allow_nan=False, separators=(",", ":"))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Binance spot for tokens at RSI extremes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES,
                        help="timeframes to scan, e.g. 4h 1d")
    parser.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    parser.add_argument("--overbought", type=float, default=OVERBOUGHT)
    parser.add_argument("--oversold", type=float, default=OVERSOLD)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS,
                        help="how far back an RSI extreme still counts")
    parser.add_argument("--top-n", type=int, default=TOP_N,
                        help="number of pairs by 24h quote volume (auto mode)")
    parser.add_argument("--quote", default=QUOTE, help="quote currency for auto mode")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS or None,
                        help="explicit symbols, e.g. BTC/USDT ETH/USDT (overrides --top-n)")
    parser.add_argument("--include-leveraged", action="store_true",
                        help="keep UP/DOWN/BULL/BEAR leveraged tokens in auto mode")
    parser.add_argument("--exchange", default=EXCHANGE_ID,
                        help="any ccxt exchange id; only tested against binance")
    parser.add_argument("--public-url", default=None,
                        help="override the public REST base, e.g. "
                             "https://data-api.binance.vision/api/v3 when the "
                             "main API is geoblocked (also $BINANCE_PUBLIC_URL)")
    parser.add_argument("--min-market-cap", default=None, metavar="SIZE",
                        help="skip coins below this market cap (50M, 1.5B, 5e7). "
                             "A size/manipulability floor, not a scam filter: a "
                             "new listing's cap is chosen, not earned")
    parser.add_argument("--show-mcap", action="store_true",
                        help="add MCAP and FLOAT columns without filtering")
    parser.add_argument("--keep-unknown-mcap", action="store_true",
                        help="keep coins CoinGecko has no data for (default: skip)")
    parser.add_argument("--cache-path", default=CACHE_PATH,
                        help="SQLite file of closed candles")
    parser.add_argument("--no-cache", action="store_true",
                        help="do not read or write the candle store")
    parser.add_argument("--offline", action="store_true",
                        help="replay from the cache with zero network calls")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore what is cached and refetch the full window")
    parser.add_argument("--format", choices=["matrix", "grouped"], default=OUTPUT_FORMAT,
                        help="matrix: token x timeframe; grouped: one table per "
                             "timeframe with price and volume")
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="write the scan document (every scanned symbol, "
                             "not just the flagged ones) as JSON")
    parser.add_argument("--html", metavar="PATH", default=None,
                        help="write the self-contained dashboard page")
    parser.add_argument("--csv-dir", default=CSV_DIR, help="where to write the CSV")
    parser.add_argument("--no-csv", action="store_true", help="skip the CSV file")
    parser.add_argument("--telegram", action="store_true",
                        help="send NEW signal alerts to Telegram (needs env vars)")
    parser.add_argument("--telegram-heartbeat", action="store_true",
                        help="also send the summary line; put this on one "
                             "scheduled run a day so silence still means broken")
    parser.add_argument("--alert-max-age-bars", type=int, default=ALERT_MAX_AGE_BARS,
                        help="crossings older than this are recorded but not "
                             "notified, so a cold database does not flood you")
    parser.add_argument("--report", action="store_true",
                        help="print how past signals actually turned out")
    parser.add_argument("--favorable-move", type=float, default=0.0, metavar="PCT",
                        help="percent move required to call a signal vindicated")
    parser.add_argument("--min-success-rate", type=float, default=MIN_SUCCESS_RATE,
                        help="warn + exit non-zero below this scan coverage")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    if args.overbought <= args.oversold:
        parser.error("--overbought must be greater than --oversold")
    if args.rsi_period < 2:
        parser.error("--rsi-period must be >= 2")
    if args.lookback_days < 1:
        parser.error("--lookback-days must be >= 1")
    if args.offline and args.no_cache:
        parser.error("--offline reads from the cache, so it cannot be used with --no-cache")
    if args.offline and args.refresh:
        parser.error("--refresh fetches, --offline does not; pick one")
    if args.min_market_cap is not None:
        try:
            args.min_market_cap = parse_money(args.min_market_cap)
        except ValueError:
            parser.error(f"--min-market-cap: cannot read {args.min_market_cap!r} "
                         f"as a size (try 50M, 1.5B or 5e7)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    run_time = datetime.now(timezone.utc)

    # --- symbol selection ---------------------------------------------------
    store: CandleStore | None = None
    try:
        exchange = build_exchange(args.exchange, args.public_url)
        if not args.no_cache:
            store = CandleStore(args.cache_path)
        volumes: dict[str, float] = {}

        if args.symbols:
            symbols = list(args.symbols)
            # Offline runs must make no network call at all, not even for volumes.
            if args.offline:
                volumes = store.volumes_for(symbols)
            else:
                exchange.load_markets()
                volumes = fetch_volumes(exchange, symbols)
            log.info("using %d explicit symbol(s)", len(symbols))
        elif args.offline:
            volumes = store.top_symbols(args.top_n, args.quote, args.include_leveraged)
            symbols = list(volumes)
            log.info("offline: top %d %s pairs ranked from the cache",
                     len(symbols), args.quote)
        else:
            volumes = fetch_top_symbols(exchange, args.top_n, args.quote,
                                        args.include_leveraged)
            symbols = list(volumes)
            log.info("selected top %d %s pairs by 24h quote volume",
                     len(symbols), args.quote)

        if store and volumes and not args.offline:
            store.save_volumes(volumes, exchange.milliseconds())
    except Exception as exc:                              # noqa: BLE001
        # Nothing to scan => nothing is salvageable. Fail loudly and distinctly.
        log.error("fatal: could not reach %s: %s: %s", args.exchange,
                  type(exc).__name__, exc)
        return EXIT_FATAL

    # --- market cap enrichment / filter (opt-in) ----------------------------
    info: dict[str, CoinInfo] = {}
    dropped: dict[str, str] = {}
    if symbols and (args.min_market_cap is not None or args.show_mcap):
        try:
            info = market_info(store, symbols, offline=args.offline,
                               refresh=args.refresh)
        except Exception as exc:                          # noqa: BLE001
            # CoinGecko is a convenience, not the job. If it is down or throttling,
            # say so and scan anyway rather than losing the week's run.
            log.error("market cap lookup failed (%s: %s)", type(exc).__name__, exc)
            if args.min_market_cap is not None:
                log.error("fatal: --min-market-cap cannot be honoured without it")
                return EXIT_FATAL

        if args.min_market_cap is not None:
            # Filter before scanning, so an excluded symbol costs no requests.
            symbols, dropped = filter_by_market_cap(
                symbols, info, args.min_market_cap, args.keep_unknown_mcap)
            log.info("market cap filter: kept %d, dropped %d (min %s)",
                     len(symbols), len(dropped), format_money(args.min_market_cap))
            for symbol, reason in sorted(dropped.items()):
                log.debug("  dropped %s: %s", symbol, reason)

    if not symbols:
        log.error("fatal: no symbols to scan%s",
                  " (cache is empty — run once without --offline)" if args.offline else "")
        return EXIT_FATAL

    # --- scan ---------------------------------------------------------------
    # The page needs every scanned symbol, so its controls can widen into ones
    # this run did not flag. Only collected when something will consume it.
    series: list[dict] | None = [] if (args.json or args.html) else None
    # Crossings are collected whenever there is somewhere to put them: recording
    # costs nothing and it is what builds the outcome history, so the question
    # "do these alerts actually work" has data behind it a month from now.
    crossings: list[Signal] | None = [] if store else None
    hits, health = scan(exchange, symbols, args.timeframes, args, volumes,
                        store=store, offline=args.offline, refresh=args.refresh,
                        series_out=series, signals_out=crossings)
    # --- signals: record, score, then alert ---------------------------------
    pending: list[tuple[int, Signal]] = []
    if store and crossings is not None:
        now_ms = int(time.time() * 1000)
        new_count = store.record_signals(crossings, now_ms, args.alert_max_age_bars)
        steps = {tf: timeframe_seconds(exchange, tf) * 1000 for tf in args.timeframes}
        scored = store.evaluate_outcomes(steps, SIGNAL_HORIZONS)
        pending = store.unalerted_signals()
        log.info("signals: %d new, %d outcome(s) scored, %d awaiting alert",
                 new_count, scored, len(pending))

    if store:
        log.info("cache %s", store.stats())

    # Attach size data to the flagged rows so it reaches the table and the CSV.
    for hit in hits:
        coin = info.get(hit.symbol)
        if coin:
            hit.market_cap, hit.float_pct = coin.market_cap, coin.float_pct

    # --- report -------------------------------------------------------------
    if args.format == "matrix":
        print(format_matrix(hits, args.timeframes, health, info or None))
    else:
        print(format_table(hits, args.timeframes))
    print()

    if series is not None:
        document = build_document(series, health, args, info)
        if args.json:
            log.info("json: %s (%d series)", dump_json(document, args.json),
                     len(document["rows"]))
        if args.html:
            log.info("page: %s", build_page(document, args.html))

    csv_path = None
    if not args.no_csv:
        csv_path = write_csv(hits, args.csv_dir, run_time)
        print(f"CSV: {csv_path}")

    if dropped:
        excluded = sum(1 for r in dropped.values() if r.startswith("mcap"))
        print(f"Market cap filter (min {format_money(args.min_market_cap)}): "
              f"{len(dropped)} symbols excluded before scanning "
              f"({excluded} too small, {len(dropped) - excluded} no data). "
              f"Run with --verbose to list them.")

    # Heartbeat: always printed, flagged or not, so silence is never ambiguous.
    print(summary_line(health, args))

    exit_code = EXIT_OK
    if health.success_rate < args.min_success_rate:
        # Loud on purpose — this is what a human greps cron mail for.
        print("=" * 72)
        print(f"WARNING: scan coverage {health.success_rate:.0%} is below the "
              f"{args.min_success_rate:.0%} threshold "
              f"({health.failed}/{health.total} tasks failed).")
        print("WARNING: results are incomplete — treat this run as unreliable.")
        print("=" * 72)
        exit_code = EXIT_LOW_COVERAGE

    if args.report and store:
        total, scored_n = store.signal_counts()
        print(format_signal_report(store.signal_stats(args.favorable_move),
                                   total, scored_n, args.favorable_move))

    if args.telegram:
        # Alerts first: these are the point. Nothing is marked as notified until
        # the send actually succeeded, so a Telegram outage retries next run
        # instead of silently swallowing a signal.
        if pending:
            if send_telegram(signal_alert_message([sig for _id, sig in pending])):
                store.mark_alerted([sid for sid, _sig in pending],
                                   int(time.time() * 1000))
                log.info("alerted %d new signal(s)", len(pending))
            else:
                log.error("alert NOT sent; will retry next run")
        if health.success_rate < args.min_success_rate:
            send_telegram(f"RSI scan degraded: coverage "
                          f"{health.success_rate:.0%} ({health.failed} of "
                          f"{health.total} tasks failed)")
        if args.telegram_heartbeat:
            send_telegram(telegram_message(hits, health, args))

    if store:
        store.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
