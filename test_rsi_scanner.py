#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"          # pandas-ta 0.4 requires 3.12+
# dependencies = ["pytest>=8.0", "pandas-ta>=0.4.67b0", "ccxt>=4.2", "pandas>=2.0",
#                 "pyyaml>=6.0"]
# ///
"""
Tests for rsi_scanner.

Two kinds of test live here:

1. Offline tests (default). They pin the RSI implementation against pandas-ta
   and against a hand-written Wilder loop, and cover the candle/window plumbing.
2. One live ccxt smoke test, marked `network` and skipped unless
   RUN_NETWORK_TESTS=1, so the suite still passes on a plane.

This file is a self-contained uv script too — it installs pytest and pandas-ta
on demand and runs itself:

    ./test_rsi_scanner.py                       # offline suite
    RUN_NETWORK_TESTS=1 ./test_rsi_scanner.py   # + the live Binance pull
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from rsi_scanner import (
    CandleStore,
    CoinGecko,
    CoinInfo,
    Hit,
    contiguous_tail,
    fetch_closed_ohlcv,
    format_matrix,
    filter_by_market_cap,
    format_money,
    build_document,
    build_page,
    dump_json,
    get_candles,
    hits_from_rsi,
    market_info,
    min_offline_bars,
    parse_money,
    ScanHealth,
    candles_needed,
    drop_forming_candle,
    find_hits,
    format_table,
    lookback_bars,
    summary_line,
    wilder_rsi,
    write_csv,
)

PERIOD = 14
# The one test that touches the network. Skipped unless explicitly enabled, so
# the suite passes offline.
NETWORK = pytest.mark.skipif(
    not os.getenv("RUN_NETWORK_TESTS"),
    reason="live network test; set RUN_NETWORK_TESTS=1 to enable",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_close() -> pd.Series:
    """A fixed (seeded) random walk — same numbers on every machine, every run."""
    rng = np.random.default_rng(42)
    steps = rng.normal(loc=0.0, scale=1.5, size=500)
    return pd.Series(100.0 + np.cumsum(steps), name="close")


def reference_wilder_rsi(closes: list[float], period: int = PERIOD) -> list[float]:
    """Wilder's RSI written as an explicit loop — an independent second opinion.

    Mirrors `ewm(alpha=1/period, adjust=False, min_periods=period)`: the running
    averages are seeded with the first gain/loss observation and then smoothed
    recursively, and output stays NaN until `period` observations exist.
    """
    alpha = 1.0 / period
    out = [math.nan] * len(closes)
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]

    avg_gain, avg_loss = gains[0], losses[0]
    for i in range(1, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        bar = i + 1                      # gains[i] is the move into close i+1
        if bar >= period:                # min_periods
            if avg_loss == 0 and avg_gain == 0:
                out[bar] = math.nan
            elif avg_loss == 0:
                out[bar] = 100.0
            else:
                out[bar] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def make_ohlcv(closes: list[float], start_ms: int = 1_600_041_600_000,
               step_ms: int = 4 * 3_600_000) -> pd.DataFrame:
    """Minimal OHLCV frame with evenly spaced timestamps.

    `start_ms` is 2020-09-14 00:00 UTC, an exact 4H boundary, so the formatted
    timestamps in a Hit round-trip without truncating seconds.
    """
    n = len(closes)
    return pd.DataFrame({
        "timestamp": [start_ms + i * step_ms for i in range(n)],
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1.0] * n,
    })


# ---------------------------------------------------------------------------
# RSI correctness
# ---------------------------------------------------------------------------

def test_matches_pandas_ta(synthetic_close: pd.Series):
    """The whole point of the Wilder/RMA formulation: agree with pandas-ta."""
    pandas_ta = pytest.importorskip("pandas_ta")
    # talib=False: if TA-Lib happens to be installed, pandas-ta would delegate to
    # it and we would be comparing against a different implementation.
    expected = pandas_ta.rsi(synthetic_close, length=PERIOD, talib=False)
    actual = wilder_rsi(synthetic_close, PERIOD)

    both = pd.concat([expected, actual], axis=1).dropna()
    assert len(both) > 400, "expected a long overlapping stretch to compare"
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9, rtol=0)

    # One intentional difference: pandas-ta's rma has no `min_periods`, so it
    # emits values from bar 1 that are seeded off a single observation (0.0,
    # 5.26, ...). We suppress those, like TradingView does. Everything from the
    # first bar we do publish onwards must agree.
    assert actual.first_valid_index() == PERIOD
    assert expected.first_valid_index() == 1
    assert not actual.iloc[PERIOD:].isna().any()


def test_matches_reference_loop(synthetic_close: pd.Series):
    """Second, dependency-free check so this stays pinned even without pandas-ta."""
    expected = reference_wilder_rsi(synthetic_close.tolist(), PERIOD)
    actual = wilder_rsi(synthetic_close, PERIOD).tolist()
    assert len(expected) == len(actual)
    for i, (want, got) in enumerate(zip(expected, actual)):
        if math.isnan(want):
            assert math.isnan(got), f"bar {i}: expected NaN, got {got}"
        else:
            assert got == pytest.approx(want, abs=1e-9), f"bar {i}"


def test_warmup_is_period_bars(synthetic_close: pd.Series):
    """RSI(14) needs 15 closes: NaN through index 13, first value at index 14."""
    rsi = wilder_rsi(synthetic_close, PERIOD)
    assert rsi.iloc[:PERIOD].isna().all()
    assert not math.isnan(rsi.iloc[PERIOD])


def test_monotonic_up_is_extreme():
    rsi = wilder_rsi(pd.Series(np.arange(1, 101, dtype=float)), PERIOD)
    assert rsi.iloc[-1] > 95


def test_monotonic_down_is_extreme():
    rsi = wilder_rsi(pd.Series(np.arange(100, 0, -1, dtype=float)), PERIOD)
    assert rsi.iloc[-1] < 5


def test_flat_series_is_nan():
    """No gains and no losses -> 0/0 -> undefined. NaN, not 50, not a crash."""
    rsi = wilder_rsi(pd.Series([42.0] * 100), PERIOD)
    assert rsi.isna().all()


def test_period_is_configurable(synthetic_close: pd.Series):
    fast = wilder_rsi(synthetic_close, 7)
    slow = wilder_rsi(synthetic_close, 21)
    assert fast.iloc[:7].isna().all() and not math.isnan(fast.iloc[7])
    assert slow.iloc[:21].isna().all() and not math.isnan(slow.iloc[21])
    # A shorter period reacts harder, so it must swing wider.
    assert fast.dropna().std() > slow.dropna().std()


# ---------------------------------------------------------------------------
# Candle handling
# ---------------------------------------------------------------------------

def test_drops_still_forming_candle():
    step = 4 * 3_600_000
    df = make_ohlcv([1.0, 2.0, 3.0], step_ms=step)
    last_open = int(df["timestamp"].iloc[-1])
    # "now" is mid-bar: the last candle has not closed yet.
    trimmed = drop_forming_candle(df, step, last_open + step // 2)
    assert len(trimmed) == 2
    assert trimmed["timestamp"].iloc[-1] == last_open - step


def test_keeps_closed_candle():
    step = 4 * 3_600_000
    df = make_ohlcv([1.0, 2.0, 3.0], step_ms=step)
    last_open = int(df["timestamp"].iloc[-1])
    kept = drop_forming_candle(df, step, last_open + step)   # bar just closed
    assert len(kept) == 3


def test_drop_forming_candle_handles_empty():
    empty = make_ohlcv([]).iloc[0:0]
    assert drop_forming_candle(empty, 1000, 1_600_000_000_000).empty


class FakeExchange:
    """Offline stand-in for the slice of the ccxt API this tool actually uses."""
    UNITS = {"m": 60, "h": 3600, "d": 86_400, "w": 604_800}

    def __init__(self, markets=None, tickers=None, candles=None, broken=()):
        self.calls: list[tuple[str, str, int]] = []   # (symbol, timeframe, limit)
        self.markets = markets or {}
        self._tickers = tickers or {}
        self._candles = candles or {}      # symbol -> list of closes
        self.broken = set(broken)          # symbols that raise on fetch
        self.now_ms = 1_600_041_600_000 + 400 * 4 * 3_600_000

    def parse_timeframe(self, timeframe: str) -> int:
        return int(timeframe[:-1]) * self.UNITS[timeframe[-1]]

    def milliseconds(self) -> int:
        return self.now_ms

    def load_markets(self):
        return self.markets

    def fetch_tickers(self, symbols=None):
        if symbols is None:
            return self._tickers
        return {s: self._tickers[s] for s in symbols if s in self._tickers}

    def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls.append((symbol, timeframe, limit))
        if symbol in self.broken:
            raise ccxt_style_error(f"{symbol} is delisted")
        closes = self._candles[symbol][-limit:]
        step = self.parse_timeframe(timeframe) * 1000
        start = self.now_ms - len(closes) * step
        return [[start + i * step, c, c, c, c, 1.0] for i, c in enumerate(closes)]


def ccxt_style_error(message: str) -> Exception:
    """ccxt raises its own exception types; any exception must be survivable."""
    import ccxt
    return ccxt.BadSymbol(message)


def _market(base, quote="USDT", spot=True, active=True):
    return {"base": base, "quote": quote, "spot": spot, "active": active}


def test_top_symbols_ranked_by_quote_volume():
    from rsi_scanner import fetch_top_symbols
    markets = {
        "BTC/USDT": _market("BTC"), "ETH/USDT": _market("ETH"),
        "DOGE/USDT": _market("DOGE"),
        "OLD/USDT": _market("OLD", active=False),        # delisted
        "BTC/EUR": _market("BTC", quote="EUR"),          # wrong quote
        "ETH/USDT:USDT": _market("ETH", spot=False),     # not spot
        "BTCUP/USDT": _market("BTCUP"),                  # leveraged token
    }
    tickers = {s: {"quoteVolume": v} for s, v in {
        "BTC/USDT": 500.0, "ETH/USDT": 900.0, "DOGE/USDT": 100.0,
        "OLD/USDT": 1e9, "BTC/EUR": 1e9, "ETH/USDT:USDT": 1e9, "BTCUP/USDT": 1e9,
    }.items()}
    exchange = FakeExchange(markets=markets, tickers=tickers)

    ranked = fetch_top_symbols(exchange, top_n=10)
    assert list(ranked) == ["ETH/USDT", "BTC/USDT", "DOGE/USDT"]
    assert ranked["ETH/USDT"] == 900.0
    assert list(fetch_top_symbols(exchange, top_n=2)) == ["ETH/USDT", "BTC/USDT"]
    # Leveraged tokens are opt-in.
    assert "BTCUP/USDT" in fetch_top_symbols(exchange, top_n=10, include_leveraged=True)


def test_lookback_bars_and_request_size():
    ex = FakeExchange()
    assert lookback_bars(ex, "1d", 30) == 30
    assert lookback_bars(ex, "4h", 30) == 180
    # Enough for the window + warmup, and never over Binance's 1000 cap.
    assert candles_needed(ex, "4h", 30) >= 180 + 14
    assert candles_needed(ex, "1d", 3650) == 1000


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def _walk_with_move(n: int, move_at: int, drift: float, seed: int,
                    move_bars: int = 10, noise: float = 0.006) -> list[float]:
    """A seeded random walk with a sustained directional move injected into it.

    Deliberately *not* a monotonic ramp: a straight line pins RSI at 0 or 100 and
    would test nothing about the window logic. The walk drifts back toward the
    middle after the move, which is what makes a stale extreme stale.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, noise, n)
    returns[move_at:move_at + move_bars] += drift
    return list(100.0 * np.exp(np.cumsum(returns)))


def _closes_with_spike(n: int = 300, spike_at: int = 250, seed: int = 7) -> list[float]:
    """Random walk with a 10-bar rip that pushes RSI(14) to ~92, then cools off."""
    return _walk_with_move(n, spike_at, drift=0.03, seed=seed)


def test_flags_overbought_with_bars_ago():
    df = make_ohlcv(_closes_with_spike())
    hits = find_hits(df, "FAKE/USDT", "4h", window_bars=180)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.side == "OVERBOUGHT"
    assert hit.rsi_extreme >= 80
    # The extreme is stale, and the current reading is well off it — exactly the
    # situation the bars_ago column exists to make visible.
    assert hit.bars_ago > 0
    assert hit.rsi_now < hit.rsi_extreme
    assert df["timestamp"].iloc[len(df) - 1 - hit.bars_ago] == int(
        pd.Timestamp(hit.extreme_time, tz="UTC").timestamp() * 1000)


def test_window_excludes_old_extremes():
    """Same data, but a window that ends before the spike: nothing flagged."""
    df = make_ohlcv(_closes_with_spike(n=400, spike_at=100, seed=6))
    # Sanity: the spike is real, it is just older than the window.
    assert find_hits(df, "FAKE/USDT", "4h", window_bars=350)
    assert find_hits(df, "FAKE/USDT", "4h", window_bars=180) == []


def test_flags_oversold():
    closes = _walk_with_move(300, move_at=250, drift=-0.03, seed=11)
    hits = find_hits(make_ohlcv(closes), "FAKE/USDT", "1d", window_bars=30)
    assert [h.side for h in hits] == ["OVERSOLD"]
    assert hits[0].rsi_extreme <= 20


def test_reports_both_sides_when_both_touched():
    """Crashed early in the window, ripped later: both readings are the signal."""
    closes = _walk_with_move(300, move_at=200, drift=-0.035, seed=11)
    tail = _walk_with_move(60, move_at=20, drift=0.035, seed=3)
    closes += [closes[-1] * value / tail[0] for value in tail[1:]]
    hits = find_hits(make_ohlcv(closes), "FAKE/USDT", "1d", window_bars=150)
    assert sorted(h.side for h in hits) == ["OVERBOUGHT", "OVERSOLD"]
    # The oversold print is the older of the two.
    by_side = {h.side: h for h in hits}
    assert by_side["OVERSOLD"].bars_ago > by_side["OVERBOUGHT"].bars_ago


def test_thresholds_are_configurable():
    df = make_ohlcv(_closes_with_spike())
    assert find_hits(df, "FAKE/USDT", "4h", 180, overbought=99.9, oversold=0.1) == []
    assert find_hits(df, "FAKE/USDT", "4h", 180, overbought=55.0, oversold=45.0)


def test_insufficient_data_raises():
    """Short series must raise so the caller counts it as a failed symbol."""
    with pytest.raises(ValueError):
        find_hits(make_ohlcv([1.0, 2.0, 3.0]), "FAKE/USDT", "4h", window_bars=180)


def test_sort_key_ranks_extreme_then_recent():
    mild = Hit("4h", "A/USDT", "OVERBOUGHT", 70, 81, 2, "t", 1, 0)
    wild = Hit("4h", "B/USDT", "OVERBOUGHT", 70, 91, 9, "t", 1, 0)
    fresh = Hit("4h", "C/USDT", "OVERBOUGHT", 70, 81, 0, "t", 1, 0)
    assert sorted([mild, wild, fresh], key=Hit.sort_key) == [wild, fresh, mild]


# ---------------------------------------------------------------------------
# Reporting / health
# ---------------------------------------------------------------------------

def _args(**overrides) -> argparse.Namespace:
    base = dict(timeframes=["4h", "1d"], rsi_period=14, overbought=80.0,
                oversold=20.0, lookback_days=30, min_success_rate=0.8)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_table_renders_empty_groups():
    """A quiet week still prints both timeframe headers — silence is explicit."""
    table = format_table([], ["4h", "1d"])
    assert "=== 4H — 0 flagged ===" in table
    assert "=== 1D — 0 flagged ===" in table
    assert "nothing at the extremes" in table


def test_table_groups_and_aligns():
    hits = [Hit("4h", "BTC/USDT", "OVERBOUGHT", 71.2, 84.3, 5, "2026-09-01 00:00",
                64000.0, 1.2e9),
            Hit("1d", "SOL/USDT", "OVERSOLD", 31.0, 18.4, 2, "2026-09-10 00:00",
                140.5, 3.4e8)]
    table = format_table(hits, ["4h", "1d"])
    assert "BTC/USDT" in table and "SOL/USDT" in table
    assert table.index("=== 4H") < table.index("BTC/USDT") < table.index("=== 1D")
    body = [ln for ln in table.splitlines() if "USDT" in ln and "===" not in ln]
    assert all("84.3" in ln or "18.4" in ln for ln in body)


def test_summary_line_is_always_emitted():
    """The heartbeat must be informative even when nothing is flagged."""
    line = summary_line(ScanHealth(total=20, succeeded=20, failed=0, flagged=0), _args())
    assert "flagged=0" in line and "coverage=100%" in line and "ok=20" in line


def test_success_rate_maths():
    assert ScanHealth(total=10, succeeded=8, failed=2).success_rate == 0.8
    assert ScanHealth().success_rate == 0.0          # no division by zero


def test_csv_written_even_when_empty(tmp_path):
    import datetime as dt
    path = write_csv([], str(tmp_path), dt.datetime(2026, 9, 13, 12, 0,
                                                    tzinfo=dt.timezone.utc))
    assert path.endswith("rsi_extremes_20260913_120000.csv")
    rows = list(pd.read_csv(path).columns)
    assert rows[:4] == ["timeframe", "symbol", "side", "rsi_now"]


def test_csv_roundtrips_hits(tmp_path):
    import datetime as dt
    hit = Hit("1d", "ETH/USDT", "OVERSOLD", 24.0, 17.5, 3, "2026-09-10 00:00",
              2400.0, 5.0e8)
    path = write_csv([hit], str(tmp_path), dt.datetime.now(dt.timezone.utc))
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.loc[0, "symbol"] == "ETH/USDT"
    assert df.loc[0, "rsi_extreme"] == pytest.approx(17.5)
    assert df.loc[0, "bars_ago"] == 3


# ---------------------------------------------------------------------------
# Scan loop + run health
# ---------------------------------------------------------------------------

# Seed 26 is a walk whose RSI stays inside 29-65 for the whole window: a market
# that is doing nothing, which is what "calm" has to mean for these tests.
CALM_SEED = 26


def _scan_exchange(broken=()):
    """Fake exchange holding one spiking symbol, two calm ones."""
    calm = _walk_with_move(300, move_at=0, drift=0.0, seed=CALM_SEED)
    candles = {
        "HOT/USDT": _closes_with_spike(n=300, spike_at=290),   # ends overbought
        "CALM/USDT": calm,
        "DEAD/USDT": calm,
    }
    return FakeExchange(candles=candles, broken=broken)


def test_calm_fixture_really_is_calm():
    """Guards the tests below: if this walk ever flags, they stop proving anything."""
    calm = _walk_with_move(300, move_at=0, drift=0.0, seed=CALM_SEED)
    assert find_hits(make_ohlcv(calm), "CALM/USDT", "4h", window_bars=180) == []


def test_scan_survives_a_broken_symbol():
    from rsi_scanner import scan
    exchange = _scan_exchange(broken=["DEAD/USDT"])
    symbols = ["HOT/USDT", "CALM/USDT", "DEAD/USDT"]

    hits, health = scan(exchange, symbols, ["4h"], _args(timeframes=["4h"]))

    # The broken symbol is counted, logged and stepped over — not fatal.
    assert health.total == 3 and health.succeeded == 2 and health.failed == 1
    assert health.success_rate == pytest.approx(2 / 3)
    assert [h.symbol for h in hits] == ["HOT/USDT"]
    assert health.flagged == 1


def test_scan_groups_timeframes_in_requested_order():
    from rsi_scanner import scan
    hits, health = scan(_scan_exchange(), ["HOT/USDT"], ["1d", "4h"],
                        _args(timeframes=["1d", "4h"]))
    assert health.total == 2
    assert [h.timeframe for h in hits] == ["1d", "4h"]


def test_low_coverage_warns_and_exits_non_zero(monkeypatch, tmp_path, capsys):
    """The cron contract: a half-broken run must be visible from the exit code."""
    import rsi_scanner
    exchange = _scan_exchange(broken=["DEAD/USDT", "CALM/USDT"])   # 1/3 succeeds
    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: exchange)
    monkeypatch.setattr(rsi_scanner, "fetch_volumes", lambda *a, **k: {})

    code = rsi_scanner.main([
        "--symbols", "HOT/USDT", "CALM/USDT", "DEAD/USDT",
        "--timeframes", "4h", "--csv-dir", str(tmp_path), "--no-cache",
    ])
    out = capsys.readouterr().out

    assert code == rsi_scanner.EXIT_LOW_COVERAGE
    assert "WARNING" in out and "below" in out
    assert "coverage=33%" in out          # heartbeat still printed


def test_healthy_run_exits_zero(monkeypatch, tmp_path, capsys):
    import rsi_scanner
    exchange = _scan_exchange()
    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: exchange)
    monkeypatch.setattr(rsi_scanner, "fetch_volumes", lambda *a, **k: {})

    code = rsi_scanner.main([
        "--symbols", "CALM/USDT", "--timeframes", "4h", "--csv-dir", str(tmp_path),
        "--no-cache",
    ])
    out = capsys.readouterr().out

    assert code == rsi_scanner.EXIT_OK
    assert "WARNING" not in out
    assert "flagged=0" in out              # heartbeat on a quiet week
    assert "No tokens at the extremes." in out
    assert list(tmp_path.glob("rsi_extremes_*.csv"))   # CSV written regardless


def test_grouped_format_still_available(monkeypatch, tmp_path, capsys):
    """--format grouped keeps the detailed per-timeframe view (price, volume)."""
    import rsi_scanner
    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: _scan_exchange())
    monkeypatch.setattr(rsi_scanner, "fetch_volumes", lambda *a, **k: {})
    rsi_scanner.main(["--symbols", "HOT/USDT", "--timeframes", "4h",
                      "--format", "grouped", "--no-csv", "--no-cache"])
    out = capsys.readouterr().out
    assert "=== 4H" in out and "24H VOL" in out
    assert "| TOKEN" not in out


def test_bad_flags_are_rejected():
    from rsi_scanner import parse_args
    for argv in (["--overbought", "20", "--oversold", "80"],
                 ["--rsi-period", "1"],
                 ["--lookback-days", "0"]):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_telegram_message_includes_heartbeat_when_quiet():
    from rsi_scanner import telegram_message
    health = ScanHealth(total=10, succeeded=10, failed=0, flagged=0)
    message = telegram_message([], health, _args())
    assert "flagged=0" in message
    assert "No RSI extremes" in message


def test_telegram_message_flags_low_coverage():
    from rsi_scanner import telegram_message
    health = ScanHealth(total=10, succeeded=5, failed=5, flagged=0)
    assert "WARNING" in telegram_message([], health, _args())


def test_telegram_send_is_a_noop_without_credentials(monkeypatch):
    """Never crash the scan just because the notifier is misconfigured."""
    from rsi_scanner import send_telegram
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert send_telegram("hello") is False


# ---------------------------------------------------------------------------
# Candle store
# ---------------------------------------------------------------------------

STEP_4H = 4 * 3_600_000


def _store(tmp_path) -> CandleStore:
    return CandleStore(str(tmp_path / "candles.sqlite"))


def test_store_roundtrips_candles(tmp_path):
    store = _store(tmp_path)
    df = make_ohlcv([1.0, 2.0, 3.0])
    now = int(df["timestamp"].iloc[-1]) + STEP_4H
    assert store.upsert_candles("BTC/USDT", "4h", df, STEP_4H, now) == 3

    loaded = store.load("BTC/USDT", "4h", 10)
    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert loaded["close"].tolist() == [1.0, 2.0, 3.0]
    assert loaded["timestamp"].dtype == "int64"
    assert store.last_ts("BTC/USDT", "4h") == int(df["timestamp"].iloc[-1])
    # Timeframes and symbols do not bleed into each other.
    assert store.load("BTC/USDT", "1d", 10).empty
    assert store.last_ts("ETH/USDT", "4h") is None


def test_store_upsert_is_idempotent_and_applies_revisions(tmp_path):
    store = _store(tmp_path)
    df = make_ohlcv([1.0, 2.0, 3.0])
    now = int(df["timestamp"].iloc[-1]) + STEP_4H
    store.upsert_candles("BTC/USDT", "4h", df, STEP_4H, now)
    store.upsert_candles("BTC/USDT", "4h", df, STEP_4H, now)
    assert len(store.load("BTC/USDT", "4h", 10)) == 3      # no duplicate rows

    revised = df.copy()
    revised.loc[2, "close"] = 99.0                          # exchange corrected a bar
    store.upsert_candles("BTC/USDT", "4h", revised, STEP_4H, now)
    assert store.load("BTC/USDT", "4h", 10)["close"].tolist() == [1.0, 2.0, 99.0]


def test_store_refuses_to_persist_a_forming_candle(tmp_path):
    """The invariant the whole cache rests on: a half-formed bar is never written.

    It would never be corrected, and every RSI computed from that series
    afterwards would be quietly wrong.
    """
    store = _store(tmp_path)
    df = make_ohlcv([1.0, 2.0, 3.0])
    mid_bar = int(df["timestamp"].iloc[-1]) + STEP_4H // 2   # last bar still open

    written = store.upsert_candles("BTC/USDT", "4h", df, STEP_4H, mid_bar)

    assert written == 2
    assert store.load("BTC/USDT", "4h", 10)["close"].tolist() == [1.0, 2.0]
    assert store.last_ts("BTC/USDT", "4h") == int(df["timestamp"].iloc[-2])


def test_store_ranks_top_symbols_offline(tmp_path):
    store = _store(tmp_path)
    store.save_volumes({"BTC/USDT": 500.0, "ETH/USDT": 900.0, "DOGE/USDT": 100.0,
                        "BTCUP/USDT": 1e9, "BTC/EUR": 1e9}, 0)

    ranked = store.top_symbols(top_n=10, quote="USDT")
    assert list(ranked) == ["ETH/USDT", "BTC/USDT", "DOGE/USDT"]   # no EUR, no leveraged
    assert ranked["ETH/USDT"] == 900.0
    assert list(store.top_symbols(2, "USDT")) == ["ETH/USDT", "BTC/USDT"]
    assert store.volumes_for(["BTC/USDT"]) == {"BTC/USDT": 500.0}


def test_contiguous_tail_trims_a_gap():
    """A hole in history yields a *wrong* RSI, not an approximate one."""
    df = make_ohlcv([float(i) for i in range(20)])
    df = df.drop(index=10).reset_index(drop=True)        # punch out one bar
    tail = contiguous_tail(df, STEP_4H, "GAPPY/USDT")

    assert len(tail) == 9                                 # only the clean tail survives
    assert tail["timestamp"].diff().dropna().unique().tolist() == [STEP_4H]
    # An ungapped frame is returned untouched.
    clean = make_ohlcv([float(i) for i in range(20)])
    assert len(contiguous_tail(clean, STEP_4H)) == 20


# ---------------------------------------------------------------------------
# Cache behaviour end to end
# ---------------------------------------------------------------------------

def test_offline_makes_zero_network_calls(tmp_path):
    """The point of the cache: replaying a scan must not touch the exchange."""
    exchange = _scan_exchange()
    store = _store(tmp_path)

    first = get_candles(exchange, store, "HOT/USDT", "4h", 300)
    assert exchange.calls, "the first run has to fetch"
    exchange.calls.clear()

    replay = get_candles(exchange, store, "HOT/USDT", "4h", 300, offline=True)
    assert exchange.calls == []
    assert replay["close"].tolist() == first["close"].tolist()


def test_offline_without_cache_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="nothing cached"):
        get_candles(_scan_exchange(), _store(tmp_path), "HOT/USDT", "4h", 300,
                    offline=True)


def test_shallow_cache_is_backfilled_not_topped_up(tmp_path):
    """A cache can be current and still too shallow for the window being asked for.

    Topping it up with the few newest bars would leave it permanently short, and
    RSI computed on the short series drifts from a live scan.
    """
    exchange = _scan_exchange()
    store = _store(tmp_path)
    get_candles(exchange, store, "HOT/USDT", "4h", 60)       # only 60 bars held

    df = get_candles(exchange, store, "HOT/USDT", "4h", 300)  # now ask for 300

    assert exchange.calls[-1][2] == 300, "must backfill, not fetch just the gap"
    assert len(df) == 300
    assert store.depth("HOT/USDT", "4h") == 300


def test_second_run_fetches_only_the_gap(tmp_path):
    """Same request count, far smaller payload — and the overlap verifies the splice."""
    exchange = _scan_exchange()
    store = _store(tmp_path)

    get_candles(exchange, store, "HOT/USDT", "4h", 300)
    first_limit = exchange.calls[-1][2]
    get_candles(exchange, store, "HOT/USDT", "4h", 300)
    second_limit = exchange.calls[-1][2]

    assert first_limit == 300
    assert second_limit < 20          # one new bar + OVERLAP_BARS of overlap
    assert len(exchange.calls) == 2   # still one request per run, as expected

    # --refresh goes back to asking for everything.
    get_candles(exchange, store, "HOT/USDT", "4h", 300, refresh=True)
    assert exchange.calls[-1][2] == 300


def test_cached_and_uncached_scans_agree(tmp_path):
    """A cached scan must produce exactly the hits a stateless one would."""
    from rsi_scanner import scan
    args = _args(timeframes=["4h"])
    stateless, _ = scan(_scan_exchange(), ["HOT/USDT"], ["4h"], args)

    store = _store(tmp_path)
    exchange = _scan_exchange()
    scan(exchange, ["HOT/USDT"], ["4h"], args, store=store)        # warm the cache
    replayed, health = scan(exchange, ["HOT/USDT"], ["4h"], args, store=store,
                            offline=True)

    assert health.succeeded == 1
    assert [(h.symbol, h.side, round(h.rsi_extreme, 9), h.bars_ago) for h in replayed] \
        == [(h.symbol, h.side, round(h.rsi_extreme, 9), h.bars_ago) for h in stateless]


def test_deeper_cache_converges_on_the_live_answer(tmp_path):
    """Why the offline depth floor exists, measured rather than asserted by faith.

    Wilder's RMA is seeded from its first observation and decays by (1 - 1/period)
    per bar — exponentially, never to exactly zero. At 60 stored bars that residue
    is still ~0.2 of an RSI point, enough to move a printed value and flip a
    threshold; by ~250 it is invisible. `min_offline_bars` draws the line.
    """
    exchange = _scan_exchange()
    full = fetch_closed_ohlcv(exchange, "HOT/USDT", "4h", 300)
    live = float(wilder_rsi(full["close"], 14).iloc[-1])

    diffs = []
    for depth in (60, 120, 240):
        store = CandleStore(str(tmp_path / f"{depth}.sqlite"))
        get_candles(exchange, store, "HOT/USDT", "4h", depth)
        # min_bars=0 bypasses the floor so the raw effect is measurable.
        shallow = get_candles(exchange, store, "HOT/USDT", "4h", 300,
                              offline=True, min_bars=0)
        assert len(shallow) == depth
        diffs.append(abs(float(wilder_rsi(shallow["close"], 14).iloc[-1]) - live))

    assert diffs[0] > diffs[1] > diffs[2], "more history must mean less deviation"
    assert diffs[0] > 0.05, "60 bars is visibly off — this is what the floor prevents"
    assert diffs[-1] < 1e-6, "by the default warmup the effect cannot be seen"


def test_offline_refuses_a_shallow_cache(tmp_path):
    """Better a counted failure than a number a live run would not reproduce."""
    exchange = _scan_exchange()
    store = _store(tmp_path)
    get_candles(exchange, store, "HOT/USDT", "4h", 60)           # thin cache

    with pytest.raises(ValueError, match="too shallow"):
        get_candles(exchange, store, "HOT/USDT", "4h", 300, offline=True,
                    min_bars=min_offline_bars(180, 14, 300))

    # Online, the same thin history is fine: it refetches what the exchange has.
    assert len(get_candles(exchange, store, "HOT/USDT", "4h", 300)) == 300


def test_min_offline_bars_never_exceeds_what_was_requested():
    # Window + period + warmup, but never more than the run itself asked for.
    assert min_offline_bars(180, 14, 300) == 294
    assert min_offline_bars(180, 14, 250) == 250


def test_shallow_cache_is_reported_as_a_failed_symbol(tmp_path):
    """A thin cache shows up in coverage, so it cannot pass unnoticed."""
    from rsi_scanner import scan
    exchange = _scan_exchange()
    store = _store(tmp_path)
    get_candles(exchange, store, "HOT/USDT", "4h", 60)

    hits, health = scan(exchange, ["HOT/USDT"], ["4h"], _args(timeframes=["4h"]),
                        store=store, offline=True)

    assert hits == []
    assert health.failed == 1
    assert ("HOT/USDT", "4h") in health.failed_tasks


def test_no_cache_leaves_no_file(tmp_path, monkeypatch):
    import rsi_scanner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: _scan_exchange())
    monkeypatch.setattr(rsi_scanner, "fetch_volumes", lambda *a, **k: {})
    rsi_scanner.main(["--symbols", "CALM/USDT", "--timeframes", "4h",
                      "--no-csv", "--no-cache"])
    assert list(tmp_path.iterdir()) == []


def test_offline_run_end_to_end(tmp_path, monkeypatch, capsys):
    """Warm the cache online, then replay it with the network removed entirely."""
    import rsi_scanner
    exchange = _scan_exchange()
    cache = str(tmp_path / "c.sqlite")
    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: exchange)
    monkeypatch.setattr(rsi_scanner, "fetch_volumes",
                        lambda *a, **k: {"HOT/USDT": 1234.0})

    online = ["--symbols", "HOT/USDT", "--timeframes", "4h", "--no-csv",
              "--cache-path", cache]
    assert rsi_scanner.main(online) == rsi_scanner.EXIT_OK
    exchange.calls.clear()

    assert rsi_scanner.main(online + ["--offline"]) == rsi_scanner.EXIT_OK
    assert exchange.calls == []
    out = capsys.readouterr().out
    assert out.count("| HOT") == 2          # same row in both runs


def test_offline_ranks_top_n_from_cache(tmp_path, monkeypatch):
    import rsi_scanner
    exchange = _scan_exchange()
    cache = str(tmp_path / "c.sqlite")
    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: exchange)
    monkeypatch.setattr(rsi_scanner, "fetch_top_symbols",
                        lambda *a, **k: {"HOT/USDT": 9.0, "CALM/USDT": 1.0})

    base = ["--timeframes", "4h", "--no-csv", "--cache-path", cache, "--top-n", "2"]
    assert rsi_scanner.main(base) == rsi_scanner.EXIT_OK
    exchange.calls.clear()
    # No --symbols: the ranking itself now comes out of the store.
    assert rsi_scanner.main(base + ["--offline"]) == rsi_scanner.EXIT_OK
    assert exchange.calls == []


def test_offline_and_no_cache_are_mutually_exclusive():
    from rsi_scanner import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--offline", "--no-cache"])
    with pytest.raises(SystemExit):
        parse_args(["--offline", "--refresh"])


# ---------------------------------------------------------------------------
# Market cap
# ---------------------------------------------------------------------------

class FakeGecko(CoinGecko):
    """CoinGecko with the network removed; records every call it would make."""

    def __init__(self, pages=None, markets=None):
        super().__init__(api_key=None, pause=0.0)
        self.pages = pages or {}          # page number -> list of ticker dicts
        self.markets_rows = markets or {}  # coin id -> /coins/markets row
        self.calls: list[tuple[str, dict]] = []

    def _get(self, path, params, tries=3):
        self.calls.append((path, dict(params)))
        if path.endswith("/tickers"):
            return {"tickers": self.pages.get(int(params["page"]), [])}
        ids = params["ids"].split(",")
        return [self.markets_rows[i] for i in ids if i in self.markets_rows]


def _ticker(base, coin_id, target="USDT"):
    return {"base": base, "target": target, "coin_id": coin_id}


def _market_row(coin_id, symbol, mcap, circ=None, total=None, fdv=None):
    return {"id": coin_id, "symbol": symbol.lower(), "name": symbol.title(),
            "market_cap": mcap, "fully_diluted_valuation": fdv,
            "circulating_supply": circ, "total_supply": total}


def test_parse_money_reads_human_sizes():
    assert parse_money("50M") == 50e6
    assert parse_money("1.5B") == 1.5e9
    assert parse_money("5e7") == 5e7
    assert parse_money("$50,000,000") == 50e6
    assert parse_money("250k") == 250e3
    with pytest.raises(ValueError):
        parse_money("a lot")


def test_format_money_is_compact():
    # Three significant figures, trailing zeros stripped: a column, not a ledger.
    assert format_money(1.686e12) == "$1.69T"
    assert format_money(32.05e9) == "$32B"
    assert format_money(860e6) == "$860M"
    assert format_money(19.21e6) == "$19.2M"
    assert format_money(0) == "-"           # no data reads as blank, not $0


def test_float_pct_handles_missing_supply():
    assert CoinInfo("x", "X", "X", 1e6, 0, 30.0, 100.0).float_pct == 30.0
    # Total supply is often absent for uncapped coins: report 0, never divide by it.
    assert CoinInfo("x", "X", "X", 1e6, 0, 30.0, 0.0).float_pct == 0.0


def test_binance_map_is_authoritative_not_ticker_matching():
    """ONE is Harmony on Binance, and a larger unrelated coin shares the ticker.

    Matching on the ticker string and taking the bigger market cap picks the
    wrong coin, which is why the mapping comes from CoinGecko's own Binance
    listing instead.
    """
    gecko = FakeGecko(pages={1: [_ticker("BTC", "bitcoin"), _ticker("ONE", "harmony")]})
    assert gecko.binance_map({"BTC", "ONE"}) == {"BTC": "bitcoin", "ONE": "harmony"}
    # Everything resolved on page 1, so it must not keep crawling.
    assert len(gecko.calls) == 1


def test_binance_map_records_misses_so_they_are_not_recrawled():
    gecko = FakeGecko(pages={1: [_ticker("BTC", "bitcoin")], 2: []})
    mapping = gecko.binance_map({"BTC", "NOTACOIN"})
    assert mapping == {"BTC": "bitcoin", "NOTACOIN": None}
    assert len(gecko.calls) == 2            # stopped at the first empty page


def test_markets_reads_caps_and_tolerates_missing_fields():
    gecko = FakeGecko(markets={
        "bitcoin": _market_row("bitcoin", "btc", 1.7e12, 19e6, 21e6, 1.8e12),
        "wrapped-bitcoin": _market_row("wrapped-bitcoin", "wbtc", 9.7e9),
    })
    caps = gecko.markets(["bitcoin", "wrapped-bitcoin"])
    assert caps["bitcoin"].market_cap == 1.7e12
    assert caps["bitcoin"].float_pct == pytest.approx(19 / 21 * 100)
    # WBTC has a ~$10B cap but no rank and no supply figures. Nothing may blow up,
    # and the cap must survive — filtering on rank would have dropped it.
    assert caps["wrapped-bitcoin"].market_cap == 9.7e9
    assert caps["wrapped-bitcoin"].float_pct == 0.0


def test_filter_splits_on_market_cap():
    info = {"BIG/USDT": CoinInfo("big", "BIG", "Big", 500e6, 0, 1, 1),
            "SMALL/USDT": CoinInfo("small", "SMALL", "Small", 19e6, 0, 1, 1)}
    symbols = ["BIG/USDT", "SMALL/USDT", "UNKNOWN/USDT"]

    kept, dropped = filter_by_market_cap(symbols, info, 50e6)
    assert kept == ["BIG/USDT"]
    assert "mcap" in dropped["SMALL/USDT"]
    assert dropped["UNKNOWN/USDT"] == "no market cap data"

    # Unmapped coins are legitimate projects often enough that this is a choice.
    kept, dropped = filter_by_market_cap(symbols, info, 50e6, keep_unknown=True)
    assert kept == ["BIG/USDT", "UNKNOWN/USDT"]
    assert list(dropped) == ["SMALL/USDT"]


def test_market_info_caches_mapping_and_caps(tmp_path):
    store = _store(tmp_path)
    gecko = FakeGecko(pages={1: [_ticker("BTC", "bitcoin")]},
                      markets={"bitcoin": _market_row("bitcoin", "btc", 1.7e12,
                                                      19e6, 21e6)})
    info = market_info(store, ["BTC/USDT"], client=gecko, now_ms=1_000_000_000_000)
    assert info["BTC/USDT"].market_cap == 1.7e12
    first_calls = len(gecko.calls)

    # Second run inside the TTL must hit SQLite, not the network.
    again = market_info(store, ["BTC/USDT"], client=gecko, now_ms=1_000_000_000_000)
    assert again["BTC/USDT"].market_cap == 1.7e12
    assert len(gecko.calls) == first_calls


def test_market_info_refetches_once_caps_go_stale(tmp_path):
    from rsi_scanner import MARKET_CAP_TTL_MS
    store = _store(tmp_path)
    gecko = FakeGecko(pages={1: [_ticker("BTC", "bitcoin")]},
                      markets={"bitcoin": _market_row("bitcoin", "btc", 1.7e12)})
    now = 1_000_000_000_000
    market_info(store, ["BTC/USDT"], client=gecko, now_ms=now)
    calls_after_first = len(gecko.calls)

    gecko.markets_rows["bitcoin"]["market_cap"] = 2.0e12
    later = market_info(store, ["BTC/USDT"], client=gecko,
                        now_ms=now + MARKET_CAP_TTL_MS + 1)

    assert later["BTC/USDT"].market_cap == 2.0e12
    assert len(gecko.calls) > calls_after_first
    # The base -> id mapping has a much longer TTL, so only the caps refetch.
    assert [c[0] for c in gecko.calls[calls_after_first:]] == ["/coins/markets"]


def test_market_info_offline_uses_cache_and_calls_nothing(tmp_path):
    store = _store(tmp_path)
    gecko = FakeGecko(pages={1: [_ticker("BTC", "bitcoin")]},
                      markets={"bitcoin": _market_row("bitcoin", "btc", 1.7e12)})
    market_info(store, ["BTC/USDT"], client=gecko, now_ms=1_000_000_000_000)
    gecko.calls.clear()

    info = market_info(store, ["BTC/USDT"], offline=True, client=gecko,
                       now_ms=1_000_000_000_000)
    assert info["BTC/USDT"].market_cap == 1.7e12
    assert gecko.calls == []


def test_bad_min_market_cap_is_rejected():
    from rsi_scanner import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--min-market-cap", "lots"])
    assert parse_args(["--min-market-cap", "50M"]).min_market_cap == 50e6


def test_market_cap_filter_runs_before_scanning(monkeypatch, tmp_path, capsys):
    """An excluded symbol must cost zero requests, not be scanned then hidden."""
    import rsi_scanner
    exchange = _scan_exchange()
    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: exchange)
    monkeypatch.setattr(rsi_scanner, "fetch_volumes", lambda *a, **k: {})
    monkeypatch.setattr(rsi_scanner, "market_info", lambda *a, **k: {
        "HOT/USDT": CoinInfo("hot", "HOT", "Hot", 500e6, 0, 8.0, 10.0),
        "CALM/USDT": CoinInfo("calm", "CALM", "Calm", 19e6, 0, 1.0, 1.0)})

    code = rsi_scanner.main([
        "--symbols", "HOT/USDT", "CALM/USDT", "--timeframes", "4h", "--no-csv",
        "--cache-path", str(tmp_path / "c.sqlite"), "--min-market-cap", "50M"])
    out = capsys.readouterr().out

    assert code == rsi_scanner.EXIT_OK
    assert [c[0] for c in exchange.calls] == ["HOT/USDT"]     # CALM never fetched
    assert "1 symbols excluded before scanning" in out
    assert "symbols=1" in out                                  # coverage reflects it
    # Size columns appear once the data is there.
    assert "MCAP" in out and "FLOAT" in out and "$500M" in out and "80%" in out


def test_matrix_has_no_size_columns_without_data():
    table = format_matrix(_matrix_hits(), ["4h", "1d"])
    assert "MCAP" not in table and "FLOAT" not in table


def test_market_cap_reaches_the_csv(tmp_path):
    import datetime as dt
    hit = Hit("1d", "ETH/USDT", "OVERSOLD", 24.0, 17.5, 3, "2026-09-10 00:00",
              2400.0, 5e8, market_cap=328e9, float_pct=100.0)
    path = write_csv([hit], str(tmp_path), dt.datetime.now(dt.timezone.utc))
    row = pd.read_csv(path).loc[0]
    assert row["market_cap"] == 328e9
    assert row["float_pct"] == 100.0


def test_missing_coingecko_is_fatal_only_when_filtering(monkeypatch, tmp_path, capsys):
    """A CoinGecko outage must not cost you the week's scan — unless you asked it
    to gate the universe, in which case scanning everything would be a lie."""
    import rsi_scanner

    def boom(*a, **k):
        raise RuntimeError("coingecko down")

    monkeypatch.setattr(rsi_scanner, "build_exchange", lambda *a, **k: _scan_exchange())
    monkeypatch.setattr(rsi_scanner, "fetch_volumes", lambda *a, **k: {})
    monkeypatch.setattr(rsi_scanner, "market_info", boom)
    base = ["--symbols", "HOT/USDT", "--timeframes", "4h", "--no-csv",
            "--cache-path", str(tmp_path / "c.sqlite")]

    assert rsi_scanner.main(base + ["--show-mcap"]) == rsi_scanner.EXIT_OK
    assert rsi_scanner.main(base + ["--min-market-cap", "50M"]) == rsi_scanner.EXIT_FATAL


# ---------------------------------------------------------------------------
# Matrix view
# ---------------------------------------------------------------------------

def _matrix_hits():
    return [
        # Extreme right now on 1D, cooled off on 4H.
        Hit("4h", "LSK/USDT", "OVERBOUGHT", 71.3, 98.2, 3, "2026-09-13 00:00", 0.9, 1),
        Hit("1d", "LSK/USDT", "OVERBOUGHT", 96.1, 96.1, 0, "2026-09-12 00:00", 0.3, 1),
        # 4H only.
        Hit("4h", "CVC/USDT", "OVERBOUGHT", 93.1, 93.1, 0, "2026-09-13 12:00", 0.0, 1),
        # Both extremes in one window: two rows for one token.
        Hit("4h", "BCH/USDT", "OVERBOUGHT", 32.3, 93.2, 138, "2026-08-21 12:00", 223.0, 1),
        Hit("4h", "BCH/USDT", "OVERSOLD", 32.3, 19.8, 18, "2026-09-10 12:00", 223.0, 1),
    ]


def test_matrix_is_one_row_per_token_and_side():
    table = format_matrix(_matrix_hits(), ["4h", "1d"])
    body = [ln for ln in table.splitlines() if ln.startswith("| ")]
    assert body[0].split("|")[1].strip() == "TOKEN"
    rows = [ln.split("|")[1].strip() for ln in body[1:]]
    assert rows == ["LSK", "CVC", "BCH", "BCH"]      # BCH appears once per side
    sides = [ln.split("|")[2].strip() for ln in body[1:]]
    assert sides == ["OB", "OB", "OB", "OS"]


def test_matrix_cells_read_correctly():
    table = format_matrix(_matrix_hits(), ["4h", "1d"])
    lsk = next(ln for ln in table.splitlines() if ln.startswith("| LSK"))
    # 4H is stale: extreme, age, and where it sits now. 1D is live.
    assert "98.2  3b ->71" in lsk
    assert "96.1  NOW" in lsk
    # A fresh hit does not repeat itself as "96.1 0b ->96".
    assert "0b" not in lsk


def test_matrix_distinguishes_calm_from_failed():
    """A delisted pair must not masquerade as a pair that simply stayed calm."""
    health = ScanHealth(failed_tasks={("CVC/USDT", "1d")})
    table = format_matrix(_matrix_hits(), ["4h", "1d"], health)
    cvc = next(ln for ln in table.splitlines() if ln.startswith("| CVC"))
    lsk = next(ln for ln in table.splitlines() if ln.startswith("| LSK"))
    assert cvc.split("|")[4].strip() == "?"          # never scanned
    bch = next(ln for ln in table.splitlines() if "| OS" in ln)
    assert bch.split("|")[4].strip() == "-"          # scanned, no extreme
    assert "?" not in lsk


def test_matrix_sorts_live_extremes_first():
    """Something extreme right now outranks a bigger number from three weeks ago."""
    hits = [
        Hit("1d", "OLD/USDT", "OVERBOUGHT", 40.0, 99.0, 25, "t", 1, 1),
        Hit("1d", "NEW/USDT", "OVERBOUGHT", 81.0, 81.0, 0, "t", 1, 1),
    ]
    rows = [ln.split("|")[1].strip() for ln in format_matrix(hits, ["1d"]).splitlines()
            if ln.startswith("| ") and "TOKEN" not in ln]
    assert rows == ["NEW", "OLD"]


def test_matrix_keeps_quote_when_ambiguous():
    hits = [Hit("1d", "BTC/USDT", "OVERBOUGHT", 81.0, 81.0, 0, "t", 1, 1),
            Hit("1d", "BTC/EUR", "OVERBOUGHT", 82.0, 82.0, 0, "t", 1, 1)]
    table = format_matrix(hits, ["1d"])
    assert "BTC/USDT" in table and "BTC/EUR" in table


def test_matrix_is_pure_ascii_and_rectangular():
    table = format_matrix(_matrix_hits(), ["4h", "1d"])
    table.encode("ascii")                            # raises if any box-drawing slipped in
    borders = [ln for ln in table.splitlines() if ln.startswith("+")]
    rows = [ln for ln in table.splitlines() if ln.startswith("|")]
    assert len(borders) == 3                         # top, under header, bottom
    assert len({len(ln) for ln in borders + rows}) == 1


def test_matrix_reports_data_freshness():
    """The 'good late' marker: say which closed candle the scan is standing on."""
    health = ScanHealth(data_through={"1d": 1_789_171_200_000})
    table = format_matrix([], ["1d"], health)
    assert "No tokens at the extremes." in table
    assert "closed candles through: 1D 2026-09-12 00:00Z" in table


# ---------------------------------------------------------------------------
# JSON document + dashboard
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_JS = os.path.join(HERE, "dashboard.js")
NODE = shutil.which("node")
NEEDS_NODE = pytest.mark.skipif(
    NODE is None, reason="node not installed; the Python<->JS parity tests need it")


def _has_jsdom() -> bool:
    """jsdom is the only way to prove the DOM layer actually runs. Optional:
    `npm install jsdom` in the project directory turns the render test on."""
    if NODE is None:
        return False
    done = subprocess.run([NODE, "-e", "require('jsdom')"], capture_output=True,
                          cwd=HERE)
    return done.returncode == 0


NEEDS_JSDOM = pytest.mark.skipif(
    not _has_jsdom(), reason="jsdom not installed (run `npm install jsdom`)")


def run_node(script: str, *args: str) -> str:
    """Run a snippet under node and return stdout."""
    done = subprocess.run([NODE, "-e", script, *args], capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, f"node failed:\n{done.stderr}"
    return done.stdout


def js_hits(cases: list[dict], tmp_path) -> list:
    """Run dashboard.js `hitsFromRsi` over `cases`, mirroring hits_from_rsi()."""
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    script = """
      const fs = require("fs");
      const { hitsFromRsi } = require(process.argv[2]);
      const cases = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
      const out = cases.map(function (c) {
        try { return hitsFromRsi(c.rsi, c.windowBars, c.ob, c.os); }
        catch (e) { return "ERROR"; }
      });
      console.log(JSON.stringify(out));
    """
    return json.loads(run_node(script, str(path), DASHBOARD_JS))


def py_hits(case: dict):
    """Same case through the Python implementation, in a comparable shape."""
    try:
        hits = hits_from_rsi(case["rsi"], "X/USDT", "4h", case["windowBars"],
                             last_ts=0, step_ms=1, overbought=case["ob"],
                             oversold=case["os"])
    except ValueError:
        return "ERROR"
    return [{"side": h.side, "rsiNow": h.rsi_now, "rsiExtreme": h.rsi_extreme,
             "barsAgo": h.bars_ago} for h in hits]


def _parity_cases(n: int = 60) -> list[dict]:
    """Random RSI series plus the edge cases that actually break ports."""
    rng = np.random.default_rng(1234)
    cases = []
    for _ in range(n):
        length = int(rng.integers(20, 200))
        rsi = [round(float(v), 2) for v in rng.uniform(0, 100, length)]
        # Sprinkle nulls: RSI is undefined for a series that never moved.
        for i in rng.choice(length, size=int(rng.integers(0, 4)), replace=False):
            if i != length - 1:                       # last bar must stay defined
                rsi[int(i)] = None
        cases.append({"rsi": rsi, "windowBars": int(rng.integers(5, length + 20)),
                      "ob": float(rng.choice([70, 80, 88])),
                      "os": float(rng.choice([12, 20, 30]))})

    cases += [
        # Ties: the first occurrence must win, in both languages.
        {"rsi": [50, 95, 50, 95, 20], "windowBars": 5, "ob": 80, "os": 20},
        {"rsi": [5, 50, 5, 50], "windowBars": 4, "ob": 80, "os": 20},
        # Exactly on the threshold: inclusive on both sides.
        {"rsi": [50, 80.0, 60], "windowBars": 3, "ob": 80, "os": 20},
        {"rsi": [50, 20.0, 60], "windowBars": 3, "ob": 80, "os": 20},
        {"rsi": [50, 79.99, 60], "windowBars": 3, "ob": 80, "os": 20},
        # Both extremes in one window -> two rows.
        {"rsi": [50, 95, 50, 5, 50], "windowBars": 5, "ob": 80, "os": 20},
        # Window larger than the series, and a window of one bar.
        {"rsi": [50, 95, 50], "windowBars": 999, "ob": 80, "os": 20},
        {"rsi": [50, 95, 85], "windowBars": 1, "ob": 80, "os": 20},
        # Nulls at the front, and an all-null window.
        {"rsi": [None, None, 95, 50], "windowBars": 4, "ob": 80, "os": 20},
        {"rsi": [None, None, None], "windowBars": 3, "ob": 80, "os": 20},
        {"rsi": [], "windowBars": 3, "ob": 80, "os": 20},
    ]
    return cases


@NEEDS_NODE
def test_python_and_js_flagging_agree_exactly(tmp_path):
    """The dashboard reimplements hits_from_rsi in JS. They must not drift.

    This is the one bug class the rest of the suite cannot catch: the CLI and the
    page would each look entirely reasonable while disagreeing about when a coin
    tagged its extreme.
    """
    cases = _parity_cases()
    from_js = js_hits(cases, tmp_path)
    assert len(from_js) == len(cases)

    for index, (case, js) in enumerate(zip(cases, from_js)):
        py = py_hits(case)
        assert js == py or (js == "ERROR" and py == "ERROR"), (
            f"case {index} diverged\n  rsi={case['rsi'][:12]}...\n"
            f"  window={case['windowBars']} ob={case['ob']} os={case['os']}\n"
            f"  py={py}\n  js={js}")


@NEEDS_NODE
def test_parity_covers_both_sides_and_ties(tmp_path):
    """Guard the guard: the corpus must actually exercise the tricky shapes."""
    cases = _parity_cases()
    results = [py_hits(c) for c in cases]
    assert any(r != "ERROR" and len(r) == 2 for r in results), "no both-sides case"
    assert any(r == "ERROR" for r in results), "no error case"
    assert any(r != "ERROR" and len(r) == 1 for r in results), "no single-hit case"

    # And the tie case must genuinely pick the earlier bar, not the later one.
    tie = {"rsi": [50, 95, 50, 95, 20], "windowBars": 5, "ob": 80, "os": 20}
    assert py_hits(tie)[0]["barsAgo"] == 3
    assert js_hits([tie], tmp_path)[0][0]["barsAgo"] == 3


@NEEDS_NODE
def test_narrowing_the_window_keeps_bars_ago_honest(tmp_path):
    """The slider re-slices the array; offsets stay relative to the latest bar."""
    rsi = [50.0] * 20 + [95.0] + [50.0] * 9        # extreme 9 bars from the end
    wide = {"rsi": rsi, "windowBars": 30, "ob": 80, "os": 20}
    narrow = {"rsi": rsi, "windowBars": 10, "ob": 80, "os": 20}
    tight = {"rsi": rsi, "windowBars": 5, "ob": 80, "os": 20}

    assert py_hits(wide)[0]["barsAgo"] == 9
    assert py_hits(narrow)[0]["barsAgo"] == 9      # still inside the window
    assert py_hits(tight) == []                    # now outside it
    assert js_hits([wide, narrow, tight], tmp_path) == [
        py_hits(wide), py_hits(narrow), []]


@NEEDS_NODE
def test_js_window_bars_matches_python(tmp_path):
    """windowBarsFor() must agree with lookback_bars() or the slider lies."""
    ex = FakeExchange()
    script = """
      const { windowBarsFor } = require(process.argv[1]);
      const out = JSON.parse(process.argv[2]).map(function (c) {
        return windowBarsFor(c[0], c[1]);
      });
      console.log(JSON.stringify(out));
    """
    cases = [(days, tf) for days in (1, 7, 14, 30, 90) for tf in ("4h", "1d", "15m")]
    payload = [[d, ex.parse_timeframe(tf) * 1000] for d, tf in cases]
    from_js = json.loads(run_node(script, DASHBOARD_JS, json.dumps(payload)))
    assert from_js == [lookback_bars(ex, tf, d) for d, tf in cases]


def test_json_rounding_stays_within_its_bound():
    """Rounding for the payload is a bounded loss, and this states the bound.

    Two things can shift. The extreme value moves by at most half a unit in the
    last decimal place. And `bars_ago` can jump when rounding collapses two
    distinct bars onto the same value, because the first-occurrence rule then
    picks the earlier one -- measured at 1 hit in 693 at 2dp, which is why the
    payload ships 4dp instead.
    """
    from rsi_scanner import JSON_RSI_DECIMALS
    epsilon = 10 ** -JSON_RSI_DECIMALS
    rng = np.random.default_rng(7)
    checked = 0

    for _ in range(300):
        exact = [float(v) for v in rng.uniform(0, 100, 80)]
        rounded = [round(v, JSON_RSI_DECIMALS) for v in exact]
        case = dict(symbol="X/USDT", timeframe="4h", window_bars=80,
                    last_ts=0, step_ms=1, overbought=80.0, oversold=20.0)

        for hit_a in hits_from_rsi(exact, **case):
            match = [h for h in hits_from_rsi(rounded, **case)
                     if h.side == hit_a.side]
            checked += 1
            if not match:
                # Only an extreme sitting right on the threshold may vanish.
                edge = 80.0 if hit_a.side == "OVERBOUGHT" else 20.0
                assert abs(hit_a.rsi_extreme - edge) <= epsilon
                continue

            assert abs(match[0].rsi_extreme - hit_a.rsi_extreme) <= epsilon
            if match[0].bars_ago != hit_a.bars_ago:
                # Allowed only when the two bars were genuinely indistinguishable
                # at the shipped precision.
                other = exact[len(exact) - 1 - match[0].bars_ago]
                assert abs(other - hit_a.rsi_extreme) <= epsilon, (
                    f"bars_ago moved {hit_a.bars_ago} -> {match[0].bars_ago} "
                    f"but the values differ by {abs(other - hit_a.rsi_extreme)}")

    assert checked > 100, "the corpus must actually produce hits to check"


def _doc(args_overrides=None, series=None):
    args = _args(**(args_overrides or {}))
    args.min_market_cap = None
    health = ScanHealth(total=2, succeeded=2, failed=0, flagged=1)
    series = series if series is not None else [{
        "symbol": "BTC/USDT", "timeframe": "4h", "last_ts": 1_789_171_200_000,
        "step_ms": 14_400_000, "rsi": [50.0, 95.0, 60.0], "price": 100.0,
        "quote_volume_24h": 1.0}]
    return build_document(series, health, args,
                          {"BTC/USDT": CoinInfo("btc", "BTC", "Bitcoin",
                                                1.7e12, 0, 19.0, 21.0)},
                          generated_ms=1_789_171_200_000)


def test_document_carries_what_the_page_needs():
    doc = _doc()
    assert doc["doc_version"] == 1
    assert doc["rsi_period"] == 14 and doc["timeframes"] == ["4h", "1d"]
    assert doc["max_lookback_days"] == 30
    assert doc["health"]["coverage"] == 1.0
    row = doc["rows"][0]
    assert row["market_cap"] == 1.7e12
    assert row["float_pct"] == pytest.approx(19 / 21 * 100)
    assert row["step_ms"] == 14_400_000


def test_document_includes_symbols_that_did_not_flag():
    """The sliders must be able to widen into symbols this run did not flag."""
    series = [{"symbol": "CALM/USDT", "timeframe": "4h", "last_ts": 0,
               "step_ms": 14_400_000, "rsi": [50.0, 51.0, 49.0],
               "price": 1.0, "quote_volume_24h": 0.0}]
    doc = _doc(series=series)
    assert [r["symbol"] for r in doc["rows"]] == ["CALM/USDT"]


def test_json_is_strictly_valid_and_rejects_nan(tmp_path):
    """Python's default json.dumps emits bare NaN, which JSON.parse throws on."""
    path = dump_json(_doc(), str(tmp_path / "d.json"))
    text = open(path, encoding="utf-8").read()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)                                   # strict parse

    broken = _doc()
    broken["rows"][0]["rsi"] = [float("nan")]
    with pytest.raises(ValueError):
        dump_json(broken, str(tmp_path / "bad.json"))


def test_page_is_self_contained(tmp_path):
    path = build_page(_doc(), str(tmp_path / "index.html"))
    html = open(path, encoding="utf-8").read()
    assert "__RSI_LOGIC__" not in html and "__RSI_DATA__" not in html
    assert "hitsFromRsi" in html                       # logic inlined
    assert "fetch(" not in html                        # no second request
    # The JSON lives inside a <script>, so "<" must not survive literally.
    data = html.split('<script id="rsi-data" type="application/json">')[1]
    data = data.split("</script>")[0]
    assert "<" not in data
    assert json.loads(data)["rows"][0]["symbol"] == "BTC/USDT"


def test_page_escaping_survives_a_hostile_symbol(tmp_path):
    """A symbol containing </script> must not be able to break out of the tag."""
    series = [{"symbol": "</script><img src=x>/USDT", "timeframe": "4h",
               "last_ts": 0, "step_ms": 14_400_000, "rsi": [50.0, 95.0],
               "price": 1.0, "quote_volume_24h": 0.0}]
    html = open(build_page(_doc(series=series), str(tmp_path / "x.html")),
                encoding="utf-8").read()
    body = html.split('<script id="rsi-data" type="application/json">')[1]
    body = body.split("</script>")[0]
    assert "</script>" not in body
    assert json.loads(body)["rows"][0]["symbol"] == "</script><img src=x>/USDT"


def test_api_key_never_reaches_the_page(tmp_path, monkeypatch):
    monkeypatch.setenv("COINGECKO_API_KEY", "super-secret-key-value")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:telegram-secret")
    html = open(build_page(_doc(), str(tmp_path / "i.html")), encoding="utf-8").read()
    assert "super-secret-key-value" not in html
    assert "telegram-secret" not in html


@NEEDS_NODE
def test_page_logic_runs_end_to_end_on_a_real_document(tmp_path):
    """Load the generated page's own data through its own logic."""
    path = dump_json(_doc(), str(tmp_path / "d.json"))
    script = """
      const fs = require("fs");
      const L = require(process.argv[2]);
      const doc = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
      const hits = L.computeHits(doc, {overbought: 80, oversold: 20,
                                       lookbackDays: 30, minMarketCap: 0});
      const rows = L.toMatrix(hits);
      console.log(JSON.stringify({
        hits: hits.length,
        first: rows[0] && [rows[0].symbol, rows[0].side, L.cellText(rows[0].cells["4h"])],
        money: L.formatMoney(rows[0] && rows[0].marketCap),
        filtered: L.computeHits(doc, {overbought: 80, oversold: 20,
                                      lookbackDays: 30, minMarketCap: 1e13}).length,
      }));
    """
    out = json.loads(run_node(script, path, DASHBOARD_JS))
    assert out["hits"] == 1
    assert out["first"] == ["BTC/USDT", "OVERBOUGHT", "95.0  1b ->60"]
    assert out["money"] == "$1.7T"
    assert out["filtered"] == 0        # market cap filter applies client-side


@NEEDS_NODE
def test_page_detects_its_own_staleness(tmp_path):
    """If the scheduled scan stops, nothing else complains. The page must."""
    script = """
      const L = require(process.argv[1]);
      const doc = {generated_ms: 0, rows: [{step_ms: 14400000}]};
      const hours = JSON.parse(process.argv[2]);
      console.log(JSON.stringify(hours.map(function (h) {
        return L.staleness(doc, h * 3600000).level;
      })));
    """
    levels = json.loads(run_node(script, DASHBOARD_JS, json.dumps([1, 4, 9, 30])))
    assert levels == ["ok", "ok", "warn", "bad"]


@NEEDS_NODE
def test_page_renders_utc_not_local_time(tmp_path):
    """Local time would disagree with the CLI on every non-UTC machine."""
    script = """
      const L = require(process.argv[1]);
      console.log(L.formatUtc(1789171200000));
    """
    env_out = subprocess.run([NODE, "-e", """
      const L = require(process.argv[1]);
      console.log(L.formatUtc(1789171200000));
    """, DASHBOARD_JS], capture_output=True, text=True,
        env={**os.environ, "TZ": "Asia/Ho_Chi_Minh"})
    assert env_out.stdout.strip() == "2026-09-12 00:00Z"


def test_public_url_override(monkeypatch):
    """The geoblock escape hatch: Binance 451s US IPs, which is most free CI."""
    from rsi_scanner import build_exchange
    mirror = "https://data-api.binance.vision/api/v3"

    assert build_exchange().urls["api"]["public"] != mirror
    assert build_exchange("binance", mirror).urls["api"]["public"] == mirror

    monkeypatch.setenv("BINANCE_PUBLIC_URL", mirror)
    assert build_exchange().urls["api"]["public"] == mirror


@NEEDS_JSDOM
def test_page_renders_and_controls_filter(tmp_path):
    """Load the real page in a DOM and drive it the way a person would.

    Everything else tests the arithmetic. This is the only test that proves the
    page actually paints, that the controls are wired, and that an empty result
    says so rather than looking like a broken page.
    """
    series = []
    for name, rsi in (("HOT", [50.0, 95.0, 60.0]),
                      ("COOL", [50.0, 51.0, 49.0]),
                      ("LOW", [50.0, 10.0, 30.0])):
        series.append({"symbol": f"{name}/USDT", "timeframe": "4h",
                       "last_ts": 1_789_171_200_000, "step_ms": 14_400_000,
                       "rsi": rsi, "price": 1.0, "quote_volume_24h": 0.0})
    page = build_page(_doc(series=series), str(tmp_path / "index.html"))

    script = """
      const fs = require("fs");
      const { JSDOM, VirtualConsole } = require("jsdom");
      const errors = [];
      const vc = new VirtualConsole()
        .on("jsdomError", function (e) { errors.push(String(e.message)); });
      const dom = new JSDOM(fs.readFileSync(process.argv[1], "utf8"),
                            { runScripts: "dangerously", virtualConsole: vc });
      const d = dom.window.document;
      const count = function () { return d.querySelectorAll("#body tr").length; };
      const set = function (id, v) {
        const el = d.getElementById(id);
        el.value = String(v);
        el.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
      };
      const out = { errors: errors, initial: count(),
                    headers: [].map.call(d.querySelectorAll("#head th"),
                                         function (t) { return t.textContent; }) };
      set("ob", 99); set("os", 1);
      out.impossible = count();
      out.emptyShown = !d.getElementById("empty").className.includes("hide");
      out.emptyText = d.getElementById("empty").textContent;
      set("ob", 80); set("os", 20);
      out.restored = count();
      set("mcap", 1000000000);          /* an option the control actually offers */
      out.filteredByMcap = count();
      out.hiddenNote = d.getElementById("count").textContent;
      console.log(JSON.stringify(out));
    """
    out = json.loads(run_node(script, page))

    assert out["errors"] == [], f"page threw: {out['errors']}"
    assert out["headers"] == ["TOKEN", "SIDE", "MCAP", "FLOAT", "4H", "1D"]
    assert out["initial"] == 2                 # HOT overbought, LOW oversold
    assert out["impossible"] == 0
    assert out["emptyShown"] and "quiet market" in out["emptyText"]
    assert out["restored"] == 2                # controls are reversible
    assert out["filteredByMcap"] == 0          # market cap gate works client-side
    # These fixtures have no CoinGecko data, and the page must say so rather
    # than letting the universe shrink without explanation.
    assert "3 hidden: no mcap data" in out["hiddenNote"]


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

WORKFLOWS = os.path.join(HERE, ".github", "workflows")


def _workflow(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    path = os.path.join(WORKFLOWS, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_probe_workflow_is_manual_only():
    doc = _workflow("probe.yml")
    # YAML 1.1 parses a bare `on:` key as the boolean True.
    triggers = doc[True] if True in doc else doc["on"]
    assert list(triggers) == ["workflow_dispatch"]


def test_scan_workflow_shape():
    doc = _workflow("scan.yml")
    triggers = doc[True] if True in doc else doc["on"]
    assert "schedule" in triggers
    assert any("*/4" in entry["cron"] for entry in triggers["schedule"])
    assert "workflow_dispatch" in triggers          # always runnable by hand
    assert "concurrency" in doc, "overlapping runs would fight over the cache"

    steps = json.dumps(doc["jobs"])
    # --offline on a cold Actions cache exits fatal, which would break deploys.
    assert "--offline" not in steps
    assert "--json" in steps or "--html" in steps


# ---------------------------------------------------------------------------
# Live smoke test (opt-in)
# ---------------------------------------------------------------------------

@NETWORK
def test_binance_ohlcv_smoke():
    """Real pull of one symbol: right shape, clean grid, and actually current."""
    from rsi_scanner import build_exchange, fetch_closed_ohlcv, timeframe_seconds

    exchange = build_exchange()
    timeframe, limit = "4h", 300
    df = fetch_closed_ohlcv(exchange, "BTC/USDT", timeframe, limit)

    # Shape: one row dropped at most (the forming candle), 6 OHLCV columns.
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert limit - 1 <= len(df) <= limit
    assert df[["open", "high", "low", "close"]].notna().all().all()

    # Strictly increasing timestamps, no gaps: every step is exactly one bar.
    step_ms = timeframe_seconds(exchange, timeframe) * 1000
    deltas = df["timestamp"].diff().dropna().unique()
    assert deltas.tolist() == [step_ms], f"irregular candle spacing: {deltas}"

    # Recency: the newest closed bar opened within the last two bar lengths.
    age_ms = time.time() * 1000 - int(df["timestamp"].iloc[-1])
    assert 0 <= age_ms <= 2 * step_ms, f"last candle is {age_ms / 3_600_000:.1f}h old"

    # And the end-to-end path produces a usable RSI.
    rsi = wilder_rsi(df["close"], PERIOD)
    assert 0 <= rsi.iloc[-1] <= 100


if __name__ == "__main__":
    # Lets the file run itself: `./test_rsi_scanner.py [-k pattern]`.
    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
