/*
 * dashboard.js - the browser half of rsi-alert.
 *
 * `hitsFromRsi` below is a direct port of `hits_from_rsi` in rsi_scanner.py and
 * MUST stay identical to it. The test suite runs both over the same arrays and
 * compares, because this is the one place where the tool can disagree with
 * itself: the CLI says a coin tagged 82 nine bars ago, the page says something
 * else, and neither is obviously wrong to look at.
 *
 * Four things are easy to get subtly wrong, so they are called out where they
 * happen: tie-breaking, inclusive thresholds, how barsAgo is counted, and the
 * fact that one symbol can produce two rows.
 *
 * No DOM access in this file - pure data in, rows out, so node can load it.
 */

"use strict";

/* Extremes inside `windowBars` of an RSI array. Mirror of hits_from_rsi(). */
function hitsFromRsi(rsi, windowBars, overbought, oversold) {
  if (!rsi || rsi.length === 0) throw new Error("empty RSI series");

  const lastIndex = rsi.length - 1;
  const start = Math.max(0, rsi.length - windowBars);

  // null entries (RSI is undefined for a series that has never moved) are
  // skipped without shifting anyone's index, so barsAgo stays honest.
  const window = [];
  for (let i = start; i < rsi.length; i++) {
    if (rsi[i] !== null && rsi[i] !== undefined) window.push([i, rsi[i]]);
  }
  if (window.length === 0) throw new Error("no RSI values in the window");

  const rsiNow = rsi[lastIndex];
  if (rsiNow === null || rsiNow === undefined) throw new Error("latest RSI undefined");

  const hits = [];
  for (const side of ["OVERBOUGHT", "OVERSOLD"]) {
    // Strict > and < (never >= / <=) so a level tagged twice keeps the FIRST
    // occurrence, matching pandas idxmax/idxmin. Using >= would report a
    // different barsAgo than the CLI for the same data.
    let bestIndex = window[0][0];
    let best = window[0][1];
    for (let k = 1; k < window.length; k++) {
      const index = window[k][0];
      const value = window[k][1];
      if (side === "OVERBOUGHT" ? value > best : value < best) {
        bestIndex = index;
        best = value;
      }
    }

    // Thresholds are inclusive: exactly 80.0 counts as overbought.
    const triggered = side === "OVERBOUGHT" ? best >= overbought : best <= oversold;
    if (!triggered) continue;

    hits.push({
      side: side,
      rsiNow: rsiNow,
      rsiExtreme: best,
      barsAgo: lastIndex - bestIndex,
    });
  }
  return hits;
}

/* How many bars cover `days` on a timeframe whose bar length is `stepMs`. */
function windowBarsFor(days, stepMs) {
  return Math.max(1, Math.ceil((days * 86400000) / stepMs));
}

/*
 * Apply the controls to a whole document.
 *
 * `minMarketCap` treats unknown (0) as excluded, mirroring the CLI default of
 * skipping coins CoinGecko could not resolve.
 */
function computeHits(doc, opts) {
  const rows = [];
  for (const row of doc.rows) {
    if (opts.minMarketCap > 0 && !(row.market_cap >= opts.minMarketCap)) continue;

    const windowBars = windowBarsFor(opts.lookbackDays, row.step_ms);
    let found;
    try {
      found = hitsFromRsi(row.rsi, windowBars, opts.overbought, opts.oversold);
    } catch (err) {
      continue; // same posture as the CLI: skip the symbol, never break the run
    }
    for (const hit of found) {
      rows.push({
        symbol: row.symbol,
        timeframe: row.timeframe,
        side: hit.side,
        rsiNow: hit.rsiNow,
        rsiExtreme: hit.rsiExtreme,
        barsAgo: hit.barsAgo,
        extremeMs: row.last_ts - hit.barsAgo * row.step_ms,
        marketCap: row.market_cap,
        floatPct: row.float_pct,
        price: row.price,
      });
    }
  }
  return rows;
}

/*
 * Collapse hits into matrix rows: one per (symbol, side), one cell per
 * timeframe. Ordering mirrors format_matrix() in the CLI - anything extreme
 * right now first, then by distance from 50, then by name - so the same data
 * reads in the same order in both places.
 */
function toMatrix(hits) {
  const grouped = new Map();
  for (const hit of hits) {
    const key = hit.symbol + " " + hit.side;
    if (!grouped.has(key)) {
      grouped.set(key, { symbol: hit.symbol, side: hit.side, cells: {} });
    }
    grouped.get(key).cells[hit.timeframe] = hit;
  }

  const rows = Array.from(grouped.values());
  for (const row of rows) {
    const cells = Object.values(row.cells);
    row.fresh = cells.some(function (c) { return c.barsAgo === 0; }) ? 0 : 1;
    row.peak = Math.max.apply(null, cells.map(function (c) {
      return Math.abs(c.rsiExtreme - 50);
    }));
    row.marketCap = cells[0].marketCap;
    row.floatPct = cells[0].floatPct;
  }
  rows.sort(function (a, b) {
    return a.fresh - b.fresh || b.peak - a.peak || a.symbol.localeCompare(b.symbol);
  });
  return rows;
}

/* Compact money for a table cell. Mirror of format_money(). */
function formatMoney(value) {
  if (!value) return "-";
  const scales = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];
  for (const pair of scales) {
    if (value >= pair[0]) {
      return "$" + Number((value / pair[0]).toPrecision(3)) + pair[1];
    }
  }
  return "$" + value.toFixed(0);
}

/* Cell text. Mirror of _cell(): a fresh hit does not repeat itself. */
function cellText(hit) {
  if (!hit) return "-";
  if (hit.barsAgo === 0) return hit.rsiExtreme.toFixed(1) + "  NOW";
  return hit.rsiExtreme.toFixed(1) + "  " + hit.barsAgo + "b ->" +
         Math.round(hit.rsiNow);
}

/* UTC, always. Local time would disagree with the CLI on every machine. */
function formatUtc(ms) {
  const d = new Date(ms);
  const pad = function (n) { return String(n).padStart(2, "0"); };
  return d.getUTCFullYear() + "-" + pad(d.getUTCMonth() + 1) + "-" +
         pad(d.getUTCDate()) + " " + pad(d.getUTCHours()) + ":" +
         pad(d.getUTCMinutes()) + "Z";
}

/*
 * How stale is this page?
 *
 * The scan's own heartbeat cannot help here: if the scheduled job stops firing
 * (GitHub disables cron after 60 days of repo inactivity) then nothing runs and
 * nothing complains. The page has to notice on its own, so silence is never
 * mistaken for a quiet market.
 */
function staleness(doc, nowMs) {
  const hours = (nowMs - doc.generated_ms) / 3600000;
  const steps = doc.rows.map(function (r) { return r.step_ms; });
  const expected = steps.length ? Math.min.apply(null, steps) / 3600000 : 4;
  let level = "ok";
  if (hours > expected * 6) level = "bad";
  else if (hours > expected * 2) level = "warn";
  return { ageHours: hours, level: level, expectedHours: expected };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    hitsFromRsi: hitsFromRsi,
    windowBarsFor: windowBarsFor,
    computeHits: computeHits,
    toMatrix: toMatrix,
    formatMoney: formatMoney,
    cellText: cellText,
    formatUtc: formatUtc,
    staleness: staleness,
  };
}
