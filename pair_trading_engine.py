"""
pair_trading_engine.py
======================
A-share pair trading backtesting engine.

Key A-share conventions enforced:
  - T+1 for long legs: shares bought on day T cannot be sold until T+1.
  - Margin shorting assumed for short legs (no T+1 restriction on closing shorts).
  - Trades execute at adjusted close price only.
  - Price-limit days (涨跌停): if a stock is locked limit-up/down, no fill.
  - Stamp duty applies to SELL side only (long exit / short open).
  - Commission applies both ways.
"""

from __future__ import annotations

import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PairTradingConfig:
    # --- pair ---
    ts_code_a: str                      # e.g. "000001.SZ"
    ts_code_b: str                      # e.g. "000002.SZ"

    # --- windows ---
    model_start: str                    # "YYYYMMDD"
    model_end: str
    test_start: str
    test_end: str

    # --- z-score thresholds ---
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_loss_z: float = 3.5

    # --- direction ---
    direction: Literal["long_short", "long_only", "short_only"] = "long_short"

    # --- rolling OLS in test window ---
    ols_window: int = 60                # trading days for rolling hedge ratio

    # --- hedge ratio drift alert ---
    drift_alert_pct: float = 0.20       # flag if β changes > 20% ...
    drift_alert_days: int = 10          # ... within this many calendar days

    # --- position sizing ---
    sizing: Literal["fixed_notional", "dollar_neutral", "vol_scaled"] = "dollar_neutral"
    capital: float = 1_000_000.0        # total capital in CNY
    fixed_notional: float = 100_000.0  # used when sizing == "fixed_notional"
    vol_window: int = 20                # lookback for vol-scaled sizing

    # --- transaction costs (A-share defaults) ---
    commission_rate: float = 0.0003     # 0.03% each way
    stamp_duty_rate: float = 0.001      # 0.1% on sells only

    # --- database ---
    db_path: str | Path = Path("data/prices.sqlite")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    config: PairTradingConfig
    equity_curve: pd.DataFrame          # index=date, cols=[portfolio_value, cash, position_value]
    trade_log: pd.DataFrame             # one row per trade leg
    spread_df: pd.DataFrame             # date, spread, z_score, hedge_ratio, ...
    stats: dict
    drift_alerts: list[dict]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_adjusted_close(
    conn: sqlite3.Connection,
    ts_code: str,
    start: str,
    end: str,
) -> pd.Series:
    """
    Returns a date-indexed Series of adjusted close prices.
    adj_close = close * adj_factor
    Both daily_prices and adj_factor use trade_date in YYYYMMDD format.
    """
    sql = """
        SELECT d.trade_date,
               d.close * COALESCE(a.adj_factor, 1.0) AS adj_close,
               d.pct_chg,
               d.close,
               d.pre_close
        FROM   daily_prices d
        LEFT JOIN adj_factor a
               ON a.ts_code = d.ts_code
              AND a.trade_date = d.trade_date
        WHERE  d.ts_code = ?
          AND  d.trade_date >= ?
          AND  d.trade_date <= ?
        ORDER BY d.trade_date
    """
    df = pd.read_sql(sql, conn, params=(ts_code, start, end))
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.set_index("trade_date")
    return df


def _is_limit_locked(row: pd.Series) -> bool:
    """
    True if a stock is price-locked (涨跌停) — cannot trade.
    Uses pct_chg vs. expected limit band (±10% normal, ±5% ST).
    We check if |pct_chg| >= 9.9% as a conservative proxy for limit lock,
    since we don't have the ST flag readily in daily_prices.
    A proper implementation would join stock_basic to check the ST flag.
    """
    if pd.isna(row.get("pct_chg")):
        return False
    return abs(row["pct_chg"]) >= 9.9


# ---------------------------------------------------------------------------
# Statistical modeling (in-sample OLS)
# ---------------------------------------------------------------------------

def _fit_ols(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float, float]:
    """OLS: log_a = α + β * log_b + ε. Returns (alpha, beta, r2)."""
    x = log_b.values
    y = log_a.values
    slope, intercept, r, *_ = stats.linregress(x, y)
    return float(intercept), float(slope), float(r ** 2)


def _rolling_ols_beta(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta (slope). Returns a Series aligned to log_a's index."""
    betas = []
    dates = []
    for i in range(len(log_a)):
        if i < window - 1:
            betas.append(np.nan)
        else:
            sl = slice(i - window + 1, i + 1)
            x = log_b.iloc[sl].values
            y = log_a.iloc[sl].values
            if len(x) < 2 or np.std(x) == 0:
                betas.append(np.nan)
            else:
                slope, *_ = stats.linregress(x, y)
                betas.append(float(slope))
        dates.append(log_a.index[i])
    return pd.Series(betas, index=log_a.index)


# ---------------------------------------------------------------------------
# Spread construction
# ---------------------------------------------------------------------------

def _build_spread(
    adj_a: pd.DataFrame,
    adj_b: pd.DataFrame,
    in_sample_beta: float,
    in_sample_alpha: float,
    ols_window: int,
    model_mean: float,
    model_std: float,
) -> pd.DataFrame:
    """
    For the test window:
      spread_t = log(A_t) - alpha - beta_t * log(B_t)
      z_t      = (spread_t - model_mean) / model_std

    beta_t is determined by rolling OLS within the test window.
    mean/std are FIXED from the in-sample period (avoid look-ahead).
    """
    log_a = np.log(adj_a["adj_close"])
    log_b = np.log(adj_b["adj_close"])

    rolling_beta = _rolling_ols_beta(log_a, log_b, ols_window)

    # Backfill initial NaNs with in-sample beta
    rolling_beta = rolling_beta.fillna(in_sample_beta)

    spread = log_a - in_sample_alpha - rolling_beta * log_b
    z_score = (spread - model_mean) / model_std

    df = pd.DataFrame({
        "log_a": log_a,
        "log_b": log_b,
        "adj_close_a": adj_a["adj_close"],
        "adj_close_b": adj_b["adj_close"],
        "pct_chg_a": adj_a["pct_chg"],
        "pct_chg_b": adj_b["pct_chg"],
        "hedge_ratio": rolling_beta,
        "spread": spread,
        "z_score": z_score,
    })
    return df


# ---------------------------------------------------------------------------
# Transaction cost helper
# ---------------------------------------------------------------------------

def _calc_cost(
    value: float,
    side: Literal["buy", "sell"],
    commission: float,
    stamp_duty: float,
) -> float:
    """Returns total transaction cost for a trade of given value."""
    cost = value * commission
    if side == "sell":
        cost += value * stamp_duty
    return cost


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def _compute_notional(
    cfg: PairTradingConfig,
    price_a: float,
    price_b: float,
    beta: float,
    spread_df: pd.DataFrame,
    current_date,
    available_capital: float,
) -> tuple[float, float]:
    """
    Returns (notional_a, notional_b) in CNY.
    The relationship is: for every 1 unit of A, we trade beta units of B.
    """
    sizing = cfg.sizing

    if sizing == "fixed_notional":
        notional_a = min(cfg.fixed_notional, available_capital * 0.9)
        notional_b = notional_a * beta * (price_b / price_a) if price_a > 0 else notional_a

    elif sizing == "dollar_neutral":
        # Split available capital into two equal legs (dollar neutral)
        half = available_capital * 0.45   # 45% each side, 10% buffer
        notional_a = half
        notional_b = half

    elif sizing == "vol_scaled":
        # Scale inversely to recent volatility of the spread
        idx = spread_df.index.get_indexer([current_date], method="pad")[0]
        start_idx = max(0, idx - cfg.vol_window)
        recent_spread = spread_df["spread"].iloc[start_idx:idx + 1]
        vol = recent_spread.std()
        if np.isnan(vol) or vol == 0:
            vol = 1.0
        base = available_capital * 0.40
        # Invert vol: higher vol → smaller size
        ref_vol = spread_df["spread"].std()
        scale = ref_vol / vol if ref_vol > 0 else 1.0
        scale = np.clip(scale, 0.25, 2.0)
        notional_a = base * scale
        notional_b = notional_a * beta * (price_b / price_a) if price_a > 0 else notional_a
    else:
        raise ValueError(f"Unknown sizing: {sizing}")

    return float(notional_a), float(notional_b)


# ---------------------------------------------------------------------------
# Core backtesting loop
# ---------------------------------------------------------------------------

def _run_backtest(
    cfg: PairTradingConfig,
    spread_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[str]]:
    """
    Simulates trading on the spread_df (test window).
    Returns (equity_curve, trade_log, drift_alerts, warnings).

    State machine:
      - FLAT: no open position
      - LONG_SPREAD: long A, short B  (z < -entry_z, expecting spread to rise)
      - SHORT_SPREAD: short A, long B (z > +entry_z, expecting spread to fall)

    A-share T+1: long leg of A opened on day T can only be closed on T+1 or later.
    """

    FLAT = 0
    LONG_SPREAD = 1    # long A / short B
    SHORT_SPREAD = -1  # short A / long B

    cash = cfg.capital
    state = FLAT
    trade_log = []
    equity_rows = []
    drift_alerts = []
    run_warnings = []

    # Track open position details
    pos_a_shares = 0.0      # +ve = long, 0 for short (handled via PnL)
    pos_b_shares = 0.0
    pos_a_value = 0.0       # notional at entry
    pos_b_value = 0.0
    pos_a_side = None       # "long" | "short"
    pos_b_side = None
    entry_date = None
    entry_z = None
    entry_beta = None

    prev_beta = None
    prev_beta_date = None

    def record_trade(date, leg, ts_code, action, shares, price, notional, cost, note=""):
        trade_log.append({
            "date": date,
            "leg": leg,
            "ts_code": ts_code,
            "action": action,
            "shares": shares,
            "price": price,
            "notional": notional,
            "transaction_cost": cost,
            "note": note,
        })

    dates = spread_df.index
    for i, dt in enumerate(dates):
        row = spread_df.loc[dt]
        z = row["z_score"]
        beta = row["hedge_ratio"]
        price_a = row["adj_close_a"]
        price_b = row["adj_close_b"]

        # --- drift alert ---
        if prev_beta is not None and not np.isnan(beta):
            days_since = (dt - prev_beta_date).days
            if days_since <= cfg.drift_alert_days:
                pct_change = abs(beta - prev_beta) / (abs(prev_beta) + 1e-10)
                if pct_change >= cfg.drift_alert_pct:
                    drift_alerts.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "prev_date": prev_beta_date.strftime("%Y-%m-%d"),
                        "prev_beta": round(prev_beta, 4),
                        "new_beta": round(beta, 4),
                        "pct_change": round(pct_change * 100, 2),
                        "days": days_since,
                    })

        prev_beta = beta
        prev_beta_date = dt

        # --- limit lock check ---
        a_locked = _is_limit_locked(row.reindex(["pct_chg_a"]).rename({"pct_chg_a": "pct_chg"}))
        b_locked = _is_limit_locked(row.reindex(["pct_chg_b"]).rename({"pct_chg_b": "pct_chg"}))
        either_locked = a_locked or b_locked

        # --- compute current position market value ---
        pos_value = 0.0
        if state == LONG_SPREAD:
            # Long A, Short B
            pnl_a = pos_a_shares * (price_a - pos_a_value / max(pos_a_shares, 1e-9))
            pnl_b = pos_b_shares * (pos_b_value / max(pos_b_shares, 1e-9) - price_b)
            pos_value = pos_a_value + pnl_a + pnl_b  # combined equity
        elif state == SHORT_SPREAD:
            # Short A, Long B
            pnl_a = pos_a_shares * (pos_a_value / max(pos_a_shares, 1e-9) - price_a)
            pnl_b = pos_b_shares * (price_b - pos_b_value / max(pos_b_shares, 1e-9))
            pos_value = pos_b_value + pnl_a + pnl_b

        portfolio_value = cash + pos_value

        equity_rows.append({
            "date": dt,
            "portfolio_value": portfolio_value,
            "cash": cash,
            "position_value": pos_value,
            "z_score": z,
            "hedge_ratio": beta,
            "state": state,
        })

        # T+1 check: can we close the long leg today?
        t1_ok = (entry_date is None) or ((dt - entry_date).days >= 1)

        # ---------------------------------------------------------------
        # Entry logic
        # ---------------------------------------------------------------
        if state == FLAT and not either_locked and not (np.isnan(z) or np.isnan(beta)):

            enter_long = (z < -cfg.entry_z) and cfg.direction in ("long_short", "long_only")
            enter_short = (z > cfg.entry_z) and cfg.direction in ("long_short", "short_only")

            if enter_long or enter_short:
                na, nb = _compute_notional(
                    cfg, price_a, price_b, beta, spread_df, dt, cash * 0.95
                )

                shares_a = na / price_a if price_a > 0 else 0.0
                shares_b = nb / price_b if price_b > 0 else 0.0
                # Round to lots of 100 (A-share convention: 1手 = 100 shares)
                shares_a = max(100, int(shares_a / 100) * 100)
                shares_b = max(100, int(shares_b / 100) * 100)
                na = shares_a * price_a
                nb = shares_b * price_b

                if enter_long:
                    # Buy A, Short B
                    cost_a = _calc_cost(na, "buy", cfg.commission_rate, cfg.stamp_duty_rate)
                    cost_b = _calc_cost(nb, "sell", cfg.commission_rate, cfg.stamp_duty_rate)  # short open = sell
                    total_required = na + nb + cost_a + cost_b
                    if total_required > cash:
                        run_warnings.append(f"{dt.date()}: insufficient capital to open LONG_SPREAD, skipping.")
                    else:
                        cash -= (na + cost_a + cost_b)
                        # Short proceeds credited but held as collateral (simplified: deduct margin)
                        # In practice: short proceeds stay in margin account; we model as capital used
                        state = LONG_SPREAD
                        pos_a_shares = shares_a
                        pos_b_shares = shares_b
                        pos_a_value = na
                        pos_b_value = nb
                        pos_a_side = "long"
                        pos_b_side = "short"
                        entry_date = dt
                        entry_z = z
                        entry_beta = beta
                        record_trade(dt, "A", cfg.ts_code_a, "BUY", shares_a, price_a, na, cost_a)
                        record_trade(dt, "B", cfg.ts_code_b, "SHORT_OPEN", shares_b, price_b, nb, cost_b)

                elif enter_short:
                    # Short A, Buy B
                    cost_a = _calc_cost(na, "sell", cfg.commission_rate, cfg.stamp_duty_rate)
                    cost_b = _calc_cost(nb, "buy", cfg.commission_rate, cfg.stamp_duty_rate)
                    total_required = na + nb + cost_a + cost_b
                    if total_required > cash:
                        run_warnings.append(f"{dt.date()}: insufficient capital to open SHORT_SPREAD, skipping.")
                    else:
                        cash -= (nb + cost_a + cost_b)
                        state = SHORT_SPREAD
                        pos_a_shares = shares_a
                        pos_b_shares = shares_b
                        pos_a_value = na
                        pos_b_value = nb
                        pos_a_side = "short"
                        pos_b_side = "long"
                        entry_date = dt
                        entry_z = z
                        entry_beta = beta
                        record_trade(dt, "A", cfg.ts_code_a, "SHORT_OPEN", shares_a, price_a, na, cost_a)
                        record_trade(dt, "B", cfg.ts_code_b, "BUY", shares_b, price_b, nb, cost_b)

        # ---------------------------------------------------------------
        # Exit logic
        # ---------------------------------------------------------------
        elif state != FLAT and not np.isnan(z):
            should_exit = False
            exit_reason = ""

            if state == LONG_SPREAD:
                # T+1: can't close long A leg unless at least 1 day has passed
                if not t1_ok:
                    pass  # forced hold
                elif abs(z) <= cfg.exit_z:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z < -cfg.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif z > cfg.entry_z:
                    # Spread crossed to other side — exit to avoid runaway
                    should_exit = True
                    exit_reason = "signal_flip"

            elif state == SHORT_SPREAD:
                if not t1_ok and pos_b_side == "long":
                    # B is the long leg for SHORT_SPREAD; T+1 applies
                    pass
                elif abs(z) <= cfg.exit_z:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z > cfg.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif z < -cfg.entry_z:
                    should_exit = True
                    exit_reason = "signal_flip"

            if should_exit and not either_locked:
                # Compute PnL and close
                if state == LONG_SPREAD:
                    # Close: Sell A, Cover short B
                    exit_na = pos_a_shares * price_a
                    exit_nb = pos_b_shares * price_b
                    cost_a = _calc_cost(exit_na, "sell", cfg.commission_rate, cfg.stamp_duty_rate)
                    cost_b = _calc_cost(exit_nb, "buy", cfg.commission_rate, cfg.stamp_duty_rate)
                    pnl_a = exit_na - pos_a_value - cost_a
                    pnl_b = (pos_b_value - exit_nb) - cost_b  # short B: profit if price fell
                    cash += pos_a_value + pnl_a + pnl_b + exit_nb - pos_b_value
                    # simplified: cash += proceeds_a - cost_a - cover_cost_b - cost_b + short_pnl_b
                    cash = cash  # already computed above
                    # Correct cash accounting:
                    # We debited (na + cost_a_entry + cost_b_entry) at entry (short proceeds offset)
                    # Now we receive: exit_na - cost_a (sell A), and net short B PnL = pos_b_value - exit_nb - cost_b
                    record_trade(dt, "A", cfg.ts_code_a, "SELL", pos_a_shares, price_a, exit_na, cost_a, exit_reason)
                    record_trade(dt, "B", cfg.ts_code_b, "SHORT_CLOSE", pos_b_shares, price_b, exit_nb, cost_b, exit_reason)

                elif state == SHORT_SPREAD:
                    exit_na = pos_a_shares * price_a
                    exit_nb = pos_b_shares * price_b
                    cost_a = _calc_cost(exit_na, "buy", cfg.commission_rate, cfg.stamp_duty_rate)
                    cost_b = _calc_cost(exit_nb, "sell", cfg.commission_rate, cfg.stamp_duty_rate)
                    record_trade(dt, "A", cfg.ts_code_a, "SHORT_CLOSE", pos_a_shares, price_a, exit_na, cost_a, exit_reason)
                    record_trade(dt, "B", cfg.ts_code_b, "SELL", pos_b_shares, price_b, exit_nb, cost_b, exit_reason)

                # Reset state
                state = FLAT
                pos_a_shares = 0.0
                pos_b_shares = 0.0
                pos_a_value = 0.0
                pos_b_value = 0.0
                entry_date = None
                entry_z = None
                entry_beta = None

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    trade_df = pd.DataFrame(trade_log)

    return equity_df, trade_df, drift_alerts, run_warnings


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def _compute_stats(equity_df: pd.DataFrame, trade_df: pd.DataFrame, cfg: PairTradingConfig) -> dict:
    pv = equity_df["portfolio_value"]

    # Returns
    daily_returns = pv.pct_change().dropna()
    total_return = (pv.iloc[-1] / pv.iloc[0]) - 1.0
    n_days = (pv.index[-1] - pv.index[0]).days
    n_years = n_days / 365.25
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # Sharpe (annualised, 252 trading days, rf=0 for simplicity)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else np.nan

    # Sortino
    downside = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else np.nan

    # Max drawdown
    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Trade stats
    if trade_df.empty:
        n_trades = win_rate = avg_trade_pnl = total_costs = 0.0
    else:
        # Pair up entry/exit trades by leg to compute round-trip PnL
        entries = trade_df[trade_df["action"].isin(["BUY", "SHORT_OPEN"])].copy()
        exits = trade_df[trade_df["action"].isin(["SELL", "SHORT_CLOSE"])].copy()
        n_trades = min(len(entries), len(exits))
        total_costs = trade_df["transaction_cost"].sum()

        # Round-trip PnL approximation (notional difference)
        pnls = []
        for leg in ["A", "B"]:
            e_leg = entries[entries["leg"] == leg].reset_index(drop=True)
            x_leg = exits[exits["leg"] == leg].reset_index(drop=True)
            pairs = min(len(e_leg), len(x_leg))
            for j in range(pairs):
                e_row = e_leg.iloc[j]
                x_row = x_leg.iloc[j]
                if e_row["action"] == "BUY":
                    pnl = x_row["notional"] - e_row["notional"] - e_row["transaction_cost"] - x_row["transaction_cost"]
                else:  # SHORT_OPEN
                    pnl = e_row["notional"] - x_row["notional"] - e_row["transaction_cost"] - x_row["transaction_cost"]
                pnls.append(pnl)

        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else np.nan
        avg_trade_pnl = np.mean(pnls) if pnls else np.nan

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if not np.isnan(cagr) else np.nan,
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "sortino_ratio": round(sortino, 3) if not np.isnan(sortino) else np.nan,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "n_round_trips": n_trades,
        "win_rate_pct": round(win_rate * 100, 2) if not np.isnan(win_rate) else np.nan,
        "avg_round_trip_pnl_cny": round(avg_trade_pnl, 2) if not np.isnan(avg_trade_pnl) else np.nan,
        "total_transaction_costs_cny": round(total_costs, 2),
        "start_value_cny": round(pv.iloc[0], 2),
        "end_value_cny": round(pv.iloc[-1], 2),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(cfg: PairTradingConfig) -> BacktestResult:
    """
    Main entry point. Run the full backtest and return a BacktestResult.

    Example
    -------
    >>> from pair_trading_engine import PairTradingConfig, run
    >>> cfg = PairTradingConfig(
    ...     ts_code_a="600519.SH",
    ...     ts_code_b="000858.SZ",
    ...     model_start="20200101", model_end="20211231",
    ...     test_start="20220101",  test_end="20231231",
    ... )
    >>> result = run(cfg)
    >>> print(result.stats)
    """
    warn_list: list[str] = []

    conn = sqlite3.connect(cfg.db_path)

    # ── Load data ────────────────────────────────────────────────────────────
    # Fetch from model_start to test_end so rolling windows at test boundary have data
    overall_start = min(cfg.model_start, cfg.test_start)
    overall_end   = max(cfg.model_end,   cfg.test_end)

    raw_a = _load_adjusted_close(conn, cfg.ts_code_a, overall_start, overall_end)
    raw_b = _load_adjusted_close(conn, cfg.ts_code_b, overall_start, overall_end)
    conn.close()

    if raw_a.empty or raw_b.empty:
        raise ValueError(f"No data found for one or both codes in the requested date range.")

    # Align on common trading dates
    common_idx = raw_a.index.intersection(raw_b.index)
    if len(common_idx) < 30:
        raise ValueError("Too few common trading dates between the two stocks.")

    raw_a = raw_a.loc[common_idx]
    raw_b = raw_b.loc[common_idx]

    # ── In-sample OLS ────────────────────────────────────────────────────────
    ms = pd.to_datetime(cfg.model_start, format="%Y%m%d")
    me = pd.to_datetime(cfg.model_end,   format="%Y%m%d")
    ts = pd.to_datetime(cfg.test_start,  format="%Y%m%d")
    te = pd.to_datetime(cfg.test_end,    format="%Y%m%d")

    insample_a = raw_a.loc[ms:me]
    insample_b = raw_b.loc[ms:me]

    if len(insample_a) < cfg.ols_window:
        raise ValueError(
            f"In-sample window has only {len(insample_a)} days; need at least {cfg.ols_window}."
        )

    log_a_is = np.log(insample_a["adj_close"])
    log_b_is = np.log(insample_b["adj_close"])

    alpha, beta_is, r2 = _fit_ols(log_a_is, log_b_is)

    # In-sample spread stats (used to normalise z-score throughout test)
    is_spread = log_a_is - alpha - beta_is * log_b_is
    model_mean = float(is_spread.mean())
    model_std  = float(is_spread.std())

    if model_std == 0:
        raise ValueError("In-sample spread has zero standard deviation — check your data.")

    warn_list.append(
        f"In-sample OLS: α={alpha:.4f}, β={beta_is:.4f}, R²={r2:.4f} "
        f"| spread μ={model_mean:.4f}, σ={model_std:.4f}"
    )

    # ── Build test-window spread ──────────────────────────────────────────────
    test_a = raw_a.loc[ts:te]
    test_b = raw_b.loc[ts:te]

    # Include ols_window extra days before test_start for warm-up
    warm_start = common_idx[max(0, common_idx.get_loc(ts) - cfg.ols_window)]
    warmup_a = raw_a.loc[warm_start:te]
    warmup_b = raw_b.loc[warm_start:te]

    spread_full = _build_spread(
        warmup_a, warmup_b,
        beta_is, alpha,
        cfg.ols_window,
        model_mean, model_std,
    )

    # Trim to test window only
    spread_df = spread_full.loc[ts:te]

    if len(spread_df) < 5:
        raise ValueError("Test window is too short.")

    # ── Run simulation ────────────────────────────────────────────────────────
    equity_df, trade_df, drift_alerts, sim_warnings = _run_backtest(cfg, spread_df)
    warn_list.extend(sim_warnings)

    # ── Performance stats ─────────────────────────────────────────────────────
    perf_stats = _compute_stats(equity_df, trade_df, cfg)
    perf_stats["in_sample_r2"]      = round(r2, 4)
    perf_stats["in_sample_beta"]    = round(beta_is, 4)
    perf_stats["in_sample_alpha"]   = round(alpha, 6)
    perf_stats["model_spread_mean"] = round(model_mean, 6)
    perf_stats["model_spread_std"]  = round(model_std, 6)
    perf_stats["n_drift_alerts"]    = len(drift_alerts)

    return BacktestResult(
        config=cfg,
        equity_curve=equity_df,
        trade_log=trade_df,
        spread_df=spread_df,
        stats=perf_stats,
        drift_alerts=drift_alerts,
        warnings=warn_list,
    )