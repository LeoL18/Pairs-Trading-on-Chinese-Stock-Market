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
    pair_name: Optional[str] = None

    # --- z-score thresholds ---
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_loss_z: float = 3.5

    # --- direction ---
    direction: Literal["long_short", "long_only", "short_only"] = "long_short"

    # --- rolling OLS ---
    ols_window: int = 60                # trading days for rolling hedge ratio

    # --- rebalancing ---
    rebalance_freq_days: int = 20       # recalculate beta & normalization every N calendar days

    # --- position sizing ---
    sizing: Literal["fixed_notional", "dollar_neutral", "vol_scaled"] = "dollar_neutral"
    capital: float = 1_000_000.0        # total capital in CNY
    fixed_notional: float = 100_000.0  # used when sizing == "fixed_notional"
    vol_window: int = 20                # lookback for vol-scaled sizing

    # --- transaction costs (A-share defaults) ---
    commission_rate: float = 0.0003     # 0.03% each way
    stamp_duty_rate: float = 0.001      # 0.1% on sells only

    # --- date range (new pipeline) ---
    start_date: Optional[str] = None    # general backtest start date (YYYYMMDD)
    end_date: Optional[str] = None      # general backtest end date (YYYYMMDD)

    # --- deprecated legacy date fields ---
    model_start: Optional[str] = None   # deprecated
    model_end: Optional[str] = None     # deprecated
    test_start: Optional[str] = None    # deprecated
    test_end: Optional[str] = None      # deprecated


@dataclass
class MultiPairTradingConfig:
    pairs: list[PairTradingConfig]
    start_date: str                     # "YYYYMMDD"
    end_date: str
    capital: float = 1_000_000.0        # total capital in CNY
    cash_buffer_pct: float = 0.05       # keep a buffer when sizing entries
    db_path: str | Path = Path("data/prices.sqlite")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    config: MultiPairTradingConfig
    equity_curve: pd.DataFrame          # index=date, cols=[portfolio_value, cash, position_value]
    trade_log: pd.DataFrame             # one row per trade leg
    spread_dfs: dict[str, pd.DataFrame]  # pair_label -> spread DataFrame
    stats: dict
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
    rebalance_freq_days: int = 20,
) -> pd.DataFrame:
    """
    For the test window, rebalance (recalculate) beta, mean, std every rebalance_freq_days.
    Between rebalancing points, use the locked-in values.
    
    This ensures the spread definition stays consistent within a period,
    avoiding the mismatch between rolling beta and fixed normalization.
    """
    log_a = np.log(adj_a["adj_close"])
    log_b = np.log(adj_b["adj_close"])

    dates = adj_a.index
    betas = []
    means = []
    stds = []
    spreads = []
    z_scores = []

    last_rebalance_date = None
    current_beta = in_sample_beta
    current_mean = model_mean
    current_std = model_std

    for i, dt in enumerate(dates):
        # Check if we should rebalance
        if last_rebalance_date is None or (dt - last_rebalance_date).days >= rebalance_freq_days:
            # Recalculate beta using rolling OLS up to this point
            if i >= ols_window - 1:
                sl = slice(i - ols_window + 1, i + 1)
                x = log_b.iloc[sl].values
                y = log_a.iloc[sl].values
                if len(x) >= 2 and np.std(x) > 0:
                    slope, *_ = stats.linregress(x, y)
                    current_beta = float(slope)
            else:
                current_beta = in_sample_beta

            # Recalculate mean/std from spread up to this rebalance point
            if i >= ols_window - 1:
                spread_to_date = log_a.iloc[:i + 1] - in_sample_alpha - current_beta * log_b.iloc[:i + 1]
                current_mean = spread_to_date.mean()
                current_std = spread_to_date.std()
                if np.isnan(current_std) or current_std == 0:
                    current_std = model_std
            else:
                current_mean = model_mean
                current_std = model_std

            last_rebalance_date = dt

        # Compute spread and z-score with locked-in beta/mean/std
        spread = log_a.iloc[i] - in_sample_alpha - current_beta * log_b.iloc[i]
        z_score = (spread - current_mean) / (current_std + 1e-9)  # avoid division by zero

        betas.append(current_beta)
        means.append(current_mean)
        stds.append(current_std)
        spreads.append(spread)
        z_scores.append(z_score)

    df = pd.DataFrame({
        "log_a": log_a,
        "log_b": log_b,
        "adj_close_a": adj_a["adj_close"],
        "adj_close_b": adj_b["adj_close"],
        "pct_chg_a": adj_a["pct_chg"],
        "pct_chg_b": adj_b["pct_chg"],
        "hedge_ratio": betas,
        "spread": spreads,
        "z_score": z_scores,
        "rebal_mean": means,
        "rebal_std": stds,
    }, index=dates)
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Simulates trading on the spread_df (test window).
    Returns (equity_curve, trade_log, warnings).

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

        # --- limit lock check ---
        a_locked = _is_limit_locked(row.reindex(["pct_chg_a"]).rename({"pct_chg_a": "pct_chg"}))
        b_locked = _is_limit_locked(row.reindex(["pct_chg_b"]).rename({"pct_chg_b": "pct_chg"}))
        either_locked = a_locked or b_locked

        # --- compute current position market value ---
        pos_value = 0.0
        if state == LONG_SPREAD:
            # Long A, Short B
            pos_value = pos_a_shares * price_a - pos_b_shares * price_b
        elif state == SHORT_SPREAD:
            # Short A, Long B
            pos_value = pos_b_shares * price_b - pos_a_shares * price_a

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
                    # Net capital required: pay for A + costs, minus proceeds from shorting B
                    net_capital_required = na + cost_a + cost_b - nb
                    if net_capital_required > cash:
                        run_warnings.append(f"{dt.date()}: insufficient capital to open LONG_SPREAD, skipping.")
                    else:
                        cash -= net_capital_required
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
                    # Net capital required: pay for B + costs, minus proceeds from shorting A
                    net_capital_required = nb + cost_b + cost_a - na
                    if net_capital_required > cash:
                        run_warnings.append(f"{dt.date()}: insufficient capital to open SHORT_SPREAD, skipping.")
                    else:
                        # Deduct net capital required
                        cash -= net_capital_required
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
                    cash += exit_na - cost_a
                    cash -= exit_nb + cost_b
                    record_trade(dt, "A", cfg.ts_code_a, "SELL", pos_a_shares, price_a, exit_na, cost_a, exit_reason)
                    record_trade(dt, "B", cfg.ts_code_b, "SHORT_CLOSE", pos_b_shares, price_b, exit_nb, cost_b, exit_reason)

                elif state == SHORT_SPREAD:
                    exit_na = pos_a_shares * price_a
                    exit_nb = pos_b_shares * price_b
                    cost_a = _calc_cost(exit_na, "buy", cfg.commission_rate, cfg.stamp_duty_rate)
                    cost_b = _calc_cost(exit_nb, "sell", cfg.commission_rate, cfg.stamp_duty_rate)
                    cash -= exit_na + cost_a
                    cash += exit_nb - cost_b
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

    return equity_df, trade_df, run_warnings


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def _compute_stats(equity_df: pd.DataFrame, trade_df: pd.DataFrame, cfg: MultiPairTradingConfig | PairTradingConfig) -> dict:
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
# Multi-pair backtest helpers
# ---------------------------------------------------------------------------

def _pair_label(pair_cfg: PairTradingConfig) -> str:
    return pair_cfg.pair_name or f"{pair_cfg.ts_code_a}/{pair_cfg.ts_code_b}"


def _prepare_pair_spread(
    pair_cfg: PairTradingConfig,
    adj_a: pd.DataFrame,
    adj_b: pd.DataFrame,
) -> tuple[str, pd.DataFrame, list[str]]:
    common_idx = adj_a.index.intersection(adj_b.index)
    if len(common_idx) < pair_cfg.ols_window:
        raise ValueError(
            f"Pair {_pair_label(pair_cfg)} has only {len(common_idx)} common trading days; need at least {pair_cfg.ols_window}."
        )

    raw_a = adj_a.loc[common_idx]
    raw_b = adj_b.loc[common_idx]

    warmup_a = raw_a.iloc[:pair_cfg.ols_window]
    warmup_b = raw_b.iloc[:pair_cfg.ols_window]
    log_a = np.log(warmup_a["adj_close"])
    log_b = np.log(warmup_b["adj_close"])

    alpha, beta, r2 = _fit_ols(log_a, log_b)
    initial_spread = log_a - alpha - beta * log_b
    model_mean = float(initial_spread.mean())
    model_std = float(initial_spread.std())

    if model_std == 0:
        raise ValueError(
            f"Pair {_pair_label(pair_cfg)} has zero initial spread std — check your data."
        )

    spread_df = _build_spread(
        raw_a,
        raw_b,
        beta,
        alpha,
        pair_cfg.ols_window,
        model_mean,
        model_std,
        pair_cfg.rebalance_freq_days,
    )

    label = _pair_label(pair_cfg)
    warn = (
        f"{label}: initial OLS on first {pair_cfg.ols_window} days: "
        f"α={alpha:.4f}, β={beta:.4f}, R²={r2:.4f} | "
        f"spread μ={model_mean:.4f}, σ={model_std:.4f}"
    )
    return label, spread_df, [warn]


def _run_multi_pair_backtest(
    cfg: MultiPairTradingConfig,
    spread_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    FLAT = 0
    LONG_SPREAD = 1
    SHORT_SPREAD = -1

    cash = cfg.capital
    trade_log = []
    equity_rows = []
    run_warnings = []

    pair_states: dict[str, dict] = {}
    for pair_cfg in cfg.pairs:
        label = _pair_label(pair_cfg)
        pair_states[label] = {
            "cfg": pair_cfg,
            "state": FLAT,
            "pos_a_shares": 0.0,
            "pos_b_shares": 0.0,
            "pos_a_value": 0.0,
            "pos_b_value": 0.0,
            "pos_a_side": None,
            "pos_b_side": None,
            "entry_date": None,
            "entry_beta": None,
            "last_price_a": None,
            "last_price_b": None,
            "last_pct_chg_a": None,
            "last_pct_chg_b": None,
        }

    all_dates = sorted({dt for df in spread_dfs.values() for dt in df.index})

    for dt in all_dates:
        total_pos_value = 0.0
        for label, state in pair_states.items():
            row = None
            if dt in spread_dfs[label].index:
                row = spread_dfs[label].loc[dt]
                state["last_price_a"] = row["adj_close_a"]
                state["last_price_b"] = row["adj_close_b"]
                state["last_pct_chg_a"] = row["pct_chg_a"]
                state["last_pct_chg_b"] = row["pct_chg_b"]

            if state["state"] != FLAT and state["last_price_a"] is not None and state["last_price_b"] is not None:
                if state["state"] == LONG_SPREAD:
                    total_pos_value += state["pos_a_shares"] * state["last_price_a"] - state["pos_b_shares"] * state["last_price_b"]
                else:
                    total_pos_value += state["pos_b_shares"] * state["last_price_b"] - state["pos_a_shares"] * state["last_price_a"]

        equity_rows.append({
            "date": dt,
            "portfolio_value": cash + total_pos_value,
            "cash": cash,
            "position_value": total_pos_value,
        })

        # Process exits first so closed capital is available for same-day re-entry.
        for label, state in pair_states.items():
            row = None
            if dt in spread_dfs[label].index:
                row = spread_dfs[label].loc[dt]

            if row is None or state["state"] == FLAT:
                continue

            pair_cfg = state["cfg"]
            z = row["z_score"]
            beta = row["hedge_ratio"]
            price_a = row["adj_close_a"]
            price_b = row["adj_close_b"]
            a_locked = _is_limit_locked(row.reindex(["pct_chg_a"]).rename({"pct_chg_a": "pct_chg"}))
            b_locked = _is_limit_locked(row.reindex(["pct_chg_b"]).rename({"pct_chg_b": "pct_chg"}))
            either_locked = a_locked or b_locked

            t1_ok = state["entry_date"] is None or ((dt - state["entry_date"]).days >= 1)
            should_exit = False
            exit_reason = ""

            if state["state"] == LONG_SPREAD:
                if not t1_ok:
                    pass
                elif abs(z) <= pair_cfg.exit_z:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z < -pair_cfg.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif z > pair_cfg.entry_z:
                    should_exit = True
                    exit_reason = "signal_flip"
            else:
                if not t1_ok and state["pos_b_side"] == "long":
                    pass
                elif abs(z) <= pair_cfg.exit_z:
                    should_exit = True
                    exit_reason = "mean_reversion"
                elif z > pair_cfg.stop_loss_z:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif z < -pair_cfg.entry_z:
                    should_exit = True
                    exit_reason = "signal_flip"

            if should_exit and not either_locked:
                exit_na = state["pos_a_shares"] * price_a
                exit_nb = state["pos_b_shares"] * price_b
                if state["state"] == LONG_SPREAD:
                    cost_a = _calc_cost(exit_na, "sell", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                    cost_b = _calc_cost(exit_nb, "buy", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                    cash += exit_na - cost_a
                    cash -= exit_nb + cost_b
                    trade_log.append({
                        "date": dt,
                        "pair": label,
                        "leg": "A",
                        "ts_code": pair_cfg.ts_code_a,
                        "action": "SELL",
                        "shares": state["pos_a_shares"],
                        "price": price_a,
                        "notional": exit_na,
                        "transaction_cost": cost_a,
                        "note": exit_reason,
                    })
                    trade_log.append({
                        "date": dt,
                        "pair": label,
                        "leg": "B",
                        "ts_code": pair_cfg.ts_code_b,
                        "action": "SHORT_CLOSE",
                        "shares": state["pos_b_shares"],
                        "price": price_b,
                        "notional": exit_nb,
                        "transaction_cost": cost_b,
                        "note": exit_reason,
                    })
                else:
                    cost_a = _calc_cost(exit_na, "buy", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                    cost_b = _calc_cost(exit_nb, "sell", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                    cash -= exit_na + cost_a
                    cash += exit_nb - cost_b
                    trade_log.append({
                        "date": dt,
                        "pair": label,
                        "leg": "A",
                        "ts_code": pair_cfg.ts_code_a,
                        "action": "SHORT_CLOSE",
                        "shares": state["pos_a_shares"],
                        "price": price_a,
                        "notional": exit_na,
                        "transaction_cost": cost_a,
                        "note": exit_reason,
                    })
                    trade_log.append({
                        "date": dt,
                        "pair": label,
                        "leg": "B",
                        "ts_code": pair_cfg.ts_code_b,
                        "action": "SELL",
                        "shares": state["pos_b_shares"],
                        "price": price_b,
                        "notional": exit_nb,
                        "transaction_cost": cost_b,
                        "note": exit_reason,
                    })

                state["state"] = FLAT
                state["pos_a_shares"] = 0.0
                state["pos_b_shares"] = 0.0
                state["pos_a_value"] = 0.0
                state["pos_b_value"] = 0.0
                state["pos_a_side"] = None
                state["pos_b_side"] = None
                state["entry_date"] = None
                state["entry_beta"] = None

        # Entry logic after exits
        for label, state in pair_states.items():
            row = None
            if dt in spread_dfs[label].index:
                row = spread_dfs[label].loc[dt]

            if row is None or state["state"] != FLAT:
                continue

            pair_cfg = state["cfg"]
            z = row["z_score"]
            beta = row["hedge_ratio"]
            price_a = row["adj_close_a"]
            price_b = row["adj_close_b"]
            a_locked = _is_limit_locked(row.reindex(["pct_chg_a"]).rename({"pct_chg_a": "pct_chg"}))
            b_locked = _is_limit_locked(row.reindex(["pct_chg_b"]).rename({"pct_chg_b": "pct_chg"}))
            either_locked = a_locked or b_locked

            if either_locked or np.isnan(z) or np.isnan(beta):
                continue

            enter_long = (z < -pair_cfg.entry_z) and pair_cfg.direction in ("long_short", "long_only")
            enter_short = (z > pair_cfg.entry_z) and pair_cfg.direction in ("long_short", "short_only")
            if not (enter_long or enter_short):
                continue

            na, nb = _compute_notional(
                pair_cfg,
                price_a,
                price_b,
                beta,
                spread_dfs[label],
                dt,
                cash * (1.0 - cfg.cash_buffer_pct),
            )

            shares_a = na / price_a if price_a > 0 else 0.0
            shares_b = nb / price_b if price_b > 0 else 0.0
            shares_a = max(100, int(shares_a / 100) * 100)
            shares_b = max(100, int(shares_b / 100) * 100)
            na = shares_a * price_a
            nb = shares_b * price_b

            if enter_long:
                cost_a = _calc_cost(na, "buy", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                cost_b = _calc_cost(nb, "sell", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                net_capital_required = na + cost_a + cost_b - nb
                if net_capital_required > cash:
                    run_warnings.append(f"{dt.date()}: insufficient capital to open LONG_SPREAD for {label}, skipping.")
                    continue

                cash -= net_capital_required
                state["state"] = LONG_SPREAD
                state["pos_a_shares"] = shares_a
                state["pos_b_shares"] = shares_b
                state["pos_a_value"] = na
                state["pos_b_value"] = nb
                state["pos_a_side"] = "long"
                state["pos_b_side"] = "short"
                state["entry_date"] = dt
                state["entry_beta"] = beta
                trade_log.append({
                    "date": dt,
                    "pair": label,
                    "leg": "A",
                    "ts_code": pair_cfg.ts_code_a,
                    "action": "BUY",
                    "shares": shares_a,
                    "price": price_a,
                    "notional": na,
                    "transaction_cost": cost_a,
                    "note": "",
                })
                trade_log.append({
                    "date": dt,
                    "pair": label,
                    "leg": "B",
                    "ts_code": pair_cfg.ts_code_b,
                    "action": "SHORT_OPEN",
                    "shares": shares_b,
                    "price": price_b,
                    "notional": nb,
                    "transaction_cost": cost_b,
                    "note": "",
                })

            elif enter_short:
                cost_a = _calc_cost(na, "sell", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                cost_b = _calc_cost(nb, "buy", pair_cfg.commission_rate, pair_cfg.stamp_duty_rate)
                net_capital_required = nb + cost_b + cost_a - na
                if net_capital_required > cash:
                    run_warnings.append(f"{dt.date()}: insufficient capital to open SHORT_SPREAD for {label}, skipping.")
                    continue

                cash -= net_capital_required
                state["state"] = SHORT_SPREAD
                state["pos_a_shares"] = shares_a
                state["pos_b_shares"] = shares_b
                state["pos_a_value"] = na
                state["pos_b_value"] = nb
                state["pos_a_side"] = "short"
                state["pos_b_side"] = "long"
                state["entry_date"] = dt
                state["entry_beta"] = beta
                trade_log.append({
                    "date": dt,
                    "pair": label,
                    "leg": "A",
                    "ts_code": pair_cfg.ts_code_a,
                    "action": "SHORT_OPEN",
                    "shares": shares_a,
                    "price": price_a,
                    "notional": na,
                    "transaction_cost": cost_a,
                    "note": "",
                })
                trade_log.append({
                    "date": dt,
                    "pair": label,
                    "leg": "B",
                    "ts_code": pair_cfg.ts_code_b,
                    "action": "BUY",
                    "shares": shares_b,
                    "price": price_b,
                    "notional": nb,
                    "transaction_cost": cost_b,
                    "note": "",
                })

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    trade_df = pd.DataFrame(trade_log)
    return equity_df, trade_df, run_warnings


def run(cfg: MultiPairTradingConfig | PairTradingConfig) -> BacktestResult:
    """
    Main entry point. Run the full backtest and return a BacktestResult.

    Example
    -------
    >>> from pair_trading_engine import PairTradingConfig, MultiPairTradingConfig, run
    >>> pair_cfg = PairTradingConfig(
    ...     ts_code_a="600519.SH",
    ...     ts_code_b="000858.SZ",
    ...     start_date="20200101",
    ...     end_date="20231231",
    ... )
    >>> cfg = MultiPairTradingConfig(pairs=[pair_cfg], start_date="20200101", end_date="20231231")
    >>> result = run(cfg)
    >>> print(result.stats)
    """
    if isinstance(cfg, PairTradingConfig):
        if cfg.start_date and cfg.end_date:
            cfg = MultiPairTradingConfig(
                pairs=[cfg],
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                capital=cfg.capital,
                db_path=cfg.db_path,
            )
        else:
            start_date = cfg.test_start or cfg.model_start
            end_date = cfg.test_end or cfg.model_end
            if start_date is None or end_date is None:
                raise ValueError("PairTradingConfig must provide start_date/end_date or legacy model/test dates.")
            cfg = MultiPairTradingConfig(
                pairs=[cfg],
                start_date=start_date,
                end_date=end_date,
                capital=cfg.capital,
                db_path=cfg.db_path,
            )

    warn_list: list[str] = []
    conn = sqlite3.connect(cfg.db_path)

    unique_codes = {code for pair_cfg in cfg.pairs for code in (pair_cfg.ts_code_a, pair_cfg.ts_code_b)}
    raw_data: dict[str, pd.DataFrame] = {}
    for code in unique_codes:
        raw_df = _load_adjusted_close(conn, code, cfg.start_date, cfg.end_date)
        if raw_df.empty:
            raise ValueError(f"No data found for {code} in the requested date range.")
        raw_data[code] = raw_df
    conn.close()

    spread_dfs: dict[str, pd.DataFrame] = {}
    for pair_cfg in cfg.pairs:
        adj_a = raw_data[pair_cfg.ts_code_a]
        adj_b = raw_data[pair_cfg.ts_code_b]
        pair_label, spread_df, pair_warnings = _prepare_pair_spread(pair_cfg, adj_a, adj_b)
        spread_dfs[pair_label] = spread_df
        warn_list.extend(pair_warnings)

    equity_df, trade_df, sim_warnings = _run_multi_pair_backtest(cfg, spread_dfs)
    warn_list.extend(sim_warnings)

    perf_stats = _compute_stats(equity_df, trade_df, cfg)
    perf_stats["rebalance_freq_days"] = list({pair_cfg.rebalance_freq_days for pair_cfg in cfg.pairs})
    perf_stats["pair_labels"] = list(spread_dfs.keys())

    return BacktestResult(
        config=cfg,
        equity_curve=equity_df,
        trade_log=trade_df,
        spread_dfs=spread_dfs,
        stats=perf_stats,
        warnings=warn_list,
    )