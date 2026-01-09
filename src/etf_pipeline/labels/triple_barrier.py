"""Triple-barrier labeling implementation.

Implements the causal triple-barrier method for generating trade labels.
All computations at time t use only data up to and including bar t.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR) causally.

    ATR at time t is computed using only bars <= t.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of close prices.
        window: Rolling window size for ATR (default 14).

    Returns:
        Series of ATR values, aligned with input index.
    """
    # True Range components
    # TR = max(high - low, |high - prev_close|, |low - prev_close|)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is rolling mean of True Range
    atr = true_range.rolling(window=window, min_periods=window).mean()

    return atr


def compute_labels(
    df: pd.DataFrame,
    atr_window: int = 14,
    k_up: float = 2.0,
    k_dn: float = 1.0,
    n_bars: int = 26,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for a single symbol.

    Label rules (MUST match exactly):
    - Row t uses data up to bar t only.
    - Entry = open of bar t+1.
    - ATR_t is computed causally from bars <= t.
    - Long barriers:
        TP = entry + k_up * ATR_t
        SL = entry - k_dn * ATR_t
    - Vertical barrier: N bars.
    - Scan bars t+1..t+N inclusive:
        - TP hit if high >= TP
        - SL hit if low <= SL
        - If TP and SL hit in same bar, assume SL first.
    - Label: +1 TP, -1 SL, 0 timeout

    Args:
        df: DataFrame with columns: open, high, low, close (single symbol).
            Must have DatetimeIndex sorted ascending.
        atr_window: Window for ATR calculation (default 14).
        k_up: Multiplier for take-profit (default 2.0).
        k_dn: Multiplier for stop-loss (default 1.0).
        n_bars: Vertical barrier in bars (default 26).

    Returns:
        DataFrame with original columns plus:
        - atr: ATR value at time t
        - entry: Entry price (open of t+1)
        - tp: Take-profit level
        - sl: Stop-loss level
        - label: +1 (TP), -1 (SL), 0 (timeout)
        - first_hit: 'tp', 'sl', or 'timeout'
        - bars_to_hit: Number of bars until hit (NaN for timeout)
        - realized_pnl_at_hit: P&L at the hit (entry to hit price)
    """
    df = df.copy()

    # Ensure sorted by time
    df = df.sort_index()

    # Compute ATR causally (at time t, using bars <= t)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], window=atr_window)

    # Entry is open of next bar (t+1)
    df["entry"] = df["open"].shift(-1)

    # Barriers based on ATR at time t
    df["tp"] = df["entry"] + k_up * df["atr"]
    df["sl"] = df["entry"] - k_dn * df["atr"]

    # Initialize label columns
    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    first_hits = np.empty(n, dtype=object)
    first_hits[:] = "timeout"
    bars_to_hit = np.full(n, np.nan)
    realized_pnl = np.full(n, np.nan)

    # Convert to numpy for speed
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    entries = df["entry"].values
    tps = df["tp"].values
    sls = df["sl"].values

    # Scan for each row t
    for t in range(n):
        entry = entries[t]
        tp = tps[t]
        sl = sls[t]

        # Skip if any barrier value is NaN
        if np.isnan(entry) or np.isnan(tp) or np.isnan(sl):
            continue

        # Scan bars t+1 to t+N (inclusive), but not beyond array
        end_idx = min(t + n_bars + 1, n)

        for j in range(t + 1, end_idx):
            h = highs[j]
            l = lows[j]
            c = closes[j]

            tp_hit = h >= tp
            sl_hit = l <= sl

            if tp_hit and sl_hit:
                # Both hit in same bar -> assume SL first
                labels[t] = -1
                first_hits[t] = "sl"
                bars_to_hit[t] = j - t
                realized_pnl[t] = sl - entry
                break
            elif sl_hit:
                labels[t] = -1
                first_hits[t] = "sl"
                bars_to_hit[t] = j - t
                realized_pnl[t] = sl - entry
                break
            elif tp_hit:
                labels[t] = 1
                first_hits[t] = "tp"
                bars_to_hit[t] = j - t
                realized_pnl[t] = tp - entry
                break
        else:
            # Timeout - no barrier hit within N bars
            # Use close of last bar in window for P&L
            if end_idx > t + 1:
                last_close = closes[end_idx - 1]
                realized_pnl[t] = last_close - entry

    df["label"] = labels
    df["first_hit"] = first_hits
    df["bars_to_hit"] = bars_to_hit
    df["realized_pnl_at_hit"] = realized_pnl

    return df


def compute_labels_multi(
    df: pd.DataFrame,
    atr_window: int = 14,
    k_up: float = 2.0,
    k_dn: float = 1.0,
    n_bars: int = 26,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for multiple symbols.

    Args:
        df: DataFrame with MultiIndex (symbol, timestamp) and bar columns.
        atr_window: Window for ATR calculation.
        k_up: Multiplier for take-profit.
        k_dn: Multiplier for stop-loss.
        n_bars: Vertical barrier in bars.

    Returns:
        DataFrame with labels computed per symbol.
    """
    results = []

    for symbol in df.index.get_level_values("symbol").unique():
        symbol_df = df.loc[symbol].copy()
        labeled = compute_labels(
            symbol_df,
            atr_window=atr_window,
            k_up=k_up,
            k_dn=k_dn,
            n_bars=n_bars,
        )
        labeled["symbol"] = symbol
        results.append(labeled)

    combined = pd.concat(results, axis=0)
    combined = combined.reset_index()
    combined = combined.rename(columns={"index": "timestamp"})
    combined = combined.set_index(["symbol", "timestamp"]).sort_index()

    return combined
