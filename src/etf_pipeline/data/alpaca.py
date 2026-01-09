"""Alpaca Market Data API utilities for fetching historical bars."""

from datetime import datetime, time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..utils.paths import get_bars_cache_path, ensure_dirs


# Regular trading hours in US/Eastern
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EASTERN = pytz.timezone("US/Eastern")


def fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Fetch 30-minute bars from Alpaca for a single symbol.

    Args:
        client: Alpaca StockHistoricalDataClient instance.
        symbol: Stock ticker symbol (e.g., 'SPY').
        start: Start datetime (timezone-aware or naive, will be localized to US/Eastern).
        end: End datetime (timezone-aware or naive, will be localized to US/Eastern).
        feed: Data feed to use ('iex' or 'sip').

    Returns:
        DataFrame with columns: open, high, low, close, volume, trade_count, vwap
        Index is a DatetimeIndex (timestamp).
    """
    # Create 30-minute timeframe
    timeframe_30min = TimeFrame(30, TimeFrameUnit.Minute)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe_30min,
        start=start,
        end=end,
        adjustment="split",
        feed=feed,
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        return df

    # Reset multi-index if present (symbol, timestamp) -> just timestamp
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    return df


def filter_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter bars to regular trading hours (09:30-16:00 US/Eastern).

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        Filtered DataFrame with only regular hours bars.
    """
    if df.empty:
        return df

    # Convert index to US/Eastern if not already
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(EASTERN)
    elif str(df.index.tz) != "US/Eastern":
        df.index = df.index.tz_convert(EASTERN)

    # Filter to regular trading hours
    mask = (df.index.time >= MARKET_OPEN) & (df.index.time <= MARKET_CLOSE)
    return df[mask].copy()


def load_or_fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load bars from cache or fetch from Alpaca if not cached.

    Args:
        client: Alpaca StockHistoricalDataClient instance.
        symbol: Stock ticker symbol.
        start: Start datetime.
        end: End datetime.
        cache: Whether to cache results to parquet.
        force_refresh: If True, ignore cache and re-fetch.

    Returns:
        DataFrame with 30-minute bars filtered to regular trading hours.
    """
    cache_path = get_bars_cache_path(symbol)

    # Try loading from cache
    if cache and cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        # Check if cache covers the requested range
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(EASTERN)
            elif str(df.index.tz) != "US/Eastern":
                df.index = df.index.tz_convert(EASTERN)
            return df

    # Fetch from Alpaca with fallback
    df = pd.DataFrame()
    last_error = None
    for feed in ["iex", "sip"]:
        try:
            print(f"  Fetching {symbol} with feed={feed}...")
            df = fetch_bars(client, symbol, start, end, feed=feed)
            if not df.empty:
                print(f"  {symbol}: fetched {len(df)} bars")
                break
            else:
                print(f"  {symbol}: no data from {feed} feed")
        except Exception as e:
            last_error = e
            print(f"  {symbol}: error with {feed} feed: {e}")
            continue

    if df.empty and last_error:
        print(f"  WARNING: Failed to fetch {symbol}: {last_error}")

    if df.empty:
        return df

    # Filter to regular hours
    df = filter_regular_hours(df)

    # Cache to parquet
    if cache and not df.empty:
        ensure_dirs()
        df.to_parquet(cache_path)

    return df


def load_all_symbols(
    client: StockHistoricalDataClient,
    symbols: List[str],
    start: datetime,
    end: datetime,
    cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load bars for multiple symbols and combine into a single DataFrame.

    Args:
        client: Alpaca StockHistoricalDataClient instance.
        symbols: List of stock ticker symbols.
        start: Start datetime.
        end: End datetime.
        cache: Whether to cache results.
        force_refresh: If True, ignore cache and re-fetch.

    Returns:
        DataFrame with MultiIndex (symbol, timestamp) and bar columns.
    """
    dfs = []
    for symbol in symbols:
        df = load_or_fetch_bars(
            client, symbol, start, end, cache=cache, force_refresh=force_refresh
        )
        if not df.empty:
            df = df.copy()
            df["symbol"] = symbol
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=0)
    combined = combined.reset_index()
    combined = combined.rename(columns={"index": "timestamp"})
    combined = combined.set_index(["symbol", "timestamp"]).sort_index()

    return combined
