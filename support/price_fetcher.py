"""
Price Fetcher Module

Fetches historical OHLC price data from DEGIRO for ETFs.
"""

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
from tqdm import tqdm
from isodate import parse_duration

from degiro_client import get_api
from degiro_connector.quotecast.tools.chart_fetcher import ChartFetcher
from degiro_connector.quotecast.models.chart import ChartRequest, Interval


class PriceFetcher:
    """
    Fetches historical price data from DEGIRO.

    Usage:
        fetcher = PriceFetcher()

        # Fetch single ETF
        df = fetcher.fetch_ohlc("590959784")

        # Fetch multiple ETFs
        prices = fetcher.fetch_all(vwd_ids)
    """

    def __init__(self, data_dir: Optional[Path] = None, verbose: bool = True):
        """
        Initialize the price fetcher.

        Args:
            data_dir: Directory to store price data (default: data/prices/)
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.trading_api = None
        self.chart_fetcher = None

        # Set up data directory
        if data_dir is None:
            data_dir = Path(__file__).parent / "data" / "prices"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _connect(self):
        """Connect to DEGIRO and initialize ChartFetcher."""
        if self.chart_fetcher is not None:
            return

        # Get existing API connection
        self.trading_api = get_api()

        # Get user token for ChartFetcher
        client_details = self.trading_api.get_client_details()

        if hasattr(client_details, 'id'):
            user_token = client_details.id
        elif isinstance(client_details, dict) and 'id' in client_details:
            user_token = client_details['id']
        else:
            config = self.trading_api.get_config()
            if hasattr(config, 'clientId'):
                user_token = config.clientId
            elif isinstance(config, dict) and 'clientId' in config:
                user_token = config['clientId']
            else:
                raise RuntimeError("Could not get user_token from DEGIRO")

        self.chart_fetcher = ChartFetcher(user_token=user_token)
        self._log("[ChartFetcher] Initialized")

    def fetch_ohlc(
        self,
        vwd_id: str,
        period: Interval = Interval.P5Y,
        resolution: Interval = Interval.P1D,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLC data for a single ETF.

        Args:
            vwd_id: DEGIRO vwdId for the ETF
            period: Time period to fetch (default: 5 years)
            resolution: Data resolution (default: daily)

        Returns:
            Pandas DataFrame with columns: date, open, high, low, close
            Returns None if fetch fails
        """
        self._connect()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                chart_request = ChartRequest(
                    culture="en-US",
                    period=period,
                    requestid="1",
                    resolution=resolution,
                    series=[f"ohlc:issueid:{vwd_id}"],
                    tz="Europe/Amsterdam",
                )

                # Use raw=True to avoid Pydantic validation issues
                chart_data = self.chart_fetcher.get_chart(chart_request=chart_request, raw=True)

                if chart_data and 'series' in chart_data:
                    for series in chart_data['series']:
                        if series.get('type') == 'ohlc' and 'data' in series and 'times' in series:
                            if 'error' in series:
                                continue

                            # Parse times string: "2020-01-01T00:00:00/P1D"
                            times_str = series['times']
                            parts = times_str.split('/')
                            base_date = datetime.fromisoformat(parts[0])
                            resolution_delta = parse_duration(parts[1])

                            # Parse OHLC data: [[offset, open, high, low, close], ...]
                            rows = []
                            for point in series['data']:
                                if len(point) >= 5:
                                    offset, o, h, l, c = point[0], point[1], point[2], point[3], point[4]
                                    timestamp = base_date + (offset * resolution_delta)
                                    rows.append({
                                        'date': timestamp,
                                        'open': o,
                                        'high': h,
                                        'low': l,
                                        'close': c
                                    })

                            if rows:
                                df = pd.DataFrame(rows)
                                df['date'] = pd.to_datetime(df['date'])
                                df = df.set_index('date')
                                return df

        except Exception as e:
            if self.verbose:
                self._log(f"Error fetching {vwd_id}: {e}")

        return None

    def fetch_all(
        self,
        vwd_ids: List[str],
        period: Interval = Interval.P5Y,
        resolution: Interval = Interval.P1D,
        delay: float = 0.15,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLC data for multiple ETFs.

        Args:
            vwd_ids: List of DEGIRO vwdIds
            period: Time period to fetch (default: 5 years)
            resolution: Data resolution (default: daily)
            delay: Delay between requests in seconds
            save: Whether to save data to disk

        Returns:
            Dictionary mapping vwdId to Pandas DataFrame
        """
        self._connect()

        results = {}
        failed = []

        for vwd_id in tqdm(vwd_ids, desc="Fetching prices", disable=not self.verbose, ncols=100):
            df = self.fetch_ohlc(str(vwd_id), period, resolution)

            if df is not None and len(df) > 0:
                results[str(vwd_id)] = df

                if save:
                    self._save_prices(str(vwd_id), df)
            else:
                failed.append(vwd_id)

            time.sleep(delay)

        self._log(f"\nFetched {len(results)}/{len(vwd_ids)} ETFs")
        if failed:
            self._log(f"Failed: {len(failed)} ETFs")

        return results

    def _save_prices(self, vwd_id: str, df: pd.DataFrame):
        """Save price data to parquet file."""
        path = self.data_dir / f"{vwd_id}.parquet"
        df.to_parquet(path)

    def load_prices(self, vwd_id: str) -> Optional[pd.DataFrame]:
        """Load price data from disk."""
        path = self.data_dir / f"{vwd_id}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def load_all(self, vwd_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all saved price data from disk.

        Args:
            vwd_ids: Optional list of vwdIds to load. If None, loads all.

        Returns:
            Dictionary mapping vwdId to Pandas DataFrame
        """
        results = {}

        if vwd_ids is None:
            # Load all parquet files
            for path in self.data_dir.glob("*.parquet"):
                vwd_id = path.stem
                results[vwd_id] = pd.read_parquet(path)
        else:
            for vwd_id in vwd_ids:
                df = self.load_prices(str(vwd_id))
                if df is not None:
                    results[str(vwd_id)] = df

        return results

    def get_combined_closes(
        self,
        vwd_ids: List[str],
        fetch_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Get a combined DataFrame with closing prices for multiple ETFs.

        Args:
            vwd_ids: List of vwdIds
            fetch_missing: Whether to fetch data for missing ETFs

        Returns:
            Pandas DataFrame with date index and one column per vwdId
        """
        # Convert to strings
        vwd_ids = [str(v) for v in vwd_ids]

        # Try to load from disk first
        prices = self.load_all(vwd_ids)

        # Fetch missing if requested
        if fetch_missing:
            missing = [v for v in vwd_ids if v not in prices]
            if missing:
                self._log(f"Fetching {len(missing)} missing ETFs...")
                fetched = self.fetch_all(missing)
                prices.update(fetched)

        if not prices:
            raise ValueError("No price data available")

        # Combine into single DataFrame
        closes = pd.DataFrame()
        for vwd_id, df in prices.items():
            closes[vwd_id] = df['close']

        closes = closes.sort_index().ffill()

        return closes


if __name__ == "__main__":
    print("Price Fetcher Test")
    print("=" * 80)

    fetcher = PriceFetcher()

    test_vwd_id = "480015513"  # iShares MSCI ACWI
    print(f"\nFetching data for vwdId: {test_vwd_id}")

    df = fetcher.fetch_ohlc(test_vwd_id)

    if df is not None:
        print(f"\nFetched {len(df)} rows")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nSample data:")
        print(df.head(5))
    else:
        print("Failed to fetch data")
