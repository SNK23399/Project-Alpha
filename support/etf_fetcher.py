"""
ETF Fetcher - Fetch and filter ETFs from DEGIRO.

Provides filtering by domicile, distribution policy, TER, fund size,
and data history. Enriches with JustETF data.
"""

import time
import warnings
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd
from tqdm import tqdm
from isodate import parse_duration
import justetf_scraping

from degiro_connector.trading.models.product_search import ETFsRequest
from degiro_connector.quotecast.tools.chart_fetcher import ChartFetcher
from degiro_connector.quotecast.models.chart import ChartRequest, Interval

from degiro_client import get_api

# Suppress noisy logging
logging.getLogger("pydantic").setLevel(logging.ERROR)

# Exchange ID to name mapping
EXCHANGE_MAP = {
    194: 'XET',   # Xetra (Germany)
    196: 'TDG',   # Tradegate
    570: 'LSE',   # London
    608: 'MIL',   # Milan
    947: 'SWX',   # Swiss
    200: 'EAM',   # Euronext Amsterdam
    710: 'EPA',   # Euronext Paris
    801: 'WSE',   # Warsaw
    590: 'MAD',   # Madrid
    1001: 'HSE',  # Helsinki
}


@dataclass
class ETFFilter:
    """
    Filter criteria for ETF selection.

    All filters are optional - set to None to disable.

    Example:
        filter = ETFFilter(
            isin_prefix="IE00",           # Irish domiciled
            distribution="Accumulating",   # Accumulating only
            max_ter=0.30,                  # TER <= 0.30%
            min_fund_size=100,             # >= 100M EUR
            min_months=60,                 # >= 60 months (5 years) history
            currency="EUR",                # EUR listings only
            exchange="XET",                # Prefer Xetra
        )
    """
    # Specific ISINs to fetch (if set, most other filters are ignored)
    isins: List[str] = field(default_factory=list)

    # Domicile filter (ISIN prefix)
    isin_prefix: Optional[str] = None  # e.g., "IE00" for Ireland

    # Distribution policy
    distribution: Optional[str] = None  # "Accumulating" or "Distributing"

    # Cost filter
    max_ter: Optional[float] = None  # Maximum TER in %

    # Fund size filter
    min_fund_size: Optional[float] = None  # Minimum AUM in millions EUR

    # Historical data filter
    min_months: Optional[int] = None  # Minimum months of price data

    # Currency filter
    currency: Optional[str] = None  # e.g., "EUR", "USD", "GBP"

    # Exchange filter
    exchange: Optional[str] = None  # e.g., "XET" for Xetra

    # Whether to deduplicate to one listing per ISIN (False keeps all vwdIds)
    # When True and min_months is set, picks the vwdId with longest history
    deduplicate: bool = True


class ETFFetcher:
    """
    Fetches and filters ETFs from DEGIRO.

    Usage:
        from etf_fetcher import ETFFetcher, ETFFilter

        fetcher = ETFFetcher()

        # Get all Irish accumulating ETFs
        df = fetcher.fetch(ETFFilter(
            isin_prefix="IE00",
            distribution="Accumulating"
        ))

        # Get specific ISINs
        df = fetcher.fetch(ETFFilter(
            isins=["IE00B5BMR087", "IE00B4K48X80"]
        ))
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.api = None
        self.chart_fetcher = None
        self._df_degiro = None
        self._df_justetf = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _connect(self):
        """Initialize API connections."""
        if self.api is not None:
            return

        self.api = get_api()

        # Initialize ChartFetcher for history checks
        client_details = self.api.get_client_details()
        user_token = getattr(client_details, 'id', None) or client_details.get('id')

        if not user_token:
            config = self.api.get_config()
            user_token = getattr(config, 'clientId', None) or config.get('clientId')

        if user_token:
            self.chart_fetcher = ChartFetcher(user_token=user_token)

    def _load_justetf(self):
        """Load JustETF overview data for enrichment."""
        if self._df_justetf is not None:
            return

        self._log("Loading JustETF data...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._df_justetf = justetf_scraping.load_overview()

        # Ensure ISIN is a column
        if self._df_justetf.index.name and self._df_justetf.index.name.lower() == 'isin':
            self._df_justetf = self._df_justetf.reset_index()

        self._log(f"  Loaded {len(self._df_justetf)} ETFs from JustETF")

    def _fetch_degiro_catalog(self) -> pd.DataFrame:
        """Fetch complete ETF catalog from DEGIRO."""
        if self._df_degiro is not None:
            return self._df_degiro.copy()

        self._connect()
        self._log("Fetching DEGIRO ETF catalog...")

        all_etfs = []
        offset = 0
        page_size = 100

        # Get initial page and total count
        request = ETFsRequest(
            popularOnly=False,
            inputAggregateTypes="",
            inputAggregateValues="",
            searchText="",
            offset=0,
            limit=page_size,
            requireTotal=True,
            sortColumns="name",
            sortTypes="asc",
        )

        result = self.api.product_search(product_request=request, raw=False)
        all_etfs.extend(result.products)
        total = getattr(result, 'total', None) or 10000

        offset = page_size

        # Fetch remaining pages
        pbar = tqdm(
            total=total,
            initial=len(all_etfs),
            desc="  Fetching",
            unit=" ETFs",
            disable=not self.verbose,
            ncols=80
        )

        while len(all_etfs) < total:
            request = ETFsRequest(
                popularOnly=False,
                inputAggregateTypes="",
                inputAggregateValues="",
                searchText="",
                offset=offset,
                limit=page_size,
                requireTotal=False,
                sortColumns="name",
                sortTypes="asc",
            )

            result = self.api.product_search(product_request=request, raw=False)

            if not result.products:
                break

            all_etfs.extend(result.products)
            pbar.update(len(result.products))

            if len(result.products) < page_size:
                break

            offset += page_size
            time.sleep(0.1)

        pbar.close()

        # Convert to DataFrame
        df = pd.DataFrame(all_etfs)
        df['symbol'] = df['symbol'].str.strip()

        # Map exchange names
        df['exchangeId'] = df['exchangeId'].astype(int)
        df['exchange'] = df['exchangeId'].map(EXCHANGE_MAP)

        # Clean up
        df = df[df['vwdId'].notna()].copy()
        df = df.drop_duplicates(subset=['vwdId'], keep='first')

        self._df_degiro = df
        self._log(f"  Found {len(df)} ETFs on DEGIRO")

        return df.copy()

    def _get_justetf_months(self, isin: str) -> Optional[int]:
        """Get months of data available on JustETF for an ISIN."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Fetch chart data from JustETF
                df = justetf_scraping.load_chart(isin, currency='EUR')

                if df is not None and len(df) > 0:
                    # Calculate months of data
                    days = (df.index.max() - df.index.min()).days
                    months = int(days / 30.44)  # Average days per month
                    return months

        except Exception:
            pass

        return None

    def _deduplicate(self, df: pd.DataFrame, by_longest_history: bool = False) -> pd.DataFrame:
        """
        Keep one listing per ISIN.

        Args:
            by_longest_history: If True, keep the vwdId with most months of data.
                               If False, just keep first occurrence.
        """
        df = df.copy()

        if by_longest_history and 'months_of_data' in df.columns:
            # Sort by ISIN, then months_of_data descending (longest first)
            df = df.sort_values(
                by=['isin', 'months_of_data'],
                ascending=[True, False]
            )
        else:
            df = df.sort_values(by=['isin'])

        df = df.drop_duplicates(subset='isin', keep='first')

        return df

    def fetch(self, filter: Optional[ETFFilter] = None) -> pd.DataFrame:
        """
        Fetch ETFs matching the filter criteria.

        Args:
            filter: ETFFilter with criteria. If None, returns all ETFs.

        Returns:
            DataFrame with columns: ISIN, Name, vwdId, exchange, currency,
            TER, fund_size, months_of_data (if min_months filter used)
        """
        if filter is None:
            filter = ETFFilter()

        # Load data sources
        df = self._fetch_degiro_catalog()
        self._load_justetf()

        self._log(f"\nFiltering {df['isin'].nunique()} ETFs...")

        # Filter by specific ISINs
        if filter.isins:
            df = df[df['isin'].isin(filter.isins)].copy()
            self._log(f"  ISINs: {df['isin'].nunique()}/{len(filter.isins)} found")

        # Filter by domicile (ISIN prefix)
        if filter.isin_prefix:
            df = df[df['isin'].str.startswith(filter.isin_prefix, na=False)].copy()
            self._log(f"  Domicile ({filter.isin_prefix}): {df['isin'].nunique()} ETFs")

        # Enrich with JustETF data
        isin_col = next((c for c in self._df_justetf.columns if c.lower() == 'isin'), None)
        dist_col = next((c for c in self._df_justetf.columns if c.lower() in ['dividends', 'dividend', 'distribution']), None)
        size_col = next((c for c in self._df_justetf.columns if 'size' in c.lower() or 'aum' in c.lower()), None)

        if isin_col:
            justetf = self._df_justetf.set_index(isin_col)
            if dist_col:
                df['distribution'] = df['isin'].map(justetf[dist_col].astype(str))
            if size_col:
                df['fund_size'] = df['isin'].map(justetf[size_col])

        # Drop ETFs not in JustETF
        df = df[df['distribution'].notna()].copy()
        self._log(f"  JustETF match: {df['isin'].nunique()} ETFs")

        # Filter by distribution policy
        if filter.distribution:
            df = df[df['distribution'] == filter.distribution].copy()
            self._log(f"  Distribution ({filter.distribution}): {df['isin'].nunique()} ETFs")

        # Filter by TER
        if filter.max_ter is not None:
            df = df[(df['totalExpenseRatio'].notna()) & (df['totalExpenseRatio'] <= filter.max_ter)].copy()
            self._log(f"  TER <= {filter.max_ter}%: {df['isin'].nunique()} ETFs")

        # Filter by fund size
        if filter.min_fund_size is not None:
            df = df[(df['fund_size'].notna()) & (df['fund_size'] >= filter.min_fund_size)].copy()
            self._log(f"  Fund size >= {filter.min_fund_size}M: {df['isin'].nunique()} ETFs")

        # Filter by currency
        if filter.currency is not None:
            df = df[df['currency'] == filter.currency].copy()
            self._log(f"  Currency ({filter.currency}): {df['isin'].nunique()} ETFs")

        # Filter by exchange
        if filter.exchange is not None:
            df = df[df['exchange'] == filter.exchange].copy()
            self._log(f"  Exchange ({filter.exchange}): {df['isin'].nunique()} ETFs")

        # Filter to numeric vwdIds only
        df = df[df['vwdId'].astype(str).str.match(r'^\d+$')].copy()

        # Always fetch months of data from JustETF (expensive - requires API calls)
        self._log(f"  Fetching data history from JustETF...")

        isins = df['isin'].unique()
        justetf_months = {}

        for isin in tqdm(isins, desc="  Checking history", disable=not self.verbose, ncols=80):
            justetf_months[isin] = self._get_justetf_months(isin)
            time.sleep(0.3)  # Rate limiting for JustETF

        df['months_of_data'] = df['isin'].map(justetf_months)

        # Filter by months of data (optional)
        if filter.min_months is not None:
            df = df[df['months_of_data'] >= filter.min_months].copy()
            self._log(f"  History >= {filter.min_months} months: {df['isin'].nunique()} ETFs")

        # Deduplicate (one listing per ISIN) - optional
        if filter.deduplicate:
            # If we have months_of_data, pick the vwdId with longest history
            by_longest = 'months_of_data' in df.columns
            df = self._deduplicate(df, by_longest_history=by_longest)
            self._log(f"  Deduplicated: {len(df)} ETFs (by {'longest history' if by_longest else 'first'})")

        # Format output
        output_cols = {
            'isin': 'ISIN',
            'name': 'Name',
            'vwdId': 'vwdId',
            'exchange': 'exchange',
            'currency': 'currency',
            'totalExpenseRatio': 'TER',
            'fund_size': 'fund_size',
        }

        if 'months_of_data' in df.columns:
            output_cols['months_of_data'] = 'months_of_data'

        available = [c for c in output_cols if c in df.columns]
        result = df[available].rename(columns=output_cols).reset_index(drop=True)

        self._log(f"\nResult: {len(result)} ETFs")

        return result


# Convenience function
def fetch_etfs(filter: Optional[ETFFilter] = None, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch ETFs matching the filter criteria.

    Args:
        filter: ETFFilter with criteria
        verbose: Print progress messages

    Returns:
        DataFrame with ETF data
    """
    fetcher = ETFFetcher(verbose=verbose)
    return fetcher.fetch(filter)


if __name__ == "__main__":
    # Test: Fetch Irish accumulating ETFs with >= 5y history
    print("\nETF Fetcher Test\n")

    df = fetch_etfs(ETFFilter(
        isin_prefix="IE00",
        distribution="Accumulating",
        min_fund_size=100,
    ))

    print(f"\nSample results:")
    print(df.head(10).to_string())
