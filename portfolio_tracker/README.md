# Portfolio Tracker

Complete DEGIRO portfolio tracking system with database maintenance and visualization.

## Structure

```
portfolio_tracker/
├── maintain_portfolio.py           # Database maintenance script (run this first)
├── visualize_portfolio.ipynb       # Visualization notebook (run this after)
├── data/                           # Data folder (created automatically)
│   └── portfolio_analysis.db       # Single unified database
└── README.md                       # This file
```

## How to Use

### Step 1: Run Database Maintenance

This script handles **all data operations**:
- Connects to DEGIRO and fetches transactions
- Looks up ISINs, fund names, and vwd_ids
- Stores transactions in `portfolio_analysis.db` (normalized schema)
- Collects historical price data from DEGIRO (via ChartFetcher)
- Calculates daily holdings
- Computes portfolio values

**Run once:**
```bash
cd portfolio_tracker
python maintain_portfolio.py
```

You'll be prompted for DEGIRO login credentials and mobile app approval.

### Step 2: View Visualizations

After the script completes, open the visualization notebook:

**In VS Code or Jupyter:**
```bash
jupyter notebook visualize_portfolio.ipynb
```

Or open `visualize_portfolio.ipynb` in your IDE.

The notebook loads data from the analysis database and displays:
- **Portfolio value over time** (gross and net of fees)
- **Cumulative fees** incurred
- **Portfolio statistics** (current value, high/low, etc.)

## Database

Single unified database stored in `portfolio_tracker/data/` (created automatically):

### portfolio_analysis.db
**Complete portfolio data** - Normalized schema for all operations
- Location: `portfolio_tracker/data/portfolio_analysis.db`
- Updated by `maintain_portfolio.py`
- 4 tables:
  - `etfs`: Master data (ISIN, name, type, currency, vwd_id)
  - `trades`: Transaction history from DEGIRO (buy/sell, quantity, price, fees)
  - `prices`: Historical daily prices from DEGIRO (stored locally to future-proof against delistings)
  - `holdings`: Daily quantity held (calculated from transactions)

## Workflow

```
┌─────────────────────────────────────────┐
│  Run: python maintain_portfolio.py      │
├─────────────────────────────────────────┤
│ 1. Connect to DEGIRO                    │
│ 2. Fetch transactions                   │
│ 3. Lookup ISINs, names, & vwd_ids       │
│ 4. Store in portfolio_analysis.db       │
│ 5. Download historical prices (DEGIRO)  │
│ 6. Calculate daily holdings             │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Run: visualize_portfolio.ipynb         │
├─────────────────────────────────────────┤
│ 1. Load portfolio data from database    │
│ 2. Calculate statistics                 │
│ 3. Display Plotly visualization         │
└─────────────────────────────────────────┘
```

## Features

✅ **Automated**: One script updates everything
✅ **Price Data**: Stored locally to handle delistings
✅ **Historical**: Tracks portfolio from inception
✅ **Daily Holdings**: Calculates quantity held each day
✅ **Fee Tracking**: Shows total fees paid
✅ **Visualizations**: Interactive Plotly charts
✅ **Idempotent**: Safe to run multiple times

## Requirements

Install dependencies:
```bash
pip install degiro-connector pandas plotly
```

**Note**: `degiro-connector` includes the ChartFetcher module used for historical price data.

## Scheduling

You can schedule `maintain_portfolio.py` to run periodically:

**Windows Task Scheduler:**
```
python C:\path\to\maintain_portfolio.py
```

**Linux/Mac Cron:**
```bash
0 20 * * * cd /path/to/portfolio_tracker && python maintain_portfolio.py
```

Then run the visualization notebook whenever you want to check your portfolio.

## Troubleshooting

**"Database not found"**
- Run `maintain_portfolio.py` first

**"No data found"**
- Ensure you have transactions in DEGIRO
- Check that DEGIRO login succeeded

**"Price data not found"**
- Check that DEGIRO login succeeded
- Some ETFs may not have ChartFetcher data available
- Script skips already-fetched data on subsequent runs

## Notes

- The script skips duplicate transactions automatically
- Price data is fetched directly from DEGIRO via ChartFetcher (Quotecast service)
- Already-fetched prices are skipped on subsequent runs
- All operations are safe to repeat - no data is lost
- vwd_id is required for DEGIRO price fetching (obtained from API)
  - For Tradegate exchange: vwd_id is in vwdkey format (e.g., `IE00B4L5Y983.TRADE,E`)
  - For other exchanges: vwd_id is in numeric issueid format (e.g., `480015513`)
  - The ChartFetcher automatically detects the format and uses the correct API call
