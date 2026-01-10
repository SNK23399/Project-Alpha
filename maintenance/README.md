# Database Maintenance Scripts

This folder contains scripts for maintaining the ETF database. These scripts are typically run **monthly** to keep the data up-to-date.

## Scripts (Run in Order)

1. **[1_collect_etf_data.py](1_collect_etf_data.py)** - Collect ETF data from DEGIRO/JustETF
2. **[2_compare_databases.py](2_compare_databases.py)** - Compare old vs new database
3. **[3_analyze_data_quality.py](3_analyze_data_quality.py)** - Analyze data quality (optional)
4. **[4_cleanup_database_by_core.py](4_cleanup_database_by_core.py)** - Historical cleanup tool (rare use)

## Monthly Update Workflow

Run these commands in order:

### Step 1: Collect New Data

```bash
python maintenance/1_collect_etf_data.py --update
```

**What it does**:
- Connects to DEGIRO API
- Fetches current ETF universe (Irish domicile, accumulating, EUR, no leverage)
- Scrapes latest prices from JustETF
- Filters data from 2009-09-25 onwards (core ETF inception)
- Saves metadata snapshot (preserves historical TER, fund_size)
- Creates new database: `data/etf_database_new.db`

**Time**: ~30-60 minutes (depending on number of ETFs)

**Important**: Requires `.env` file with DEGIRO credentials

### Step 2: Verify New Database

```bash
python maintenance/2_compare_databases.py
```

**What it does**:
- Compares old vs new database
- Checks ETF metadata (new/removed/updated)
- Validates price data in overlap period
- Reports any mismatches or issues
- Gives verdict: PASSED or WARNINGS

**Expected results**:
- A few new ETFs (normal)
- Some price "mismatches" on recent dates (data corrections - normal)
- Date range should start at 2009-09-25
- Should show "[OK] VERIFICATION PASSED"

### 3. Replace Database (if verification passes)

**Windows**:
```bash
move data\etf_database.db data\etf_database_backup.db
move data\etf_database_new.db data\etf_database.db
```

**macOS/Linux**:
```bash
mv data/etf_database.db data/etf_database_backup.db
mv data/etf_database_new.db data/etf_database.db
```

### Step 4: Verify Data Quality (optional)

```bash
python maintenance/3_analyze_data_quality.py
```

**What it does**:
- Analyzes missing data patterns
- Detects price outliers
- Checks data coverage over time
- Validates core ETF (IE00B4L5Y983)
- Provides recommendations

**When to run**: After replacing database, or if you suspect data issues

## Detailed Script Documentation

### 1_collect_etf_data.py

**Purpose**: Fetch and store ETF data from DEGIRO/JustETF

**Usage**:
```bash
# Full collection (creates new database from scratch)
python maintenance/1_collect_etf_data.py

# Monthly update (saves metadata snapshot first)
python maintenance/1_collect_etf_data.py --update
```

**Output**: `data/etf_database_new.db`

**Filter criteria**:
- Irish domicile (IE00) - tax efficient
- Accumulating distribution
- EUR currency
- No leveraged ETFs
- Data filtered from 2009-09-25 onwards

### 2_compare_databases.py

**Purpose**: Compare old and new databases before replacing

**Usage**:
```bash
python maintenance/2_compare_databases.py
```

**Checks**:
1. ETF metadata comparison
2. Price data validation
3. Date range verification
4. Overall statistics
5. Verification verdict

**Safe to replace if**: "[OK] VERIFICATION PASSED"

### 3_analyze_data_quality.py

**Purpose**: Analyze data quality and identify issues

**Usage**:
```bash
python maintenance/3_analyze_data_quality.py
```

**Checks**:
1. Missing data patterns
2. Price movement outliers
3. Data availability by time period
4. Price level analysis
5. Core ETF validation
6. Recommendations

**When to use**: Monthly check or when investigating data issues

### 4_cleanup_database_by_core.py

**Purpose**: Remove data before core ETF inception (historical cleanup)

**Usage**:
```bash
# Dry run (shows what would be deleted)
python maintenance/4_cleanup_database_by_core.py

# Actually delete data
python maintenance/4_cleanup_database_by_core.py --execute
```

**⚠️ WARNING**: This permanently deletes data. Make backup first.

**When to use**: Only if you need to clean an old database that has pre-2009 data

## Important Notes

### Database Behavior

**When you re-run data collection**:
- ✅ Always creates `data/etf_database_new.db` (never overwrites production)
- ✅ New ETFs are automatically added
- ✅ Existing ETFs have metadata updated (TER, fund_size, etc.)
- ✅ Price history is completely replaced with fresh data
- ✅ Removed/delisted ETFs stay in database (historical preservation)
- ✅ You control when to replace the old database

### Prerequisites

1. **DEGIRO Account**: Active account with credentials
2. **Environment File**: `.env` in project root with:
   ```
   DEGIRO_USERNAME=your_username
   DEGIRO_PASSWORD=your_password
   DEGIRO_TOTP_SECRET=your_totp_secret  # Optional
   ```
3. **Python Packages**: See main project requirements
4. **Mobile App**: Have DEGIRO app ready to approve login (in-app approval flow)

### Troubleshooting

**Issue**: "Connection failed to DEGIRO"
- **Solution**: Check `.env` file exists and credentials are correct

**Issue**: "No approval received"
- **Solution**: Approve login on mobile app within 2 minutes

**Issue**: Many price mismatches in comparison
- **Check**: Are mismatches only on recent dates? (Normal - data corrections)
- **Check**: Are mismatches > €0.01? (Small differences are OK)
- **Action**: If concerned, investigate specific ISINs

**Issue**: Script is very slow
- **Normal**: JustETF scraping has rate limiting (0.3s per ETF)
- **Expected**: 30-60 minutes for 800+ ETFs
- **Tip**: Run during lunch break or overnight

## Monthly Checklist

- [ ] Run `1_collect_etf_data.py --update`
- [ ] Run `2_compare_databases.py`
- [ ] Check verification verdict
- [ ] Review any warnings
- [ ] Replace database if verification passes
- [ ] Optional: Run `3_analyze_data_quality.py`
- [ ] Delete old backup after confirming new database works

## Questions?

See main project documentation: [../CLAUDE.md](../CLAUDE.md)
