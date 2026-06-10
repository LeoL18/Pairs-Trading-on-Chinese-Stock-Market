"""
fetch_tushare_macroeconomics.py

Pulls all available macroeconomic data from Tushare Pro and stores it in
macroeconomics.sqlite (SQLite, WAL mode) alongside prices.sqlite.

Endpoints covered
-----------------
  Daily / high-frequency
    shibor           – Shanghai Interbank Offered Rate (daily, 8 tenors)
    shibor_lpr       – LPR Loan Prime Rate (daily)
    shibor_quote     – Shibor bank-level quotes (daily)
    libor            – USD LIBOR (daily, 7 tenors)
    hibor            – HKD HIBOR (daily, 8 tenors)

  Monthly
    cn_cpi           – CPI YoY / MoM (national + urban + rural)
    cn_ppi           – PPI YoY (factory-gate & sub-indices)
    cn_m             – Money supply M0 / M1 / M2
    cn_pmi           – PMI (manufacturing + non-manufacturing composites)
    sf_month         – Social Financing (total + components)

  Quarterly
    cn_gdp           – GDP + primary / secondary / tertiary breakdown

Usage
-----
    python fetch_tushare_macroeconomics.py --token YOUR_TOKEN
    python fetch_tushare_macroeconomics.py          # reads TUSHARE_TOKEN env var
    python fetch_tushare_macroeconomics.py --full    # ignore existing data, refetch all history
"""

import argparse
import logging
import os
import sqlite3
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import tushare as ts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "macroeconomics.sqlite"
LOG_PATH: Optional[Path] = None          # set via --log flag

# ---------------------------------------------------------------------------
# Historical start dates (go as far back as Tushare data allows)
# ---------------------------------------------------------------------------
DAILY_START   = "20160101"   # Shibor published since 2006 (20060101)
MONTHLY_START = "201601"     # monthly series; some go back to 1996 (199601)
QUARTERLY_START = "2016Q1"   # GDP goes back to early 1990s (1992Q1)
LIBOR_START   = "20160101"   # LIBOR data generally available from 2000 (20000101)
HIBOR_START   = "20160101"   # HIBOR data starts around 2013 (20130101)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path: Optional[Path]) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def upsert_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str,
              pk_cols: list[str]) -> int:
    """Insert-or-replace rows; returns number of rows written."""
    if df.empty:
        return 0
    df = df.copy()
    # write to a staging temp table then INSERT OR REPLACE into main
    staging = f"_staging_{table}"
    df.to_sql(staging, conn, if_exists="replace", index=False)
    cols = ", ".join(f"`{c}`" for c in df.columns)
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM {staging}"
    )
    conn.execute(f"DROP TABLE IF EXISTS {staging}")
    conn.commit()
    return len(df)


def ensure_table(conn: sqlite3.Connection, df: pd.DataFrame, table: str,
                 pk_cols: list[str]) -> None:
    """Create table from DataFrame schema if it doesn't exist; add missing columns."""
    if not table_exists(conn, table):
        df.head(0).to_sql(table, conn, if_exists="fail", index=False)
        pk = ", ".join(f"`{c}`" for c in pk_cols)
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS uix_{table} ON `{table}`({pk})"
        )
        conn.commit()
    else:
        existing = {
            row[1] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        for col in df.columns:
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
        conn.commit()


# ---------------------------------------------------------------------------
# Latest-date helpers  (avoid re-fetching data we already have)
# ---------------------------------------------------------------------------
def latest_date_in_table(conn: sqlite3.Connection, table: str,
                         date_col: str) -> Optional[str]:
    if not table_exists(conn, table):
        return None
    cur = conn.execute(f"SELECT MAX({date_col}) FROM {table}")
    val = cur.fetchone()[0]
    return val  # may be None if table is empty


# ---------------------------------------------------------------------------
# Generic fetch wrapper with retry
# ---------------------------------------------------------------------------
def fetch_with_retry(fn, retries: int = 3, pause: float = 1.5, **kwargs):
    attempt = 0
    while attempt < retries:
        try:
            df = fn(**kwargs)
            return df
        except Exception as exc:
            if "频率超限" in str(exc):
                logging.getLogger(__name__).warning("Rate limited, waiting 61s...")
                time.sleep(61)
                # don't increment attempt — this wasn't a real failure, just a wait
                continue
            elif attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
                attempt += 1
            else:
                raise exc


# ---------------------------------------------------------------------------
# Individual endpoint fetchers
# ---------------------------------------------------------------------------

def fetch_shibor(pro, conn, logger, full: bool) -> None:
    table = "shibor"
    date_col = "date"
    start = DAILY_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            # restart from day after latest
            dt = datetime.strptime(latest, "%Y%m%d") if len(latest) == 8 else datetime.strptime(latest, "%Y-%m-%d")
            start = (dt).strftime("%Y%m%d")  # re-fetch same day to be safe

    end = date.today().strftime("%Y%m%d")
    logger.info(f"[shibor] fetching {start} → {end}")
    df = fetch_with_retry(pro.shibor, start_date=start, end_date=end)
    if df is None or df.empty:
        logger.info("[shibor] no data returned")
        return
    # Normalise date column to YYYYMMDD string
    df["date"] = df["date"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[shibor] upserted {n} rows")


def fetch_shibor_lpr(pro, conn, logger, full: bool) -> None:
    table = "shibor_lpr"
    date_col = "date"
    start = DAILY_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = latest.replace("-", "")

    end = date.today().strftime("%Y%m%d")
    logger.info(f"[shibor_lpr] fetching {start} → {end}")
    df = fetch_with_retry(pro.shibor_lpr, start_date=start, end_date=end)
    if df is None or df.empty:
        logger.info("[shibor_lpr] no data returned")
        return
    df["date"] = df["date"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[shibor_lpr] upserted {n} rows")


def fetch_shibor_quote(pro, conn, logger, full: bool) -> None:
    """
    shibor_quote has a bank dimension so PK is (date, bank).
    Tushare limits to 2000 rows per call; chunk by year.
    """
    table = "shibor_quote"
    pk_cols = ["date", "bank"]
    start_year = int(DAILY_START[:4])
    if not full:
        latest = latest_date_in_table(conn, table, "date")
        if latest:
            start_year = int(str(latest)[:4])

    end_year = date.today().year
    all_dfs = []
    for yr in range(start_year, end_year + 1):
        s = f"{yr}0101"
        e = f"{yr}1231"
        logger.info(f"[shibor_quote] year {yr}")
        df = fetch_with_retry(pro.shibor_quote, start_date=s, end_date=e)
        if df is not None and not df.empty:
            all_dfs.append(df)
        time.sleep(0.3)

    if not all_dfs:
        logger.info("[shibor_quote] no data returned")
        return
    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = df["date"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, pk_cols)
    n = upsert_df(conn, df, table, pk_cols)
    logger.info(f"[shibor_quote] upserted {n} rows")


def fetch_libor(pro, conn, logger, full: bool) -> None:
    """
    LIBOR has a curr_type dimension; Tushare default is USD.
    PK: (date, curr_type)
    """
    table = "libor"
    pk_cols = ["date", "curr_type"]
    start = LIBOR_START
    if not full:
        latest = latest_date_in_table(conn, table, "date")
        if latest:
            start = latest.replace("-", "")

    end = date.today().strftime("%Y%m%d")
    currencies = ["USD", "EUR", "GBP", "JPY"]
    all_dfs = []
    for curr in currencies:
        logger.info(f"[libor] {curr} {start} → {end}")
        df = fetch_with_retry(pro.libor, start_date=start, end_date=end, curr_type=curr)
        if df is not None and not df.empty:
            df["curr_type"] = curr
            all_dfs.append(df)
        time.sleep(0.3)

    if not all_dfs:
        logger.info("[libor] no data returned")
        return
    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = df["date"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, pk_cols)
    n = upsert_df(conn, df, table, pk_cols)
    logger.info(f"[libor] upserted {n} rows")


def fetch_hibor(pro, conn, logger, full: bool) -> None:
    table = "hibor"
    date_col = "date"
    start = HIBOR_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = latest.replace("-", "")

    end = date.today().strftime("%Y%m%d")
    logger.info(f"[hibor] fetching {start} → {end}")
    df = fetch_with_retry(pro.hibor, start_date=start, end_date=end)
    if df is None or df.empty:
        logger.info("[hibor] no data returned")
        return
    df["date"] = df["date"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[hibor] upserted {n} rows")


# ---- Monthly series --------------------------------------------------------

def _current_month_str() -> str:
    return date.today().strftime("%Y%m")


def fetch_cn_cpi(pro, conn, logger, full: bool) -> None:
    table = "cn_cpi"
    date_col = "month"
    start = MONTHLY_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = str(latest).replace("-", "")[:6]

    end = _current_month_str()
    logger.info(f"[cn_cpi] fetching {start} → {end}")
    df = fetch_with_retry(pro.cn_cpi, start_m=start, end_m=end)
    if df is None or df.empty:
        logger.info("[cn_cpi] no data returned")
        return
    df["month"] = df["month"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[cn_cpi] upserted {n} rows")


def fetch_cn_ppi(pro, conn, logger, full: bool) -> None:
    table = "cn_ppi"
    date_col = "month"
    start = MONTHLY_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = str(latest).replace("-", "")[:6]

    end = _current_month_str()
    logger.info(f"[cn_ppi] fetching {start} → {end}")
    df = fetch_with_retry(pro.cn_ppi, start_m=start, end_m=end)
    if df is None or df.empty:
        logger.info("[cn_ppi] no data returned")
        return
    df["month"] = df["month"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[cn_ppi] upserted {n} rows")


def fetch_cn_m(pro, conn, logger, full: bool) -> None:
    """Money supply: M0, M1, M2 and their YoY / MoM growth."""
    table = "cn_m"
    date_col = "month"
    start = MONTHLY_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = str(latest).replace("-", "")[:6]

    end = _current_month_str()
    logger.info(f"[cn_m] fetching {start} → {end}")
    df = fetch_with_retry(pro.cn_m, start_m=start, end_m=end)
    if df is None or df.empty:
        logger.info("[cn_m] no data returned")
        return
    df["month"] = df["month"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[cn_m] upserted {n} rows")


def fetch_cn_pmi(pro, conn, logger, full: bool) -> None:
    table = "cn_pmi"
    date_col = "month"
    start = MONTHLY_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = str(latest).replace("-", "")[:6]

    end = _current_month_str()
    logger.info(f"[cn_pmi] fetching {start} → {end}")
    df = fetch_with_retry(pro.cn_pmi, start_m=start, end_m=end)
    if df is None or df.empty:
        logger.info("[cn_pmi] no data returned")
        return
    print(df.columns)
    df["month"] = df["month"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[cn_pmi] upserted {n} rows")


def fetch_sf_month(pro, conn, logger, full: bool) -> None:
    """Social Financing aggregate (月度社会融资规模)."""
    table = "sf_month"
    date_col = "month"
    start = "200201"   # Social Financing data starts around 2002
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = str(latest).replace("-", "")[:6]

    end = _current_month_str()
    logger.info(f"[sf_month] fetching {start} → {end}")
    df = fetch_with_retry(pro.sf_month, start_m=start, end_m=end)
    if df is None or df.empty:
        logger.info("[sf_month] no data returned")
        return
    df["month"] = df["month"].astype(str).str.replace("-", "")
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[sf_month] upserted {n} rows")


# ---- Quarterly series ------------------------------------------------------

def _current_quarter_str() -> str:
    today = date.today()
    q = (today.month - 1) // 3 + 1
    return f"{today.year}Q{q}"


def fetch_cn_gdp(pro, conn, logger, full: bool) -> None:
    table = "cn_gdp"
    date_col = "quarter"
    start = QUARTERLY_START
    if not full:
        latest = latest_date_in_table(conn, table, date_col)
        if latest:
            start = str(latest)  # already in YYYYQN format

    end = _current_quarter_str()
    logger.info(f"[cn_gdp] fetching {start} → {end}")
    df = fetch_with_retry(pro.cn_gdp, start_q=start, end_q=end)
    if df is None or df.empty:
        logger.info("[cn_gdp] no data returned")
        return
    ensure_table(conn, df, table, [date_col])
    n = upsert_df(conn, df, table, [date_col])
    logger.info(f"[cn_gdp] upserted {n} rows")


# ---------------------------------------------------------------------------
# Metadata table: records last successful run per endpoint
# ---------------------------------------------------------------------------
def record_run(conn: sqlite3.Connection, endpoint: str, status: str,
               rows: int = 0) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _fetch_log (
            endpoint  TEXT PRIMARY KEY,
            last_run  TEXT,
            status    TEXT,
            rows      INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO _fetch_log (endpoint, last_run, status, rows)
        VALUES (?, ?, ?, ?)
        """,
        (endpoint, datetime.utcnow().isoformat(timespec="seconds"), status, rows),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ENDPOINTS = [
    # (name,            fetcher_fn)
    ("shibor",          fetch_shibor),
    ("shibor_lpr",      fetch_shibor_lpr),
    ("shibor_quote",    fetch_shibor_quote),
    ("libor",           fetch_libor),
    ("hibor",           fetch_hibor),
    ("cn_cpi",          fetch_cn_cpi),
    ("cn_ppi",          fetch_cn_ppi),
    ("cn_m",            fetch_cn_m),
    ("cn_pmi",          fetch_cn_pmi),
    ("sf_month",        fetch_sf_month),
    ("cn_gdp",          fetch_cn_gdp),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Tushare macro data → macroeconomics.sqlite")
    p.add_argument("--token", default=os.getenv("TUSHARE_TOKEN", ""),
                   help="Tushare Pro token (or set TUSHARE_TOKEN env var)")
    p.add_argument("--db", default=str(DB_PATH),
                   help=f"Path to SQLite DB (default: {DB_PATH})")
    p.add_argument("--log", default=None,
                   help="Optional path for log file")
    p.add_argument("--full", action="store_true",
                   help="Ignore existing data and re-fetch full history")
    p.add_argument("--endpoints", nargs="*",
                   default=[e[0] for e in ENDPOINTS],
                   help="Subset of endpoints to fetch (default: all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log_path = Path(args.log) if args.log else None
    logger = setup_logging(log_path)

    if not args.token:
        logger.error("No Tushare token provided. Use --token or set TUSHARE_TOKEN.")
        raise SystemExit(1)

    ts.set_token(args.token)
    pro = ts.pro_api()

    db_path = Path(args.db)
    conn = get_connection(db_path)
    logger.info(f"Connected to DB: {db_path}")
    logger.info(f"Full history mode: {args.full}")

    requested = set(args.endpoints)
    for name, fn in ENDPOINTS:
        if name not in requested:
            logger.info(f"[{name}] skipped (not in --endpoints list)")
            continue
        try:
            fn(pro, conn, logger, full=args.full)
            record_run(conn, name, "ok")
        except Exception as exc:
            logger.error(f"[{name}] FAILED: {exc}", exc_info=True)
            record_run(conn, name, f"error: {exc}")

    conn.close()
    logger.info("All done.")


if __name__ == "__main__":
    main()