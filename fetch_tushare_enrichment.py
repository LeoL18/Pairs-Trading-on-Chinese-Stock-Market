from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "prices.sqlite"
LOG_PATH: Path | None = None


METADATA_ENDPOINTS = {
    "stock_basic": {
        "params": {"exchange": "", "list_status": ""},
        "fields": (
            "ts_code,symbol,name,area,industry,market,list_date,delist_date,"
            "list_status,is_hs,act_name,act_ent_type"
        ),
        "table": "stock_basic",
    },
    "namechange": {
        "params": {},
        "fields": "ts_code,name,start_date,end_date,ann_date,change_reason",
        "table": "stock_namechange",
    },
    "stock_company": {
        "params": {"exchange": ""},
        "fields": (
            "ts_code,exchange,chairman,manager,secretary,reg_capital,setup_date,"
            "province,city,website,email,office,business_scope,employees,main_business"
        ),
        "table": "stock_company",
    },
    "daily_basic": {
        "params": {},
        "fields": (
            "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,"
            "pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,"
            "total_mv,circ_mv"
        ),
        "table": "latest_daily_basic",
    },
}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def get_pro_api(token: str | None = None):
    token = token or os.environ.get("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        raise SystemExit(
            "No Tushare token found. Set TUSHARE_TOKEN or run ts.set_token(...) once."
        )
    return ts.pro_api(token=token, timeout=60)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def existing_symbols(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT ts_code FROM daily_prices ORDER BY ts_code"
    ).fetchall()
    return [row[0] for row in rows]


def symbols_missing_adj_factor(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT p.ts_code
        FROM daily_prices p
        LEFT JOIN adj_factor a
          ON a.ts_code = p.ts_code
         AND a.trade_date = p.trade_date
        WHERE a.ts_code IS NULL
        GROUP BY p.ts_code
        ORDER BY p.ts_code
        """
    ).fetchall()
    return [row[0] for row in rows]


def missing_adj_date_range(conn: sqlite3.Connection) -> tuple[str, str]:
    first_date, last_date = conn.execute(
        """
        SELECT MIN(p.trade_date), MAX(p.trade_date)
        FROM daily_prices p
        LEFT JOIN adj_factor a
          ON a.ts_code = p.ts_code
         AND a.trade_date = p.trade_date
        WHERE a.ts_code IS NULL
        """
    ).fetchone()
    return str(first_date), str(last_date)


def trading_dates(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT trade_date FROM daily_prices ORDER BY trade_date"
    ).fetchall()
    return [str(row[0]) for row in rows]


def latest_trade_date(conn: sqlite3.Connection) -> str:
    return str(conn.execute("SELECT MAX(trade_date) FROM daily_prices").fetchone()[0])


def price_date_range(conn: sqlite3.Connection) -> tuple[str, str]:
    first_date, last_date = conn.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM daily_prices"
    ).fetchone()
    return str(first_date), str(last_date)


def write_frame(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)


def emit(message: str) -> None:
    print(message, flush=True)
    if LOG_PATH is None:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def classify_tushare_error(exc: Exception) -> str:
    message = str(exc).lower()
    transient_markers = (
        "httpconnectionpool",
        "proxyerror",
        "connectionreseterror",
        "connection reset",
        "max retries exceeded",
        "unable to connect",
        "timed out",
        "timeout",
        "remote host",
    )
    if any(marker in message for marker in transient_markers):
        return "failed"
    quota_markers = (
        "quota",
        "limit",
        "rate",
        "exceed",
        "exceeded",
        "exceeds",
        "too many",
        "积分",
        "权限",
        "每分钟",
        "访问次数",
        "调用频次",
        "超过",
        "最多访问",
    )
    if any(marker in message for marker in quota_markers):
        return "quota_exceeded"
    return "failed"


def progress_line(
    *,
    label: str,
    done: int,
    total: int,
    rows: int,
    started_at: float | None = None,
    current: str | None = None,
) -> str:
    pct = (done / total * 100) if total else 100.0
    parts = [f"{label}: {done:,}/{total:,} ({pct:.1f}%)"]
    if current:
        parts.append(f"current={current}")
    parts.append(f"rows={rows:,}")
    if started_at and done:
        elapsed = time.time() - started_at
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate else 0
        parts.append(f"elapsed={format_seconds(elapsed)}")
        parts.append(f"eta={format_seconds(remaining)}")
    return " | ".join(parts)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def current_adj_row_count(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "adj_factor"):
        return 0
    return conn.execute("SELECT COUNT(*) FROM adj_factor").fetchone()[0]


def print_progress_snapshot(conn: sqlite3.Connection) -> None:
    total_dates = conn.execute(
        "SELECT COUNT(DISTINCT trade_date) FROM daily_prices"
    ).fetchone()[0]
    total_symbols = conn.execute(
        "SELECT COUNT(DISTINCT ts_code) FROM daily_prices"
    ).fetchone()[0]
    price_rows = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    adj_rows = current_adj_row_count(conn)

    date_status = pd.DataFrame()
    if table_exists(conn, "tushare_adj_date_fetch_status"):
        date_status = pd.read_sql_query(
            """
            SELECT status, COUNT(*) AS dates
            FROM tushare_adj_date_fetch_status
            GROUP BY status
            ORDER BY status
            """,
            conn,
        )
        ok_dates = conn.execute(
            "SELECT COUNT(*) FROM tushare_adj_date_fetch_status WHERE status='ok'"
        ).fetchone()[0]
        last_ok = conn.execute(
            """
            SELECT trade_date, rows, fetched_at
            FROM tushare_adj_date_fetch_status
            WHERE status='ok'
            ORDER BY trade_date DESC
            LIMIT 1
            """
        ).fetchone()
    else:
        ok_dates = 0
        last_ok = None

    emit("Current Tushare enrichment progress")
    emit(f"- price rows: {price_rows:,}")
    emit(f"- symbols in price table: {total_symbols:,}")
    emit(f"- trading dates in price table: {total_dates:,}")
    emit(f"- adj_factor rows stored: {adj_rows:,}")
    emit(f"- adj_factor date progress: {ok_dates:,}/{total_dates:,} ({ok_dates / total_dates * 100:.1f}%)")
    if last_ok:
        emit(f"- latest completed adj_factor date: {last_ok[0]} ({last_ok[1]:,} API rows, fetched {last_ok[2]})")
    if not date_status.empty:
        emit("\nDate fetch status:")
        emit(date_status.to_string(index=False))


def pull_metadata(pro, conn: sqlite3.Connection) -> None:
    report_rows = []
    latest_date = latest_trade_date(conn)

    for endpoint, spec in METADATA_ENDPOINTS.items():
        params = dict(spec["params"])
        if endpoint == "daily_basic":
            params["trade_date"] = latest_date

        try:
            df = pro.query(endpoint, fields=spec["fields"], **params)
            if df is None or df.empty:
                report_rows.append((endpoint, spec["table"], 0, "empty"))
                continue
            write_frame(conn, df, spec["table"])
            report_rows.append((endpoint, spec["table"], len(df), "ok"))
        except Exception as exc:
            status = classify_tushare_error(exc)
            report_rows.append((endpoint, spec["table"], 0, f"{status}: {exc}"))
            if status == "quota_exceeded":
                emit(f"Quota/access limit hit while fetching {endpoint}: {exc}")

    pd.DataFrame(
        report_rows, columns=["endpoint", "table_name", "rows", "status"]
    ).to_sql("tushare_fetch_report", conn, if_exists="replace", index=False)


def prepare_adj_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adj_factor (
            ts_code TEXT NOT NULL,
            trade_date INTEGER NOT NULL,
            adj_factor REAL NOT NULL,
            PRIMARY KEY (ts_code, trade_date)
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tushare_adj_date_fetch_status (
            trade_date INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0,
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            message TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tushare_adj_fetch_status (
            ts_code TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0,
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            message TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_adj_factor_trade_date ON adj_factor(trade_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_adj_factor_ts_trade ON adj_factor(ts_code, trade_date)"
    )


def done_symbols(conn: sqlite3.Connection) -> set[str]:
    if not table_exists(conn, "tushare_adj_fetch_status"):
        return set()
    rows = conn.execute(
        "SELECT ts_code FROM tushare_adj_fetch_status WHERE status='ok'"
    ).fetchall()
    return {row[0] for row in rows}


def done_dates(conn: sqlite3.Connection) -> set[str]:
    if not table_exists(conn, "tushare_adj_date_fetch_status"):
        return set()
    rows = conn.execute(
        "SELECT trade_date FROM tushare_adj_date_fetch_status WHERE status='ok'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def chunks(values: list[str], limit: int | None) -> Iterable[str]:
    count = 0
    for value in values:
        if limit is not None and count >= limit:
            break
        yield value
        count += 1


def pull_adj_factor(
    pro,
    conn: sqlite3.Connection,
    symbols: list[str],
    sleep_seconds: float,
    limit: int | None,
    retry: int,
    only_missing: bool = False,
) -> None:
    prepare_adj_tables(conn)
    if only_missing:
        start_date, end_date = missing_adj_date_range(conn)
        remaining = symbols
    else:
        start_date, end_date = price_date_range(conn)
        remaining = [symbol for symbol in symbols if symbol not in done_symbols(conn)]
    total = len(remaining) if limit is None else min(limit, len(remaining))
    started_at = time.time()
    emit(progress_line(label="adj_factor symbols", done=0, total=total, rows=current_adj_row_count(conn)))

    for idx, ts_code in enumerate(chunks(remaining, limit), 1):
        last_error = None
        for attempt in range(1, retry + 1):
            try:
                df = pro.query(
                    "adj_factor",
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields="ts_code,trade_date,adj_factor",
                )
                if df is None:
                    df = pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
                if not df.empty:
                    df["trade_date"] = df["trade_date"].astype(int)
                    df["adj_factor"] = df["adj_factor"].astype(float)
                    df.to_sql("_adj_factor_stage", conn, if_exists="replace", index=False)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO adj_factor (ts_code, trade_date, adj_factor)
                        SELECT s.ts_code, s.trade_date, s.adj_factor
                        FROM _adj_factor_stage s
                        JOIN daily_prices p
                          ON p.ts_code = s.ts_code
                         AND p.trade_date = s.trade_date
                        """
                    )
                    conn.execute("DROP TABLE IF EXISTS _adj_factor_stage")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tushare_adj_fetch_status
                    (ts_code, status, rows, fetched_at, message)
                    VALUES (?, 'ok', ?, CURRENT_TIMESTAMP, NULL)
                    """,
                    (ts_code, len(df)),
                )
                conn.commit()
                emit(
                    progress_line(
                        label="adj_factor symbols",
                        done=idx,
                        total=total,
                        rows=current_adj_row_count(conn),
                        started_at=started_at,
                        current=f"{ts_code} ({len(df):,} API rows)",
                    ),
                )
                break
            except Exception as exc:
                conn.rollback()
                last_error = str(exc)
                if classify_tushare_error(exc) == "quota_exceeded":
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO tushare_adj_fetch_status
                        (ts_code, status, rows, fetched_at, message)
                        VALUES (?, 'quota_exceeded', 0, CURRENT_TIMESTAMP, ?)
                        """,
                        (ts_code, last_error),
                    )
                    conn.commit()
                    emit(f"Quota/access limit hit at symbol {ts_code}: {last_error}")
                    raise SystemExit("Stopping because Tushare quota/access limit was reached.")
                time.sleep(max(sleep_seconds, 1.0) * attempt)
        else:
            status = classify_tushare_error(Exception(last_error or ""))
            conn.execute(
                """
                INSERT OR REPLACE INTO tushare_adj_fetch_status
                (ts_code, status, rows, fetched_at, message)
                VALUES (?, ?, 0, CURRENT_TIMESTAMP, ?)
                """,
                (ts_code, status, last_error),
            )
            conn.commit()
            emit(f"adj_factor symbol {idx}/{total} {ts_code}: {status}: {last_error}")

        if sleep_seconds:
            time.sleep(sleep_seconds)


def pull_adj_factor_by_date(
    pro,
    conn: sqlite3.Connection,
    dates: list[str],
    sleep_seconds: float,
    limit: int | None,
    retry: int,
) -> None:
    prepare_adj_tables(conn)
    remaining = [date for date in dates if date not in done_dates(conn)]
    total = len(remaining) if limit is None else min(limit, len(remaining))
    started_at = time.time()
    emit(progress_line(label="adj_factor dates", done=0, total=total, rows=current_adj_row_count(conn)))

    for idx, trade_date in enumerate(chunks(remaining, limit), 1):
        last_error = None
        for attempt in range(1, retry + 1):
            try:
                df = pro.query(
                    "adj_factor",
                    trade_date=trade_date,
                    fields="ts_code,trade_date,adj_factor",
                )
                if df is None:
                    df = pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
                if not df.empty:
                    df["trade_date"] = df["trade_date"].astype(int)
                    df["adj_factor"] = df["adj_factor"].astype(float)
                    df.to_sql("_adj_factor_stage", conn, if_exists="replace", index=False)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO adj_factor (ts_code, trade_date, adj_factor)
                        SELECT s.ts_code, s.trade_date, s.adj_factor
                        FROM _adj_factor_stage s
                        JOIN daily_prices p
                          ON p.ts_code = s.ts_code
                         AND p.trade_date = s.trade_date
                        """
                    )
                    conn.execute("DROP TABLE IF EXISTS _adj_factor_stage")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tushare_adj_date_fetch_status
                    (trade_date, status, rows, fetched_at, message)
                    VALUES (?, 'ok', ?, CURRENT_TIMESTAMP, NULL)
                    """,
                    (int(trade_date), len(df)),
                )
                conn.commit()
                emit(
                    progress_line(
                        label="adj_factor dates",
                        done=idx,
                        total=total,
                        rows=current_adj_row_count(conn),
                        started_at=started_at,
                        current=f"{trade_date} ({len(df):,} API rows)",
                    ),
                )
                break
            except Exception as exc:
                conn.rollback()
                last_error = str(exc)
                if classify_tushare_error(exc) == "quota_exceeded":
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO tushare_adj_date_fetch_status
                        (trade_date, status, rows, fetched_at, message)
                        VALUES (?, 'quota_exceeded', 0, CURRENT_TIMESTAMP, ?)
                        """,
                        (int(trade_date), last_error),
                    )
                    conn.commit()
                    emit(f"Quota/access limit hit at trade_date {trade_date}: {last_error}")
                    raise SystemExit("Stopping because Tushare quota/access limit was reached.")
                time.sleep(max(sleep_seconds, 1.0) * attempt)
        else:
            status = classify_tushare_error(Exception(last_error or ""))
            conn.execute(
                """
                INSERT OR REPLACE INTO tushare_adj_date_fetch_status
                (trade_date, status, rows, fetched_at, message)
                VALUES (?, ?, 0, CURRENT_TIMESTAMP, ?)
                """,
                (int(trade_date), status, last_error),
            )
            conn.commit()
            emit(f"adj_factor date {idx}/{total} {trade_date}: {status}: {last_error}")

        if sleep_seconds:
            time.sleep(sleep_seconds)


def create_analysis_views(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP VIEW IF EXISTS daily_prices_adjusted;
        CREATE VIEW daily_prices_adjusted AS
        SELECT
            p.ts_code,
            p.trade_date,
            p.open,
            p.high,
            p.low,
            p.close,
            p.pre_close,
            p.change,
            p.pct_chg,
            p.vol,
            p.amount,
            a.adj_factor,
            p.open * a.adj_factor AS adj_open,
            p.high * a.adj_factor AS adj_high,
            p.low * a.adj_factor AS adj_low,
            p.close * a.adj_factor AS adj_close,
            b.symbol,
            b.name,
            b.area,
            b.industry,
            b.market,
            b.list_date,
            b.delist_date,
            b.list_status,
            b.is_hs,
            b.act_name,
            b.act_ent_type
        FROM daily_prices p
        LEFT JOIN adj_factor a
          ON a.ts_code = p.ts_code
         AND a.trade_date = p.trade_date
        LEFT JOIN stock_basic b
          ON b.ts_code = p.ts_code;
        """
    )


def verify_enrichment(conn: sqlite3.Connection) -> pd.DataFrame:
    checks = [
        (
            "price_rows",
            "SELECT COUNT(*) FROM daily_prices",
        ),
        (
            "adj_factor_rows",
            "SELECT COUNT(*) FROM adj_factor",
        ),
        (
            "price_rows_with_adj_factor",
            """
            SELECT COUNT(*)
            FROM daily_prices p
            JOIN adj_factor a
              ON a.ts_code = p.ts_code
             AND a.trade_date = p.trade_date
            """,
        ),
        (
            "symbols_with_adj_factor",
            "SELECT COUNT(DISTINCT ts_code) FROM adj_factor",
        ),
        (
            "stock_basic_rows",
            "SELECT COUNT(*) FROM stock_basic",
        ),
    ]
    rows = []
    for name, query in checks:
        try:
            value = conn.execute(query).fetchone()[0]
        except Exception as exc:
            value = f"failed: {exc}"
        rows.append((name, value))
    report = pd.DataFrame(rows, columns=["check_name", "value"])
    report.to_sql("tushare_enrichment_summary", conn, if_exists="replace", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None)
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--limit-symbols", type=int, default=None)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--adj-only", action="store_true")
    parser.add_argument("--reset-adj", action="store_true")
    parser.add_argument("--adj-by-symbol", action="store_true")
    parser.add_argument("--fill-missing-adj", action="store_true")
    parser.add_argument("--progress-only", action="store_true")
    parser.add_argument(
        "--log-file",
        default=str(ROOT / "data" / "tushare_enrichment_progress.log"),
    )
    args = parser.parse_args()

    global LOG_PATH
    LOG_PATH = Path(args.log_file) if args.log_file else None
    emit("")
    emit(f"=== fetch_tushare_enrichment.py {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    symbols = []
    with connect() as conn:
        if args.progress_only:
            print_progress_snapshot(conn)
            return

    pro = get_pro_api(args.token)
    with connect() as conn:
        emit(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if not args.adj_only:
            pull_metadata(pro, conn)
        if not args.metadata_only:
            if args.reset_adj:
                conn.execute("DROP TABLE IF EXISTS adj_factor")
                conn.execute("DROP TABLE IF EXISTS tushare_adj_fetch_status")
                conn.execute("DROP TABLE IF EXISTS tushare_adj_date_fetch_status")
                conn.commit()
            if args.adj_by_symbol:
                symbols = (
                    symbols_missing_adj_factor(conn)
                    if args.fill_missing_adj
                    else existing_symbols(conn)
                )
                pull_adj_factor(
                    pro,
                    conn,
                    symbols,
                    sleep_seconds=args.sleep,
                    limit=args.limit_symbols,
                    retry=args.retry,
                    only_missing=args.fill_missing_adj,
                )
            else:
                pull_adj_factor_by_date(
                    pro,
                    conn,
                    trading_dates(conn),
                    sleep_seconds=args.sleep,
                    limit=args.limit_symbols,
                    retry=args.retry,
                )
        create_analysis_views(conn)
        report = verify_enrichment(conn)
        emit(report.to_string(index=False))


if __name__ == "__main__":
    main()
