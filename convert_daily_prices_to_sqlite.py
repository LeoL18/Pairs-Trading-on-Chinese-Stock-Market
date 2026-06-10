from __future__ import annotations

import csv
import hashlib
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "data" / "daily_prices"
DB_PATH = ROOT / "data" / "prices.sqlite"
EXPECTED_HEADER = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
BATCH_SIZE = 10_000


def as_float(value: str) -> float | None:
    return None if value == "" else float(value)


def prepare_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;

        DROP TABLE IF EXISTS daily_prices;
        DROP TABLE IF EXISTS csv_manifest;

        CREATE TABLE daily_prices (
            ts_code TEXT NOT NULL,
            trade_date INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (ts_code, trade_date)
        ) WITHOUT ROWID;

        CREATE TABLE csv_manifest (
            source_file TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            rows INTEGER NOT NULL,
            bytes INTEGER NOT NULL
        );
        """
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def insert_file(conn: sqlite3.Connection, path: Path) -> int:
    rows: list[tuple[object, ...]] = []
    row_count = 0
    source_file = path.relative_to(ROOT).as_posix()

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise ValueError(f"{path} has unexpected header: {header!r}")

        for row in reader:
            if len(row) != len(EXPECTED_HEADER):
                raise ValueError(f"{path} has malformed row {row_count + 2}: {row!r}")
            rows.append(
                (
                    row[0],
                    int(row[1]),
                    as_float(row[2]),
                    as_float(row[3]),
                    as_float(row[4]),
                    as_float(row[5]),
                    as_float(row[6]),
                    as_float(row[7]),
                    as_float(row[8]),
                    as_float(row[9]),
                    as_float(row[10]),
                    source_file,
                )
            )
            row_count += 1
            if len(rows) >= BATCH_SIZE:
                conn.executemany(
                    """
                    INSERT INTO daily_prices
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                rows.clear()

    if rows:
        conn.executemany(
            """
            INSERT INTO daily_prices
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    conn.execute(
        """
        INSERT INTO csv_manifest (source_file, sha256, rows, bytes)
        VALUES (?, ?, ?, ?)
        """,
        (source_file, file_hash(path), row_count, path.stat().st_size),
    )
    return row_count


def verify(conn: sqlite3.Connection, expected_files: int, expected_rows: int) -> None:
    manifest_files, manifest_rows = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM csv_manifest"
    ).fetchone()
    stored_rows = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    mismatched_files = conn.execute(
        """
        SELECT COUNT(*)
        FROM csv_manifest m
        WHERE m.rows != (
            SELECT COUNT(*)
            FROM daily_prices d
            WHERE d.source_file = m.source_file
        )
        """
    ).fetchone()[0]

    if manifest_files != expected_files:
        raise RuntimeError(f"manifest file count {manifest_files} != {expected_files}")
    if manifest_rows != expected_rows:
        raise RuntimeError(f"manifest row total {manifest_rows} != {expected_rows}")
    if stored_rows != expected_rows:
        raise RuntimeError(f"database row total {stored_rows} != {expected_rows}")
    if mismatched_files:
        raise RuntimeError(f"{mismatched_files} files have mismatched row counts")


def main() -> None:
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {CSV_DIR}")

    if DB_PATH.exists():
        DB_PATH.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{DB_PATH}{suffix}")
        if sidecar.exists():
            sidecar.unlink()

    expected_rows = 0
    with sqlite3.connect(DB_PATH) as conn:
        prepare_database(conn)
        with conn: # rollback the entire load if any file fails to insert
            for index, path in enumerate(csv_files, 1):
                expected_rows += insert_file(conn, path)
                if index % 250 == 0 or index == len(csv_files):
                    print(f"converted {index}/{len(csv_files)} files")

        conn.execute("CREATE INDEX idx_daily_prices_trade_date ON daily_prices(trade_date)")
        conn.execute("CREATE INDEX idx_daily_prices_source_file ON daily_prices(source_file)")
        conn.execute("CREATE INDEX idx_daily_prices_ts_trade ON daily_prices(ts_code, trade_date);")
        conn.execute("PRAGMA optimize")
        verify(conn, len(csv_files), expected_rows)

    print(f"verified {len(csv_files)} files and {expected_rows} rows in {DB_PATH}")


if __name__ == "__main__":
    main()
