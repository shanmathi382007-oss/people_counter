"""
SQLite and CSV Crossing Event Logger
Provides persistent logging of line crossing events and on-demand CSV export.
Uses Python's native sqlite3 and csv modules for zero-dependency reliability.
"""

from __future__ import annotations
import os
import csv
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from .line_counter import CrossingEvent


class CrossingLogger:
    """
    Manages persistent logging of line crossing events to an SQLite database
    and provides functionality to export to CSV format.
    """

    def __init__(self, db_path: str = "people_counter.db", default_csv_path: str = "outputs/crossings_log.csv") -> None:
        """
        Initialize logger with database and CSV target paths.

        Args:
            db_path: Path to SQLite database file.
            default_csv_path: Default output path for CSV export.
        """
        self.db_path = db_path
        self.default_csv_path = default_csv_path

        # Ensure directory for database exists if path contains directories
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database with timeout."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the crossings database table if it doesn't already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crossings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    in_total INTEGER NOT NULL,
                    out_total INTEGER NOT NULL,
                    occupancy INTEGER NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON crossings(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_track_id ON crossings(track_id)
            """)
            conn.commit()

    def log_event(self, event: CrossingEvent) -> int:
        """
        Record a line crossing event to the database.

        Args:
            event: CrossingEvent object.

        Returns:
            The inserted database record ID.
        """
        iso_time = event.timestamp.isoformat(sep=" ", timespec="seconds")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO crossings (timestamp, track_id, direction, in_total, out_total, occupancy)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (iso_time, event.track_id, event.direction, event.in_total, event.out_total, event.occupancy))
            conn.commit()
            return cursor.lastrowid or 0

    def get_latest_counts(self) -> Tuple[int, int, int]:
        """
        Get the latest (in_total, out_total, occupancy) from the database.

        Returns:
            Tuple of (in_total, out_total, occupancy). Defaults to (0, 0, 0) if empty.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT in_total, out_total, occupancy FROM crossings
                ORDER BY id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return int(row["in_total"]), int(row["out_total"]), int(row["occupancy"])
            return 0, 0, 0

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent crossing events as a list of dictionaries.

        Args:
            limit: Maximum number of events to fetch.

        Returns:
            List of event dictionaries.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, track_id, direction, in_total, out_total, occupancy
                FROM crossings
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_events(self) -> List[Dict[str, Any]]:
        """
        Retrieve all crossing events as a list of dictionaries.

        Returns:
            List of all event records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, track_id, direction, in_total, out_total, occupancy
                FROM crossings
                ORDER BY id ASC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_dataframe(self) -> Any:
        """
        Load all crossing records into a Polars or pandas DataFrame.
        Falls back to list of dicts if neither is available.
        """
        rows = self.get_all_events()
        if not rows:
            try:
                import polars as pl
                return pl.DataFrame(schema={
                    "id": pl.Int64, "timestamp": pl.String, "track_id": pl.Int64,
                    "direction": pl.String, "in_total": pl.Int64, "out_total": pl.Int64,
                    "occupancy": pl.Int64
                })
            except Exception:
                return []

        try:
            import polars as pl
            df = pl.DataFrame(rows)
            if "timestamp" in df.columns:
                df = df.with_columns(pl.col("timestamp").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False))
            return df
        except Exception:
            try:
                import pandas as pd
                df = pd.DataFrame(rows)
                if not df.empty and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            except Exception:
                return rows

    def export_csv(self, csv_path: Optional[str] = None) -> str:
        """
        Export all database records to a CSV file using native csv module.

        Args:
            csv_path: Path to target CSV file (defaults to default_csv_path).

        Returns:
            The absolute path of the written CSV file.
        """
        target_path = csv_path or self.default_csv_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        rows = self.get_all_events()
        headers = ["id", "timestamp", "track_id", "direction", "in_total", "out_total", "occupancy"]

        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        return os.path.abspath(target_path)

    def clear_all(self) -> None:
        """Clears all records from the database (useful for test resets)."""
        with self._get_connection() as conn:
            conn.cursor().execute("DELETE FROM crossings")
            conn.commit()
