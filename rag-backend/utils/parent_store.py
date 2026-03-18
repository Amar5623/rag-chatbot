# utils/parent_store.py
# SQLite-backed parent chunk store.
# Replaces the pickle file — no schema fragility, proper transactions,
# JSON-stable serialization.

import sqlite3
import json
from pathlib import Path


class ParentStore:
    """
    Persistent key-value store for parent chunks using SQLite.

    Keys   : parent_id strings (set by HierarchicalChunker)
    Values : parent chunk dicts (content, source, page, heading, etc.)

    Why SQLite over pickle:
      - Schema changes don't silently corrupt old data
      - JSON serialization is stable across Python versions
      - Batch fetch in one SQL query instead of N dict lookups
      - Proper reset/wipe with DELETE, not file deletion
    """

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parents (
                    parent_id TEXT PRIMARY KEY,
                    data      TEXT NOT NULL
                )
            """)
            conn.commit()

    # ── Write ─────────────────────────────────────────────────

    def add(self, parents: dict) -> None:
        """Insert or replace parent chunks. Safe to call incrementally."""
        if not parents:
            return
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO parents (parent_id, data) VALUES (?, ?)",
                [(k, json.dumps(v)) for k, v in parents.items()],
            )
            conn.commit()
        print(f"  [PARENT STORE] Saved {len(parents)} parents")

    def reset(self) -> None:
        """Delete all parents. Called on KB wipe."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM parents")
            conn.commit()
        print("  [PARENT STORE] Cleared")

    # ── Read ──────────────────────────────────────────────────

    def get(self, parent_id: str) -> dict | None:
        """Fetch a single parent by ID. Returns None if not found."""
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT data FROM parents WHERE parent_id = ?", (parent_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_batch(self, parent_ids: list[str]) -> dict[str, dict]:
        """
        Fetch multiple parents in one query.
        Returns {parent_id: parent_dict} for found IDs only.
        """
        if not parent_ids:
            return {}
        unique = list(set(parent_ids))
        placeholders = ",".join("?" * len(unique))
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                f"SELECT parent_id, data FROM parents WHERE parent_id IN ({placeholders})",
                unique,
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    # ── Stats ─────────────────────────────────────────────────

    def count(self) -> int:
        with sqlite3.connect(self.path) as conn:
            return conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    def __len__(self) -> int:
        return self.count()