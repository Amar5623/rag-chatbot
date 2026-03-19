# auth/user_store.py
# SQLite-backed user registry.
# Same pattern as ParentStore — no ORM, just raw sqlite3.

import sqlite3
import threading
from pathlib import Path


class UserStore:
    """
    Persistent SQLite user registry.

    Schema:
        users(id TEXT PK, email TEXT UNIQUE, hashed_password TEXT, created_at TEXT)

    Thread-safe via lock — FastAPI runs handlers concurrently.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS users (
        id               TEXT PRIMARY KEY,
        email            TEXT UNIQUE NOT NULL,
        hashed_password  TEXT NOT NULL,
        created_at       TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        print(f"  [USER STORE] Ready at {path}")

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(self.DDL)
            conn.commit()

    # ── WRITE ─────────────────────────────────────────────

    def create_user(self, user_id: str, email: str, hashed_password: str) -> bool:
        """
        Insert a new user. Returns True on success, False if email already exists.
        """
        try:
            with self._lock:
                with sqlite3.connect(self.path) as conn:
                    conn.execute(
                        "INSERT INTO users (id, email, hashed_password) VALUES (?, ?, ?)",
                        (user_id, email.lower().strip(), hashed_password),
                    )
                    conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    # ── READ ──────────────────────────────────────────────

    def get_by_email(self, email: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email.lower().strip(),),
            ).fetchone()
        return dict(row) if row else None

    def get_by_id(self, user_id: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def email_exists(self, email: str) -> bool:
        return self.get_by_email(email) is not None

    def __len__(self) -> int:
        with sqlite3.connect(self.path) as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


__all__ = ["UserStore"]