from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class Memory:
    id: int
    text: str
    importance: float


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    text TEXT NOT NULL UNIQUE,
                    importance REAL NOT NULL DEFAULT 0.5
                );
                """
            )

    def add_turn(self, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO turns(role, content) VALUES (?, ?)", (role, content))

    def recent_turns(self, limit: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def add_memory(self, text: str, importance: float = 0.6) -> int:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("Memory text cannot be empty.")
        importance = max(0.0, min(1.0, float(importance)))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO memories(text, importance) VALUES (?, ?)",
                (cleaned, importance),
            )
            row = conn.execute("SELECT id FROM memories WHERE text = ?", (cleaned,)).fetchone()
        if row is None:
            # Not reachable through normal use - the INSERT above is OR IGNORE,
            # so the row exists either way. It becomes reachable if the database
            # is swapped or truncated between the two statements. An `assert`
            # was wrong here: python -O strips it, and the failure would then be
            # an int(None) TypeError from a stack frame that explains nothing.
            raise RuntimeError(
                f"Memory {cleaned!r} vanished between insert and read - "
                f"the store at {self.path} may have been modified concurrently."
            )
        return int(row["id"])

    def delete_memory(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def list_memories(self, limit: int = 100) -> list[Memory]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, text, importance FROM memories ORDER BY importance DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Memory(int(r["id"]), str(r["text"]), float(r["importance"])) for r in rows]

    def retrieve(self, query: str, limit: int) -> list[Memory]:
        if limit <= 0:
            return []
        query_tokens = _tokens(query)
        candidates = self.list_memories(limit=500)
        scored: list[tuple[float, Memory]] = []

        for memory in candidates:
            memory_tokens = _tokens(memory.text)
            overlap = len(query_tokens & memory_tokens)
            union = max(1, len(query_tokens | memory_tokens))
            score = (overlap / union) * 0.8 + memory.importance * 0.2
            if overlap > 0 or memory.importance >= 0.85:
                scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    def clear_turns(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM turns")
