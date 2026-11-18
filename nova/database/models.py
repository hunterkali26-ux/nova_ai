"""Database schema definitions and store classes for NOVA AI.

Stores:
- conversations / messages (chat history)
- documents (RAG knowledge metadata)
- model_registry (registered models)
- settings (application settings)
- training_history (training runs)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from pathlib import Path

from .db import DatabaseManager

TABLES: Dict[str, str] = {
    "conversations": """
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'New Conversation',
        model_name TEXT NOT NULL DEFAULT 'nova-tiny',
        rag_enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "messages": """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
        content TEXT NOT NULL,
        sources TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    )""",
}


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def init_schema(db_path: str | Path) -> None:
    """Create core tables for the given database if they do not exist."""
    db_path = Path(db_path)
    with DatabaseManager(db_path) as conn:
        for sql in TABLES.values():
            conn.execute(sql)
        conn.commit()
TABLES["documents"] = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_path TEXT,
        source_url TEXT,
        retrieved_at TEXT,
        license TEXT,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        indexed INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )"""
TABLES["model_registry"] = """
    CREATE TABLE IF NOT EXISTS model_registry (
        name TEXT PRIMARY KEY,
        identifier TEXT NOT NULL,
        model_type TEXT NOT NULL,
        architecture TEXT NOT NULL,
        file_path TEXT,
        quantization TEXT,
        context_length INTEGER,
        model_format TEXT,
        description TEXT,
        created_at TEXT NOT NULL
    )"""
TABLES["settings"] = """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )"""
TABLES["training_history"] = """
    CREATE TABLE IF NOT EXISTS training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        preset TEXT NOT NULL,
        epochs INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        learning_rate REAL NOT NULL,
        steps INTEGER NOT NULL,
        final_train_loss REAL,
        final_val_loss REAL,
        best_val_loss REAL,
        best_epoch INTEGER,
        created_at TEXT NOT NULL,
        checkpoint_path TEXT
    )"""


class DocumentStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, doc: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(id, title, source_type, source_path, source_url, retrieved_at, license, chunk_count, indexed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc["id"],
                doc["title"],
                doc["source_type"],
                doc.get("source_path"),
                doc.get("source_url"),
                doc.get("retrieved_at", now_iso()),
                doc.get("license"),
                int(doc.get("chunk_count", 0)),
                int(doc.get("indexed", 0)),
                doc.get("created_at", now_iso()),
            ),
        )
        self.conn.commit()

    def list(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()

    def get(self, doc_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def set_indexed(self, doc_id: str, chunk_count: int, indexed: bool = True) -> None:
        self.conn.execute(
            "UPDATE documents SET indexed = ?, chunk_count = ? WHERE id = ?",
            (int(indexed), int(chunk_count), doc_id),
        )
        self.conn.commit()

    def delete(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()


class ModelRegistryStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, entry: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO model_registry "
            "(name, identifier, model_type, architecture, file_path, quantization, context_length, model_format, description, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry["name"],
                entry["identifier"],
                entry["model_type"],
                entry["architecture"],
                entry.get("file_path"),
                entry.get("quantization"),
                entry.get("context_length"),
                entry.get("model_format"),
                entry.get("description"),
                entry.get("created_at", now_iso()),
            ),
        )
        self.conn.commit()

    def list(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM model_registry ORDER BY name").fetchall()

    def get(self, name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM model_registry WHERE name = ?", (name,)).fetchone()

    def delete(self, name: str) -> None:
        self.conn.execute("DELETE FROM model_registry WHERE name = ?", (name,))
        self.conn.commit()



class ConversationStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, conversation_id: str, title: str, model_name: str, rag_enabled: bool = True) -> None:
        self.conn.execute(
            "INSERT INTO conversations (id, title, model_name, rag_enabled, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, title, model_name, int(rag_enabled), now_iso(), now_iso()),
        )
        self.conn.commit()

    def list(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ).fetchall()

    def get(self, conversation_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()

    def update_touch(self, conversation_id: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now_iso(), conversation_id),
        )
        self.conn.commit()

    def delete(self, conversation_id: str) -> None:
        self.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()


class MessageStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(
        self,
        message_id: str,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, conversation_id, role, content, json.dumps(sources or []), now_iso()),
        )
        self.conn.commit()

    def list_for_conversation(self, conversation_id: str) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()

    def delete_for_conversation(self, conversation_id: str) -> None:
        self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        self.conn.commit()


class SettingsStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def set(self, key: str, value: Any) -> None:
        if not isinstance(value, str):
            value = json.dumps(value)
        self.conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        value = row["value"]
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def delete(self, key: str) -> None:
        self.conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        self.conn.commit()


class TrainingHistoryStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, entry: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO training_history "
            "(model_name, preset, epochs, batch_size, learning_rate, steps, final_train_loss, final_val_loss, best_val_loss, best_epoch, created_at, checkpoint_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.get("model_name"),
                entry.get("preset"),
                int(entry.get("epochs", 0)),
                int(entry.get("batch_size", 0)),
                float(entry.get("learning_rate", 0.0)),
                int(entry.get("steps", 0)),
                entry.get("final_train_loss"),
                entry.get("final_val_loss"),
                entry.get("best_val_loss"),
                entry.get("best_epoch"),
                entry.get("created_at", now_iso()),
                entry.get("checkpoint_path"),
            ),
        )
        self.conn.commit()

    def list(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM training_history ORDER BY created_at DESC"
        ).fetchall()

