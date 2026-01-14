import os
import sqlite3
import torch
from config import config


def _get_connection():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round INTEGER NOT NULL,
                train_loss REAL,
                train_accuracy REAL,
                train_precision REAL,
                train_recall REAL,
                train_f1 REAL,
                test_accuracy REAL,
                test_precision REAL,
                test_recall REAL,
                test_f1 REAL,
                test_auc REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                risk_score REAL,
                risk_category TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def record_training_round(round_number, train_metrics, test_metrics):
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO training_history (
                round,
                train_loss,
                train_accuracy,
                train_precision,
                train_recall,
                train_f1,
                test_accuracy,
                test_precision,
                test_recall,
                test_f1,
                test_auc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_number,
                train_metrics.get("loss"),
                train_metrics.get("accuracy"),
                train_metrics.get("precision"),
                train_metrics.get("recall"),
                train_metrics.get("f1"),
                test_metrics.get("accuracy"),
                test_metrics.get("precision"),
                test_metrics.get("recall"),
                test_metrics.get("f1"),
                test_metrics.get("auc"),
            ),
        )


def get_training_history():
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM training_history
            ORDER BY round ASC, id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def record_prediction(risk_score, risk_category):
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions (risk_score, risk_category)
            VALUES (?, ?)
            """,
            (risk_score, risk_category),
        )


def get_prediction_count():
    with _get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM predictions").fetchone()
    return int(row["count"])


def record_model_version(version, path):
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO model_versions (version, path)
            VALUES (?, ?)
            """,
            (version, path),
        )


def list_model_versions():
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT version, path, created_at
            FROM model_versions
            ORDER BY version DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_model_version(version):
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT version, path, created_at
            FROM model_versions
            WHERE version = ?
            LIMIT 1
            """,
            (version,),
        ).fetchone()
    return dict(row) if row else None


def get_latest_model_version():
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT version, path, created_at
            FROM model_versions
            ORDER BY version DESC
            LIMIT 1
            """
        ).fetchone()
    return dict(row) if row else None


def get_next_model_version():
    latest = get_latest_model_version()
    if not latest:
        return 1
    return int(latest["version"]) + 1


def save_model_version(model):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    version = get_next_model_version()
    filename = f"model_v{version}.pt"
    path = os.path.join(config.MODEL_DIR, filename)
    torch.save(model.state_dict(), path)
    record_model_version(version, path)
    return {
        "version": version,
        "path": path
    }
