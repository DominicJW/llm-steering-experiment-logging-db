from dataclasses import fields
import sqlite3
from typing import Dict, List, Optional, Tuple

from .dto import (
    ExperimentLiveInstanceDTO,
    ExperimentSnapshotDTO,
    ExperimentTemplateDTO,
    GeneratedOutputDTO,
    GroupPromptLinkDTO,
    MetricDTO,
    PromptDTO,
    PromptGroupDTO,
    VectorDTO,
    _table_name_for_dto,
)

DB_PATH = "experiments.db"

_TABLE_REGISTRATION_ORDER: List[type] = [
    ExperimentTemplateDTO,
    VectorDTO,
    ExperimentLiveInstanceDTO,
    ExperimentSnapshotDTO,
    PromptDTO,
    PromptGroupDTO,
    GeneratedOutputDTO,
    MetricDTO,
    GroupPromptLinkDTO,
]


def _persisted_field_defs(dto_type: type):
    return [f for f in fields(dto_type) if f.metadata.get("persist", True)]


def _column_sql_and_signature(dto_type: type):
    column_sql = []
    signature = []
    for f in _persisted_field_defs(dto_type):
        sql_type = str(f.metadata.get("sql_type", "")).upper().strip()
        if not sql_type:
            raise ValueError(f"Field '{f.name}' in {dto_type.__name__} is missing sql_type metadata")

        is_pk = bool(f.metadata.get("primary_key", False))
        is_autoincrement = bool(f.metadata.get("autoincrement", False))

        parts = [f.name, sql_type]
        if is_pk:
            parts.append("PRIMARY KEY")
        if is_autoincrement:
            parts.append("AUTOINCREMENT")

        column_sql.append(" ".join(parts))
        signature.append((f.name, sql_type, 1 if is_pk else 0))

    return column_sql, signature


def _table_exists(cur: sqlite3.Cursor, table_name: str) -> bool:
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _actual_table_signature(cur: sqlite3.Cursor, table_name: str):
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    signature = []
    for row in rows:
        col_type = (row["type"] or "").upper().strip()
        signature.append((row["name"], col_type, int(row["pk"])))
    return signature


def _create_table(cur: sqlite3.Cursor, dto_type: type) -> None:
    table_name = _table_name_for_dto(dto_type)
    column_sql, _ = _column_sql_and_signature(dto_type)
    fk_sql = [
        f"FOREIGN KEY({f.name}) REFERENCES {_table_name_for_dto(f.metadata['foreign_dto_type'])}({f.metadata['foreign_field']})"
        for f in fields(dto_type)
        if f.metadata.get("foreign_key", False)
    ]
    all_defs = column_sql + fk_sql
    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    " + ",\n    ".join(all_defs) + "\n);"
    cur.execute(create_sql)


def _validate_existing_table(cur: sqlite3.Cursor, dto_type: type) -> None:
    table_name = _table_name_for_dto(dto_type)
    _, expected = _column_sql_and_signature(dto_type)
    actual = _actual_table_signature(cur, table_name)

    if actual != expected:
        raise ValueError(
            f"Schema mismatch for table '{table_name}'. Expected columns/types/pk {expected}, found {actual}. "
            "This project is configured for new DB files only; recreate the SQLite database."
        )


def get_connection(path: Optional[str] = None) -> sqlite3.Connection:
    """
    Return a sqlite3.Connection with row_factory set to sqlite3.Row so
    repository code can use row['colname'].
    """
    conn = sqlite3.connect(path or DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(path: Optional[str] = None) -> None:
    """
    Create the tables expected by the repositories and DTOs.
    DTO metadata is the source of truth for columns and types.

    Existing table validation is strict (new-DB-only policy): if a table exists
    with mismatched columns/types/pk metadata, an error is raised.
    """
    conn = get_connection(path)
    cur = conn.cursor()

    for dto_type in _TABLE_REGISTRATION_ORDER:
        table_name = _table_name_for_dto(dto_type)
        if _table_exists(cur, table_name):
            _validate_existing_table(cur, dto_type)
        else:
            _create_table(cur, dto_type)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_schema()
