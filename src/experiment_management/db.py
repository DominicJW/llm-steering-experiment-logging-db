import sqlite3
from typing import Iterator, Optional

DB_PATH = "experiments.db"


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
    Columns now match the fields in dto.py.
    """
    conn = get_connection(path)
    cur = conn.cursor()

    # ----------------------------------------------------------------------
    # ExperimentTemplate – fields aligned with ExperimentTemplateDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ExperimentTemplate (
        experiment_template_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_group INTEGER,
        loss_name TEXT,
        loss_additional_parameters TEXT,
        optimizer_name TEXT,
        optimizer_additional_parameters TEXT,
        module_name TEXT,
        batch_size INTEGER,
        normalization REAL
    );
    """)

    # ----------------------------------------------------------------------
    # Vectors – matches VectorDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Vectors (
        vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vector_data BLOB,
        seed INTEGER
    );
    """)

    # ----------------------------------------------------------------------
    # ExperimentLiveInstance – matches ExperimentLiveInstanceDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ExperimentLiveInstance (
        experiment_instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vector_data BLOB,
        initial_vector_id INTEGER,
        iteration_count INTEGER,
        experiment_template_id INTEGER,
        FOREIGN KEY(initial_vector_id) REFERENCES Vectors(vector_id),
        FOREIGN KEY(experiment_template_id) REFERENCES ExperimentTemplate(experiment_template_id)
    );
    """)

    # ----------------------------------------------------------------------
    # ExperimentSnapshot – matches ExperimentSnapshotDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ExperimentSnapshot (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vector_id INTEGER,
        iteration_count INTEGER,
        experiment_instance_id INTEGER,
        FOREIGN KEY(vector_id) REFERENCES Vectors(vector_id),
        FOREIGN KEY(experiment_instance_id) REFERENCES ExperimentLiveInstance(experiment_instance_id)
    );
    """)

    # ----------------------------------------------------------------------
    # GeneratedOutput – matches GeneratedOutputDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS GeneratedOutput (
        output_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_id INTEGER,
        text TEXT,
        snapshot_id INTEGER,
        vanilla INTEGER,
        generation_details TEXT,
        FOREIGN KEY(snapshot_id) REFERENCES ExperimentSnapshot(snapshot_id)
    );
    """)

    # ----------------------------------------------------------------------
    # Metric – matches MetricDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Metric (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        value REAL,
        description TEXT,
        snapshot_id INTEGER,
        prompt_id INTEGER,
        generated_output_id INTEGER,
        FOREIGN KEY(snapshot_id) REFERENCES ExperimentSnapshot(snapshot_id),
        FOREIGN KEY(generated_output_id) REFERENCES GeneratedOutput(output_id)
    );
    """)

    # ----------------------------------------------------------------------
    # Prompt – matches PromptDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Prompt (
        prompt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT
    );
    """)

    # ----------------------------------------------------------------------
    # PromptGroup – matches PromptGroupDTO
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS PromptGroup (
        group_id INTEGER PRIMARY KEY AUTOINCREMENT
    );
    """)

    # ----------------------------------------------------------------------
    # GroupPrompts – many‑to‑many linking PromptGroup ↔ Prompt
    # ----------------------------------------------------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS GroupPrompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER,
        prompt_id INTEGER,
        FOREIGN KEY(group_id) REFERENCES PromptGroup(group_id),
        FOREIGN KEY(prompt_id) REFERENCES Prompt(prompt_id)
    );
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_schema()

