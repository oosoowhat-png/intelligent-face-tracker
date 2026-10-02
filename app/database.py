import sqlite3
from pathlib import Path

import numpy as np


class Database:

    def __init__(self, db_path):

        self.db_path = db_path

        Path(
            db_path
        ).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            db_path
        )

        self.create_tables()

    # ======================================================
    # CREATE TABLES
    # ======================================================

    def create_tables(self):

        cursor = self.connection.cursor()

        # --------------------------------------------------
        # Visitors
        # --------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS visitors (
                person_id TEXT PRIMARY KEY,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                visit_count INTEGER DEFAULT 1
            )
            """
        )

        # --------------------------------------------------
        # Visits / events
        # --------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                track_id INTEGER
            )
            """
        )

        # --------------------------------------------------
        # Face embeddings
        #
        # One persistent embedding is stored for each
        # registered person.
        # --------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                person_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                dtype TEXT NOT NULL
            )
            """
        )

        self.connection.commit()

    # ======================================================
    # VISITOR METHODS
    # ======================================================

    def add_visitor(
        self,
        person_id,
        timestamp
    ):
        """
        Add a visitor if they do not already exist.

        This does NOT increment visit_count.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT OR IGNORE INTO visitors
            (
                person_id,
                first_seen,
                last_seen,
                visit_count
            )
            VALUES (?, ?, ?, 1)
            """,
            (
                person_id,
                timestamp,
                timestamp
            )
        )

        self.connection.commit()

    def record_entry(
        self,
        person_id,
        timestamp
    ):
        """
        Record a new visit/session.

        First-ever visitor:
            visit_count = 1

        Returning visitor:
            visit_count += 1
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT person_id
            FROM visitors
            WHERE person_id = ?
            """,
            (person_id,)
        )

        existing = cursor.fetchone()

        if existing is None:

            cursor.execute(
                """
                INSERT INTO visitors
                (
                    person_id,
                    first_seen,
                    last_seen,
                    visit_count
                )
                VALUES (?, ?, ?, 1)
                """,
                (
                    person_id,
                    timestamp,
                    timestamp
                )
            )

        else:

            cursor.execute(
                """
                UPDATE visitors
                SET
                    last_seen = ?,
                    visit_count = visit_count + 1
                WHERE person_id = ?
                """,
                (
                    timestamp,
                    person_id
                )
            )

        self.connection.commit()

    def update_visitor(
        self,
        person_id,
        timestamp
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            UPDATE visitors
            SET last_seen = ?
            WHERE person_id = ?
            """,
            (
                timestamp,
                person_id
            )
        )

        self.connection.commit()

    # ======================================================
    # VISIT EVENTS
    # ======================================================

    def add_visit_event(
        self,
        person_id,
        event_type,
        timestamp,
        image_path=None,
        track_id=None
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO visits
            (
                person_id,
                event_type,
                timestamp,
                image_path,
                track_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                person_id,
                event_type,
                timestamp,
                image_path,
                track_id
            )
        )

        self.connection.commit()

    # ======================================================
    # EMBEDDING STORAGE
    # ======================================================

    def save_embedding(
        self,
        person_id,
        embedding
    ):
        """
        Store a face embedding as a SQLite BLOB.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        embedding_blob = (
            embedding.tobytes()
        )

        dimension = (
            embedding.shape[0]
        )

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO face_embeddings
            (
                person_id,
                embedding,
                dimension,
                dtype
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                person_id,
                embedding_blob,
                dimension,
                "float32"
            )
        )

        self.connection.commit()

    def get_all_embeddings(self):
        """
        Load all persistent face embeddings.

        Returns:
            dict:
                {
                    "PERSON_001": numpy_array,
                    "PERSON_002": numpy_array
                }
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT
                person_id,
                embedding,
                dimension,
                dtype
            FROM face_embeddings
            ORDER BY person_id
            """
        )

        rows = cursor.fetchall()

        embeddings = {}

        for (
            person_id,
            embedding_blob,
            dimension,
            dtype
        ) in rows:

            if dtype != "float32":
                continue

            embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float32
            ).copy()

            if embedding.size != dimension:
                continue

            embeddings[
                person_id
            ] = embedding

        return embeddings

    # ======================================================
    # STATISTICS
    # ======================================================

    def get_unique_visitor_count(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM visitors
            """
        )

        result = cursor.fetchone()

        return result[0]

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        self.connection.close()