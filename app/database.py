import sqlite3
from pathlib import Path


class Database:
    def __init__(self, db_path):
        self.db_path = db_path

        Path(db_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            db_path
        )

        self.create_tables()

    def create_tables(self):

        cursor = self.connection.cursor()

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

        self.connection.commit()

    def add_visitor(
        self,
        person_id,
        timestamp
    ):

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

    def close(self):

        self.connection.close()