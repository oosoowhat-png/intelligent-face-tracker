import sqlite3
import os


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

        directory = os.path.dirname(db_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )

        self.create_tables()

    def create_tables(self):
        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT UNIQUE NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                embedding BLOB
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT
            )
        """)

        self.connection.commit()

    def add_visitor(
        self,
        face_id,
        first_seen,
        embedding
    ):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO visitors
            (face_id, first_seen, last_seen, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (
                face_id,
                first_seen,
                first_seen,
                embedding
            )
        )

        self.connection.commit()

    def update_last_seen(self, face_id, timestamp):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            UPDATE visitors
            SET last_seen = ?
            WHERE face_id = ?
            """,
            (timestamp, face_id)
        )

        self.connection.commit()

    def add_event(
        self,
        face_id,
        event_type,
        timestamp,
        image_path
    ):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO events
            (face_id, event_type, timestamp, image_path)
            VALUES (?, ?, ?, ?)
            """,
            (
                face_id,
                event_type,
                timestamp,
                image_path
            )
        )

        self.connection.commit()

    def get_visitor_count(self):
        cursor = self.connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM visitors"
        )

        return cursor.fetchone()[0]

    def close(self):
        self.connection.close()