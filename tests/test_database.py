from app.database import Database


DB_PATH = "data/test_visitors.db"


db = Database(DB_PATH)


db.add_visitor(
    "PERSON_001",
    "2026-10-02 22:00:00"
)

db.update_visitor(
    "PERSON_001",
    "2026-10-02 22:05:00"
)

db.add_visit_event(
    person_id="PERSON_001",
    event_type="ENTRY",
    timestamp="2026-10-02 22:00:00",
    image_path="logs/entries/PERSON_001.jpg",
    track_id=1
)

db.add_visit_event(
    person_id="PERSON_001",
    event_type="EXIT",
    timestamp="2026-10-02 22:05:00",
    image_path="logs/exits/PERSON_001.jpg",
    track_id=1
)


print(
    "Unique visitor count:",
    db.get_unique_visitor_count()
)


db.close()

print("DATABASE TEST PASSED")