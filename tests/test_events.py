from app.database import Database
from app.logger import EventLogger
from app.event_manager import EventManager


DB_PATH = "data/test_events.db"
LOG_PATH = "logs/test_events.log"


db = Database(DB_PATH)

logger = EventLogger(
    LOG_PATH
)

manager = EventManager(
    database=db,
    logger=logger,
    entry_directory="logs/entries",
    exit_directory="logs/exits",
    max_lost_frames=3
)


# Simulate PERSON_001 entering.
manager.process_detection(
    person_id="PERSON_001",
    track_id=1,
    face_crop=None,
    similarity=1.0,
    is_new=True
)


# Simulate the person being visible.
manager.update_frame(
    current_frame_number=1,
    visible_track_ids=[1]
)

manager.update_frame(
    current_frame_number=2,
    visible_track_ids=[1]
)


# Person disappears.
manager.update_frame(
    current_frame_number=3,
    visible_track_ids=[]
)

manager.update_frame(
    current_frame_number=4,
    visible_track_ids=[]
)

manager.update_frame(
    current_frame_number=5,
    visible_track_ids=[]
)


print(
    "Unique visitor count:",
    db.get_unique_visitor_count()
)

db.close()

print("EVENT TEST PASSED")