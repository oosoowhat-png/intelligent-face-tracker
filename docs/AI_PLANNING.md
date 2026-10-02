# AI Planning and Development Workflow

## 1. Purpose

This document describes the planning process used to build the Intelligent Face Tracker with Auto Registration and Visitor Counting.

AI coding assistance was used throughout development to help with architecture planning, modular implementation, debugging, testing, and documentation.

All generated code was reviewed and tested as part of the development process.

---

# 2. Requirement Analysis

The initial requirements were analyzed into the following functional requirements:

1. Detect faces from video.
2. Track detected faces.
3. Generate face embeddings.
4. Automatically register new visitors.
5. Recognize previously registered visitors.
6. Maintain persistent visitor identities.
7. Detect entry events.
8. Detect exit events.
9. Save timestamped face crops.
10. Store metadata in a database.
11. Maintain a mandatory event log.
12. Prevent duplicate counting of returning visitors.
13. Support configurable frame skipping.
14. Process multiple videos.
15. Support future RTSP input.
16. Provide clear documentation and reproducible setup.

---

# 3. Feature Breakdown

The application was divided into the following modules:

| Feature                | Module               |
| ---------------------- | -------------------- |
| Configuration          | `config.py`          |
| Face Detection         | `detector.py`        |
| Tracking               | `tracker.py`         |
| Face Embedding         | `recognizer.py`      |
| Persistent Identity    | `face_registry.py`   |
| Database               | `database.py`        |
| Events                 | `event_manager.py`   |
| Logging                | `logger.py`          |
| Single Video Pipeline  | `video_processor.py` |
| Multi-Video Processing | `batch_processor.py` |

This modular approach makes it possible to improve individual components without redesigning the entire application.

---

# 4. Architecture Planning

The architecture was planned as a sequential AI video-processing pipeline:

```text
Video Input
    ↓
OpenCV
    ↓
YOLO Face Detection
    ↓
ByteTrack
    ↓
Face Crop
    ↓
InsightFace
    ↓
Face Embedding
    ↓
Face Registry
    ↓
Recognition / Registration
    ↓
Event Manager
    ↓
Database + Logs + Images
```

The most important architectural decision was to separate temporary tracking identity from persistent face identity.

ByteTrack produces temporary track IDs.

For example:

```text
Track ID = 139
```

may later change to:

```text
Track ID = 2
```

The persistent identity remains:

```text
PERSON_001
```

This prevents tracker-ID changes from creating duplicate visitors.

---

# 5. AI Development Workflow

## Phase 1 – Planning

AI assistance was used to break the requirements into manageable software components.

The goal was to avoid implementing the entire application as one large Python script.

The application was therefore divided into independent modules.

---

## Phase 2 – Face Detection

YOLO was selected for face detection.

The detector receives an OpenCV frame and returns:

```text
Bounding box
Confidence
```

The detection confidence is configurable.

---

## Phase 3 – Tracking

ByteTrack was selected for multi-object tracking.

Tracking provides temporary IDs for faces detected across consecutive frames.

The tracking layer is independent from persistent identity recognition.

---

## Phase 4 – Face Recognition

InsightFace was selected for face embedding generation.

The face crop is passed through InsightFace and converted into a numerical embedding vector.

The embedding represents facial characteristics in a form that can be compared with previously stored embeddings.

---

## Phase 5 – Identity Matching

Cosine similarity is used to compare a new embedding with stored visitor embeddings.

Conceptually:

```text
New Face Embedding
        ↓
Compare with stored embeddings
        ↓
Find highest similarity
        ↓
Similarity >= threshold?
      /       \
    YES        NO
     ↓          ↓
Existing      Register
PERSON ID     New PERSON ID
```

The similarity threshold is configurable through `config.json`.

---

# 6. Persistent Identity Design

A major requirement was that a returning person should not be counted as a new visitor.

Therefore the system stores face embeddings in SQLite.

Example:

```text
PERSON_001 → embedding
PERSON_002 → embedding
PERSON_003 → embedding
```

When another video is processed, the Face Registry loads these embeddings from the database.

This allows identity matching to continue across separate videos and application instances.

---

# 7. Entry and Exit Design

Entry and exit handling was implemented separately from face recognition.

A recognized person becomes an active visitor session.

An ENTRY event is generated when the person becomes active.

The system continuously updates the active tracking state.

When the person is no longer tracked for the configured timeout, an EXIT event is generated.

If the video ends while the person is still active, the application flushes the remaining active sessions and records EXIT events.

---

# 8. Event Logging Plan

The hackathon requires mandatory event logging.

Therefore an event logger was included.

Important events include:

```text
REGISTRATION
RECOGNITION
EMBEDDING
TRACKING
ENTRY
EXIT
```

The events are written to:

```text
logs/events.log
```

This provides an auditable record of system activity.

---

# 9. Database Planning

SQLite was selected because it is:

* Lightweight
* Built into Python
* Easy to deploy
* Suitable for the hackathon workload
* Persistent across application runs

Three tables were designed:

```text
visitors
visits
face_embeddings
```

The `visitors` table maintains persistent visitor identities.

The `visits` table stores event history.

The `face_embeddings` table stores the numerical representation used for re-identification.

---

# 10. Multi-Video Planning

The Google Drive dataset contains multiple videos.

Instead of manually processing each video, a batch processor was created.

The batch processor:

1. Scans the `sample/` directory.
2. Finds supported video files.
3. Sorts the files.
4. Creates a processing instance for each video.
5. Processes each video.
6. Reuses the persistent SQLite database.
7. Reports successful and failed videos.

This allows the system to process the complete dataset automatically.

---

# 11. Configuration Planning

Runtime parameters were moved into:

```text
config.json
```

This avoids hard-coding important values.

Configurable parameters include:

```text
frame_skip
confidence_threshold
image_size
similarity_threshold
max_lost_frames
database path
log paths
```

---

# 12. Testing Workflow

Testing was performed incrementally.

### Test 1 – Single Video

The initial pipeline was tested using a sample video.

The expected behavior was:

```text
Face detected
↓
Face tracked
↓
Embedding generated
↓
PERSON ID assigned
↓
ENTRY
↓
Recognition
↓
EXIT
```

### Test 2 – Persistent Identity

The same video was processed again after identities had been stored.

The goal was to verify that existing identities were recognized rather than generating new visitor IDs.

### Test 3 – Tracker ID Changes

The system was tested with ByteTrack IDs changing during a person's presence.

The persistent `PERSON_xxx` identity was kept independent of the temporary tracker ID.

### Test 4 – Multiple Videos

The batch processor was tested with the complete provided dataset.

### Final Validation

All 23 sample videos were processed.

Final database result:

```text
97 unique visitors

144 ENTRY events

144 EXIT events
```

The database also contained visitors with multiple visit counts, demonstrating persistent identity recognition across sessions.

---

# 13. Compute Planning

The application consists of both CPU and neural-network workloads.

## CPU

CPU operations:

* Video decoding
* Frame processing
* Cropping
* Database operations
* Logging
* File writing
* Application control

## GPU

If GPU acceleration is available, neural-network inference can use:

* YOLO
* InsightFace

The current implementation also supports CPU inference through InsightFace's CPU execution provider.

---

# 14. AI Prompts Used During Development

The following are representative development prompts used to guide AI-assisted implementation.

## Prompt 1 – Architecture

```text
Design a modular Python architecture for an intelligent face tracker that uses YOLO for face detection, ByteTrack for tracking, InsightFace for face embeddings, SQLite for persistent visitor identities, and file/database logging for entry and exit events.
```

## Prompt 2 – Face Recognition

```text
Implement a Python class using InsightFace Buffalo_L to generate face embeddings from OpenCV face crops. The implementation should handle invalid crops and return a numerical embedding suitable for cosine similarity comparison.
```

## Prompt 3 – Persistent Identity

```text
Design a persistent face registry that assigns PERSON_001 style IDs to new faces, compares new InsightFace embeddings with stored embeddings using cosine similarity, and loads existing embeddings from SQLite when the application starts.
```

## Prompt 4 – Event Management

```text
Design an event manager for a face tracking application that generates exactly one ENTRY and one EXIT event for each active visitor session, handles ByteTrack ID changes, saves timestamped face crops, and flushes active visitors when a video ends.
```

## Prompt 5 – Database

```text
Design a SQLite schema for persistent visitor identities, visit events, and face embeddings. The database should support unique visitor counting and re-identification across multiple application runs.
```

## Prompt 6 – Multi-Video Processing

```text
Create a Python batch processor that automatically discovers all supported videos in a sample directory and processes them sequentially using the same persistent face database so identities can be recognized across videos.
```

## Prompt 7 – Debugging

```text
Review this face tracking event management code and identify why the same person may receive duplicate entry events when ByteTrack changes the temporary track ID. Modify the design so persistent PERSON IDs are independent from temporary tracker IDs.
```

---

# 15. AI-Assisted Development Principle

AI-generated code was treated as development assistance rather than as an unchecked final implementation.

The implementation was:

1. Generated or suggested with AI assistance.
2. Reviewed in the development environment.
3. Integrated into the modular project.
4. Tested against sample videos.
5. Debugged when unexpected behavior occurred.
6. Validated using database records and event logs.

The final application was tested using the provided multi-video dataset.

---

# 16. Future Improvements

Possible future improvements include:

* Multiple embeddings per visitor
* Better embedding aggregation
* GPU acceleration
* Asynchronous processing
* PostgreSQL support
* Web dashboard
* Real-time analytics
* Docker deployment
* Distributed video processing
* Advanced person re-identification
* Better handling of difficult lighting and occlusion conditions

---

# 17. Final Development Outcome

The final implementation combines:

```text
YOLO
+
ByteTrack
+
InsightFace
+
Persistent Face Registry
+
SQLite
+
Event Logging
+
OpenCV
+
Multi-Video Processing
```

The system was validated against 23 sample videos and produced:

```text
97 unique visitor identities
144 ENTRY events
144 EXIT events
```

This completes the core intelligent video-processing workflow.
