from app.config import load_config
from app.video_processor import VideoProcessor


def main():

    print("Loading configuration...")

    config = load_config()

    processor = VideoProcessor(
        config
    )

    processor.process_video()

    processor.database.close()

    print()
    print("VIDEO PROCESSOR TEST PASSED")


if __name__ == "__main__":
    main()