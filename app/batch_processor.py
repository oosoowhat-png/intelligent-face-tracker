from pathlib import Path

from app.config import load_config
from app.video_processor import VideoProcessor


SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
}


def find_videos(sample_directory):
    """
    Find all supported video files inside the sample directory.
    """

    sample_path = Path(
        sample_directory
    )

    if not sample_path.exists():

        print(
            f"ERROR: Directory does not exist: "
            f"{sample_directory}"
        )

        return []

    videos = []

    for file_path in sample_path.iterdir():

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            videos.append(
                file_path
            )

    videos.sort(
        key=lambda path: path.name.lower()
    )

    return videos


def main():

    print()
    print("=" * 70)
    print("INTELLIGENT FACE TRACKER")
    print("MULTI-VIDEO PROCESSING")
    print("=" * 70)
    print()

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------

    config = load_config()

    sample_directory = "sample"

    # --------------------------------------------------
    # Find all videos
    # --------------------------------------------------

    videos = find_videos(
        sample_directory
    )

    if not videos:

        print(
            "ERROR: No supported video files found "
            "inside the sample directory."
        )

        print()
        print(
            "Supported formats:"
        )

        print(
            ", ".join(
                sorted(
                    SUPPORTED_EXTENSIONS
                )
            )
        )

        return

    print(
        f"Found {len(videos)} video(s):"
    )

    for index, video_path in enumerate(
        videos,
        start=1
    ):

        print(
            f"  {index:02d}. "
            f"{video_path.name}"
        )

    print()

    # --------------------------------------------------
    # Process videos one at a time
    # --------------------------------------------------

    successful_videos = 0
    failed_videos = []

    for index, video_path in enumerate(
        videos,
        start=1
    ):

        print()
        print(
            "#" * 70
        )

        print(
            f"VIDEO {index}/{len(videos)}"
        )

        print(
            f"File: {video_path.name}"
        )

        print(
            "#" * 70
        )

        try:

            # --------------------------------------------------
            # IMPORTANT:
            #
            # Create a fresh VideoProcessor for every video.
            #
            # This resets ByteTrack state while the SQLite
            # database keeps the persistent face identities.
            # --------------------------------------------------

            processor = VideoProcessor(
                config
            )

            success = (
                processor.process_video(
                    video_source=str(
                        video_path
                    )
                )
            )

            processor.database.close()

            if success:

                successful_videos += 1

                print()
                print(
                    f"SUCCESS: "
                    f"{video_path.name}"
                )

            else:

                failed_videos.append(
                    video_path.name
                )

                print()
                print(
                    f"FAILED: "
                    f"{video_path.name}"
                )

        except Exception as error:

            failed_videos.append(
                video_path.name
            )

            print()
            print(
                f"ERROR while processing "
                f"{video_path.name}:"
            )

            print(
                str(error)
            )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("MULTI-VIDEO PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Total videos: "
        f"{len(videos)}"
    )

    print(
        f"Successful: "
        f"{successful_videos}"
    )

    print(
        f"Failed: "
        f"{len(failed_videos)}"
    )

    if failed_videos:

        print()
        print(
            "Failed videos:"
        )

        for filename in failed_videos:

            print(
                f"  - {filename}"
            )

    print()
    print(
        "Check data/visitors.db for "
        "the final unique visitor count."
    )

    print(
        "Check logs/events.log for "
        "the complete event history."
    )

    print()


if __name__ == "__main__":
    main()