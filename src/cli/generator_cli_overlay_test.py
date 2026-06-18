# FILE: src/cli/generator_cli_overlay_test.py

import argparse
import os
from src.visualisation.frame_overlay_gpu_test import FrameOverlayGPU

def main():
    parser = argparse.ArgumentParser(description="Test only the frame overlay step.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an existing stitched video (e.g., artifacts/stitched.mp4)"
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="assets/Pickleverse_frame_1.png",
        help="Path to the PNG overlay frame"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/test_overlay_output.mp4",
        help="Where to save the overlay test output"
    )

    args = parser.parse_args()

    # Validate input video
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")

    # Validate frame PNG
    if not os.path.exists(args.frame):
        raise FileNotFoundError(f"Frame PNG not found: {args.frame}")

    print("\n==============================")
    print(" FRAME OVERLAY TEST MODE")
    print("==============================")
    print(f"Input video:  {args.input}")
    print(f"Frame PNG:    {args.frame}")
    print(f"Output video: {args.output}")
    print("==============================\n")

    # Run overlay
    overlay = FrameOverlayGPU(args.frame)
    overlay.apply_to_video(args.input, args.output)

    print("\n[TEST COMPLETE]")
    print(f"Overlay test output saved → {args.output}\n")


if __name__ == "__main__":
    main()
