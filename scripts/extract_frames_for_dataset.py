import cv2
import os

def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: int = 5
):
    """
    Extract frames from a video at a target FPS and save them with strict naming.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where frames will be saved.
        target_fps (int): Frames per second to extract.
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / target_fps)

    frame_idx = 0
    saved_idx = 0

    print(f"Extracting frames from {video_path}")
    print(f"Video FPS: {video_fps}, Target FPS: {target_fps}, Interval: {frame_interval}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save every Nth frame
        if frame_idx % frame_interval == 0:
            filename = f"frame_{saved_idx:06d}.jpg"
            save_path = os.path.join(output_dir, filename)

            try:
                cv2.imwrite(save_path, frame)
                saved_idx += 1
            except Exception as e:
                print(f"Skipping corrupted frame {frame_idx}: {e}")

        frame_idx += 1

    cap.release()
    print(f"Done! Saved {saved_idx} frames to {output_dir}")

if __name__ == "__main__":
    extract_frames(
        video_path="videos/match1.mp4",
        output_dir="datasets/pickleball-ball/images/train",
        target_fps=5
    )
