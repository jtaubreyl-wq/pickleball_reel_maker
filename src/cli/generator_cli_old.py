import asyncio
from pathlib import Path

import click
from rich.progress import Progress

from src.extractor.async_frame_extractor import extract_frames_async
from src.models.yolo_detector import run_yolo_tracking
from src.rally.segmentation.rally_segment_builder import detect_rallies
from src.rally.scoring.highlight_scoring_engine import score_rallies
from src.highlights.highlight_selector import select_top_highlights
from src.video.clip_extractor import extract_clip
from src.video.reel_assembler import assemble_reel


@click.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="artifacts/highlight_reel.mp4")
@click.option("--max", "max_highlights", default=10, show_default=True)
def generate(video, output, max_highlights):
    """
    Unified end-to-end highlight generator:
    video → frames → tracking → rallies → scoring → selection → clips → reel.
    """
    video_path = Path(video)
    output_path = Path(output)
    clips_dir = output_path.parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:

        # 1. Extract frames (async)
        t1 = progress.add_task("Extracting frames…", total=1)
        frames = asyncio.run(extract_frames_async(video_path))
        progress.update(t1, advance=1)

        # 2. YOLO tracking
        t2 = progress.add_task("Running ball tracking…", total=1)
        tracking = run_yolo_tracking(frames)
        progress.update(t2, advance=1)

        # 3. Rally segmentation
        t3 = progress.add_task("Detecting rallies…", total=1)
        rallies = detect_rallies(tracking)
        progress.update(t3, advance=1)

        # 4. Scoring
        t4 = progress.add_task("Scoring rallies…", total=1)
        scored = score_rallies(rallies)
        progress.update(t4, advance=1)

        # 5. Selection
        t5 = progress.add_task("Selecting highlights…", total=1)
        selected = select_top_highlights(scored, max_highlights=max_highlights)
        progress.update(t5, advance=1)

        # 6. Clip extraction
        t6 = progress.add_task("Extracting clips…", total=len(selected))
        clip_paths = []
        for i, r in enumerate(selected):
            clip_path = clips_dir / f"clip_{i+1}.mp4"
            extract_clip(video_path, r.start_time, r.end_time, clip_path)
            clip_paths.append(clip_path)
            progress.update(t6, advance=1)

        # 7. Reel assembly
        t7 = progress.add_task("Assembling final reel…", total=1)
        assemble_reel(clip_paths, output_path)
        progress.update(t7, advance=1)

    click.echo(f"🎉 Highlight reel created → {output_path}")


if __name__ == "__main__":
    generate()
