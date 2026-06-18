flowchart TD

    A[Raw Match Video<br/>(GoPro / Phone)] --> B[Video Loader<br/>FFmpeg + OpenCV]

    B --> C[Frame Extraction<br/>FPS Downsampling]

    C --> D[Player Detection<br/>YOLOv8 / YOLOv10]

    D --> E[Tracking<br/>ByteTrack / Supervision]

    E --> F[Rally Segmentation<br/>Ball Movement + Player Motion]

    F --> G[Highlight Logic<br/>Scoring Events / Long Rallies]

    G --> H[Clip Extraction<br/>MoviePy]

    H --> I[Highlight Stitching<br/>Transitions + Music]

    I --> J[Final Highlight Reel<br/>MP4 Export]
