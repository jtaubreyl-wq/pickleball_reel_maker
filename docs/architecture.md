 ┌──────────────────────┐
 │   Raw Match Video    │
 │ (GoPro / Phone Input)│
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │    Video Loader      │
 │  (FFmpeg + OpenCV)   │
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │   Frame Extraction   │
 │   (FPS Downsample)   │
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │   Player Detection   │
 │   (YOLOv8/YOLOv10)   │
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │      Tracking        │
 │ (ByteTrack / SV)     │
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │  Rally Segmentation  │
 │ (Ball + Motion Logic)│
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │   Highlight Logic    │
 │ (Events + Scoring)   │
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │   Clip Extraction    │
 │      (MoviePy)       │
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │  Highlight Stitching │
 │ (Transitions + Music)│
 └────────────┬─────────┘
              │
              ▼
 ┌──────────────────────┐
 │ Final Highlight Reel │
 │       (MP4)          │
 └──────────────────────┘
