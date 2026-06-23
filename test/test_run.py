from src.visualisation.frame_overlay_gpu_test import FrameOverlayGPU

FRAME = r"assets\Pickleverse_frame_1.png"
VIDEO = r"assets\PB1.mp4"
OUTPUT = r"artifacts\output_test.mp4"

overlay = FrameOverlayGPU(FRAME)
overlay.apply_to_video(VIDEO, OUTPUT)

print("DONE → Saved:", OUTPUT)
