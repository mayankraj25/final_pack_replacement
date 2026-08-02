"""
s1_frames.py  -  STAGE 1: video -> numbered frames

Extracts every frame as a numbered JPEG and records the video's
fps / dimensions for later stages.

Filenames are PURE NUMERIC (0000.jpg). SAM2's loader runs int() on
every filename in the folder, so "frame_0000.jpg" crashes it, and so
does any stray json or preview image sitting in the same folder.
This folder holds frames and nothing else - the script wipes it on
each run to guarantee that.

RUN:  py s1_frames.py
"""

import os
import json
import shutil

import cv2

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

VIDEO_PATH   = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\input\Video Project 5.mp4"
OUTPUT_ROOT  = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

JPEG_QUALITY = 95
LIMIT        = 0        # 0 = all frames. Set e.g. 30 for a quick test.

# ==================================================================

FRAMES_DIR = os.path.join(OUTPUT_ROOT, "frames")
META_PATH  = os.path.join(OUTPUT_ROOT, "video_meta.json")
PREVIEWS   = os.path.join(OUTPUT_ROOT, "previews")


def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found:\n  {VIDEO_PATH}")

    # Wipe frames folder so a re-run can't leave stale frames from a
    # previous, longer video mixed in with the new ones.
    if os.path.isdir(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(PREVIEWS, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video:\n  {VIDEO_PATH}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("WARNING: could not read fps, defaulting to 30.0")
        fps = 30.0

    print(f"Source : {os.path.basename(VIDEO_PATH)}")
    print(f"         {width}x{height} @ {fps:.3f} fps, ~{reported} frames")

    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imwrite(os.path.join(FRAMES_DIR, f"{idx:04d}.jpg"), frame, params)
        idx += 1

        if idx % 25 == 0:
            print(f"  extracted {idx}")
        if LIMIT and idx >= LIMIT:
            print(f"  stopping at LIMIT={LIMIT}")
            break

    cap.release()

    if idx == 0:
        raise RuntimeError("No frames extracted. Is the video file valid?")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "source_video": VIDEO_PATH,
            "fps": float(fps),
            "width": width,
            "height": height,
            "frame_count": idx,
        }, f, indent=2)

    print()
    print(f"Done. {idx} frames -> {FRAMES_DIR}")
    print(f"Metadata      -> {META_PATH}")


if __name__ == "__main__":
    main()