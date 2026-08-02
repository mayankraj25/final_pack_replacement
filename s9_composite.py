"""
s9_composite.py  -  STAGE 9: put the new label on the video

Takes the smoothed corner data from stage 6 and a new label image,
warps the label to fit the tracked position on each frame, and
writes out a finished video with the label replaced.

THIS IS OPTIONAL
----------------
The intended workflow is stage 8's AE export, where the editor does
the compositing in After Effects with full artistic control. This
stage exists for cases where you want a quick preview without opening
AE, or where you need a batch-processed result.

HOW THE WARP WORKS
------------------
Your new label is a flat rectangular image. The tracked corners tell
you where the label should sit on each frame. OpenCV's perspective
transform (getPerspectiveTransform + warpPerspective) stretches your
flat rectangle so its 4 corners land exactly on the tracked positions.

This is mathematically identical to what AE's Corner Pin effect does.

WHAT IT WON'T LOOK LIKE
------------------------
The composited label will look flat - no lighting match, no
reflections, no shadow. For a quick test or batch preview that's
fine. For client-facing output, use the AE path and let the editor
add shading.

RUN:  py s9_composite.py
"""

import os
import json

import cv2
import numpy as np

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

OUTPUT_ROOT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

# The new label image. Must be a PNG, ideally with a transparent
# background. If it has no alpha channel, the entire rectangle is
# used as-is.
NEW_LABEL_PATH = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\new_label.png"

OBJ_ID = 1

# How much to feather the edges of the composited label (pixels).
# 0 = hard edge (looks like a sticker). 2-4 = blends into the plate.
EDGE_FEATHER = 3

# Output video codec. Use "mp4v" for broad compatibility.
OUTPUT_CODEC = "mp4v"

# ==================================================================

FRAMES_DIR  = os.path.join(OUTPUT_ROOT, "frames")
SMOOTH_PATH = os.path.join(OUTPUT_ROOT, "geometry_smoothed.json")
META_PATH   = os.path.join(OUTPUT_ROOT, "video_meta.json")
OUTPUT_VIDEO = os.path.join(OUTPUT_ROOT, "composited_output.mp4")


def load_label(path):
    """
    Load the new label as BGRA. If the source has no alpha channel,
    add a fully opaque one so the compositing math works the same way
    regardless.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read label image:\n  {path}")

    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
        img = np.concatenate([img, alpha], axis=2)
        print("Label has no alpha channel - treating as fully opaque")

    print(f"Label loaded: {img.shape[1]}x{img.shape[0]} px")
    return img


def warp_and_composite(frame, label_bgra, dst_corners, feather):
    """
    Warp the flat label onto the frame at the tracked position.

    frame:        BGR image (the original video frame)
    label_bgra:   the new label with alpha
    dst_corners:  4 corners [TL, TR, BR, BL] where the label should go
    feather:      edge softness in pixels

    Returns the composited BGR frame.
    """
    lh, lw = label_bgra.shape[:2]
    fh, fw = frame.shape[:2]

    # Source corners = the 4 corners of the flat label image
    src = np.array([
        [0, 0],
        [lw - 1, 0],
        [lw - 1, lh - 1],
        [0, lh - 1]
    ], dtype=np.float32)

    dst = np.array(dst_corners, dtype=np.float32)

    # Perspective transform: stretches the rectangle so its corners
    # land on the tracked positions. This is mathematically identical
    # to what AE's Corner Pin effect does.
    H = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        label_bgra, H, (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Separate the warped label into colour and alpha
    warped_bgr = warped[:, :, :3]
    warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0

    # Feather the alpha edge so it blends instead of hard-cutting
    if feather > 0:
        ksize = feather * 2 + 1
        warped_alpha = cv2.GaussianBlur(warped_alpha, (ksize, ksize), 0)

    # Alpha composite: new label on top of original frame
    alpha_3ch = warped_alpha[:, :, np.newaxis]
    composited = (warped_bgr.astype(np.float32) * alpha_3ch +
                  frame.astype(np.float32) * (1.0 - alpha_3ch))

    return composited.astype(np.uint8)


def main():
    if not os.path.exists(SMOOTH_PATH):
        raise FileNotFoundError(
            f"{SMOOTH_PATH} missing.\nRun s6_smooth.py first."
        )

    with open(SMOOTH_PATH, "r", encoding="utf-8") as f:
        geom = json.load(f)

    n_frames = geom["num_frames"]
    frame_w = geom["frame_width"]
    frame_h = geom["frame_height"]

    fps = 30.0
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            fps = json.load(f).get("fps", 30.0)

    label = load_label(NEW_LABEL_PATH)

    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer.\n"
            f"If '{OUTPUT_CODEC}' is not supported, try 'XVID' or 'avc1'."
        )

    valid_count = 0
    skip_count = 0

    print(f"Compositing {n_frames} frames...")

    for i in range(n_frames):
        frame = cv2.imread(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"))
        if frame is None:
            print(f"  WARNING: missing frame {i}, writing black")
            writer.write(np.zeros((frame_h, frame_w, 3), np.uint8))
            continue

        fr = geom["frames"][i]

        if fr["valid"] and fr["corners"] is not None:
            corners = fr["corners"]
            composited = warp_and_composite(frame, label, corners, EDGE_FEATHER)
            writer.write(composited)
            valid_count += 1
        else:
            # No valid track on this frame - write the original unchanged
            writer.write(frame)
            skip_count += 1

        if i % 20 == 0:
            print(f"  frame {i}/{n_frames}")

    writer.release()

    print()
    print("-" * 62)
    print("COMPOSITE SUMMARY")
    print(f"  frames with new label : {valid_count}")
    print(f"  frames unchanged      : {skip_count}  (no valid track)")
    print(f"  output                : {OUTPUT_VIDEO}")
    print("-" * 62)
    print()
    print("This is a FLAT composite - no lighting, no shadows.")
    print("For production quality, use the AE path (s8_export.py)")
    print("where the editor can add shading and colour matching.")


if __name__ == "__main__":
    main()