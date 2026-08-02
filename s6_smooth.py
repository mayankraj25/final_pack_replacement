"""
s6_smooth.py  -  STAGE 6: remove the frame-to-frame wobble

Takes the per-frame corners from stage 5 and smooths each corner's
path through time.

WHY THIS STAGE EXISTS AT ALL
----------------------------
A 99% inlier rate says the fit agrees with the points. It does not
say the result is temporally stable. Sub-pixel noise in the mask and
in the point tracks still lands as a pixel or two of wobble per
frame, and at 30fps that reads as shimmer on the composited label.

Jitter is the thing editors spend hours fixing by hand. This is the
stage that removes it, and it's about twenty lines of real work.

HOW SAVITZKY-GOLAY WORKS
------------------------
Take a window of consecutive frames - say 7. Fit the smoothest simple
curve (a parabola) through the corner's position across those 7
frames. Replace the middle frame's value with the curve's value.
Slide the window forward, repeat.

Real motion is smooth, so a curve follows it. Noise is random, so a
curve physically cannot follow it. That's the whole trick.

A plain moving average would also remove noise, but it flattens
genuine acceleration - if the box speeds up, averaging drags the
estimate backwards. Fitting a curve keeps the acceleration.

THE ONE RULE THAT MATTERS
-------------------------
Never smooth across invalid frames. If frames 40-45 have no valid
fit, a window spanning that gap drags garbage into frames 38 and 47.
This script splits the sequence into runs of consecutive valid frames
and smooths each run independently.

RUN:  py s6_smooth.py
"""

import os
import json

import cv2
import numpy as np
from scipy.signal import savgol_filter

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

OUTPUT_ROOT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

ENABLED = True

# Window must be odd. Larger = smoother but starts lagging real motion.
#   5  gentle
#   7  good default
#   11+ will visibly soften fast camera moves
WINDOW = 7

# Polynomial order. Must be less than WINDOW, with a decent gap.
#   1 = straight line, too rigid, flattens curves
#   2 = parabola, almost always right
#   4+ starts following the noise you're trying to remove
POLYORDER = 2

RENDER_PREVIEW = True
PREVIEW_WIDTH  = 1280

# ==================================================================

FRAMES_DIR     = os.path.join(OUTPUT_ROOT, "frames")
GEOM_PATH      = os.path.join(OUTPUT_ROOT, "geometry.json")
SMOOTH_PATH    = os.path.join(OUTPUT_ROOT, "geometry_smoothed.json")
META_PATH      = os.path.join(OUTPUT_ROOT, "video_meta.json")
PREVIEWS       = os.path.join(OUTPUT_ROOT, "previews")
SMOOTH_PREVIEW = os.path.join(PREVIEWS, "smoothing_compare.mp4")


def find_valid_runs(valid_flags):
    """
    Split the sequence into runs of consecutive valid frames.

    [T,T,T,F,F,T,T]  ->  [(0,2), (5,6)]

    Each run gets smoothed on its own so a gap can't contaminate the
    good frames either side of it.
    """
    runs = []
    start = None
    for i, ok in enumerate(valid_flags):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(valid_flags) - 1))
    return runs


def jitter_metric(corners, valid_flags):
    """
    Measure how much the corners shake, so we can report a
    before/after number rather than just claiming it improved.

    We use second difference - the frame-to-frame change in velocity.
    Smooth motion has low acceleration; shake has high acceleration
    that flips sign constantly. Averaged over all four corners.

    Units are pixels per frame squared.
    """
    vals = []
    for a, b in find_valid_runs(valid_flags):
        if b - a < 2:
            continue
        seg = corners[a:b + 1]                    # (n, 4, 2)
        accel = np.diff(seg, n=2, axis=0)         # (n-2, 4, 2)
        vals.append(np.linalg.norm(accel, axis=2).mean())
    return float(np.mean(vals)) if vals else 0.0


def smooth_corners(corners, valid_flags, window, polyorder):
    """
    Smooth each corner's x and y path through time, run by run.

    corners: (T, 4, 2) array. Invalid frames are passed through
    untouched - stage 8 keyframes their opacity to zero anyway.
    """
    out = corners.copy()

    for a, b in find_valid_runs(valid_flags):
        n = b - a + 1

        # Savitzky-Golay needs at least `window` samples. Short runs
        # get a shrunken window rather than being skipped entirely.
        w = window
        if n < w:
            w = n if n % 2 == 1 else n - 1
        if w < 3 or w <= polyorder:
            continue      # too short to smooth meaningfully

        seg = corners[a:b + 1]                    # (n, 4, 2)
        for c in range(4):
            for axis in range(2):
                out[a:b + 1, c, axis] = savgol_filter(
                    seg[:, c, axis], w, polyorder
                )

    return out


def recompute_homographies(reference_quad, smoothed_corners, valid_flags):
    """
    The homographies from stage 5 no longer match the smoothed
    corners, so refit them from the reference quad to each frame's
    smoothed quad.

    Stage 8 mainly needs the corners, but keeping the homographies
    consistent means anything downstream that wants to warp an image
    gets the right transform.
    """
    src = np.asarray(reference_quad, np.float32).reshape(-1, 1, 2)
    out = []

    for i, ok in enumerate(valid_flags):
        if not ok:
            out.append(None)
            continue
        dst = smoothed_corners[i].astype(np.float32).reshape(-1, 1, 2)
        H = cv2.getPerspectiveTransform(src, dst)
        out.append(H.tolist())

    return out


def render_preview(raw_corners, smooth_corners_arr, valid_flags, n_frames, fps):
    """
    Amber = raw quad from stage 5
    Green = smoothed quad

    On a clean shot these sit almost on top of each other. Where they
    separate is exactly where the wobble was.
    """
    import imageio.v2 as imageio

    first = cv2.imread(os.path.join(FRAMES_DIR, "0000.jpg"))
    oh, ow = first.shape[:2]
    s = PREVIEW_WIDTH / ow
    pw = PREVIEW_WIDTH
    ph = int(round(oh * s))
    ph -= ph % 2

    os.makedirs(PREVIEWS, exist_ok=True)
    writer = imageio.get_writer(SMOOTH_PREVIEW, fps=fps, macro_block_size=None)

    for i in range(n_frames):
        img = cv2.imread(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"))
        if img is None:
            continue
        img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

        if valid_flags[i]:
            raw = (raw_corners[i] * s).astype(np.int32).reshape(-1, 1, 2)
            sm = (smooth_corners_arr[i] * s).astype(np.int32).reshape(-1, 1, 2)

            cv2.polylines(img, [raw], True, (0, 190, 255), 1)   # amber, thin
            cv2.polylines(img, [sm], True, (0, 230, 0), 2)      # green, bold

            shift = np.linalg.norm(
                smooth_corners_arr[i] - raw_corners[i], axis=1
            ).mean()
            label = f"frame {i}   correction {shift:.2f} px"
        else:
            cv2.rectangle(img, (0, 0), (pw - 1, ph - 1), (0, 0, 255), 6)
            label = f"frame {i}   INVALID"

        cv2.rectangle(img, (0, 0), (pw, 34), (0, 0, 0), -1)
        cv2.putText(img, label, (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "amber = raw    green = smoothed",
                    (14, ph - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    writer.close()


def main():
    if not os.path.exists(GEOM_PATH):
        raise FileNotFoundError(f"{GEOM_PATH} missing.\nRun s5_geometry.py first.")

    with open(GEOM_PATH, "r", encoding="utf-8") as f:
        geom = json.load(f)

    n_frames = geom["num_frames"]
    frames = geom["frames"]
    ref_quad = np.array(geom["reference_quad"], np.float32)

    fps = 30.0
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            fps = json.load(f).get("fps", 30.0)

    # ---- pull corners into an array --------------------------------
    valid_flags = [bool(fr["valid"]) for fr in frames]
    corners = np.zeros((n_frames, 4, 2), np.float32)

    for i, fr in enumerate(frames):
        if fr["corners"] is not None:
            corners[i] = np.array(fr["corners"], np.float32)
        elif i > 0:
            corners[i] = corners[i - 1]      # hold, so arrays stay shaped

    n_valid = sum(valid_flags)
    print(f"{n_valid}/{n_frames} valid frames")

    runs = find_valid_runs(valid_flags)
    if len(runs) > 1:
        print(f"Sequence has {len(runs)} valid runs - each smoothed separately "
              f"so gaps can't contaminate good frames:")
        for a, b in runs:
            print(f"    frames {a}-{b}  ({b - a + 1} frames)")

    # ---- smooth -----------------------------------------------------
    before = jitter_metric(corners, valid_flags)

    if ENABLED:
        smoothed = smooth_corners(corners, valid_flags, WINDOW, POLYORDER)
    else:
        print("Smoothing disabled - passing corners through unchanged.")
        smoothed = corners.copy()

    after = jitter_metric(smoothed, valid_flags)

    # ---- write -------------------------------------------------------
    new_homographies = recompute_homographies(ref_quad, smoothed, valid_flags)

    out_frames = []
    for i, fr in enumerate(frames):
        out_frames.append({
            "frame": i,
            "valid": fr["valid"],
            "reason": fr.get("reason", ""),
            "points_used": fr.get("points_used", 0),
            "inlier_ratio": fr.get("inlier_ratio", 0.0),
            "corners": smoothed[i].tolist() if fr["valid"] else None,
            "corners_raw": fr["corners"],
            "homography": new_homographies[i],
        })

    with open(SMOOTH_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "obj_id": geom["obj_id"],
            "reference_frame": geom["reference_frame"],
            "reference_quad": geom["reference_quad"],
            "num_frames": n_frames,
            "frame_width": geom["frame_width"],
            "frame_height": geom["frame_height"],
            "smoothing": {
                "enabled": ENABLED,
                "window": WINDOW,
                "polyorder": POLYORDER,
                "jitter_before": round(before, 4),
                "jitter_after": round(after, 4),
            },
            "frames": out_frames,
        }, f, indent=2)

    print(f"Wrote smoothed geometry -> {SMOOTH_PATH}")

    # ---- report -------------------------------------------------------
    shifts = np.linalg.norm(smoothed - corners, axis=2)
    mean_shift = float(shifts[[i for i in range(n_frames) if valid_flags[i]]].mean()) \
        if n_valid else 0.0
    max_shift = float(shifts.max())

    print()
    print("-" * 62)
    print("SMOOTHING SUMMARY")
    print(f"  window / order      : {WINDOW} / {POLYORDER}")
    print(f"  jitter before       : {before:.3f} px/frame^2")
    print(f"  jitter after        : {after:.3f} px/frame^2")
    if before > 0:
        print(f"  reduction           : {(1 - after / before) * 100:.0f}%")
    print(f"  mean corner shift   : {mean_shift:.2f} px")
    print(f"  largest shift       : {max_shift:.2f} px")

    if max_shift > 15:
        print()
        print("  Some corners moved a lot. Either the raw track was noisy")
        print("  there, or WINDOW is too large and is cutting a real fast")
        print("  move. Check the preview around the biggest corrections.")
    print("-" * 62)

    # ---- preview -------------------------------------------------------
    if RENDER_PREVIEW:
        print("Rendering comparison preview...")
        render_preview(corners, smoothed, valid_flags, n_frames, fps)
        print(f"Preview -> {SMOOTH_PREVIEW}")

    print()
    print("Amber is the raw quad, green is smoothed. On a clean shot they")
    print("nearly overlap. Where they separate is where the wobble was.")


if __name__ == "__main__":
    main()