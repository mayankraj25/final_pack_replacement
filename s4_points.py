"""
s4_points.py  -  STAGE 4: dense point tracking

This is the stage that makes the pipeline better than Mocha.

WHY IT MATTERS
--------------
Mocha tracks 3-4 points. A flat-surface transform has 8 unknowns, and
4 points give exactly 8 equations - a perfect but fragile fit with no
margin. One point drifting on a reflection shifts the whole solution.
That is the mathematical origin of Mocha's jitter, and fixing it by
hand is most of the 8-12 hours on a complex shot.

This tracks 300-800 points instead. That's an over-determined system,
so bad points get identified and thrown out (stage 5 does that with
MAGSAC++) while the solution barely moves. The tracker also reports
per-point visibility, so a hand crossing the label excludes those
points rather than corrupting the fit.

THE RISK THIS ALSO TESTS
------------------------
Point trackers cannot track featureless surfaces. A plain cream
shampoo label has almost no texture in the middle. So instead of a
naive uniform grid, this seeds points three ways and reports how many
genuinely trackable features it found. If that number is low, the
console says so - and that changes stage 5's design.

RUN
  py s4_points.py --seed-only     fast texture check, no tracking
  py s4_points.py                 the real run (slow on CPU)
"""

import os
import json
import argparse

import cv2
import numpy as np

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

OUTPUT_ROOT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

OBJ_ID = 1            # which label to track

# Video is downscaled to this width before tracking, then results are
# scaled back up. 4K tracking would need ~10GB and take forever.
# 512 is sane on CPU. Raise to 768/1024 if you get a GPU.
TRACKING_WIDTH = 512

TARGET_POINT_COUNT = 500

# Where points go. Because a plain label has no texture in the middle,
# we deliberately bias toward corners and the label outline.
CORNER_FRACTION   = 0.5    # real trackable features (Shi-Tomasi)
BOUNDARY_FRACTION = 0.3    # along the mask outline (always high contrast)
GRID_FRACTION     = 0.2    # even coverage fallback

CORNER_QUALITY      = 0.01   # lower = accepts weaker corners
CORNER_MIN_DISTANCE = 8

VISIBILITY_THRESHOLD = 0.5
MODEL_VARIANT = "cotracker3_offline"   # or "cotracker3_online"

PREVIEW_WIDTH = 1280

# ==================================================================

FRAMES_DIR    = os.path.join(OUTPUT_ROOT, "frames")
MASKS_RAW     = os.path.join(OUTPUT_ROOT, "masks_raw")
META_PATH     = os.path.join(OUTPUT_ROOT, "video_meta.json")
TRACKS_PATH   = os.path.join(OUTPUT_ROOT, "point_tracks.json")
PREVIEWS      = os.path.join(OUTPUT_ROOT, "previews")
SEED_DIAG     = os.path.join(PREVIEWS, "seed_points.jpg")
TRACK_PREVIEW = os.path.join(PREVIEWS, "point_tracks.mp4")


# ==================================================================
# SEEDING - deciding where to put the tracking points
# ==================================================================

def seed_corner_points(gray, mask, want):
    """
    Strategy 1: find genuinely trackable features.

    Shi-Tomasi looks for spots where the image changes in two
    directions at once - corners of letters, edges of graphics,
    texture. These are what a tracker can actually lock onto.

    On a printed label this finds plenty. On a blank label it finds
    very few, which is exactly the diagnostic we want.
    """
    if want <= 0:
        return np.zeros((0, 2), np.float32)

    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=int(want), qualityLevel=CORNER_QUALITY,
        minDistance=CORNER_MIN_DISTANCE, mask=mask,
    )
    if corners is None:
        return np.zeros((0, 2), np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def seed_boundary_points(mask, want, inset_px=4):
    """
    Strategy 2: sample along the label's outline.

    The edge between label and bottle is high contrast by definition,
    so it stays trackable even when the middle of the label is blank.
    This is the fallback that saves us on featureless labels.

    Points are pulled slightly INWARD - a point sitting exactly on the
    edge is half on the background, and the tracker gets confused
    about which side it belongs to.
    """
    if want <= 0:
        return np.zeros((0, 2), np.float32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), np.float32)

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    want = min(int(want), len(contour))

    idx = np.linspace(0, len(contour) - 1, want).astype(int)
    pts = contour[idx]

    centre = contour.mean(axis=0)
    direction = centre - pts
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    pts = pts + (direction / norm) * inset_px

    h, w = mask.shape
    keep = []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 127:
            keep.append([x, y])

    return np.array(keep, np.float32) if keep else np.zeros((0, 2), np.float32)


def seed_grid_points(mask, want):
    """Strategy 3: even coverage so the fit isn't dominated by one region."""
    if want <= 0:
        return np.zeros((0, 2), np.float32)

    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)

    spacing = max(4, int(np.sqrt(len(xs) / max(want, 1))))

    pts = []
    for y in range(ys.min(), ys.max() + 1, spacing):
        for x in range(xs.min(), xs.max() + 1, spacing):
            if mask[y, x] > 127:
                pts.append([float(x), float(y)])

    if not pts:
        return np.zeros((0, 2), np.float32)

    pts = np.array(pts, np.float32)
    if len(pts) > want:
        idx = np.linspace(0, len(pts) - 1, int(want)).astype(int)
        pts = pts[idx]
    return pts


def dedupe(points, min_dist=3.0):
    """Drop points that landed almost on top of each other."""
    if len(points) == 0:
        return points
    kept = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(np.array(kept) - p, axis=1).min() >= min_dist:
            kept.append(p)
    return np.array(kept, np.float32)


def build_seed_points(frame_bgr, mask):
    total = TARGET_POINT_COUNT
    want_corner   = int(total * CORNER_FRACTION)
    want_boundary = int(total * BOUNDARY_FRACTION)
    want_grid     = int(total * GRID_FRACTION)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    corner_pts   = seed_corner_points(gray, mask, want_corner)
    boundary_pts = seed_boundary_points(mask, want_boundary)
    grid_pts     = seed_grid_points(mask, want_grid)

    parts = [p for p in (corner_pts, boundary_pts, grid_pts) if len(p) > 0]
    all_pts = np.vstack(parts) if parts else np.zeros((0, 2), np.float32)
    all_pts = dedupe(all_pts)

    yield_ratio = len(corner_pts) / max(want_corner, 1)
    if yield_ratio < 0.25:
        verdict = "VERY LOW"
    elif yield_ratio < 0.6:
        verdict = "LOW"
    else:
        verdict = "OK"

    diag = {
        "corner_points_requested": want_corner,
        "corner_points_found": len(corner_pts),
        "corner_yield": round(float(yield_ratio), 3),
        "boundary_points": len(boundary_pts),
        "grid_points": len(grid_pts),
        "total_after_dedupe": len(all_pts),
        "texture_verdict": verdict,
    }
    return all_pts, corner_pts, boundary_pts, grid_pts, diag


def save_seed_preview(frame_bgr, corner_pts, boundary_pts, grid_pts):
    """
    Green = real trackable features, Blue = boundary, Grey = grid fill.
    If this is mostly blue and grey with almost no green, the label has
    no texture and stage 5 needs the contour-based path.
    """
    img = frame_bgr.copy()
    for x, y in grid_pts:
        cv2.circle(img, (int(x), int(y)), 3, (150, 150, 150), -1)
    for x, y in boundary_pts:
        cv2.circle(img, (int(x), int(y)), 3, (255, 120, 0), -1)
    for x, y in corner_pts:
        cv2.circle(img, (int(x), int(y)), 4, (0, 230, 0), -1)

    legend = [("green = trackable features", (0, 230, 0)),
              ("blue  = label boundary",     (255, 120, 0)),
              ("grey  = grid fill",          (150, 150, 150))]
    for i, (text, colour) in enumerate(legend):
        cv2.putText(img, text, (20, 40 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

    os.makedirs(os.path.dirname(SEED_DIAG), exist_ok=True)
    cv2.imwrite(SEED_DIAG, img)


# ==================================================================
# TRACKING
# ==================================================================

def load_video_tensor(n_frames):
    """Load frames as a downscaled tensor. Returns (tensor, scale, dims)."""
    import torch

    first = cv2.imread(os.path.join(FRAMES_DIR, "0000.jpg"))
    if first is None:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}")

    oh, ow = first.shape[:2]
    scale = TRACKING_WIDTH / ow
    nw = TRACKING_WIDTH
    nh = int(round(oh * scale))
    nh -= nh % 2

    print(f"Loading {n_frames} frames at {nw}x{nh} "
          f"(source {ow}x{oh}, scale {scale:.3f})")

    buf = np.zeros((n_frames, nh, nw, 3), np.uint8)
    for i in range(n_frames):
        img = cv2.imread(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"Missing frame {i}")
        buf[i] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    rgb = buf[..., ::-1].copy()                       # BGR -> RGB
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float()
    return tensor, scale, (ow, oh), (nw, nh)


def run_cotracker(video_tensor, query_points):
    """
    Run CoTracker3. Loads the model from torch.hub, which downloads
    from GitHub on first run only.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CoTracker3 ({MODEL_VARIANT})...")
    model = torch.hub.load("facebookresearch/co-tracker", MODEL_VARIANT)
    model = model.to(device)

    n = len(query_points)
    queries = np.zeros((n, 3), np.float32)
    queries[:, 0] = 0.0                # all seeded on frame 0
    queries[:, 1:] = query_points
    queries_t = torch.from_numpy(queries)[None].to(device)

    video_tensor = video_tensor.to(device)

    if device == "cpu":
        print("WARNING: running on CPU. Expect 20-40 minutes for ~100 frames.")

    print(f"Tracking {n} points across {video_tensor.shape[1]} frames...")
    with torch.no_grad():
        tracks, visibility = model(video_tensor, queries=queries_t)

    return tracks[0].cpu().numpy(), visibility[0].cpu().numpy()


def save_tracks_preview(tracks, visibility, scale, n_frames, fps):
    """
    Green dot = visible and tracking.  Red dot = tracker knows it's occluded.

    Watch this once. If green dots stay glued to the label through the
    clip, tracking works. If they slide off or scatter, it doesn't -
    and no downstream geometry will fix that.
    """
    import imageio.v2 as imageio

    first = cv2.imread(os.path.join(FRAMES_DIR, "0000.jpg"))
    oh, ow = first.shape[:2]
    pv = PREVIEW_WIDTH / ow
    pw = PREVIEW_WIDTH
    ph = int(round(oh * pv))
    ph -= ph % 2

    os.makedirs(os.path.dirname(TRACK_PREVIEW), exist_ok=True)
    writer = imageio.get_writer(TRACK_PREVIEW, fps=fps, macro_block_size=None)

    for i in range(n_frames):
        img = cv2.imread(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"))
        img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

        pts = tracks[i] / scale * pv
        vis = visibility[i]

        for (x, y), v in zip(pts, vis):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            colour = (0, 230, 0) if v > VISIBILITY_THRESHOLD else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 2, colour, -1)

        n_vis = int((vis > VISIBILITY_THRESHOLD).sum())
        cv2.putText(img, f"frame {i}   visible {n_vis}/{len(vis)}",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    writer.close()


# ==================================================================
# MAIN
# ==================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-only", action="store_true",
                    help="seed points and save the preview image only, "
                         "don't run the tracker")
    args = ap.parse_args()

    os.makedirs(PREVIEWS, exist_ok=True)

    n_frames = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")]) \
        if os.path.isdir(FRAMES_DIR) else 0
    if n_frames == 0:
        raise RuntimeError(f"No frames in {FRAMES_DIR}\nRun s1_frames.py first.")

    fps = 30.0
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            fps = json.load(f).get("fps", 30.0)

    # ---- seed on frame 0 ------------------------------------------
    frame0 = cv2.imread(os.path.join(FRAMES_DIR, "0000.jpg"))
    h0, w0 = frame0.shape[:2]

    # Build the mask directly from the polygon drawn in stage 2,
    # bypassing SAM2 entirely. SAM2 segments whole objects, not
    # sub-regions like a colour band on a box. The polygon you
    # drew IS the region definition.
    SEED_PATH = os.path.join(OUTPUT_ROOT, "seed_boxes.json")
    with open(SEED_PATH, "r") as f:
        seed = json.load(f)
    obj = [o for o in seed["objects"] if o["obj_id"] == OBJ_ID][0]

    if obj.get("polygon"):
        mask0 = np.zeros((h0, w0), np.uint8)
        pts = np.array(obj["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask0, [pts], 255)
        print("Using polygon from stage 2 as mask (SAM2 bypassed)")
    else:
        # Fallback: if you used a box not a polygon, try the SAM2 mask
        mask0_path = os.path.join(MASKS_RAW, f"obj_{OBJ_ID}", "0000.png")
        mask0 = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE)
        if mask0 is None:
            raise FileNotFoundError(
                f"No mask and no polygon. Either draw a polygon in "
                f"s2_seed.py (press P) or run s3_masks.py first."
            )
        print("Using SAM2 mask from stage 3")

    all_pts, corner_pts, boundary_pts, grid_pts, diag = \
        build_seed_points(frame0, mask0)

    save_seed_preview(frame0, corner_pts, boundary_pts, grid_pts)

    print()
    print("-" * 62)
    print("SEEDING DIAGNOSTIC")
    print(f"  trackable features found : {diag['corner_points_found']} of "
          f"{diag['corner_points_requested']} requested "
          f"({diag['corner_yield']*100:.0f}%)")
    print(f"  boundary points          : {diag['boundary_points']}")
    print(f"  grid fill points         : {diag['grid_points']}")
    print(f"  total after dedupe       : {diag['total_after_dedupe']}")
    print(f"  label texture            : {diag['texture_verdict']}")

    if diag["texture_verdict"] == "VERY LOW":
        print()
        print("  This label has almost no trackable texture. Interior")
        print("  points will drift, so the fit has to rely on boundary")
        print("  points - stage 5 should use contour-based fitting")
        print("  rather than dense interior correspondence.")
        print(f"  Check {SEED_DIAG} - if you see almost no green dots,")
        print("  that's what's happening.")
    print("-" * 62)

    if len(all_pts) < 20:
        raise RuntimeError(
            f"Only {len(all_pts)} seed points. Something is wrong with the "
            f"mask - open {mask0_path} and check it actually shows the label."
        )

    if args.seed_only:
        print()
        print(f"Seed preview -> {SEED_DIAG}")
        print("Stopping (--seed-only). Look at that image before the full run.")
        return

    # ---- track -----------------------------------------------------
    video, scale, (ow, oh), (nw, nh) = load_video_tensor(n_frames)

    # Seed points were found at ORIGINAL resolution; scale down for
    # the tracker, then scale results back up afterwards.
    tracks, visibility = run_cotracker(video, all_pts * scale)
    tracks_full = tracks / scale

    frames_out = []
    for i in range(tracks_full.shape[0]):
        vis_i = visibility[i] > VISIBILITY_THRESHOLD
        frames_out.append({
            "frame": i,
            "points": tracks_full[i].tolist(),
            "visible": vis_i.tolist(),
            "visible_count": int(vis_i.sum()),
        })

    with open(TRACKS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "obj_id": OBJ_ID,
            "num_points": int(tracks_full.shape[1]),
            "num_frames": int(tracks_full.shape[0]),
            "frame_width": ow,
            "frame_height": oh,
            "tracking_width": nw,
            "tracking_scale": float(scale),
            "seeding_diagnostic": diag,
            "frames": frames_out,
        }, f, indent=2)

    print(f"Wrote tracks -> {TRACKS_PATH}")

    print("Rendering preview video...")
    save_tracks_preview(tracks, visibility, scale, n_frames, fps)
    print(f"Preview      -> {TRACK_PREVIEW}")

    # ---- survival summary -------------------------------------------
    t = VISIBILITY_THRESHOLD
    first_vis = int((visibility[0] > t).sum())
    mid_vis   = int((visibility[len(visibility) // 2] > t).sum())
    last_vis  = int((visibility[-1] > t).sum())

    print()
    print("-" * 62)
    print("POINT SURVIVAL")
    print(f"  frame 0    : {first_vis} visible")
    print(f"  frame {len(visibility)//2:<4} : {mid_vis} visible")
    print(f"  frame {len(visibility)-1:<4} : {last_vis} visible")

    survival = last_vis / max(first_vis, 1)
    if survival < 0.3:
        print("  Most points died. Watch the preview - either the label")
        print("  leaves frame, or the tracker lost it.")
    elif survival < 0.6:
        print("  Moderate attrition. Normal if the label rotates away.")
    else:
        print("  Good survival rate.")
    print("-" * 62)


if __name__ == "__main__":
    main()