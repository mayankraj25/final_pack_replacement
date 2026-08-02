"""
s5_geometry.py  -  STAGE 5: turn point tracks into a per-frame shape

Takes the hundreds of tracked points from stage 4 and works out, for
every frame, how the label surface has moved and warped relative to
the reference frame.

THIS IS WHERE WE BEAT MOCHA
---------------------------
Mocha fits a transform to 3-4 points. A flat-surface transform has 8
unknowns, so 4 points give exactly 8 equations - a perfect fit with
zero margin. One point drifting on a reflection moves the entire
solution, and that is where the jitter comes from.

Here we fit to several hundred points. That is massively
over-determined, so MAGSAC++ can identify which points disagree with
the consensus and discard them. Fifty bad points out of four hundred
barely move the answer.

Two useful properties fall out of this approach:

1. The corner ordering is automatically stable. We transform ONE
   reference quad by each frame's homography, rather than extracting
   corners from each frame's mask independently. So "top-left" stays
   top-left even through heavy rotation - no corner-flip artefacts.

2. The inlier ratio is a free confidence signal. If only 30% of
   points agree with the fitted model, something is wrong on that
   frame even if everything else looks fine. Stage 7 uses this.

WHAT IT WRITES
--------------
geometry.json   per frame: 4 corners, the 3x3 homography, inlier
                ratio, and a valid flag
previews/geometry.mp4   the fitted quad drawn on the footage

RUN:  py s5_geometry.py
"""

import os
import json

import cv2
import numpy as np

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

OUTPUT_ROOT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

OBJ_ID = 1

# Which frame the points were seeded on. Everything is measured
# relative to this frame's label shape.
REFERENCE_FRAME = 0

# MAGSAC++ reprojection threshold, in pixels at FULL resolution.
# A point has to land within this distance of where the fitted model
# predicts to count as an inlier. Larger = more forgiving.
RANSAC_THRESHOLD = 4.0

# A frame needs at least this many usable points to attempt a fit.
# Below this the geometry is under-constrained and we mark the frame
# invalid rather than emitting a garbage shape.
MIN_POINTS_FOR_FIT = 12

# Frames where fewer than this fraction of points agree with the
# fitted model get flagged. Not discarded - flagged - so stage 7 can
# tell the editor which frames to look at.
MIN_INLIER_RATIO = 0.40

# Sanity limits on the fitted quad. Catches blown-up solutions that
# are mathematically valid but physically nonsense.
MAX_AREA_RATIO = 6.0    # quad can't grow more than 6x the reference
MIN_AREA_RATIO = 0.05   # or shrink below 5%

RENDER_PREVIEW = True
PREVIEW_WIDTH  = 1280

# ==================================================================

FRAMES_DIR   = os.path.join(OUTPUT_ROOT, "frames")
MASKS_RAW    = os.path.join(OUTPUT_ROOT, "masks_raw")
TRACKS_PATH  = os.path.join(OUTPUT_ROOT, "point_tracks.json")
META_PATH    = os.path.join(OUTPUT_ROOT, "video_meta.json")
GEOM_PATH    = os.path.join(OUTPUT_ROOT, "geometry.json")
PREVIEWS     = os.path.join(OUTPUT_ROOT, "previews")
GEOM_PREVIEW = os.path.join(PREVIEWS, "geometry.mp4")


# ------------------------------------------------------------------
# Reference shape
# ------------------------------------------------------------------

def reference_quad_from_mask(mask):
    """
    The label's shape on the reference frame, as 4 corners.

    minAreaRect gives the tightest rotated rectangle around the mask.
    Every other frame's quad is this shape pushed through that
    frame's homography, which is what keeps corner identity stable
    across the whole clip.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Reference frame mask is empty.")

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)

    return order_quad(box)


def order_quad(pts):
    """
    Put 4 corners into a consistent order:
    top-left, top-right, bottom-right, bottom-left.

    Only needs to be correct once, on the reference frame. Every
    other frame inherits this ordering through the homography.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]

    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]   # top-left      smallest x+y
    out[2] = pts[np.argmax(s)]   # bottom-right  largest  x+y
    out[1] = pts[np.argmax(d)]   # top-right     largest  x-y
    out[3] = pts[np.argmin(d)]   # bottom-left   smallest x-y
    return out


def quad_area(quad):
    return abs(cv2.contourArea(np.asarray(quad, np.float32).reshape(-1, 1, 2)))


def quad_is_sane(quad, reference_area):
    """
    Reject solutions that are mathematically valid but physically
    absurd - inverted quads, or ones that ballooned to fill the frame.
    These happen when the points are nearly collinear and the
    homography is under-determined.
    """
    q = np.asarray(quad, np.float32)

    if not np.all(np.isfinite(q)):
        return False, "non-finite coordinates"

    area = quad_area(q)
    if area <= 1.0:
        return False, "degenerate (zero area)"

    ratio = area / max(reference_area, 1.0)
    if ratio > MAX_AREA_RATIO:
        return False, f"area exploded ({ratio:.1f}x reference)"
    if ratio < MIN_AREA_RATIO:
        return False, f"area collapsed ({ratio:.2f}x reference)"

    # A valid perspective view of a rectangle stays convex.
    hull = cv2.convexHull(q.reshape(-1, 1, 2))
    if len(hull) < 4:
        return False, "quad folded on itself"

    return True, ""


# ------------------------------------------------------------------
# Fitting
# ------------------------------------------------------------------

def fit_frame(ref_pts, cur_pts, ref_quad, reference_area):
    """
    Fit the transform from the reference frame to this frame.

    ref_pts / cur_pts are matched arrays - same point, two frames.
    Returns (corners, H, inlier_ratio, valid, reason).
    """
    if len(ref_pts) < MIN_POINTS_FOR_FIT:
        return None, None, 0.0, False, f"only {len(ref_pts)} usable points"

    # USAC_MAGSAC is MAGSAC++, already built into OpenCV 4.5+.
    # It avoids the hard inlier/outlier cutoff that plain RANSAC needs,
    # which means one less threshold to hand-tune per shot.
    H, inlier_mask = cv2.findHomography(
        ref_pts.reshape(-1, 1, 2),
        cur_pts.reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=5000,
        confidence=0.999,
    )

    if H is None:
        return None, None, 0.0, False, "homography fit failed"

    inlier_ratio = (float(inlier_mask.sum()) / len(inlier_mask)
                    if inlier_mask is not None else 0.0)

    corners = cv2.perspectiveTransform(
        ref_quad.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(4, 2)

    sane, reason = quad_is_sane(corners, reference_area)
    if not sane:
        return corners, H, inlier_ratio, False, reason

    if inlier_ratio < MIN_INLIER_RATIO:
        return corners, H, inlier_ratio, False, \
            f"low agreement ({inlier_ratio:.0%} of points)"

    return corners, H, inlier_ratio, True, ""


def points_inside_mask(points, mask):
    """
    A tracked point can stay 'visible' while drifting off the label
    onto the background. Filtering against the current frame's mask
    catches that.
    """
    h, w = mask.shape
    keep = np.zeros(len(points), dtype=bool)
    for i, (x, y) in enumerate(points):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 127:
            keep[i] = True
    return keep


# ------------------------------------------------------------------
# Preview
# ------------------------------------------------------------------

def render_preview(results, n_frames, fps):
    """
    Green quad  = good fit
    Amber quad  = fitted but low confidence, worth checking
    Red banner  = no valid fit on this frame
    """
    import imageio.v2 as imageio

    first = cv2.imread(os.path.join(FRAMES_DIR, f"{REFERENCE_FRAME:04d}.jpg"))
    oh, ow = first.shape[:2]
    s = PREVIEW_WIDTH / ow
    pw = PREVIEW_WIDTH
    ph = int(round(oh * s))
    ph -= ph % 2

    os.makedirs(PREVIEWS, exist_ok=True)
    writer = imageio.get_writer(GEOM_PREVIEW, fps=fps, macro_block_size=None)

    for i in range(n_frames):
        img = cv2.imread(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"))
        if img is None:
            continue
        img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

        r = results[i]
        if r["corners"] is not None:
            q = (np.array(r["corners"], np.float32) * s).astype(np.int32)

            if r["valid"]:
                colour = (0, 230, 0)
            else:
                colour = (0, 190, 255)

            cv2.polylines(img, [q.reshape(-1, 1, 2)], True, colour, 2)
            for j, (x, y) in enumerate(q):
                cv2.circle(img, (int(x), int(y)), 5, colour, -1)
                cv2.putText(img, str(j), (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        status = (f"frame {i}   inliers {r['inlier_ratio']:.0%}   "
                  f"pts {r['points_used']}")
        if not r["valid"]:
            cv2.rectangle(img, (0, 0), (pw - 1, ph - 1), (0, 0, 255), 6)
            status += f"   INVALID: {r['reason']}"

        cv2.rectangle(img, (0, 0), (pw, 34), (0, 0, 0), -1)
        cv2.putText(img, status, (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    writer.close()


# ------------------------------------------------------------------

def main():
    if not os.path.exists(TRACKS_PATH):
        raise FileNotFoundError(
            f"{TRACKS_PATH} missing.\nRun s4_points.py first (without --seed-only)."
        )

    with open(TRACKS_PATH, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)

    n_frames = tracks_data["num_frames"]
    n_points = tracks_data["num_points"]

    fps = 30.0
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            fps = json.load(f).get("fps", 30.0)

    print(f"Loaded {n_points} points across {n_frames} frames")

    # ---- reference shape ------------------------------------------
    # Build reference mask from the polygon, same as stage 4
    SEED_PATH = os.path.join(OUTPUT_ROOT, "seed_boxes.json")
    with open(SEED_PATH, "r") as f:
        seed = json.load(f)
    obj = [o for o in seed["objects"] if o["obj_id"] == OBJ_ID][0]

    if obj.get("polygon"):
        ref_mask = np.zeros((tracks_data["frame_height"],
                             tracks_data["frame_width"]), np.uint8)
        pts = np.array(obj["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(ref_mask, [pts], 255)
        print("Using polygon from stage 2 as reference mask")
    else:
        ref_mask_path = os.path.join(MASKS_RAW, f"obj_{OBJ_ID}",
                                     f"{REFERENCE_FRAME:04d}.png")
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_mask is None:
            raise FileNotFoundError(f"No reference mask at {ref_mask_path}")
        print("Using SAM2 mask as reference")

    ref_quad = reference_quad_from_mask(ref_mask)
    reference_area = quad_area(ref_quad)
    print(f"Reference quad from frame {REFERENCE_FRAME}, "
          f"area {reference_area:,.0f} px")

    # Point positions on the reference frame
    ref_frame_entry = tracks_data["frames"][REFERENCE_FRAME]
    ref_all = np.array(ref_frame_entry["points"], np.float32)
    ref_visible = np.array(ref_frame_entry["visible"], bool)

    # ---- fit every frame -------------------------------------------
    results = []
    print("Fitting geometry...")

    for i in range(n_frames):
        entry = tracks_data["frames"][i]
        cur_all = np.array(entry["points"], np.float32)
        cur_visible = np.array(entry["visible"], bool)

        # A point is usable if it was visible on BOTH the reference
        # frame and this one.
        usable = ref_visible & cur_visible

        # If SAM2 masks exist, also filter points that drifted outside
        # the label. If we're in polygon-direct mode (no SAM2), skip
        # this — MAGSAC++ handles drifted points via outlier rejection.
        mask_path = os.path.join(MASKS_RAW, f"obj_{OBJ_ID}", f"{i:04d}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 0:
                usable &= points_inside_mask(cur_all, mask)

        ref_pts = ref_all[usable]
        cur_pts = cur_all[usable]

        corners, H, inlier_ratio, valid, reason = fit_frame(
            ref_pts, cur_pts, ref_quad, reference_area
        )

        results.append({
            "frame": i,
            "valid": bool(valid),
            "reason": reason,
            "points_used": int(usable.sum()),
            "inlier_ratio": round(float(inlier_ratio), 4),
            "corners": corners.tolist() if corners is not None else None,
            "homography": H.tolist() if H is not None else None,
        })

        if i % 20 == 0:
            print(f"  frame {i}/{n_frames}  "
                  f"pts {int(usable.sum())}  inliers {inlier_ratio:.0%}")

    # ---- write ------------------------------------------------------
    with open(GEOM_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "obj_id": OBJ_ID,
            "reference_frame": REFERENCE_FRAME,
            "reference_quad": ref_quad.tolist(),
            "num_frames": n_frames,
            "frame_width": tracks_data["frame_width"],
            "frame_height": tracks_data["frame_height"],
            "frames": results,
        }, f, indent=2)

    print(f"Wrote geometry -> {GEOM_PATH}")

    # ---- summary ----------------------------------------------------
    valid_frames = [r for r in results if r["valid"]]
    invalid = [r for r in results if not r["valid"]]
    ratios = [r["inlier_ratio"] for r in valid_frames]

    print()
    print("-" * 62)
    print("GEOMETRY SUMMARY")
    print(f"  valid frames    : {len(valid_frames)}/{n_frames}")
    if ratios:
        print(f"  mean inlier rate: {np.mean(ratios):.0%}")
        print(f"  worst frame     : {min(ratios):.0%}")

    if invalid:
        nums = [r["frame"] for r in invalid]
        # Collapse consecutive frame numbers into ranges for readability
        runs, start, prev = [], nums[0], nums[0]
        for n in nums[1:]:
            if n == prev + 1:
                prev = n
                continue
            runs.append((start, prev))
            start = prev = n
        runs.append((start, prev))
        pretty = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)

        print(f"  invalid frames  : {pretty}")
        reasons = {}
        for r in invalid:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"      {count:>3} x  {reason}")
    print("-" * 62)

    # ---- preview -----------------------------------------------------
    if RENDER_PREVIEW:
        print("Rendering preview...")
        render_preview(results, n_frames, fps)
        print(f"Preview -> {GEOM_PREVIEW}")

    print()
    print("Watch the preview. The green quad should sit on the label and")
    print("follow it smoothly. Some wobble is expected at this stage -")
    print("stage 6 smooths it out. What you're checking here is whether")
    print("the quad is on the right thing at all.")


if __name__ == "__main__":
    main()