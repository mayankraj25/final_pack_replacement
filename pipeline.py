"""
pipeline.py  -  pack replacement pipeline, combined into one file.

"""

import os
import json
import shutil
import argparse

import cv2
import numpy as np
from scipy.signal import savgol_filter

# ==================================================================
# GLOBAL SETTINGS  -  shared by every stage
# ==================================================================

OUTPUT_ROOT = "/Users/mayankraj/Desktop/packRep/output"

OBJ_ID = 1   # which tracked label s4 / s5 / s9 operate on

MAX_DISPLAY_W = 1400   # interactive window sizing - used by s2 and s2b
MAX_DISPLAY_H = 800

PREVIEW_WIDTH = 1280   # QA preview video width - used by s4, s5, s6

# ==================================================================
# DERIVED PATHS  -  the JSON / image interface between stages.
# Centralised here so every stage agrees on where things live -
# previously each script redeclared these from its own OUTPUT_ROOT,
# which could silently drift out of sync between files.
# ==================================================================

FRAMES_DIR     = os.path.join(OUTPUT_ROOT, "frames")
META_PATH      = os.path.join(OUTPUT_ROOT, "video_meta.json")
PREVIEWS       = os.path.join(OUTPUT_ROOT, "previews")

SEED_PATH      = os.path.join(OUTPUT_ROOT, "seed_boxes.json")
PREVIEW_PATH   = os.path.join(PREVIEWS, "seed_frame0.jpg")

OCCLUDER_DIR   = os.path.join(OUTPUT_ROOT, "occluder_masks")
OCCLUDER_SEED  = os.path.join(OUTPUT_ROOT, "occluder_seed.json")

# Only ever read as an optional fallback / point filter. Nothing in
# this pipeline writes it anymore (that was s3_masks.py, removed) -
# if it doesn't exist, s4 and s5 both fall back to the polygon.
MASKS_RAW      = os.path.join(OUTPUT_ROOT, "masks_raw")

TRACKS_PATH    = os.path.join(OUTPUT_ROOT, "point_tracks.json")
SEED_DIAG      = os.path.join(PREVIEWS, "seed_points.jpg")
TRACK_PREVIEW  = os.path.join(PREVIEWS, "point_tracks.mp4")

GEOM_PATH      = os.path.join(OUTPUT_ROOT, "geometry.json")
GEOM_PREVIEW   = os.path.join(PREVIEWS, "geometry.mp4")

SMOOTH_PATH    = os.path.join(OUTPUT_ROOT, "geometry_smoothed.json")
SMOOTH_PREVIEW = os.path.join(PREVIEWS, "smoothing_compare.mp4")

OUTPUT_VIDEO   = os.path.join(OUTPUT_ROOT, "composited_output.mp4")


# ==================================================================
# STAGE 1  -  video -> numbered frames
# ==================================================================

S1_VIDEO_PATH   = r"/Users/mayankraj/Desktop/packRep/input/Vichy.mp4"
S1_JPEG_QUALITY = 95
S1_LIMIT        = 0        # 0 = all frames. Set e.g. 30 for a quick test.


def stage_s1():
    if not os.path.exists(S1_VIDEO_PATH):
        raise FileNotFoundError(f"Video not found:\n  {S1_VIDEO_PATH}")

    # Wipe frames folder so a re-run can't leave stale frames from a
    # previous, longer video mixed in with the new ones.
    if os.path.isdir(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(PREVIEWS, exist_ok=True)

    cap = cv2.VideoCapture(S1_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video:\n  {S1_VIDEO_PATH}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("WARNING: could not read fps, defaulting to 30.0")
        fps = 30.0

    print(f"Source : {os.path.basename(S1_VIDEO_PATH)}")
    print(f"         {width}x{height} @ {fps:.3f} fps, ~{reported} frames")

    params = [int(cv2.IMWRITE_JPEG_QUALITY), S1_JPEG_QUALITY]

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imwrite(os.path.join(FRAMES_DIR, f"{idx:04d}.jpg"), frame, params)
        idx += 1

        if idx % 25 == 0:
            print(f"  extracted {idx}")
        if S1_LIMIT and idx >= S1_LIMIT:
            print(f"  stopping at S1_LIMIT={S1_LIMIT}")
            break

    cap.release()

    if idx == 0:
        raise RuntimeError("No frames extracted. Is the video file valid?")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "source_video": S1_VIDEO_PATH,
            "fps": float(fps),
            "width": width,
            "height": height,
            "frame_count": idx,
        }, f, indent=2)

    print()
    print(f"Done. {idx} frames -> {FRAMES_DIR}")
    print(f"Metadata      -> {META_PATH}")


# ==================================================================
# STAGE 2  -  tell the pipeline which label to track
# ==================================================================

S2_BANNER_H = 34


def s2_draw_banner(canvas, text):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], S2_BANNER_H), (0, 0, 0), -1)
    cv2.putText(canvas, text, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def s2_draw_polygon(display_img, obj_index, scale):
    verts = []
    win = f"Label {obj_index} POLYGON   click outline   U=undo   ENTER=done   ESC=cancel"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            verts.append([x, y])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = display_img.copy()

        if len(verts) >= 2:
            pts = np.array(verts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (0, 220, 255), 2)
            cv2.line(canvas, tuple(verts[-1]), tuple(verts[0]),
                     (0, 140, 200), 1)

        for i, (x, y) in enumerate(verts):
            cv2.circle(canvas, (x, y), 5, (0, 220, 255), -1)
            cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1)
            cv2.putText(canvas, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        s2_draw_banner(canvas,
                       f"POLYGON  {len(verts)} points   "
                       f"LEFT=add   U=undo   ENTER=done (need 3+)   ESC=cancel")

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 and len(verts) >= 3:
            break
        elif key == 27:
            cv2.destroyWindow(win)
            return None
        elif key in (ord("u"), ord("U")) and verts:
            verts.pop()

    cv2.destroyWindow(win)
    return [[v[0] / scale, v[1] / scale] for v in verts]


def s2_collect_clicks(display_img, obj_index, scale):
    positives, negatives, history = [], [], []
    win = f"Label {obj_index}   L=is label   R=is NOT   U=undo   ENTER=done"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            positives.append([x / scale, y / scale])
            history.append("pos")
        elif event == cv2.EVENT_RBUTTONDOWN:
            negatives.append([x / scale, y / scale])
            history.append("neg")

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = display_img.copy()

        for x, y in positives:
            p = (int(x * scale), int(y * scale))
            cv2.circle(canvas, p, 7, (0, 230, 0), -1)
            cv2.circle(canvas, p, 7, (255, 255, 255), 1)
        for x, y in negatives:
            p = (int(x * scale), int(y * scale))
            cv2.circle(canvas, p, 7, (0, 0, 255), -1)
            cv2.circle(canvas, p, 7, (255, 255, 255), 1)

        s2_draw_banner(canvas,
                       "LEFT = label    RIGHT = not label    U = undo    "
                       "ENTER = done    ESC = discard")

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:
            break
        if key == 27:
            cv2.destroyWindow(win)
            return None, None
        if key in (ord("u"), ord("U")) and history:
            last = history.pop()
            if last == "pos" and positives:
                positives.pop()
            elif last == "neg" and negatives:
                negatives.pop()

    cv2.destroyWindow(win)
    return positives, negatives


def s2_choose_mode(display_img, obj_index, existing, scale):
    win = "Choose how to mark this label"

    while True:
        canvas = display_img.copy()

        for det in existing:
            if det.get("_poly_display"):
                pts = np.array(det["_poly_display"], np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, (0, 200, 0), 2)
            elif det.get("_box_display"):
                b = det["_box_display"]
                cv2.rectangle(canvas, (int(b[0]), int(b[1])),
                              (int(b[2]), int(b[3])), (0, 200, 0), 2)

        s2_draw_banner(canvas,
                       f"Label {obj_index}:   B = box    "
                       f"P = polygon (tilted/odd shape)    Q = finish")

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("b"), ord("B")):
            cv2.destroyWindow(win)
            return "box"
        if key in (ord("p"), ord("P")):
            cv2.destroyWindow(win)
            return "poly"
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(win)
            return None


def stage_s2():
    os.makedirs(PREVIEWS, exist_ok=True)

    f0_path = os.path.join(FRAMES_DIR, "0000.jpg")
    frame = cv2.imread(f0_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read {f0_path}\nRun stage s1 first.")

    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    display = (cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
               if scale < 1.0 else frame.copy())

    print()
    print("=" * 68)
    print("  SELECT THE LABELS TO TRACK")
    print("=" * 68)
    print()
    print("  B - BOX      drag a rectangle. Fast, fine for most shots.")
    print("  P - POLYGON  click round the outline. Use when the label")
    print("               is tilted or oddly shaped.")
    print("  Q - finish")
    print()
    print("  Then add refinement clicks:")
    print("    LEFT-click inside the label     -> this IS it")
    print("    RIGHT-click on the bottle/hand  -> this is NOT it")
    print()
    print("  TIP: one positive click in the middle of the label plus")
    print("  one negative click on the bottle body usually stops SAM2")
    print("  from grabbing the whole bottle.")
    print("=" * 68)
    print()

    detections = []
    obj_id = 1

    while True:
        mode = s2_choose_mode(display, obj_id, detections, scale)
        if mode is None:
            break

        marked_display = display.copy()
        polygon = None
        box_orig = None

        if mode == "box":
            box_win = f"Drag box around label {obj_id}   (ENTER=confirm, ESC=cancel)"
            roi = cv2.selectROI(box_win, display, showCrosshair=True,
                                fromCenter=False)
            cv2.destroyWindow(box_win)

            if roi[2] == 0 or roi[3] == 0:
                print(f"Label {obj_id} cancelled.")
                continue

            x, y, bw, bh = roi
            box_orig = [x / scale, y / scale, (x + bw) / scale, (y + bh) / scale]
            cv2.rectangle(marked_display, (int(x), int(y)),
                          (int(x + bw), int(y + bh)), (255, 160, 0), 2)

        else:
            polygon = s2_draw_polygon(display, obj_id, scale)
            if polygon is None:
                print(f"Label {obj_id} cancelled.")
                continue

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            box_orig = [min(xs), min(ys), max(xs), max(ys)]

            pts = np.array([[int(px * scale), int(py * scale)]
                            for px, py in polygon], np.int32).reshape(-1, 1, 2)
            cv2.polylines(marked_display, [pts], True, (255, 160, 0), 2)

        positives, negatives = s2_collect_clicks(marked_display, obj_id, scale)
        if positives is None:
            print(f"Label {obj_id} discarded.")
            continue

        det = {
            "obj_id": obj_id,
            "mode": mode,
            "box": [float(v) for v in box_orig],
            "polygon": ([[float(a), float(b)] for a, b in polygon]
                        if polygon else None),
            "positive_points": [[float(a), float(b)] for a, b in positives],
            "negative_points": [[float(a), float(b)] for a, b in negatives],
            "_box_display": [v * scale for v in box_orig],
            "_poly_display": ([[p[0] * scale, p[1] * scale] for p in polygon]
                              if polygon else None),
        }

        detections.append(det)

        shape_desc = (f"polygon ({len(polygon)} pts)" if polygon else "box")
        print(f"Label {obj_id}: {shape_desc} + {len(positives)} positive, "
              f"{len(negatives)} negative clicks")
        obj_id += 1

    cv2.destroyAllWindows()

    if not detections:
        print("No labels selected. Nothing written.")
        return

    clean = []
    for det in detections:
        clean.append({k: v for k, v in det.items() if not k.startswith("_")})

    with open(SEED_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "source_frame": "0000.jpg",
            "frame_width": w,
            "frame_height": h,
            "num_objects": len(clean),
            "objects": clean,
        }, f, indent=2)

    overlay = frame.copy()
    for det in clean:
        if det.get("polygon"):
            pts = np.array(det["polygon"], np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], True, (0, 200, 255), 3)
            anchor = det["polygon"][0]
        else:
            b = det["box"]
            cv2.rectangle(overlay, (int(b[0]), int(b[1])),
                          (int(b[2]), int(b[3])), (0, 200, 255), 3)
            anchor = [b[0], b[1]]

        cv2.putText(overlay, f"obj {det['obj_id']}",
                    (int(anchor[0]), max(int(anchor[1]) - 12, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        for px, py in det["positive_points"]:
            cv2.circle(overlay, (int(px), int(py)), 10, (0, 230, 0), -1)
        for nx, ny in det["negative_points"]:
            cv2.circle(overlay, (int(nx), int(ny)), 10, (0, 0, 255), -1)

    cv2.imwrite(PREVIEW_PATH, overlay)

    print()
    print(f"Saved {len(clean)} label(s) -> {SEED_PATH}")
    print(f"Preview               -> {PREVIEW_PATH}")


# ==================================================================
# STAGE 2B  -  mark an occluder (hand/arm/etc) and propagate its mask
# ==================================================================

S2B_SAM2_CHECKPOINT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\checkpoints\sam2.1_hiera_base_plus.pt"
S2B_SAM2_CONFIG      = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Which frame you want to mark the occluder on. Pick one where the
# occluder is fully in view and clearly separated from the label -
# usually a middle frame is better than frame 0 for a hand that
# enters partway through.
S2B_SEED_FRAME_INDEX = 0


def s2b_draw_polygon(display_img, scale):
    """Same polygon UI as stage 2."""
    verts = []
    win = "Draw polygon around the OCCLUDER (hand/arm/etc)   ENTER=done   U=undo   ESC=cancel"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            verts.append([x, y])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = display_img.copy()

        if len(verts) >= 2:
            pts = np.array(verts, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (0, 100, 255), 2)
            cv2.line(canvas, tuple(verts[-1]), tuple(verts[0]),
                     (0, 60, 200), 1)
        for i, (x, y) in enumerate(verts):
            cv2.circle(canvas, (x, y), 5, (0, 100, 255), -1)
            cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1)

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(canvas,
                    f"OCCLUDER  {len(verts)} points   LEFT=add   U=undo   "
                    f"ENTER=done (need 3+)   ESC=cancel",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 and len(verts) >= 3:
            break
        if key == 27:
            cv2.destroyWindow(win)
            return None
        if key in (ord("u"), ord("U")) and verts:
            verts.pop()

    cv2.destroyWindow(win)
    return [[v[0] / scale, v[1] / scale] for v in verts]


def stage_s2b():
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    os.makedirs(OCCLUDER_DIR, exist_ok=True)
    os.makedirs(PREVIEWS, exist_ok=True)

    seed_path = os.path.join(FRAMES_DIR, f"{S2B_SEED_FRAME_INDEX:04d}.jpg")
    frame = cv2.imread(seed_path)
    if frame is None:
        raise FileNotFoundError(f"Frame not found: {seed_path}")

    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    display = cv2.resize(frame, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()

    print(f"Draw polygon around the OCCLUDER on frame {S2B_SEED_FRAME_INDEX}.")
    print("Doesn't have to be exact - SAM2 refines it.")

    polygon = s2b_draw_polygon(display, scale)
    if polygon is None:
        print("Cancelled.")
        return

    # Save what you drew (useful if you want to re-run without redrawing)
    with open(OCCLUDER_SEED, "w") as f:
        json.dump({"seed_frame": S2B_SEED_FRAME_INDEX, "polygon": polygon}, f, indent=2)

    # Fill polygon into a mask for SAM2 to refine
    seed_mask = np.zeros((h, w), np.uint8)
    pts = np.array(polygon, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(seed_mask, [pts], 255)

    # Run SAM2 video predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 on {device}...")
    predictor = build_sam2_video_predictor(S2B_SAM2_CONFIG, S2B_SAM2_CHECKPOINT, device=device)

    n_frames = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])

    print(f"Initialising on {n_frames} frames...")
    state = predictor.init_state(video_path=FRAMES_DIR)
    predictor.reset_state(state)

    # Register the occluder mask on our seed frame
    predictor.add_new_mask(
        inference_state=state,
        frame_idx=S2B_SEED_FRAME_INDEX,
        obj_id=99,   # arbitrary ID, just needs to be unique
        mask=seed_mask > 127,
    )

    # Propagate forward from seed frame
    print("Propagating forward...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            logits = mask_logits[i].cpu().numpy().squeeze()
            binary = (logits > 0.0).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(OCCLUDER_DIR, f"{frame_idx:04d}.png"), binary)

    # Propagate backward from seed frame to cover earlier frames
    print("Propagating backward...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=True):
        for i, oid in enumerate(obj_ids):
            logits = mask_logits[i].cpu().numpy().squeeze()
            binary = (logits > 0.0).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(OCCLUDER_DIR, f"{frame_idx:04d}.png"), binary)

    # Any frame not covered by either pass gets an empty mask
    for i in range(n_frames):
        out = os.path.join(OCCLUDER_DIR, f"{i:04d}.png")
        if not os.path.exists(out):
            cv2.imwrite(out, np.zeros((h, w), np.uint8))

    print(f"\nOccluder masks written to {OCCLUDER_DIR}")
    print("Now run stage s9. If you're running it as a separate command, set")
    print("S9_USE_OCCLUDER_MASKS = True near the top of pipeline.py first -")
    print("s9 won't pick these up on its own. `python pipeline.py all")
    print("--include-occluder` does this for you automatically.")


# ==================================================================
# STAGE 4  -  seed and track points on the label (CoTracker3)
# ==================================================================

S4_TRACKING_WIDTH = 512
# Video is downscaled to this width before tracking, then results are
# scaled back up. 4K tracking would need ~10GB and take forever.
# 512 is sane on CPU. Raise to 768/1024 if you get a GPU.

S4_TARGET_POINT_COUNT = 500

# Where points go. Because a plain label has no texture in the middle,
# we deliberately bias toward corners and the label outline.
S4_CORNER_FRACTION   = 0.5    # real trackable features (Shi-Tomasi)
S4_BOUNDARY_FRACTION = 0.3    # along the mask outline (always high contrast)
S4_GRID_FRACTION     = 0.2    # even coverage fallback

S4_CORNER_QUALITY      = 0.01   # lower = accepts weaker corners
S4_CORNER_MIN_DISTANCE = 8

S4_VISIBILITY_THRESHOLD = 0.5
S4_MODEL_VARIANT = "cotracker3_offline"   # or "cotracker3_online"


# ---- seeding: deciding where to put the tracking points -----------

def s4_seed_corner_points(gray, mask, want):
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
        gray, maxCorners=int(want), qualityLevel=S4_CORNER_QUALITY,
        minDistance=S4_CORNER_MIN_DISTANCE, mask=mask,
    )
    if corners is None:
        return np.zeros((0, 2), np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def s4_seed_boundary_points(mask, want, inset_px=4):
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


def s4_seed_grid_points(mask, want):
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


def s4_dedupe(points, min_dist=3.0):
    """Drop points that landed almost on top of each other."""
    if len(points) == 0:
        return points
    kept = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(np.array(kept) - p, axis=1).min() >= min_dist:
            kept.append(p)
    return np.array(kept, np.float32)


def s4_build_seed_points(frame_bgr, mask):
    total = S4_TARGET_POINT_COUNT
    want_corner   = int(total * S4_CORNER_FRACTION)
    want_boundary = int(total * S4_BOUNDARY_FRACTION)
    want_grid     = int(total * S4_GRID_FRACTION)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    corner_pts   = s4_seed_corner_points(gray, mask, want_corner)
    boundary_pts = s4_seed_boundary_points(mask, want_boundary)
    grid_pts     = s4_seed_grid_points(mask, want_grid)

    parts = [p for p in (corner_pts, boundary_pts, grid_pts) if len(p) > 0]
    all_pts = np.vstack(parts) if parts else np.zeros((0, 2), np.float32)
    all_pts = s4_dedupe(all_pts)

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


def s4_save_seed_preview(frame_bgr, corner_pts, boundary_pts, grid_pts):
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


# ---- tracking -------------------------------------------------------

def s4_load_video_tensor(n_frames):
    """Load frames as a downscaled tensor. Returns (tensor, scale, dims)."""
    import torch

    first = cv2.imread(os.path.join(FRAMES_DIR, "0000.jpg"))
    if first is None:
        raise FileNotFoundError(f"No frames in {FRAMES_DIR}")

    oh, ow = first.shape[:2]
    scale = S4_TRACKING_WIDTH / ow
    nw = S4_TRACKING_WIDTH
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


def s4_run_cotracker(video_tensor, query_points):
    """
    Run CoTracker3. Loads the model from torch.hub, which downloads
    from GitHub on first run only.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CoTracker3 ({S4_MODEL_VARIANT})...")
    model = torch.hub.load("facebookresearch/co-tracker", S4_MODEL_VARIANT)
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


def s4_save_tracks_preview(tracks, visibility, scale, n_frames, fps):
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
            colour = (0, 230, 0) if v > S4_VISIBILITY_THRESHOLD else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 2, colour, -1)

        n_vis = int((vis > S4_VISIBILITY_THRESHOLD).sum())
        cv2.putText(img, f"frame {i}   visible {n_vis}/{len(vis)}",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    writer.close()


def stage_s4(seed_only=False):
    os.makedirs(PREVIEWS, exist_ok=True)

    n_frames = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")]) \
        if os.path.isdir(FRAMES_DIR) else 0
    if n_frames == 0:
        raise RuntimeError(f"No frames in {FRAMES_DIR}\nRun stage s1 first.")

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
    with open(SEED_PATH, "r") as f:
        seed = json.load(f)
    obj = [o for o in seed["objects"] if o["obj_id"] == OBJ_ID][0]

    mask0_path = None
    if obj.get("polygon"):
        mask0 = np.zeros((h0, w0), np.uint8)
        pts = np.array(obj["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask0, [pts], 255)
        print("Using polygon from stage 2 as mask (SAM2 bypassed)")
    else:
        # Fallback: if you used a box not a polygon, try a SAM2 mask -
        # nothing in this pipeline writes masks_raw/ anymore, so this
        # will only work if you produced it some other way.
        mask0_path = os.path.join(MASKS_RAW, f"obj_{OBJ_ID}", "0000.png")
        mask0 = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE)
        if mask0 is None:
            raise FileNotFoundError(
                f"No mask and no polygon. Draw a polygon in stage s2 "
                f"(press P) - the box-only fallback needs a mask at "
                f"{mask0_path}, which nothing in this pipeline produces."
            )
        print("Using SAM2 mask from masks_raw/ (fallback)")

    all_pts, corner_pts, boundary_pts, grid_pts, diag = \
        s4_build_seed_points(frame0, mask0)

    s4_save_seed_preview(frame0, corner_pts, boundary_pts, grid_pts)

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
            f"mask - open {mask0_path or '(polygon-derived mask)'} and check "
            f"it actually shows the label."
        )

    if seed_only:
        print()
        print(f"Seed preview -> {SEED_DIAG}")
        print("Stopping (--seed-only). Look at that image before the full run.")
        return

    # ---- track -----------------------------------------------------
    video, scale, (ow, oh), (nw, nh) = s4_load_video_tensor(n_frames)

    # Seed points were found at ORIGINAL resolution; scale down for
    # the tracker, then scale results back up afterwards.
    tracks, visibility = s4_run_cotracker(video, all_pts * scale)
    tracks_full = tracks / scale

    frames_out = []
    for i in range(tracks_full.shape[0]):
        vis_i = visibility[i] > S4_VISIBILITY_THRESHOLD
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
    s4_save_tracks_preview(tracks, visibility, scale, n_frames, fps)
    print(f"Preview      -> {TRACK_PREVIEW}")

    # ---- survival summary -------------------------------------------
    t = S4_VISIBILITY_THRESHOLD
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


# ==================================================================
# STAGE 5  -  fit a per-frame homography / 4-corner quad
# ==================================================================

# Which frame the points were seeded on. Everything is measured
# relative to this frame's label shape.
S5_REFERENCE_FRAME = 0

# MAGSAC++ reprojection threshold, in pixels at FULL resolution.
# A point has to land within this distance of where the fitted model
# predicts to count as an inlier. Larger = more forgiving.
S5_RANSAC_THRESHOLD = 4.0

# A frame needs at least this many usable points to attempt a fit.
# Below this the geometry is under-constrained and we mark the frame
# invalid rather than emitting a garbage shape.
S5_MIN_POINTS_FOR_FIT = 12

# Frames where fewer than this fraction of points agree with the
# fitted model get flagged. Not discarded - flagged - so an editor
# can tell which frames to look at.
S5_MIN_INLIER_RATIO = 0.40

# Sanity limits on the fitted quad. Catches blown-up solutions that
# are mathematically valid but physically nonsense.
S5_MAX_AREA_RATIO = 10.0   # quad can't grow more than 10x the reference
S5_MIN_AREA_RATIO = 0.03   # or shrink below 3%

S5_RENDER_PREVIEW = True


# ---- reference shape ------------------------------------------------

def s5_reference_quad_from_points(ref_points, ref_visible):
    """
    Build the reference quad from the actual tracked points rather
    than from the mask. The tracked points sit on real label features
    (text corners, graphic edges), so their extremes match the label
    boundaries more tightly than a mask-fitted rectangle.
    """
    pts = ref_points[ref_visible]
    if len(pts) < 4:
        raise RuntimeError("Not enough visible points on reference frame")

    # Convex hull of all visible points
    hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.float32))
    hull = hull.reshape(-1, 2)

    # Fit a rotated rectangle to the hull — tighter than fitting to
    # the mask because the points only cover the printed label area,
    # not any blank margin the mask might include
    rect = cv2.minAreaRect(hull.reshape(-1, 1, 2).astype(np.float32))
    box = cv2.boxPoints(rect)

    return s5_order_quad(box)


def s5_reference_quad_from_mask(mask):
    """
    The label's shape as 4 corners, fitted to a binary mask.

    minAreaRect gives the tightest rotated rectangle around the mask.
    Every other frame's quad is this shape pushed through that frame's
    homography, which is what keeps corner identity stable across the
    whole clip.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Reference mask is empty.")

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)

    return s5_order_quad(box)


def s5_order_quad(pts):
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


def s5_quad_area(quad):
    return abs(cv2.contourArea(np.asarray(quad, np.float32).reshape(-1, 1, 2)))


def s5_quad_is_sane(quad, reference_area):
    """
    Reject solutions that are mathematically valid but physically
    absurd - inverted quads, or ones that ballooned to fill the frame.
    These happen when the points are nearly collinear and the
    homography is under-determined.
    """
    q = np.asarray(quad, np.float32)

    if not np.all(np.isfinite(q)):
        return False, "non-finite coordinates"

    area = s5_quad_area(q)
    if area <= 1.0:
        return False, "degenerate (zero area)"

    ratio = area / max(reference_area, 1.0)
    if ratio > S5_MAX_AREA_RATIO:
        return False, f"area exploded ({ratio:.1f}x reference)"
    if ratio < S5_MIN_AREA_RATIO:
        return False, f"area collapsed ({ratio:.2f}x reference)"

    # A valid perspective view of a rectangle stays convex.
    hull = cv2.convexHull(q.reshape(-1, 1, 2))
    if len(hull) < 4:
        return False, "quad folded on itself"

    return True, ""


# ---- fitting ----------------------------------------------------------

def s5_fit_frame(ref_pts, cur_pts, ref_quad, reference_area):
    """
    Fit the transform from the reference frame to this frame.

    ref_pts / cur_pts are matched arrays - same point, two frames.
    Returns (corners, H, inlier_ratio, valid, reason).
    """
    if len(ref_pts) < S5_MIN_POINTS_FOR_FIT:
        return None, None, 0.0, False, f"only {len(ref_pts)} usable points"

    # USAC_MAGSAC is MAGSAC++, already built into OpenCV 4.5+.
    # It avoids the hard inlier/outlier cutoff that plain RANSAC needs,
    # which means one less threshold to hand-tune per shot.
    H, inlier_mask = cv2.findHomography(
        ref_pts.reshape(-1, 1, 2),
        cur_pts.reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=S5_RANSAC_THRESHOLD,
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

    sane, reason = s5_quad_is_sane(corners, reference_area)
    if not sane:
        return corners, H, inlier_ratio, False, reason

    if inlier_ratio < S5_MIN_INLIER_RATIO:
        return corners, H, inlier_ratio, False, \
            f"low agreement ({inlier_ratio:.0%} of points)"

    return corners, H, inlier_ratio, True, ""


def s5_points_inside_mask(points, mask):
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


# ---- preview ------------------------------------------------------------

def s5_render_preview(results, n_frames, fps):
    """
    Green quad  = good fit
    Amber quad  = fitted but low confidence, worth checking
    Red banner  = no valid fit on this frame
    """
    import imageio.v2 as imageio

    first = cv2.imread(os.path.join(FRAMES_DIR, f"{S5_REFERENCE_FRAME:04d}.jpg"))
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


def stage_s5():
    if not os.path.exists(TRACKS_PATH):
        raise FileNotFoundError(
            f"{TRACKS_PATH} missing.\nRun stage s4 first (without --seed-only)."
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
                                     f"{S5_REFERENCE_FRAME:04d}.png")
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_mask is None:
            raise FileNotFoundError(f"No reference mask at {ref_mask_path}")
        print("Using SAM2 mask as reference")

    ref_frame_entry = tracks_data["frames"][S5_REFERENCE_FRAME]
    ref_all = np.array(ref_frame_entry["points"], np.float32)
    ref_visible = np.array(ref_frame_entry["visible"], bool)

    # The polygon you drew in stage 2 IS the label boundary - use it as
    # ground truth. Point-based hulls can under-represent the shape if
    # texture is uneven (e.g. corners cluster near a logo, and the
    # plain-colour background between logo and edges finds nothing),
    # which produces a quad smaller than the real label. That was the
    # cause of the misshapen composite - the tracked shape didn't match
    # the actual label extent.
    with open(SEED_PATH, "r") as f:
        seed = json.load(f)
    seed_obj = [o for o in seed["objects"] if o["obj_id"] == OBJ_ID][0]

    if seed_obj.get("polygon"):
        poly_mask = np.zeros(
            (tracks_data["frame_height"], tracks_data["frame_width"]), np.uint8
        )
        pts = np.array(seed_obj["polygon"], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(poly_mask, [pts], 255)
        ref_quad = s5_reference_quad_from_mask(poly_mask)
        print("Reference quad from the polygon you drew (authoritative shape)")
    elif ref_visible.sum() >= 20:
        ref_quad = s5_reference_quad_from_points(ref_all, ref_visible)
        print("Reference quad from tracked points (no polygon available)")
    else:
        ref_quad = s5_reference_quad_from_mask(ref_mask)
        print("Reference quad from SAM2 mask (fallback)")
    reference_area = s5_quad_area(ref_quad)
    print(f"Reference quad from frame {S5_REFERENCE_FRAME}, "
          f"area {reference_area:,.0f} px")

    # Point positions on the reference frame
    ref_frame_entry = tracks_data["frames"][S5_REFERENCE_FRAME]
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

        # If a SAM2 mask exists for this frame, also filter points that
        # drifted outside the label. In polygon-direct mode (the normal
        # case now) this file never exists, so it's skipped - MAGSAC++
        # handles drifted points via outlier rejection instead.
        mask_path = os.path.join(MASKS_RAW, f"obj_{OBJ_ID}", f"{i:04d}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 0:
                usable &= s5_points_inside_mask(cur_all, mask)

        ref_pts = ref_all[usable]
        cur_pts = cur_all[usable]

        corners, H, inlier_ratio, valid, reason = s5_fit_frame(
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
            "reference_frame": S5_REFERENCE_FRAME,
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
    if S5_RENDER_PREVIEW:
        print("Rendering preview...")
        s5_render_preview(results, n_frames, fps)
        print(f"Preview -> {GEOM_PREVIEW}")

    print()
    print("Watch the preview. The green quad should sit on the label and")
    print("follow it smoothly. Some wobble is expected at this stage -")
    print("stage 6 smooths it out. What you're checking here is whether")
    print("the quad is on the right thing at all.")


# ==================================================================
# STAGE 6  -  smooth the corner paths (Savitzky-Golay)
# ==================================================================

S6_ENABLED = True

# Window must be odd. Larger = smoother but starts lagging real motion.
#   5  gentle
#   7  good default
#   11+ will visibly soften fast camera moves
S6_WINDOW = 7

# Polynomial order. Must be less than WINDOW, with a decent gap.
#   1 = straight line, too rigid, flattens curves
#   2 = parabola, almost always right
#   4+ starts following the noise you're trying to remove
S6_POLYORDER = 2

S6_RENDER_PREVIEW = True


def s6_find_valid_runs(valid_flags):
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


def s6_jitter_metric(corners, valid_flags):
    """
    Measure how much the corners shake, so we can report a
    before/after number rather than just claiming it improved.

    We use second difference - the frame-to-frame change in velocity.
    Smooth motion has low acceleration; shake has high acceleration
    that flips sign constantly. Averaged over all four corners.

    Units are pixels per frame squared.
    """
    vals = []
    for a, b in s6_find_valid_runs(valid_flags):
        if b - a < 2:
            continue
        seg = corners[a:b + 1]                    # (n, 4, 2)
        accel = np.diff(seg, n=2, axis=0)         # (n-2, 4, 2)
        vals.append(np.linalg.norm(accel, axis=2).mean())
    return float(np.mean(vals)) if vals else 0.0


def s6_smooth_corners(corners, valid_flags, window, polyorder):
    """
    Smooth each corner's x and y path through time, run by run.

    corners: (T, 4, 2) array. Invalid frames are passed through
    untouched - stage 9 skips compositing on them anyway.
    """
    out = corners.copy()

    for a, b in s6_find_valid_runs(valid_flags):
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


def s6_recompute_homographies(reference_quad, smoothed_corners, valid_flags):
    """
    The homographies from stage 5 no longer match the smoothed
    corners, so refit them from the reference quad to each frame's
    smoothed quad.

    Stage 9 mainly needs the corners, but keeping the homographies
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


def s6_render_preview(raw_corners, smooth_corners_arr, valid_flags, n_frames, fps):
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


def stage_s6():
    if not os.path.exists(GEOM_PATH):
        raise FileNotFoundError(f"{GEOM_PATH} missing.\nRun stage s5 first.")

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

    runs = s6_find_valid_runs(valid_flags)
    if len(runs) > 1:
        print(f"Sequence has {len(runs)} valid runs - each smoothed separately "
              f"so gaps can't contaminate good frames:")
        for a, b in runs:
            print(f"    frames {a}-{b}  ({b - a + 1} frames)")

    # ---- smooth -----------------------------------------------------
    before = s6_jitter_metric(corners, valid_flags)

    if S6_ENABLED:
        smoothed = s6_smooth_corners(corners, valid_flags, S6_WINDOW, S6_POLYORDER)
    else:
        print("Smoothing disabled - passing corners through unchanged.")
        smoothed = corners.copy()

    after = s6_jitter_metric(smoothed, valid_flags)

    # ---- write -------------------------------------------------------
    new_homographies = s6_recompute_homographies(ref_quad, smoothed, valid_flags)

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
                "enabled": S6_ENABLED,
                "window": S6_WINDOW,
                "polyorder": S6_POLYORDER,
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
    print(f"  window / order      : {S6_WINDOW} / {S6_POLYORDER}")
    print(f"  jitter before       : {before:.3f} px/frame^2")
    print(f"  jitter after        : {after:.3f} px/frame^2")
    if before > 0:
        print(f"  reduction           : {(1 - after / before) * 100:.0f}%")
    print(f"  mean corner shift   : {mean_shift:.2f} px")
    print(f"  largest shift       : {max_shift:.2f} px")

    if max_shift > 15:
        print()
        print("  Some corners moved a lot. Either the raw track was noisy")
        print("  there, or S6_WINDOW is too large and is cutting a real fast")
        print("  move. Check the preview around the biggest corrections.")
    print("-" * 62)

    # ---- preview -------------------------------------------------------
    if S6_RENDER_PREVIEW:
        print("Rendering comparison preview...")
        s6_render_preview(corners, smoothed, valid_flags, n_frames, fps)
        print(f"Preview -> {SMOOTH_PREVIEW}")

    print()
    print("Amber is the raw quad, green is smoothed. On a clean shot they")
    print("nearly overlap. Where they separate is where the wobble was.")


# ==================================================================
# STAGE 9  -  composite the new label onto the footage
# ==================================================================

# The new label image. Must be a PNG, ideally with a transparent
# background. If it has no alpha channel, the entire rectangle is
# used as-is.
S9_NEW_LABEL_PATH = "/Users/mayankraj/Desktop/packRep/input/new_label1.png"

S9_TRANSFORM_MODE = "homography"   # "similarity" = no stretch, "homography" = old behaviour
S9_FIT_TO = "width"                # "width" | "height" | "area"
S9_ROTATION_SIGN = -1              # flip to 1 if the label rotates the wrong way

# How much to feather the edges of the composited label (pixels).
# 0 = hard edge (looks like a sticker). 2-4 = blends into the plate.
S9_EDGE_FEATHER = 3

# Output video codec. Use "mp4v" for broad compatibility.
S9_OUTPUT_CODEC = "mp4v"

# ---- occlusion settings ----

# Set True to use masks produced by stage s2b. Only relevant if the
# clip actually has a hand/object crossing the label - leave False
# otherwise. If the masks folder doesn't exist, this is auto-disabled
# for the run rather than erroring, so it's safe to leave True even on
# clips you haven't run s2b on.
#
# `python pipeline.py all --include-occluder` flips this to True for
# you at runtime (see run_all()) since it also runs s2b in that same
# call. Running s2b as its own command and then `python pipeline.py s9`
# separately does NOT flip it automatically - edit this constant by
# hand first, or s9 will composite as if there were no occluder.
S9_USE_OCCLUDER_MASKS = False

# How much to soften the occluder's edge before subtracting it from
# the label alpha, so the boundary blends instead of hard-cutting.
S9_OCCLUDER_FEATHER = 7


# ==================================================================
# OCCLUSION  -  reads masks produced by stage s2b (SAM2, propagated
# from a single hand-drawn frame)
# ==================================================================

def s9_apply_occlusion(warped_alpha, frame_idx, feather=S9_OCCLUDER_FEATHER):
    if not S9_USE_OCCLUDER_MASKS:
        return warped_alpha

    mask_path = os.path.join(OCCLUDER_DIR, f"{frame_idx:04d}.png")
    if not os.path.exists(mask_path):
        return warped_alpha

    occluder = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if occluder is None or occluder.max() == 0:
        return warped_alpha

    # Threshold to clean binary first - kills speckle/noise from
    # SAM2's raw logits before any blur touches it
    _, occluder = cv2.threshold(occluder, 127, 255, cv2.THRESH_BINARY)

    # Morphological cleanup - same treatment as the main label masks
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    occluder = cv2.morphologyEx(occluder, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    occluder = cv2.morphologyEx(occluder, cv2.MORPH_OPEN, k_open)

    if occluder.max() == 0:
        return warped_alpha   # cleanup removed everything - treat as no occlusion

    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        occluder = cv2.GaussianBlur(occluder, (k, k), 0)

    occluder_norm = occluder.astype(np.float32) / 255.0
    return warped_alpha * (1.0 - occluder_norm)


# ==================================================================
# LABEL LOADING / CHECKS
# ==================================================================

def s9_crop_to_content(label_bgra):
    """
    Crop away any transparent padding around the actual visible
    artwork, using the alpha channel to find real content bounds.

    If the source PNG's canvas is bigger than the artwork itself,
    every downstream calculation (scale, centroid) was measuring the
    wrong thing - the padded canvas instead of the actual label.
    This is very likely why the composite looked like a tiny
    misplaced sliver even though the tracked quad was correct.
    """
    alpha = label_bgra[:, :, 3]
    ys, xs = np.where(alpha > 10)   # anything not fully transparent

    if len(xs) == 0:
        print("  WARNING: label PNG appears fully transparent - "
             "check the file")
        return label_bgra

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    original_size = (label_bgra.shape[1], label_bgra.shape[0])
    cropped = label_bgra[y0:y1, x0:x1]

    print(f"  Cropped label from {original_size[0]}x{original_size[1]} "
         f"to {cropped.shape[1]}x{cropped.shape[0]} "
         f"(removed transparent padding)")

    return cropped


def s9_load_label(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read label image:\n  {path}")

    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
        img = np.concatenate([img, alpha], axis=2)
        print("Label has no alpha channel - treating as fully opaque")

    print(f"Label loaded: {img.shape[1]}x{img.shape[0]} px")

    img = s9_crop_to_content(img)

    return img


def s9_check_aspect_ratio(label_shape, corners):
    """Warn if the new label's aspect ratio doesn't match the tracked region."""
    lh, lw = label_shape[:2]
    label_ratio = lw / max(lh, 1)

    c = np.array(corners, np.float32)
    top_edge = np.linalg.norm(c[1] - c[0])
    left_edge = np.linalg.norm(c[3] - c[0])
    quad_ratio = top_edge / max(left_edge, 1)

    diff = abs(label_ratio - quad_ratio) / max(quad_ratio, 0.01)
    if diff > 0.15:
        print(f"  WARNING: aspect ratio mismatch")
        print(f"    new label   : {label_ratio:.2f}  ({lw}x{lh})")
        print(f"    tracked quad: {quad_ratio:.2f}")
        print(f"    the label will look stretched by ~{diff*100:.0f}%")
        print(f"    resize your label PNG to match, or accept the distortion")
    else:
        print(f"  Aspect ratio OK  (label {label_ratio:.2f}  vs "
              f"quad {quad_ratio:.2f})")


# ==================================================================
# TRANSFORMS
# ==================================================================

def s9_quad_dims(quad):
    """Average width and height of a 4-corner quad [TL, TR, BR, BL]."""
    q = np.array(quad, np.float32)
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    return width, height


def s9_quad_angle_deg(quad):
    """Angle of the top edge (TL -> TR), in degrees."""
    q = np.array(quad, np.float32)
    dx, dy = q[1] - q[0]
    return np.degrees(np.arctan2(dy, dx))


def s9_compute_similarity_params(ref_quad, cur_quad, label_w, label_h,
                                 fit_to="width"):
    """
    One scale, one rotation, one position - derived from how the
    tracked quad changed relative to the reference frame. No shear,
    no independent x/y stretch, ever.

    fit_to controls how the label is initially sized against the
    reference quad:
      "width"  - match the label's width to the quad's width
      "height" - match the label's height to the quad's height
      "area"   - match total area (a compromise when neither
                 dimension lines up well)
    """
    ref_w, ref_h = s9_quad_dims(ref_quad)
    cur_w, cur_h = s9_quad_dims(cur_quad)

    if fit_to == "width":
        base_scale = ref_w / max(label_w, 1)
    elif fit_to == "height":
        base_scale = ref_h / max(label_h, 1)
    else:
        base_scale = np.sqrt((ref_w * ref_h) / max(label_w * label_h, 1))

    # How much bigger/smaller the quad got, relative to the reference -
    # averaged across width and height so it stays a single uniform
    # number, not two separate stretch factors.
    growth = np.sqrt((cur_w / max(ref_w, 1e-6)) * (cur_h / max(ref_h, 1e-6)))

    scale = base_scale * growth
    angle = s9_quad_angle_deg(cur_quad) - s9_quad_angle_deg(ref_quad)

    centroid = np.array(cur_quad, np.float32).mean(axis=0)

    return scale, angle, centroid


def s9_composite_similarity(frame, label_bgra, scale, angle_deg, centroid,
                            feather, rotation_sign, frame_idx):
    """
    Place the label at (scale, rotation, position) with NO shear and
    NO independent x/y stretch. The label always keeps its own
    proportions.
    """
    lh, lw = label_bgra.shape[:2]
    fh, fw = frame.shape[:2]

    # Rotate + scale about the label's own centre, in label-local space
    M = cv2.getRotationMatrix2D((lw / 2.0, lh / 2.0),
                                rotation_sign * angle_deg, scale)
    # Then move that centre to where it belongs in the frame
    M[0, 2] += centroid[0] - lw / 2.0
    M[1, 2] += centroid[1] - lh / 2.0

    warped = cv2.warpAffine(
        label_bgra, M, (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    warped_bgr = warped[:, :, :3]
    warped_alpha = warped[:, :, 3].astype(np.float32) / 255.0

    # Occlusion applied BEFORE the edge feather, so the feather also
    # softens the occluder boundary rather than leaving it hard-edged
    warped_alpha = s9_apply_occlusion(warped_alpha, frame_idx)

    if feather > 0:
        ksize = feather * 2 + 1
        warped_alpha = cv2.GaussianBlur(warped_alpha, (ksize, ksize), 0)

    alpha_3ch = warped_alpha[:, :, np.newaxis]
    composited = (warped_bgr.astype(np.float32) * alpha_3ch +
                  frame.astype(np.float32) * (1.0 - alpha_3ch))
    return composited.astype(np.uint8)


def s9_warp_and_composite(frame, label_bgra, dst_corners, feather, frame_idx):
    """
    Warp the flat label onto the frame at the tracked position.

    frame:        BGR image (the original video frame)
    label_bgra:   the new label with alpha
    dst_corners:  4 corners [TL, TR, BR, BL] where the label should go
    feather:      edge softness in pixels
    frame_idx:    used to look up this frame's occluder mask, if any

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

    warped_alpha = s9_apply_occlusion(warped_alpha, frame_idx)

    # Feather the alpha edge so it blends instead of hard-cutting
    if feather > 0:
        ksize = feather * 2 + 1
        warped_alpha = cv2.GaussianBlur(warped_alpha, (ksize, ksize), 0)

    # Alpha composite: new label on top of original frame
    alpha_3ch = warped_alpha[:, :, np.newaxis]
    composited = (warped_bgr.astype(np.float32) * alpha_3ch +
                  frame.astype(np.float32) * (1.0 - alpha_3ch))

    return composited.astype(np.uint8)


def stage_s9():
    global S9_USE_OCCLUDER_MASKS

    if not os.path.exists(SMOOTH_PATH):
        raise FileNotFoundError(
            f"{SMOOTH_PATH} missing.\nRun stage s6 first."
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

    label = s9_load_label(S9_NEW_LABEL_PATH)

    # Check the fit against the reference frame's tracked quad before
    # processing the whole video - cheap, and tells you up front if
    # the PNG needs resizing.
    ref_idx = geom.get("reference_frame", 0)
    ref_entry = geom["frames"][ref_idx]
    if ref_entry["valid"] and ref_entry["corners"] is not None:
        print("Checking aspect ratio against reference frame...")
        s9_check_aspect_ratio(label.shape, ref_entry["corners"])
    print()

    if S9_USE_OCCLUDER_MASKS:
        if os.path.isdir(OCCLUDER_DIR) and os.listdir(OCCLUDER_DIR):
            print(f"Occlusion masking: ON  (using masks from {OCCLUDER_DIR})")
        else:
            print("Occlusion masking requested but no occluder_masks found - "
                 "disabling for this run. Run stage s2b if this clip "
                 "has a hand/object crossing the label.")
            S9_USE_OCCLUDER_MASKS = False
    else:
        print("Occlusion masking: off")
    print()

    fourcc = cv2.VideoWriter_fourcc(*S9_OUTPUT_CODEC)
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer.\n"
            f"If '{S9_OUTPUT_CODEC}' is not supported, try 'XVID' or 'avc1'."
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
            if S9_TRANSFORM_MODE == "similarity":
                scale, angle, centroid = s9_compute_similarity_params(
                    geom["reference_quad"], corners,
                    label.shape[1], label.shape[0], S9_FIT_TO
                )
                composited = s9_composite_similarity(
                    frame, label, scale, angle, centroid,
                    S9_EDGE_FEATHER, S9_ROTATION_SIGN, i
                )
            else:
                composited = s9_warp_and_composite(frame, label, corners, S9_EDGE_FEATHER, i)
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
    print(f"  occlusion masking     : {'on' if S9_USE_OCCLUDER_MASKS else 'off'}")
    print(f"  output                : {OUTPUT_VIDEO}")
    print("-" * 62)
    print()
    print("This is a FLAT composite - no lighting, no shadows.")


# ==================================================================
# CLI DRIVER
# ==================================================================

STAGE_FUNCS = {
    "s1":  stage_s1,
    "s2":  stage_s2,
    "s2b": stage_s2b,
    "s4":  stage_s4,
    "s5":  stage_s5,
    "s6":  stage_s6,
    "s9":  stage_s9,
}

# s2b is optional and only relevant if the clip has a hand/object
# occluding the label at some point - it's not part of the default
# "all" run. S9_USE_OCCLUDER_MASKS defaults to off in stage 9 to match.
PIPELINE_ORDER = ["s1", "s2", "s4", "s5", "s6", "s9"]


def run_all(include_occluder=False, seed_only=False):
    global S9_USE_OCCLUDER_MASKS

    order = PIPELINE_ORDER[:]
    if include_occluder:
        order.insert(order.index("s2") + 1, "s2b")
        # Generating occluder masks (s2b) without also telling s9 to use
        # them would composite as if there were no occlusion at all -
        # the whole point of --include-occluder is that both happen
        # together. Standalone `python pipeline.py s9` still defaults to
        # off, matching S9_USE_OCCLUDER_MASKS's declared default.
        S9_USE_OCCLUDER_MASKS = True

    for name in order:
        print()
        print("#" * 70)
        print(f"#  STAGE {name}")
        print("#" * 70)
        if name == "s4":
            stage_s4(seed_only=seed_only)
        else:
            STAGE_FUNCS[name]()


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Pack replacement pipeline - run one stage, or 'all' "
                    "to run the full sequence in order."
    )
    p.add_argument("stage", choices=list(STAGE_FUNCS) + ["all"],
                   help="which stage to run, or 'all' for the full pipeline")
    p.add_argument("--seed-only", action="store_true",
                   help="(s4 / all) seed points and save the preview image "
                        "only, don't run the tracker")
    p.add_argument("--include-occluder", action="store_true",
                   help="(all only) also run s2b between s2 and s4")
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.stage == "all":
        run_all(include_occluder=args.include_occluder, seed_only=args.seed_only)
    elif args.stage == "s4":
        stage_s4(seed_only=args.seed_only)
    else:
        STAGE_FUNCS[args.stage]()


if __name__ == "__main__":
    main()
