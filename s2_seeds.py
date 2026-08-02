"""
s2_seed.py  -  STAGE 2: tell the pipeline which label to track

TWO WAYS TO MARK A LABEL
-------------------------
BOX      Press B. Drag an axis-aligned rectangle. Fast, fine for
         most shots.

POLYGON  Press P. Click around the label's outline. Use this when
         the label is tilted, is an odd shape, or when a rectangle
         would swallow a neighbouring product.

After marking the shape, you add refinement clicks:
  LEFT-click   = this IS the label   (green dot)
  RIGHT-click  = this is NOT it      (red dot)

Press Q at the mode picker when you've done all labels.

RUN:  py s2_seed.py
"""

import os
import json

import cv2
import numpy as np

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

OUTPUT_ROOT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 800

# ==================================================================

FRAMES_DIR   = os.path.join(OUTPUT_ROOT, "frames")
SEED_PATH    = os.path.join(OUTPUT_ROOT, "seed_boxes.json")
PREVIEWS     = os.path.join(OUTPUT_ROOT, "previews")
PREVIEW_PATH = os.path.join(PREVIEWS, "seed_frame0.jpg")

BANNER_H = 34


def draw_banner(canvas, text):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], BANNER_H), (0, 0, 0), -1)
    cv2.putText(canvas, text, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


# ------------------------------------------------------------------
# POLYGON
# ------------------------------------------------------------------

def draw_polygon(display_img, obj_index, scale):
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

        draw_banner(canvas,
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


# ------------------------------------------------------------------
# REFINEMENT CLICKS
# ------------------------------------------------------------------

def collect_clicks(display_img, obj_index, scale):
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

        draw_banner(canvas,
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


# ------------------------------------------------------------------
# MODE PICKER
# ------------------------------------------------------------------

def choose_mode(display_img, obj_index, existing, scale):
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

        draw_banner(canvas,
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


# ------------------------------------------------------------------

def main():
    os.makedirs(PREVIEWS, exist_ok=True)

    f0_path = os.path.join(FRAMES_DIR, "0002.jpg")
    frame = cv2.imread(f0_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read {f0_path}\nRun s1_frames.py first.")

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
        mode = choose_mode(display, obj_id, detections, scale)
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
            polygon = draw_polygon(display, obj_id, scale)
            if polygon is None:
                print(f"Label {obj_id} cancelled.")
                continue

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            box_orig = [min(xs), min(ys), max(xs), max(ys)]

            pts = np.array([[int(px * scale), int(py * scale)]
                            for px, py in polygon], np.int32).reshape(-1, 1, 2)
            cv2.polylines(marked_display, [pts], True, (255, 160, 0), 2)

        positives, negatives = collect_clicks(marked_display, obj_id, scale)
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


if __name__ == "__main__":
    main()