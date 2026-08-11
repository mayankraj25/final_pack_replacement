

import os
import json
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

OUTPUT_ROOT     = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"
SAM2_CHECKPOINT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Which frame you want to mark the occluder on. Pick one where the
# occluder is fully in view and clearly separated from the label -
# usually a middle frame is better than frame 0 for a hand that
# enters partway through.
SEED_FRAME_INDEX = 0

MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 800

FRAMES_DIR      = os.path.join(OUTPUT_ROOT, "frames")
OCCLUDER_DIR    = os.path.join(OUTPUT_ROOT, "occluder_masks")
OCCLUDER_SEED   = os.path.join(OUTPUT_ROOT, "occluder_seed.json")
PREVIEWS        = os.path.join(OUTPUT_ROOT, "previews")


def draw_polygon(display_img, scale):
    """Same polygon UI as s2_seed.py."""
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


def main():
    os.makedirs(OCCLUDER_DIR, exist_ok=True)
    os.makedirs(PREVIEWS, exist_ok=True)

    seed_path = os.path.join(FRAMES_DIR, f"{SEED_FRAME_INDEX:04d}.jpg")
    frame = cv2.imread(seed_path)
    if frame is None:
        raise FileNotFoundError(f"Frame not found: {seed_path}")

    h, w = frame.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    display = cv2.resize(frame, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()

    print(f"Draw polygon around the OCCLUDER on frame {SEED_FRAME_INDEX}.")
    print("Doesn't have to be exact - SAM2 refines it.")

    polygon = draw_polygon(display, scale)
    if polygon is None:
        print("Cancelled.")
        return

    # Save what you drew (useful if you want to re-run without redrawing)
    with open(OCCLUDER_SEED, "w") as f:
        json.dump({"seed_frame": SEED_FRAME_INDEX, "polygon": polygon}, f, indent=2)

    # Fill polygon into a mask for SAM2 to refine
    seed_mask = np.zeros((h, w), np.uint8)
    pts = np.array(polygon, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(seed_mask, [pts], 255)

    # Run SAM2 video predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 on {device}...")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)

    n_frames = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])

    print(f"Initialising on {n_frames} frames...")
    state = predictor.init_state(video_path=FRAMES_DIR)
    predictor.reset_state(state)

    # Register the occluder mask on our seed frame
    predictor.add_new_mask(
        inference_state=state,
        frame_idx=SEED_FRAME_INDEX,
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
    print("Now run s9_composite.py - it'll pick them up automatically.")


if __name__ == "__main__":
    main()