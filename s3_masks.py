"""
s3_masks.py  -  STAGE 3: propagate the label mask across every frame

Takes the frame-0 seed from stage 2 and uses SAM2's video predictor
to produce a pixel-accurate mask of the label on every frame.

BOX vs POLYGON SEEDS
--------------------
If stage 2 gave a box, it goes to SAM2 as a box prompt.
If stage 2 gave a polygon, it is filled into a binary mask and passed
as a mask prompt - which handles tilted or odd-shaped labels that an
axis-aligned box would enclose badly.

Either way the refinement clicks are sent as a second prompt on the
same frame, which is what stops SAM2 grabbing the whole bottle.

TWO SETS OF MASKS ARE WRITTEN
-----------------------------
masks_raw/    hard binary, no feathering.
              Stage 4 needs these - it places tracking points based
              on the mask, and a feathered edge would put points in
              the blurry transition zone where they don't belong.

masks_clean/  morphology + feathered edges.
              Stage 8 turns these into the After Effects track matte.
              A hard binary edge composites like a paper cutout; a
              1-2px feather blends into the plate properly.

Feathering early would destroy information stage 4 needs, so both
versions are kept.

RUN:  py s3_masks.py
"""

import os
import json

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

OUTPUT_ROOT     = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

SAM2_CHECKPOINT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# If you switch to the tiny checkpoint for CPU speed, change BOTH:
#   SAM2_CHECKPOINT = r"...\checkpoints\sam2.1_hiera_tiny.pt"
#   SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"

# Mask cleanup
CLOSE_SIZE  = 5     # fills small holes inside the mask
OPEN_SIZE   = 3     # removes stray specks outside it
FEATHER_PX  = 2     # softens the edge (0 = hard binary)

RENDER_PREVIEW = True
PREVIEW_WIDTH  = 1280

# ==================================================================

FRAMES_DIR  = os.path.join(OUTPUT_ROOT, "frames")
SEED_PATH   = os.path.join(OUTPUT_ROOT, "seed_boxes.json")
META_PATH   = os.path.join(OUTPUT_ROOT, "video_meta.json")
MASKS_RAW   = os.path.join(OUTPUT_ROOT, "masks_raw")
MASKS_CLEAN = os.path.join(OUTPUT_ROOT, "masks_clean")
SCORES_PATH = os.path.join(OUTPUT_ROOT, "mask_scores.json")
PREVIEWS    = os.path.join(OUTPUT_ROOT, "previews")


def clean_mask(binary_255):
    """
    CLOSE  - grow then shrink: fills small holes inside the label
    OPEN   - shrink then grow: removes stray specks outside it
    BLUR   - softens the edge from a hard 0/255 cut to a gradient,
             which is what makes the composite look natural in AE
             instead of like a sticker
    """
    m = binary_255.copy()

    if CLOSE_SIZE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if OPEN_SIZE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_SIZE, OPEN_SIZE))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    if FEATHER_PX > 0:
        ksize = int(FEATHER_PX) * 2 + 1
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)

    return m


def polygon_to_mask(polygon, height, width):
    """Fill a polygon into a boolean mask for SAM2's mask prompt."""
    mask = np.zeros((height, width), np.uint8)
    pts = np.array(polygon, np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask > 127


def logit_confidence(logits, binary):
    """
    Turn SAM2's raw mask logits into a rough 0-1 confidence.

    Logits are the model's pre-threshold output. Large magnitude means
    it's sure which side of the boundary a pixel falls on; values near
    zero mean it's guessing. Stage 7 uses this as one input to the
    per-frame confidence score. It's a proxy, not an official
    confidence head, but it correlates well and costs nothing.
    """
    if binary.sum() == 0:
        return 0.0
    vals = np.abs(logits[binary])
    if vals.size == 0:
        return 0.0
    return float(np.clip(float(vals.mean()) / 8.0, 0.0, 1.0))


def render_preview(obj_id, n_frames, fps, out_path):
    """Overlay the mask on the footage so you can check it in one watch."""
    import imageio.v2 as imageio

    first = cv2.imread(os.path.join(FRAMES_DIR, "0000.jpg"))
    oh, ow = first.shape[:2]
    s = PREVIEW_WIDTH / ow
    pw = PREVIEW_WIDTH
    ph = int(round(oh * s))
    ph -= ph % 2

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)

    for i in range(n_frames):
        img = cv2.imread(os.path.join(FRAMES_DIR, f"{i:04d}.jpg"))
        if img is None:
            continue
        m = cv2.imread(os.path.join(MASKS_CLEAN, f"obj_{obj_id}", f"{i:04d}.png"),
                       cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

        if m is not None:
            m = cv2.resize(m, (pw, ph), interpolation=cv2.INTER_NEAREST)
            tint = np.zeros_like(img)
            tint[:, :, 1] = 255
            alpha = (m.astype(np.float32) / 255.0)[..., None] * 0.45
            img = (img * (1 - alpha) + tint * alpha).astype(np.uint8)

            contours, _ = cv2.findContours((m > 127).astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 255, 255), 2)

        cv2.putText(img, f"frame {i}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    writer.close()


def main():
    # ---- checks ---------------------------------------------------
    n_frames = (len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
                if os.path.isdir(FRAMES_DIR) else 0)
    if n_frames == 0:
        raise RuntimeError(f"No frames in {FRAMES_DIR}\nRun s1_frames.py first.")

    if not os.path.exists(SEED_PATH):
        raise FileNotFoundError(f"{SEED_PATH} missing.\nRun s2_seed.py first.")

    if not os.path.exists(SAM2_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found:\n  {SAM2_CHECKPOINT}\n\n"
            f"Download it with:\n"
            f'  Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/'
            f'segment_anything_2/092824/sam2.1_hiera_base_plus.pt" '
            f'-OutFile "checkpoints\\sam2.1_hiera_base_plus.pt"'
        )

    with open(SEED_PATH, "r", encoding="utf-8") as f:
        seed_data = json.load(f)
    objects = seed_data["objects"]
    frame_w = seed_data["frame_width"]
    frame_h = seed_data["frame_height"]

    fps = 30.0
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            fps = json.load(f).get("fps", 30.0)

    os.makedirs(PREVIEWS, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. SAM2 on CPU is slow -")
        print("         expect several minutes for ~100 frames.")

    # ---- build predictor -------------------------------------------
    print(f"Loading SAM2 ({os.path.basename(SAM2_CHECKPOINT)}) on {device}")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT,
                                           device=device)

    print(f"Initialising on {n_frames} frames...")
    state = predictor.init_state(video_path=FRAMES_DIR)
    predictor.reset_state(state)

    # ---- register seeds on frame 0 ----------------------------------
    for obj in objects:
        obj_id = obj["obj_id"]
        polygon = obj.get("polygon")
        pos = obj.get("positive_points", [])
        neg = obj.get("negative_points", [])

        if polygon:
            # Polygon path: fill it into a mask and use that as the prompt.
            # Handles tilted / odd-shaped labels a rectangle would enclose
            # badly.
            mask_prompt = polygon_to_mask(polygon, frame_h, frame_w)
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask_prompt,
            )
            shape_desc = f"polygon ({len(polygon)} pts)"

            # Refinement clicks go as a separate prompt on the same frame.
            if pos or neg:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array(pos + neg, dtype=np.float32),
                    labels=np.array([1] * len(pos) + [0] * len(neg),
                                    dtype=np.int32),
                )
        else:
            # Box path: box and clicks can go in one call.
            kwargs = {
                "inference_state": state,
                "frame_idx": 0,
                "obj_id": obj_id,
                "box": np.array(obj["box"], dtype=np.float32),
            }
            if pos or neg:
                kwargs["points"] = np.array(pos + neg, dtype=np.float32)
                kwargs["labels"] = np.array([1] * len(pos) + [0] * len(neg),
                                            dtype=np.int32)
            predictor.add_new_points_or_box(**kwargs)
            shape_desc = "box"

        print(f"  seeded obj {obj_id}  {shape_desc}  "
              f"({len(pos)} positive, {len(neg)} negative clicks)")

        os.makedirs(os.path.join(MASKS_RAW, f"obj_{obj_id}"), exist_ok=True)
        os.makedirs(os.path.join(MASKS_CLEAN, f"obj_{obj_id}"), exist_ok=True)

    # ---- propagate ---------------------------------------------------
    scores, areas = {}, {}

    print("Propagating masks through the video...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        for i, obj_id in enumerate(obj_ids):
            logits = mask_logits[i].cpu().numpy().squeeze()
            binary = logits > 0.0
            raw = binary.astype(np.uint8) * 255

            cv2.imwrite(os.path.join(MASKS_RAW, f"obj_{obj_id}",
                                     f"{frame_idx:04d}.png"), raw)
            cv2.imwrite(os.path.join(MASKS_CLEAN, f"obj_{obj_id}",
                                     f"{frame_idx:04d}.png"), clean_mask(raw))

            scores.setdefault(str(obj_id), {})[str(frame_idx)] = \
                logit_confidence(logits, binary)
            areas.setdefault(str(obj_id), {})[str(frame_idx)] = int(binary.sum())

        if frame_idx % 20 == 0:
            print(f"  frame {frame_idx}/{n_frames}")

    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump({"num_frames": n_frames,
                   "confidence": scores,
                   "mask_area_px": areas}, f, indent=2)

    # ---- summary ------------------------------------------------------
    print()
    print("-" * 62)
    for obj in objects:
        oid = str(obj["obj_id"])
        a = areas.get(oid, {})
        vals = [a[str(i)] for i in range(n_frames) if str(i) in a]
        if not vals:
            continue
        empty = sum(1 for v in vals if v == 0)
        print(f"obj {oid}: mask present on {len(vals) - empty}/{len(vals)} frames"
              f"   avg area {int(np.mean(vals)):,} px")
        if empty:
            print(f"  WARNING: {empty} empty frames - label likely rotated "
                  f"out of view, or tracking dropped")
    print("-" * 62)

    # ---- previews -----------------------------------------------------
    if RENDER_PREVIEW:
        for obj in objects:
            oid = obj["obj_id"]
            out = os.path.join(PREVIEWS, f"masks_obj{oid}.mp4")
            print(f"Rendering preview for obj {oid}...")
            render_preview(oid, n_frames, fps, out)
            print(f"  -> {out}")

    print()
    print("Stage 3 complete.")
    print("Watch the preview before continuing. The green overlay should")
    print("sit on the LABEL only. If it covers the whole bottle, re-run")
    print("s2_seed.py and right-click on the cap and on the glass above")
    print("and below the label.")


if __name__ == "__main__":
    main()