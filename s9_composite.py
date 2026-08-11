
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
NEW_LABEL_PATH = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\input\065338054811_PREFERENCE 9.png"

OBJ_ID = 1

TRANSFORM_MODE = "homography"   # "similarity" = no stretch, "homography" = old behaviour
FIT_TO = "width"                # "width" | "height" | "area"
ROTATION_SIGN = -1              # flip to 1 if the label rotates the wrong way

# How much to feather the edges of the composited label (pixels).
# 0 = hard edge (looks like a sticker). 2-4 = blends into the plate.
EDGE_FEATHER = 3

# Output video codec. Use "mp4v" for broad compatibility.
OUTPUT_CODEC = "mp4v"

# ---- occlusion settings ----

# Set True to use masks produced by s2b_occluder.py. If the masks
# folder doesn't exist, this is auto-disabled for the run rather
# than erroring - safe to leave True even on clips you haven't run
# s2b_occluder.py on.
USE_OCCLUDER_MASKS = False

# How much to soften the occluder's edge before subtracting it from
# the label alpha, so the boundary blends instead of hard-cutting.
OCCLUDER_FEATHER = 7

# ==================================================================

FRAMES_DIR   = os.path.join(OUTPUT_ROOT, "frames")
SMOOTH_PATH  = os.path.join(OUTPUT_ROOT, "geometry_smoothed.json")
META_PATH    = os.path.join(OUTPUT_ROOT, "video_meta.json")
OCCLUDER_DIR = os.path.join(OUTPUT_ROOT, "occluder_masks")
OUTPUT_VIDEO = os.path.join(OUTPUT_ROOT, "composited_output.mp4")


# ==================================================================
# OCCLUSION  -  reads masks produced by s2b_occluder.py (SAM2,
# propagated from a single hand-drawn frame)
# ==================================================================

def apply_occlusion(warped_alpha, frame_idx, feather=OCCLUDER_FEATHER):
    if not USE_OCCLUDER_MASKS:
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

def crop_to_content(label_bgra):
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


def load_label(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read label image:\n  {path}")

    if img.shape[2] == 3:
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, np.uint8)
        img = np.concatenate([img, alpha], axis=2)
        print("Label has no alpha channel - treating as fully opaque")

    print(f"Label loaded: {img.shape[1]}x{img.shape[0]} px")

    img = crop_to_content(img)

    return img


def check_aspect_ratio(label_shape, corners):
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

def quad_dims(quad):
    """Average width and height of a 4-corner quad [TL, TR, BR, BL]."""
    q = np.array(quad, np.float32)
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    return width, height


def quad_angle_deg(quad):
    """Angle of the top edge (TL -> TR), in degrees."""
    q = np.array(quad, np.float32)
    dx, dy = q[1] - q[0]
    return np.degrees(np.arctan2(dy, dx))


def compute_similarity_params(ref_quad, cur_quad, label_w, label_h,
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
    ref_w, ref_h = quad_dims(ref_quad)
    cur_w, cur_h = quad_dims(cur_quad)

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
    angle = quad_angle_deg(cur_quad) - quad_angle_deg(ref_quad)

    centroid = np.array(cur_quad, np.float32).mean(axis=0)

    return scale, angle, centroid


def composite_similarity(frame, label_bgra, scale, angle_deg, centroid,
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
    warped_alpha = apply_occlusion(warped_alpha, frame_idx)

    if feather > 0:
        ksize = feather * 2 + 1
        warped_alpha = cv2.GaussianBlur(warped_alpha, (ksize, ksize), 0)

    alpha_3ch = warped_alpha[:, :, np.newaxis]
    composited = (warped_bgr.astype(np.float32) * alpha_3ch +
                  frame.astype(np.float32) * (1.0 - alpha_3ch))
    return composited.astype(np.uint8)


def warp_and_composite(frame, label_bgra, dst_corners, feather, frame_idx):
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

    warped_alpha = apply_occlusion(warped_alpha, frame_idx)

    # Feather the alpha edge so it blends instead of hard-cutting
    if feather > 0:
        ksize = feather * 2 + 1
        warped_alpha = cv2.GaussianBlur(warped_alpha, (ksize, ksize), 0)

    # Alpha composite: new label on top of original frame
    alpha_3ch = warped_alpha[:, :, np.newaxis]
    composited = (warped_bgr.astype(np.float32) * alpha_3ch +
                  frame.astype(np.float32) * (1.0 - alpha_3ch))

    return composited.astype(np.uint8)


# ==================================================================
# MAIN
# ==================================================================

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

    # Check the fit against the reference frame's tracked quad before
    # processing the whole video - cheap, and tells you up front if
    # the PNG needs resizing.
    ref_idx = geom.get("reference_frame", 0)
    ref_entry = geom["frames"][ref_idx]
    if ref_entry["valid"] and ref_entry["corners"] is not None:
        print("Checking aspect ratio against reference frame...")
        check_aspect_ratio(label.shape, ref_entry["corners"])
    print()

    global USE_OCCLUDER_MASKS
    if USE_OCCLUDER_MASKS:
        if os.path.isdir(OCCLUDER_DIR) and os.listdir(OCCLUDER_DIR):
            print(f"Occlusion masking: ON  (using masks from {OCCLUDER_DIR})")
        else:
            print("Occlusion masking requested but no occluder_masks found - "
                 "disabling for this run. Run s2b_occluder.py if this clip "
                 "has a hand/object crossing the label.")
            USE_OCCLUDER_MASKS = False
    else:
        print("Occlusion masking: off")
    print()

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
            if TRANSFORM_MODE == "similarity":
                scale, angle, centroid = compute_similarity_params(
                    geom["reference_quad"], corners,
                    label.shape[1], label.shape[0], FIT_TO
                )
                composited = composite_similarity(
                    frame, label, scale, angle, centroid,
                    EDGE_FEATHER, ROTATION_SIGN, i
                )
            else:
                composited = warp_and_composite(frame, label, corners, EDGE_FEATHER, i)
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
    print(f"  occlusion masking     : {'on' if USE_OCCLUDER_MASKS else 'off'}")
    print(f"  output                : {OUTPUT_VIDEO}")
    print("-" * 62)
    print()
    print("This is a FLAT composite - no lighting, no shadows.")
    print("For production quality, use the AE path (s8_export.py)")
    print("where the editor can add shading and colour matching.")


if __name__ == "__main__":
    main()