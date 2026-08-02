"""
s8_export.py  -  STAGE 8: hand it to After Effects

Turns everything the pipeline computed into two files an editor can
actually use.

WHAT GETS WRITTEN
-----------------
matte_objN.mov    ProRes 4444 with a real alpha channel. Opaque where
                  the label is, transparent everywhere else. Drops
                  into AE as a track matte, which is the native way
                  editors already restrict a layer to a moving shape.

ae_setup.jsx      An ExtendScript. The editor runs it from
                  File > Scripts > Run Script File and the project
                  builds itself: matte imported, placeholder layer
                  created, Corner Pin effect keyframed on every frame,
                  opacity zeroed where tracking failed, and markers
                  dropped on frames worth checking.

WHY CORNER PIN
--------------
Tracking data in After Effects has no separate file format - it lives
as keyframes on layer properties. A Mocha planar track ends up as
keyframes on a Corner Pin effect. So does this. From the editor's
side the project looks exactly like one they tracked themselves,
except the tracking is already done.

That also means every part of it stays editable. If a keyframe is
slightly off, they nudge it. Nothing here is baked.

CODEC WARNING
-------------
ProRes 4444 is used because it carries an alpha channel. H.264 does
not - it will encode without error and silently discard the
transparency, and the matte will be useless.

RUN:  py s8_export.py
"""

import os
import json

import cv2
import numpy as np

# ==================================================================
# SETTINGS  -  edit these
# ==================================================================

OUTPUT_ROOT = r"C:\Users\mraj01\OneDrive - dentsu\Desktop\pack_rep\output"

OBJ_ID = 2

# Name given to the layers the script creates inside After Effects
LAYER_PREFIX = "LabelTrack"

# Frames below this confidence get a marker on the AE timeline so the
# editor can jump straight to them. Match s7_qa.py.
REVIEW_THRESHOLD = 0.60

# Write the alpha matte MOV. Slow on 4K - skip it if you only want
# to regenerate the .jsx after a tweak.
WRITE_MATTE = True

# ==================================================================

MASKS_CLEAN = os.path.join(OUTPUT_ROOT, "masks_clean")
SMOOTH_PATH = os.path.join(OUTPUT_ROOT, "geometry_smoothed.json")
CONF_PATH   = os.path.join(OUTPUT_ROOT, "confidence.json")
META_PATH   = os.path.join(OUTPUT_ROOT, "video_meta.json")
EXPORT_DIR  = os.path.join(OUTPUT_ROOT, "for_after_effects")
MATTE_PATH  = os.path.join(EXPORT_DIR, f"matte_obj{OBJ_ID}.mov")
JSX_PATH    = os.path.join(EXPORT_DIR, "ae_setup.jsx")


def write_matte_mov(n_frames, fps):
    """
    Mask PNG sequence -> ProRes 4444 with alpha.

    RGB is filled white and the mask value goes into the alpha
    channel. AE only reads the alpha for a track matte, but a white
    fill means the clip is also readable on its own if anyone opens
    it to check.
    """
    import imageio.v2 as imageio

    os.makedirs(EXPORT_DIR, exist_ok=True)

    first = cv2.imread(os.path.join(MASKS_CLEAN, f"obj_{OBJ_ID}", "0000.png"),
                       cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise FileNotFoundError(
            f"No masks in {MASKS_CLEAN}/obj_{OBJ_ID}\nRun s3_masks.py first."
        )
    h, w = first.shape[:2]

    writer = imageio.get_writer(
        MATTE_PATH,
        fps=fps,
        codec="prores_ks",
        pixelformat="yuva444p10le",     # the 'a' is the alpha channel
        ffmpeg_params=["-profile:v", "4"],   # profile 4 = ProRes 4444
        macro_block_size=None,
    )

    for i in range(n_frames):
        m = cv2.imread(os.path.join(MASKS_CLEAN, f"obj_{OBJ_ID}", f"{i:04d}.png"),
                       cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((h, w), np.uint8)

        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[:, :, :3] = 255
        rgba[:, :, 3] = m
        writer.append_data(rgba)

        if i % 25 == 0:
            print(f"  matte frame {i}/{n_frames}")

    writer.close()
    print(f"Matte -> {MATTE_PATH}")


def build_jsx(geom, confidences, fps, width, height, n_frames):
    """
    Generate the ExtendScript.

    Corner Pin's property order in AE is Upper Left, Upper Right,
    Lower Left, Lower Right - note LL comes before LR. Our corners
    are stored clockwise as TL, TR, BR, BL, so the mapping is
    0, 1, 3, 2. Getting this wrong twists the label into a bowtie.
    """
    frames = geom["frames"]

    ul, ur, ll, lr = [], [], [], []
    invalid_frames = []
    review_frames = []

    for i, fr in enumerate(frames):
        conf = confidences.get(i, 1.0)

        if fr["valid"] and fr["corners"]:
            c = fr["corners"]
            ul.append([i, c[0][0], c[0][1]])
            ur.append([i, c[1][0], c[1][1]])
            ll.append([i, c[3][0], c[3][1]])   # BL in our ordering
            lr.append([i, c[2][0], c[2][1]])   # BR in our ordering
        else:
            invalid_frames.append(i)

        if fr["valid"] and conf < REVIEW_THRESHOLD:
            review_frames.append([i, round(conf, 2)])

    def js_array(rows):
        return "[" + ",".join(
            "[" + ",".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                           for v in row) + "]"
            for row in rows
        ) + "]"

    matte_name = os.path.basename(MATTE_PATH)

    jsx = f"""// ==================================================================
// ae_setup.jsx  -  generated by the pack replacement pipeline
//
// WHAT THIS DOES
//   1. Finds or creates a composition at the source resolution/fps
//   2. Imports the alpha matte MOV that sits next to this file
//   3. Creates a placeholder solid with a Corner Pin effect, already
//      keyframed to follow the label on every frame
//   4. Sets the matte as an alpha track matte on that layer
//   5. Zeroes opacity on frames where tracking was not valid
//   6. Drops timeline markers on frames worth a look
//
// WHAT YOU DO AFTER RUNNING IT
//   Select the "{LAYER_PREFIX}_{OBJ_ID}" layer, then use
//   Layer > Replace Source or drag your label artwork onto it while
//   holding Alt. Everything else is already set up.
//
// Everything this creates is a normal AE layer. Nudge any keyframe
// you disagree with - nothing is baked.
// ==================================================================

app.beginUndoGroup("Pack Replacement Setup");

(function () {{

    var COMP_W   = {width};
    var COMP_H   = {height};
    var FPS      = {fps:.6f};
    var N_FRAMES = {n_frames};
    var DURATION = N_FRAMES / FPS;

    // ---- composition ---------------------------------------------
    var comp = null;
    if (app.project.activeItem instanceof CompItem) {{
        comp = app.project.activeItem;
    }} else {{
        comp = app.project.items.addComp(
            "PackReplacement", COMP_W, COMP_H, 1.0, DURATION, FPS
        );
        comp.openInViewer();
    }}

    // ---- import the matte ----------------------------------------
    var scriptFile = new File($.fileName);
    var matteFile  = new File(scriptFile.parent.fsName + "/{matte_name}");

    if (!matteFile.exists) {{
        alert("Could not find {matte_name}\\n\\nIt should sit in the same "
            + "folder as this script:\\n" + scriptFile.parent.fsName);
        return;
    }}

    var io = new ImportOptions(matteFile);
    var matteFootage = app.project.importFile(io);
    matteFootage.name = "{LAYER_PREFIX}_{OBJ_ID}_matte";

    var matteLayer = comp.layers.add(matteFootage);
    matteLayer.name = "{LAYER_PREFIX}_{OBJ_ID}_matte";

    // ---- placeholder layer for the new label ---------------------
    var target = comp.layers.addSolid(
        [0.85, 0.15, 0.55], "{LAYER_PREFIX}_{OBJ_ID}",
        COMP_W, COMP_H, 1.0, DURATION
    );

    // Matte must sit directly above the layer it mattes
    matteLayer.moveBefore(target);

    // ---- corner pin ----------------------------------------------
    var cp = target.property("ADBE Effect Parade")
                   .addProperty("ADBE Corner Pin");

    var UL = {js_array(ul)};
    var UR = {js_array(ur)};
    var LL = {js_array(ll)};
    var LR = {js_array(lr)};

    function keyframe(prop, rows) {{
        for (var i = 0; i < rows.length; i++) {{
            var t = rows[i][0] / FPS;
            prop.setValueAtTime(t, [rows[i][1], rows[i][2]]);
        }}
    }}

    keyframe(cp.property("Upper Left"),  UL);
    keyframe(cp.property("Upper Right"), UR);
    keyframe(cp.property("Lower Left"),  LL);
    keyframe(cp.property("Lower Right"), LR);

    // ---- hide the frames where tracking failed -------------------
    var INVALID = {json.dumps(invalid_frames)};
    var opacity = target.property("ADBE Transform Group")
                        .property("ADBE Opacity");

    if (INVALID.length > 0) {{
        opacity.setValueAtTime(0, 100);
        for (var i = 0; i < INVALID.length; i++) {{
            var t = INVALID[i] / FPS;
            opacity.setValueAtTime(t, 0);
            // Hold interpolation, so opacity snaps rather than fading
            var k = opacity.nearestKeyIndex(t);
            opacity.setInterpolationTypeAtKey(
                k, KeyframeInterpolationType.HOLD, KeyframeInterpolationType.HOLD
            );
        }}
    }}

    // ---- track matte ---------------------------------------------
    try {{
        target.trackMatteType = TrackMatteType.ALPHA;
    }} catch (e) {{
        // Newer AE versions changed the API. Fall back, and if that
        // also fails just tell the editor to set it by hand.
        try {{
            target.setTrackMatte(matteLayer, MaskMode.ALPHA);
        }} catch (e2) {{
            alert("Could not set the track matte automatically.\\n\\n"
                + "Set the TrkMat column on the "
                + "\\"{LAYER_PREFIX}_{OBJ_ID}\\" layer to Alpha Matte "
                + "manually - everything else is done.");
        }}
    }}

    // ---- markers on frames worth checking ------------------------
    var REVIEW = {js_array(review_frames)};
    if (REVIEW.length > 0) {{
        for (var i = 0; i < REVIEW.length; i++) {{
            var mk = new MarkerValue("check  conf " + REVIEW[i][1]);
            comp.markerProperty.setValueAtTime(REVIEW[i][0] / FPS, mk);
        }}
    }}

    // ---- summary --------------------------------------------------
    alert("Pack replacement setup complete.\\n\\n"
        + "Keyframed  : " + UL.length + " frames\\n"
        + "Not tracked: " + INVALID.length + " frames (opacity 0)\\n"
        + "Flagged    : " + REVIEW.length + " frames (see timeline markers)\\n\\n"
        + "Now replace the source of the \\"{LAYER_PREFIX}_{OBJ_ID}\\" "
        + "layer with your label artwork.");

}})();

app.endUndoGroup();
"""

    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(JSX_PATH, "w", encoding="utf-8") as f:
        f.write(jsx)

    return len(ul), len(invalid_frames), len(review_frames)


def main():
    if not os.path.exists(SMOOTH_PATH):
        raise FileNotFoundError(f"{SMOOTH_PATH} missing.\nRun s6_smooth.py first.")

    with open(SMOOTH_PATH, "r", encoding="utf-8") as f:
        geom = json.load(f)

    confidences = {}
    if os.path.exists(CONF_PATH):
        with open(CONF_PATH, "r", encoding="utf-8") as f:
            for fr in json.load(f)["frames"]:
                confidences[fr["frame"]] = fr["confidence"]

    fps, width, height = 30.0, geom["frame_width"], geom["frame_height"]
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
            fps = m.get("fps", 30.0)
            width = m.get("width", width)
            height = m.get("height", height)

    n_frames = geom["num_frames"]

    os.makedirs(EXPORT_DIR, exist_ok=True)

    if WRITE_MATTE:
        print(f"Writing ProRes 4444 matte ({width}x{height}, {n_frames} frames)...")
        write_matte_mov(n_frames, fps)
    else:
        print("Skipping matte (WRITE_MATTE is False)")

    n_keys, n_invalid, n_review = build_jsx(
        geom, confidences, fps, width, height, n_frames
    )

    print()
    print("-" * 62)
    print("EXPORT SUMMARY")
    print(f"  keyframed frames  : {n_keys}")
    print(f"  untracked frames  : {n_invalid}   (opacity keyed to 0)")
    print(f"  flagged for review: {n_review}   (timeline markers)")
    print("-" * 62)
    print()
    print(f"Folder to hand over: {EXPORT_DIR}")
    print()
    print("EDITOR INSTRUCTIONS")
    print("  1. Open the project containing the source footage")
    print("  2. File > Scripts > Run Script File...  ->  ae_setup.jsx")
    print("  3. Select the LabelTrack layer and replace its source with")
    print("     the new label artwork (Layer > Replace Source, or drag")
    print("     the artwork onto the layer holding Alt)")
    print("  4. Check the timeline markers, adjust anything that needs it")
    print()
    print("Keep ae_setup.jsx and the .mov in the same folder - the script")
    print("looks for the matte next to itself.")


if __name__ == "__main__":
    main()