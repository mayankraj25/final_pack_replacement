# Pack Replacement Pipeline

Replace a product label/pack on video footage: track the label through the
clip, fit its perspective on every frame, and composite new label artwork on
top — output as a flat MP4. No lighting or shading is added; this is a
tracking + compositing tool, not a full VFX renderer.

The whole pipeline is one file, `pipeline.py`, built as a sequence of
numbered stages you run one at a time (watching a QA preview after each),
or all together once you trust the settings.

## How it works

1. **s1 — extract frames** from the source video into numbered JPEGs.
2. **s2 — mark the label** on frame 0: draw a polygon around it (or a box),
   then click a couple of refinement points.
3. **s2b — mark an occluder** *(optional)* if a hand or object crosses in
   front of the label at some point in the clip.
4. **s4 — track points** on the label across every frame with CoTracker3.
5. **s5 — fit geometry**: turn the tracked points into a 4-corner quad per
   frame (a homography from the reference frame).
6. **s6 — smooth** the corner paths so the tracked shape doesn't jitter.
7. **s9 — composite** your new label artwork onto the footage using the
   smoothed geometry, and write the final MP4.

Every stage that touches the whole clip renders its own QA preview video
into `output/previews/` — watch it before moving to the next stage. There's
no automated correctness check; a bad seed or a bad track will silently
propagate downstream if you don't catch it.

## Requirements

```
pip install -r requirements.txt
```

That covers numpy, OpenCV, scipy, pillow, PyYAML, imageio, torch, and
torchvision. Two more things are needed but not pip-installable from
`requirements.txt` directly:

- **SAM2** (or the [SAMURAI](https://github.com/yangchris11/samurai) fork),
  used only by stage `s2b` for occlusion masking:
  ```
  pip install "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip"
  ```
  Also grab a checkpoint, e.g.:
  ```
  Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" -OutFile "checkpoints\sam2.1_hiera_base_plus.pt"
  ```
- **CoTracker3**, used by stage `s4` — no install needed, it's pulled via
  `torch.hub` on first run (downloads code + weights from GitHub/Hugging
  Face, so the machine needs internet access at least once).

imageio ships its own ffmpeg binary, so no separate ffmpeg install is
required for reading/writing video.

`s2` and `s2b` are interactive OpenCV windows (click to mark the label,
drag a box, etc.) — you need a real display; they won't run headless over
SSH.

## Configuration

There's no config file or env vars. Every setting is a constant near the
top of `pipeline.py`, grouped under a `STAGE N` banner comment, named
`S<N>_...` (e.g. `S6_WINDOW`, `S9_TRANSFORM_MODE`). Open the file and edit
the constant for the stage you're changing.

Before your first run, you need to point four paths at your own files (all
near the top of `pipeline.py`):

| Constant | What it is |
|---|---|
| `OUTPUT_ROOT` | folder where every stage reads/writes its output |
| `S1_VIDEO_PATH` | your source video |
| `S2B_SAM2_CHECKPOINT` | the SAM2 checkpoint file (only needed if you use `s2b`) |
| `S9_NEW_LABEL_PATH` | the new label artwork, ideally a PNG with alpha |

A few settings are shared across stages rather than duplicated: `OBJ_ID`
(which tracked label to use, from `seed_boxes.json`), `MAX_DISPLAY_W/H`
(interactive window size), and `PREVIEW_WIDTH` (QA video width).

## Usage

Run one stage at a time:

```
python pipeline.py s1
python pipeline.py s2
python pipeline.py s2b            # optional, only if something occludes the label
python pipeline.py s4             # add --seed-only to check point seeding before the full track
python pipeline.py s5
python pipeline.py s6
python pipeline.py s9
```

Or run the whole thing in order:

```
python pipeline.py all
python pipeline.py all --include-occluder   # also runs s2b, and tells s9 to use its masks
python pipeline.py all --seed-only          # stop after s4's seeding check
```

`s2` and `s2b` will still pop up their interactive windows when run via
`all` — the run pauses there until you finish marking, same as running them
by hand.

### Occlusion handling

Only needed if a hand, arm, or other object passes in front of the label
at some point in the clip. `s2b` lets you draw one polygon around the
occluder on a single frame and propagates it forward and backward through
the whole clip with SAM2.

- `python pipeline.py all --include-occluder` runs `s2b` **and** turns on
  `S9_USE_OCCLUDER_MASKS` for that same run, so the masks it generates
  actually get used.
- If you run `s2b` on its own (`python pipeline.py s2b`), you must set
  `S9_USE_OCCLUDER_MASKS = True` by hand in `pipeline.py` before running
  `s9` — it does not turn on by itself outside of `all --include-occluder`.

### Multiple labels

`s2` lets you mark more than one label on frame 0 (`obj_id` 1, 2, 3...).
`s4`, `s5`, and `s9` all operate on whichever one `OBJ_ID` points to — to
process a second label, change `OBJ_ID` and re-run `s4` through `s9`.

## Output

Everything lands under `OUTPUT_ROOT`:

```
frames/                  numbered source frames (0000.jpg, 0001.jpg, ...)
video_meta.json          fps / resolution / frame count
seed_boxes.json          label polygon(s)/box(es) + refinement clicks from s2
occluder_masks/          per-frame occluder masks from s2b (optional)
point_tracks.json        CoTracker3 point tracks from s4
geometry.json            per-frame 4-corner quad + homography from s5
geometry_smoothed.json   smoothed version from s6
composited_output.mp4    final output from s9
previews/                QA videos/images rendered by every stage
```

## Notes

- This produces a **flat composite** — no relighting, shadows, or color
  matching. It's meant as a fast preview / rough cut, not a finished shot.
- Point tracking runs at a downscaled resolution (`S4_TRACKING_WIDTH`,
  512px by default) for speed on CPU; raise it if you have a GPU.
- Frames where tracking fails are flagged rather than guessed at — `s9`
  leaves the original footage untouched on those frames instead of
  compositing a bad fit.
