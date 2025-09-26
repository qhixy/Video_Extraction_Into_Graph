Here's a clean, drop-in **README.md** for your CLI (the snippet you pasted lives in `facegraph/cli.py`). It documents install, folder layout, arguments, examples, outputs, and common gotchas.

---

# facegraph — Multi-group FaceMesh extractor (official meshes)

Extract MediaPipe FaceMesh landmarks for selected **groups** (`GLOBAL`, `EYES`, `LIPS`) and export:

* **NPZ**: nodes (T,N,3), official **edges** per region, node ids, frame mask, fps, frame indices, **engineered features** per group, and their names.
* **CSV (features)**: per-frame engineered features (EAR, mouth metrics, face oval, etc).
* **CSV (raw landmarks)**: per-node coordinates (MediaPipe or normalized).
* **Preview/Overlay**: show landmarks+edges while extracting or save an overlay MP4.

All edges use **MediaPipe official region meshes** (face oval, lips, eyes, eyebrows, nose) — no kNN or tessellation — so visuals are neat and anatomically consistent.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # opencv-python, mediapipe, numpy
```

Optional (recommended for stable imports):

```bash
# from project root (that contains facegraph/ and scripts/)
pip install -e .
```

---

## Project layout (recommended)

```
your-project/
  facegraph/
    __init__.py
    indices.py
    edges.py
    features.py
    overlay.py
    io_utils.py
    extractor.py
    cli.py
  scripts/
    extract.py
  requirements.txt
  README.md
```

---

## How to run

### Option A — as a module (robust)

From project root:

```bash
python -m scripts.extract \
  --video "/path/in.mp4" \
  --out "/path/out.npz" \
  --groups GLOBAL,EYES \
  --preview --draw_idx \
  --feat_csv "/path/features.csv" \
  --raw_csv "/path/raw.csv" \
  --raw_mode mp
```

### Option B — plain script with PYTHONPATH

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/extract.py --video "/path/in.mp4" --out "/path/out.npz" --groups GLOBAL,LIPS
```

---

## CLI arguments

| Argument      | Type / Choices                | Default | Description                                                                                                         |
| ------------- | ----------------------------- | ------: | ------------------------------------------------------------------------------------------------------------------- |
| `--video`     | path (required)               |       – | Input video file.                                                                                                   |
| `--out`       | path (required)               |       – | Output NPZ path.                                                                                                    |
| `--groups`    | CSV of `GLOBAL`,`EYES`,`LIPS` |       – | Which groups to extract (union of nodes & edges). Examples: `GLOBAL,EYES`, `GLOBAL,LIPS`, `EYES`.                   |
| `--preview`   | flag                          | `False` | Live preview window with overlay. Press `Q`/`Esc` to stop.                                                          |
| `--draw_idx`  | flag                          | `False` | Draw global landmark indices on preview/overlay.                                                                    |
| `--save_vis`  | path                          |  `None` | Save overlay video (MP4).                                                                                           |
| `--feat_csv`  | path                          |  `None` | Save engineered features per frame to CSV.                                                                          |
| `--raw_csv`   | path                          |  `None` | Save raw per-node coordinates to CSV.                                                                               |
| `--raw_mode`  | `mp` | `normalized`           |    `mp` | `mp`: MediaPipe coords (x,y∈[0,1], z relative). `normalized`: recentered to nose + scaled by inter-ocular distance. |
| `--stride`    | int ≥1                        |     `1` | Sample every Nth frame.                                                                                             |
| `--refine`    | flag                          | `False` | Enable MediaPipe refine landmarks (478 pts). Iris edges are not used, safe to leave off.                            |
| `--min_det`   | float                         |   `0.5` | MediaPipe detection confidence.                                                                                     |
| `--min_track` | float                         |   `0.5` | MediaPipe tracking confidence.                                                                                      |

---

## Groups & included regions

* **GLOBAL** → `FACE_OVAL`, `LEFT_EYEBROW`, `RIGHT_EYEBROW`, `NOSE`
* **EYES** → `LEFT_EYE`, `RIGHT_EYE`, `LEFT_EYEBROW`, `RIGHT_EYEBROW`
* **LIPS** → `LIPS`, `NOSE` (nose is included to aid normalization/geometry)

Nodes are the union of the group indices you pick; **edges** are the union of **official MediaPipe region edges** filtered to that node set.

---

## Outputs

### NPZ (`--out`)

* `nodes`: `(T, N, 3)` — per-frame `(x, y, z)` landmarks in MediaPipe space.
* `edges`: `(E, 2)` — undirected edges (indices 0..N-1) mapped to `node_ids` order.
* `node_ids`: `(N,)` — global Mediapipe indices (0..467) of the nodes stored.
* `frame_mask`: `(T,) bool` — whether a face was detected at each stored frame.
* `fps`: scalar array — effective FPS after stride.
* `frame_indices`: `(T,)` — original frame indices from the video.
* `features`: `(T, F)` — engineered features per frame (NaN for invalid frames).
* `feature_names`: `(F,) object` — names like `GLOBAL:face_w`, `EYES:EAR_mean`, `LIPS:mouth_open_ratio`.
* `groups`: selected group names.

### CSV (features) — `--feat_csv`

```
frame_idx,valid,GLOBAL:face_w,GLOBAL:face_h,...,EYES:EAR_mean,...
```

### CSV (raw landmarks) — `--raw_csv`

```
frame_idx,valid,<id1>_x,...,<idN>_x,<id1>_y,...,<idN>_y,<id1>_z,...,<idN>_z
```

`idK` are global MediaPipe indices (e.g., `33`, `263`, …). If `--raw_mode normalized`, x,y are recentered to nose and scaled by inter-ocular distance (stable across subjects).

---

## Examples

### GLOBAL + EYES with preview and indices

```bash
python -m scripts.extract \
  --video "/path/in.mp4" \
  --out "/path/ge_official.npz" \
  --groups GLOBAL,EYES \
  --preview --draw_idx \
  --feat_csv "/path/ge_features.csv" \
  --raw_csv "/path/ge_raw.csv" \
  --raw_mode mp
```

### GLOBAL + LIPS with overlay MP4, normalized raw CSV

```bash
python -m scripts.extract \
  --video "/path/in.mp4" \
  --out "/path/gl_official.npz" \
  --groups GLOBAL,LIPS \
  --save_vis "/path/gl_overlay.mp4" \
  --raw_csv "/path/gl_raw_norm.csv" \
  --raw_mode normalized
```

### Faster pass (process every 2nd frame)

```bash
python -m scripts.extract \
  --video "/path/in.mp4" \
  --out "/path/stride2.npz" \
  --groups EYES \
  --stride 2
```

---

## Loading the NPZ in Python

```python
import numpy as np
npz = np.load("/path/out.npz", allow_pickle=True)
nodes = npz["nodes"]          # (T,N,3)
edges = npz["edges"]          # (E,2)
node_ids = npz["node_ids"]    # (N,)
features = npz["features"]    # (T,F)
names = npz["feature_names"]  # (F,)
mask = npz["frame_mask"]      # (T,)
```

---

## Tips & troubleshooting

* **`ModuleNotFoundError: facegraph`**
  Run from project root with `python -m scripts.extract` **or** set `PYTHONPATH=$PWD` **or** `pip install -e .`.
* **No face detected / many invalid frames**
  Increase lighting, use higher `--min_det` only if needed, try `--refine`. Consider lowering `--min_track` slightly.
* **Mirrored preview**
  The preview is drawn over the raw frame; if you pre-flip frames elsewhere, indices LEFT/RIGHT remain MediaPipe’s canonical (not mirrored).
* **Iris support**
  Iris landmarks require `--refine`, but iris edges are not drawn/used here by design.
