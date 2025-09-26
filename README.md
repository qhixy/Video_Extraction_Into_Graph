# facegraph

Ekstraksi FaceMesh multi-group (GLOBAL/EYES/LIPS) memakai **mesh resmi MediaPipe**:
- Simpan NPZ: nodes, edges, node_ids, frame_mask, fps, frame_indices, features, feature_names, groups
- Simpan CSV: fitur (`--feat_csv`) & raw landmarks (`--raw_csv`), raw bisa `mp` atau `normalized`
- Preview overlay & simpan video overlay (`--save_vis`)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
