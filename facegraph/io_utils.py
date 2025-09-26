import os, json
import numpy as np
from .features import interocular_scale, nose_center

def save_npz(outpath, nodes, edges, node_ids, frame_mask, fps_eff, frame_indices, features, feature_names, groups):
    if os.path.dirname(outpath): os.makedirs(os.path.dirname(outpath), exist_ok=True)
    np.savez_compressed(
        outpath,
        nodes=nodes.astype(np.float32),
        edges=edges.astype(np.int64),
        node_ids=np.array(node_ids, np.int64),
        frame_mask=frame_mask,
        fps=np.array([fps_eff], np.float32),
        frame_indices=frame_indices,
        features=features.astype(np.float32),
        feature_names=np.array(feature_names, dtype=object),
        groups=np.array(groups, dtype=object)
    )

def save_features_csv(path, frame_indices, frame_mask, features, feature_names):
    with open(path, "w", encoding="utf-8") as f:
        cols = ["frame_idx", "valid"] + (feature_names if len(feature_names)>0 else [])
        f.write(",".join(cols) + "\n")
        for i in range(features.shape[0]):
            row = [str(int(frame_indices[i])), "1" if frame_mask[i] else "0"]
            if features.shape[1] > 0:
                row += [f"{v:.6f}" if v==v else "" for v in features[i]]
            f.write(",".join(row) + "\n")

def save_raw_csv(path, nodes, node_ids, frame_indices, frame_mask, mode="mp"):
    ids = list(node_ids)
    x_cols = [f"{gid}_x" for gid in ids]
    y_cols = [f"{gid}_y" for gid in ids]
    z_cols = [f"{gid}_z" for gid in ids]
    header = ["frame_idx", "valid"] + x_cols + y_cols + z_cols

    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for t in range(nodes.shape[0]):
            valid = frame_mask[t]
            P = nodes[t].copy()  # (N,3)
            if mode == "normalized" and valid:
                s = interocular_scale(P, ids)
                nc = nose_center(P, ids)
                if s > 0 and not np.isnan(nc).any():
                    P[:, :2] -= nc[:2]
                    P[:, :2] /= s
            row = [str(int(frame_indices[t])), "1" if valid else "0"]
            row += [("" if not valid or np.isnan(P[i,0]) else f"{P[i,0]:.6f}") for i in range(P.shape[0])]
            row += [("" if not valid or np.isnan(P[i,1]) else f"{P[i,1]:.6f}") for i in range(P.shape[0])]
            row += [("" if not valid or np.isnan(P[i,2]) else f"{P[i,2]:.6f}") for i in range(P.shape[0])]
            f.write(",".join(row) + "\n")

def save_meta(json_path, meta: dict):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
