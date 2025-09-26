import numpy as np
from .indices import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_EYEBROW_IDX, RIGHT_EYEBROW_IDX,
    FACE_OVAL_IDX, GROUP_DEFS
)

def _dist2d(a, b): return float(np.linalg.norm(a - b))

def interocular_scale(pts3, node_ids):
    id2 = {g: i for i, g in enumerate(node_ids)}
    if 33 in id2 and 263 in id2:
        return max(1e-6, _dist2d(pts3[id2[33], :2], pts3[id2[263], :2]))
    if 61 in id2 and 291 in id2:
        return max(1e-6, _dist2d(pts3[id2[61], :2], pts3[id2[291], :2]))
    xs = pts3[:, 0]
    return max(1e-6, float(xs.max() - xs.min()))

def nose_center(pts3, node_ids):
    id2 = {g: i for i, g in enumerate(node_ids)}
    for nid in (1, 4, 2):
        if nid in id2:
            return pts3[id2[nid]]
    return np.nan * np.ones(3, np.float32)

def mouth_metrics(pts3, node_ids):
    id2 = {g: i for i, g in enumerate(node_ids)}
    pairs = [(13, 14), (82, 312), (81, 311), (80, 310), (191, 95)]
    vals = []
    for a, b in pairs:
        if a in id2 and b in id2:
            vals.append(_dist2d(pts3[id2[a], :2], pts3[id2[b], :2]))
        else:
            vals.append(np.nan)
    width = np.nan
    if 61 in id2 and 291 in id2:
        width = _dist2d(pts3[id2[61], :2], pts3[id2[291], :2])
    return vals, width

def eye_EAR(pts3, node_ids, left=True):
    id2 = {g: i for i, g in enumerate(node_ids)}
    if left:
        req = [263, 362, 386, 374, 387, 373, 385, 380]
        if not all(r in id2 for r in req): return np.nan
        A = _dist2d(pts3[id2[386], :2], pts3[id2[374], :2])
        B = _dist2d(pts3[id2[387], :2], pts3[id2[373], :2])
        C = _dist2d(pts3[id2[385], :2], pts3[id2[380], :2])
        D = _dist2d(pts3[id2[263], :2], pts3[id2[362], :2])
    else:
        req = [33, 133, 159, 145, 160, 144, 158, 153]
        if not all(r in id2 for r in req): return np.nan
        A = _dist2d(pts3[id2[159], :2], pts3[id2[145], :2])
        B = _dist2d(pts3[id2[160], :2], pts3[id2[144], :2])
        C = _dist2d(pts3[id2[158], :2], pts3[id2[153], :2])
        D = _dist2d(pts3[id2[33], :2], pts3[id2[133], :2])
    if D < 1e-6: return np.nan
    return (A + B + C) / (3.0 * D)

def brow_eye_distance(pts3, node_ids, left=True):
    eye = LEFT_EYE_IDX if left else RIGHT_EYE_IDX
    brow = LEFT_EYEBROW_IDX if left else RIGHT_EYEBROW_IDX
    id2 = {g: i for i, g in enumerate(node_ids)}
    E = [id2[e] for e in eye if e in id2]
    B = [id2[b] for b in brow if b in id2]
    if not E or not B: return np.nan
    mins = []
    for bi in B:
        mins.append(min(_dist2d(pts3[bi, :2], pts3[ei, :2]) for ei in E))
    return float(np.mean(mins))

def face_oval_metrics(pts3, node_ids):
    idx = [node_ids.index(g) for g in FACE_OVAL_IDX if g in node_ids]
    xs, ys = (pts3[:, 0], pts3[:, 1]) if len(idx) < 4 else (pts3[idx, 0], pts3[idx, 1])
    w = float(xs.max() - xs.min()); h = float(ys.max() - ys.min())
    return w, h, (w / h if h > 1e-6 else np.nan)

def compute_features_one_group(group, pts3_union, node_ids_union):
    # Normalisasi di union (pusat hidung + skala interocular)
    s = interocular_scale(pts3_union, node_ids_union)
    nc = nose_center(pts3_union, node_ids_union)
    P = pts3_union.copy()
    if not np.isnan(nc).any(): P[:, :2] -= nc[:2]
    if s > 0: P[:, :2] /= s

    subset = GROUP_DEFS[group]
    id2u = {g: i for i, g in enumerate(node_ids_union)}
    sub_pos = [id2u[g] for g in subset if g in id2u]
    if not sub_pos:
        return np.zeros((0,), np.float32), []

    Psub = P[sub_pos]
    sub_ids = [node_ids_union[i] for i in sub_pos]
    feats, names = [], []

    if group == "LIPS":
        vals, width = mouth_metrics(Psub, sub_ids)
        for k, v in enumerate(vals): feats.append(v); names.append(f"{group}:mouth_open_{k}")
        feats.append(width); names.append(f"{group}:mouth_width")
        feats.append((vals[0] / (width + 1e-6)) if (width == width and vals[0] == vals[0]) else np.nan); names.append(f"{group}:mouth_open_ratio")
        id2 = {g: i for i, g in enumerate(sub_ids)}
        feats.append(Psub[id2[1], 2] if 1 in id2 else np.nan); names.append(f"{group}:nose_z")

    elif group == "EYES":
        ear_l = eye_EAR(Psub, sub_ids, left=True)
        ear_r = eye_EAR(Psub, sub_ids, left=False)
        feats += [ear_l, ear_r, np.nanmean([ear_l, ear_r]), (ear_l - ear_r) if (ear_l == ear_l and ear_r == ear_r) else np.nan]
        names += [f"{group}:EAR_left", f"{group}:EAR_right", f"{group}:EAR_mean", f"{group}:EAR_asym"]
        be_l = brow_eye_distance(Psub, sub_ids, left=True)
        be_r = brow_eye_distance(Psub, sub_ids, left=False)
        feats += [be_l / s if be_l == be_l else np.nan, be_r / s if be_r == be_r else np.nan]
        names += [f"{group}:broweye_left", f"{group}:broweye_right"]

    elif group == "GLOBAL":
        w, h, ratio = face_oval_metrics(Psub, sub_ids)
        feats += [w, h, ratio]; names += [f"{group}:face_w", f"{group}:face_h", f"{group}:face_ratio"]
        id2 = {g: i for i, g in enumerate(sub_ids)}
        feats.append(Psub[id2[1], 2] if 1 in id2 else np.nan); names.append(f"{group}:nose_z")
        vals, width = mouth_metrics(Psub, sub_ids)
        feats.append(vals[0] if vals else np.nan); names.append(f"{group}:mouth_open_center")
        ear_l = eye_EAR(Psub, sub_ids, left=True); ear_r = eye_EAR(Psub, sub_ids, left=False)
        feats += [ear_l, ear_r]; names += [f"{group}:EAR_left", f"{group}:EAR_right"]

    return np.array(feats, np.float32), names
