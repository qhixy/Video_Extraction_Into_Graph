import numpy as np
import mediapipe as mp
from .indices import FACE_OVAL_IDX

def official_edges_for_regions():
    mp_face = mp.solutions.face_mesh #type: ignore
    return {
        "FACE_OVAL":        list(mp_face.FACEMESH_FACE_OVAL),
        "LEFT_EYE":         list(mp_face.FACEMESH_LEFT_EYE),
        "RIGHT_EYE":        list(mp_face.FACEMESH_RIGHT_EYE),
        "LEFT_EYEBROW":     list(mp_face.FACEMESH_LEFT_EYEBROW),
        "RIGHT_EYEBROW":    list(mp_face.FACEMESH_RIGHT_EYEBROW),
        "NOSE":             list(mp_face.FACEMESH_NOSE),
        "LIPS":             list(mp_face.FACEMESH_LIPS),
    }

def regions_needed_for_group(group_name: str):
    if group_name == "GLOBAL":
        return {"FACE_OVAL", "LEFT_EYEBROW", "RIGHT_EYEBROW", "NOSE"}
    if group_name == "LIPS":
        return {"LIPS", "NOSE"}
    if group_name == "EYES":
        return {"LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW", "RIGHT_EYEBROW"}
    raise ValueError("Unknown group")

def build_official_edges_union(node_ids_sorted, groups, edges_by_region):
    node_set = set(node_ids_sorted)
    id2pos = {gid: i for i, gid in enumerate(node_ids_sorted)}
    needed = set()
    for g in groups:
        needed |= regions_needed_for_group(g)
    E = []
    for region in needed:
        for a, b in edges_by_region[region]:
            if a in node_set and b in node_set:
                E.append((id2pos[a], id2pos[b]))
    if not E:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.array(E, dtype=np.int64), axis=0)
