import os, cv2, time
import numpy as np
import mediapipe as mp
from .indices import GROUP_DEFS
from .edges import official_edges_for_regions, build_official_edges_union
from .features import compute_features_one_group
from .overlay import draw_overlay_official
from .io_utils import save_npz, save_features_csv, save_raw_csv, save_meta

class MultiGroupExtractor:
    def __init__(self, groups, preview=False, draw_idx=False, save_vis=None,
                 stride=1, refine=False, min_det=0.5, min_track=0.5):
        self.groups = sorted(set([g.upper() for g in groups]))
        self.preview = preview
        self.draw_idx = draw_idx
        self.save_vis = save_vis
        self.stride = max(1, int(stride))
        self.refine = bool(refine)
        self.min_det = float(min_det)
        self.min_track = float(min_track)

    def run(self, video_path, out_npz, feat_csv=None, raw_csv=None, raw_mode="mp"):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

        # union node ids
        node_ids = sorted(set().union(*[GROUP_DEFS[g] for g in self.groups]))
        N = len(node_ids)

        mp_face = mp.solutions.face_mesh #type:ignore
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise RuntimeError(f"Gagal membuka video: {video_path}")

        fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = None
        if self.save_vis:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") #type:ignore
            out_fps = max(1.0, fps_in / self.stride)
            writer = cv2.VideoWriter(self.save_vis, fourcc, out_fps, (W, H))
            if not writer.isOpened(): raise RuntimeError(f"Gagal open VideoWriter: {self.save_vis}")

        face_mesh = mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=self.refine,
            min_detection_confidence=self.min_det, min_tracking_confidence=self.min_track
        )

        # official edges (union)
        edges_by_region = official_edges_for_regions()
        edges = build_official_edges_union(node_ids, self.groups, edges_by_region)

        nodes_list, frame_mask, frame_indices = [], [], []
        feats_list, feat_names = [], None

        t_idx = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok: break
            if (t_idx % self.stride) != 0:
                t_idx += 1; continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            frame_indices.append(t_idx)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts = np.zeros((N,3), np.float32)
                valid = True
                for i,g in enumerate(node_ids):
                    if g >= len(lm): valid = False; break
                    pts[i] = [lm[g].x, lm[g].y, lm[g].z]

                nodes_list.append(pts if valid else np.full((N,3), np.nan, np.float32))
                frame_mask.append(valid)

                if valid:
                    f_all, names_all = [], []
                    for gname in self.groups:
                        fvec, names = compute_features_one_group(gname, pts.copy(), node_ids)
                        f_all.append(fvec); names_all += names
                    f_all = np.concatenate(f_all, axis=0) if len(f_all)>0 else np.zeros((0,), np.float32)
                    feats_list.append(f_all)
                    if feat_names is None: feat_names = names_all

                    if self.preview or writer is not None:
                        overlay = frame.copy()
                        if edges.shape[0] > 0:
                            draw_overlay_official(overlay, lm, node_ids, edges, draw_idx=self.draw_idx)
                        else:
                            for gid in node_ids:
                                x, y = int(lm[gid].x*w), int(lm[gid].y*h)
                                cv2.circle(overlay, (x,y), 2, (0,255,255), -1)
                        cv2.putText(overlay, f"{'+'.join(self.groups)} | N={N} E={edges.shape[0]} | frame {t_idx+1}/{total_frames}",
                                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
                        if writer is not None: writer.write(overlay)
                        if self.preview: cv2.imshow("Preview (official mesh)", overlay)
                else:
                    feats_list.append(np.full((0,), np.nan, np.float32))
                    if self.preview: cv2.imshow("Preview (official mesh)", frame)
                    if writer is not None: writer.write(frame)
            else:
                nodes_list.append(np.full((N,3), np.nan, np.float32))
                frame_mask.append(False)
                feats_list.append(np.full((0,), np.nan, np.float32))
                if self.preview: cv2.imshow("Preview (official mesh)", frame)
                if writer is not None: writer.write(frame)

            if self.preview:
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')): break
            t_idx += 1

        cap.release(); face_mesh.close()
        if writer is not None: writer.release()
        if self.preview: cv2.destroyAllWindows()

        nodes = np.stack(nodes_list, axis=0) if nodes_list else np.zeros((0,N,3), np.float32)
        frame_mask = np.array(frame_mask, bool)
        frame_indices = np.array(frame_indices, np.int64)

        # pad features untuk frame invalid
        if feat_names is None:
            feature_names = []
            features = np.zeros((nodes.shape[0], 0), np.float32)
        else:
            F = len(feat_names)
            features = np.full((nodes.shape[0], F), np.nan, np.float32)
            vi = 0
            for t, v in enumerate(frame_mask):
                if v:
                    features[t] = feats_list[vi]
                    vi += 1
            feature_names = feat_names

        fps_eff = float(max(1.0, (fps_in / self.stride)))

        # save NPZ
        save_npz(out_npz, nodes, edges, node_ids, frame_mask, fps_eff, frame_indices, features, feature_names, self.groups)

        # CSV opsional
        if feat_csv:
            save_features_csv(feat_csv, frame_indices, frame_mask, features, feature_names)
        if raw_csv:
            save_raw_csv(raw_csv, nodes, node_ids, frame_indices, frame_mask, mode="normalized" if raw_mode=="normalized" else "mp")

        meta = {
            "video": os.path.abspath(video_path),
            "out_npz": os.path.abspath(out_npz),
            "groups": self.groups,
            "T": int(nodes.shape[0]),
            "N": int(N),
            "E": int(edges.shape[0]),
            "fps_in": float(fps_in),
            "effective_fps": float(fps_eff),
            "stride": int(self.stride),
            "feature_count": int(features.shape[1]),
            "save_vis": os.path.abspath(self.save_vis) if self.save_vis else None,
            "feat_csv": os.path.abspath(feat_csv) if feat_csv else None,
            "raw_csv": os.path.abspath(raw_csv) if raw_csv else None,
            "raw_mode": raw_mode if raw_csv else None,
            "refine": bool(self.refine),
        }
        save_meta(os.path.splitext(out_npz)[0] + ".json", meta)

        return meta
