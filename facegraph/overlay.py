import cv2

def draw_overlay_official(frame, lm, node_ids, edges_pos, draw_idx=False):
    h, w = frame.shape[:2]
    for a, b in edges_pos:
        ax, ay = int(lm[node_ids[a]].x * w), int(lm[node_ids[a]].y * h)
        bx, by = int(lm[node_ids[b]].x * w), int(lm[node_ids[b]].y * h)
        cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 1)
    for gid in node_ids:
        x, y = int(lm[gid].x * w), int(lm[gid].y * h)
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        if draw_idx:
            cv2.putText(frame, str(gid), (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
