import argparse
from .extractor import MultiGroupExtractor

def build_parser():
    ap = argparse.ArgumentParser(description="Extract MULTI groups (GLOBAL/EYES/LIPS) -> graph + features (official MediaPipe meshes).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", required=True, help="Comma-separated, contoh: GLOBAL,EYES atau GLOBAL,LIPS")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--draw_idx", action="store_true")
    ap.add_argument("--save_vis", default=None)
    ap.add_argument("--feat_csv", default=None)
    ap.add_argument("--raw_csv", default=None)
    ap.add_argument("--raw_mode", choices=["mp","normalized"], default="mp")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--min_track", type=float, default=0.5)
    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    ex = MultiGroupExtractor(
        groups=groups,
        preview=args.preview,
        draw_idx=args.draw_idx,
        save_vis=args.save_vis,
        stride=args.stride,
        refine=args.refine,
        min_det=args.min_det,
        min_track=args.min_track,
    )
    meta = ex.run(
        video_path=args.video,
        out_npz=args.out,
        feat_csv=args.feat_csv,
        raw_csv=args.raw_csv,
        raw_mode=args.raw_mode,
    )
    print("[OK] Done.")
    for k,v in meta.items():
        print(f"{k}: {v}")
