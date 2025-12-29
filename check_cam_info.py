#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import h5py

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image, ImageDraw


def to_homo(p3: np.ndarray) -> np.ndarray:
    return np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float64)


def project_point(K: np.ndarray, p_cam: np.ndarray):
    """
    K: (3,3)
    p_cam: (3,) in camera frame
    return (u,v) or None if behind camera
    """
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if z <= 1e-9 or not np.isfinite(z):
        return None
    u = float(K[0, 0]) * (x / z) + float(K[0, 2])
    v = float(K[1, 1]) * (y / z) + float(K[1, 2])
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (u, v)


def draw_point(img_uint8: np.ndarray, uv, label: str = "B", radius: int = 5):
    """
    img_uint8: (H,W,3) uint8  (assumed RGB in numpy)
    """
    H, W = img_uint8.shape[:2]
    out = img_uint8.copy()

    if uv is None:
        return out

    u, v = uv
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        return out

    if HAS_CV2:
        # OpenCV uses BGR; choose green
        bgr = (0, 255, 0)
        # cv2 draws in-place
        cv2.circle(out, (ui, vi), radius, bgr, -1)
        cv2.putText(out, label, (ui + 6, vi - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
        return out
    else:
        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        rgb = (0, 255, 0)
        dr.ellipse((ui - radius, vi - radius, ui + radius, vi + radius), fill=rgb)
        dr.text((ui + 6, vi - 6), label, fill=rgb)
        return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to .hdf5/.h5")
    ap.add_argument("--demo", default="demo_0_1", help="Demo key under /data (e.g., demo_0_1)")
    ap.add_argument("--outdir", default="eef_overlay_B", help="Output directory for overlay images")
    ap.add_argument("--stride", type=int, default=5, help="Save every N frames")
    ap.add_argument("--t0", type=int, default=0, help="Start frame index")
    ap.add_argument("--max_frames", type=int, default=10_000, help="Max frames to process")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img_path = f"/data/{args.demo}/obs/agentview_rgb"
    ee_path  = f"/data/{args.demo}/obs/ee_pos"
    K_path   = f"/data/{args.demo}/intrinsic_matrices"
    T_path   = f"/data/{args.demo}/extrinsic_matrices"

    inside = 0
    valid = 0

    with h5py.File(args.h5, "r") as h5:
        imgs = np.array(h5[img_path][()])   # (T,H,W,3)
        ee   = np.array(h5[ee_path][()])    # (T,3)
        Ks   = np.array(h5[K_path][()])     # (T,3,3)
        Ts   = np.array(h5[T_path][()])     # (T,4,4)

        Tlen = imgs.shape[0]
        H, W = imgs.shape[1], imgs.shape[2]
        Tuse = min(Tlen, args.max_frames)

        print("Shapes:",
              "imgs", imgs.shape,
              "ee", ee.shape,
              "K", Ks.shape,
              "T", Ts.shape)

        for t in range(args.t0, Tuse):
            if (t - args.t0) % args.stride != 0:
                continue

            img = imgs[t]
            K = Ks[t]
            T = Ts[t]
            p_w = ee[t].astype(np.float64)

            # ---- B: assume T is cam->world, so world->cam = inv(T) ----
            p_cam_h = np.linalg.inv(T) @ to_homo(p_w)
            p_cam = p_cam_h[:3]

            uv = project_point(K, p_cam)  # uv in original image coordinate system

            # ---- Visualization flips ----
            img_vis = np.flipud(img)

            if uv is not None:
                valid += 1
                u, v = uv
                if 0 <= u < W and 0 <= v < H:
                    inside += 1

            out = draw_point(img_vis, uv, label="B")

            out_path = os.path.join(args.outdir, f"{args.demo}_t{t:04d}.png")

            if HAS_CV2:
                # out is a numpy array; cv2.imwrite expects BGR but will still save.
                # If colors look odd, try: cv2.imwrite(out_path, out[..., ::-1])
                cv2.imwrite(out_path, out)
            else:
                Image.fromarray(out).save(out_path)

        print(f"\nSummary (stride applied): valid_z={valid}, inside_image={inside}  ({inside}/{max(valid,1)})")
        print("Saved to:", args.outdir)


if __name__ == "__main__":
    main()
