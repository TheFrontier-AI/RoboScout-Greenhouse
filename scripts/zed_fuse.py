"""Dense 3-D map from the R181-S ZED 2i: unproject each depth frame with the pose cuVSLAM estimated for it (no ground
truth), colour from the left camera, voxel-downsample. Output: a PLY in the scene frame (aligned once with the Sim3 that
fits cuVSLAM's path to the recorded path, so the map can be drawn over the farm plan) and a per-point frame index for
"map grows as the robot drives" videos.

  python3 scripts/zed_fuse.py --zed <render_out>/zed --cuvslam cuvslam_stereo_q.json --gt groundtruth.csv --calib calib.json --out cuvslam_map"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image


def quat_to_R(q):                                   # xyzw
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def sim3(P, G):
    mp, mg = P.mean(0), G.mean(0); U, D, Vt = np.linalg.svd((G - mg).T @ (P - mp) / len(P)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt; s = np.trace(np.diag(D) @ S) / (((P - mp) ** 2).sum() / len(P))
    return s, R, mp, mg


def write_ply(path: Path, P, C):
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        buf = np.zeros(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); buf["p"] = P; buf["c"] = C; f.write(buf.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zed", type=Path, required=True); ap.add_argument("--cuvslam", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True); ap.add_argument("--calib", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=4, help="use every Nth frame"); ap.add_argument("--stride", type=int, default=8, help="pixel stride")
    ap.add_argument("--max-depth", type=float, default=8.0); ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--flip", default="", help="axis convention fix, e.g. 'y,z' negates camera y and z before posing")
    a = ap.parse_args()
    cal = json.loads(a.calib.read_text()); K = cal["K"]; fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    cu = json.load(open(a.cuvslam)); poses = {int(p[0]): (np.array(p[1:4]), quat_to_R(q)) for p, q in zip(cu["poses"], cu["quat_xyzw"])}
    gt = {int(r[0]): r[1:4] for r in np.loadtxt(a.gt, delimiter=",", comments="#")}
    ns_all = sorted(n for n in poses if n in gt)
    P = np.array([poses[n][0] for n in ns_all]); G = np.array([gt[n] for n in ns_all]); s, R, mp, mg = sim3(P, G)
    print(f"Sim3 scale {s:.4f}; {len(ns_all)} poses")
    flip = np.ones(3); [flip.__setitem__("xyz".index(c), -1.0) for c in a.flip.split(",") if c]
    pts, cols, fidx = [], [], []
    used = ns_all[::a.every]
    for i, n in enumerate(used):
        dp = a.zed / "depth0" / f"{n}.png"; cp = a.zed / "cam0" / f"{n}.png"
        if not dp.exists() or not cp.exists():
            continue
        d = np.asarray(Image.open(dp), dtype=np.float32)[::a.stride, ::a.stride] / 1000.0
        c = np.asarray(Image.open(cp).convert("RGB"))[::a.stride, ::a.stride]
        h, w = d.shape; v, u = np.mgrid[0:h, 0:w]; u = u * a.stride; v = v * a.stride
        ok = (d > 0.3) & (d < a.max_depth)
        X = np.stack([(u[ok] - cx) / fx * d[ok], (v[ok] - cy) / fy * d[ok], d[ok]], 1) * flip
        t, Rc = poses[n]; Xw = X @ Rc.T + t
        pts.append(Xw.astype(np.float32)); cols.append(c[ok]); fidx.append(np.full(ok.sum(), i, np.int32))
    Pw = np.concatenate(pts); C = np.concatenate(cols); F = np.concatenate(fidx)
    A = ((s * (R @ (Pw - mp).T)).T + mg).astype(np.float32)                   # scene frame
    key = np.round(A / a.voxel).astype(np.int32); _, keep = np.unique(key, axis=0, return_index=True); keep.sort()
    A, C, F = A[keep], C[keep], F[keep]
    path = (s * (R @ (np.array([poses[n][0] for n in used]) - mp).T)).T + mg
    write_ply(a.out.with_suffix(".ply"), A, C); np.save(a.out.with_suffix(".reveal.npy"), F); np.save(a.out.with_suffix(".path.npy"), path)
    print(f"{len(Pw):,} raw -> {len(A):,} points after {a.voxel} m voxel; {len(used)} frames; z range {np.percentile(A[:,2],[1,99]).round(2)}")


if __name__ == "__main__":
    main()
