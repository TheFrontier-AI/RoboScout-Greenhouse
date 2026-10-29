#!/usr/bin/env python3
"""Second observer: fuse the robot's own estimate (wheels + gyro + LiDAR scan-to-map, recorded in steps.jsonl) with
cuVSLAM stereo poses (scripts/cuvslam_r181.py output) on the same route, offline, and score everything against the
recorded ground truth.  Shows what the robot gains from carrying a stereo camera:

  observer A  robot estimate      xy from steps.jsonl (what actually drove)
  observer B  cuVSLAM stereo      camera poses, placed in the farm frame with the dock pose only (no ground truth)
  fused       cuVSLAM relative motion as the odometry, the robot's scan-to-map corrections kept as recorded
  monitor     |A - B| per step: a disagreement above --disagree-m flags wheel slip / a LiDAR ambiguity

  python3 scripts/fuse_localisation.py --rec <run>/rec --cuvslam <dir>/cuvslam_stereo_q.json [--out fusion/]"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np


def rmse(E):
    return float(np.sqrt((E ** 2).mean())), float(np.median(E)), float(E.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True); ap.add_argument("--cuvslam", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None); ap.add_argument("--disagree-m", type=float, default=0.5)
    ap.add_argument("--anchor-s", type=float, default=90.0, help="window of scan-to-map fixes used to keep the camera->farm alignment current")
    ap.add_argument("--align-s", type=float, default=20.0, help="seconds of robot heading used to find the camera yaw offset")
    a = ap.parse_args()
    S = [json.loads(l) for l in open(a.rec / "steps.jsonl")]
    S = [s for s in S if s.get("xy") and s.get("gt_scene_pose")]
    t = np.array([s["t"] for s in S]); A = np.array([s["xy"] for s in S], float)
    G = np.array([s["gt_scene_pose"][:2] for s in S], float); hdg = np.array([s["heading_deg"] for s in S], float)
    legs = [s.get("leg") for s in S]
    q = json.load(open(a.cuvslam)); P = np.array([[p[0] / 1e9, p[1], p[2], p[3]] for p in q["poses"]], float)
    # The stereo frames come from a re-render of the same route whose clock is not the robot's (holds/revisits were
    # dropped). Synchronise the two recordings with their ground-truth positions (a data-sync step, not localisation):
    # robot step k <-> re-render frame with the nearest truth position within +-win_s of the same route time.
    gt_csv = a.cuvslam.parent / "groundtruth.csv"
    GT = np.loadtxt(gt_csv, delimiter=",", comments="#") if gt_csv.exists() else None
    if GT is not None:
        gt_t = GT[:, 0] / 1e9; gt_xy = GT[:, 1:3]; win = 8.0
        jmap = np.full(len(t), -1)
        for k in range(len(t)):
            lo, hi = np.searchsorted(gt_t, t[k] - win), np.searchsorted(gt_t, t[k] + win)
            if hi <= lo:
                continue
            d = np.linalg.norm(gt_xy[lo:hi] - G[k], axis=1); jj = lo + int(d.argmin())
            if d.min() < 0.15:
                jmap[k] = jj
        keep = jmap >= 0
        S = [s for s, kp in zip(S, keep) if kp]; t, A, G, hdg = t[keep], A[keep], G[keep], hdg[keep]; legs = [l for l, kp in zip(legs, keep) if kp]
        j = jmap[keep]
    else:
        j = np.array([min(np.searchsorted(P[:, 0], ti), len(P) - 1) for ti in t])
    # camera frame -> farm frame: planar motion lives in the camera's (x, z); in-plane rotation/translation from the
    # robot's OWN estimate over the first align-s seconds (dock pose/heading are known by design; truth not used here)
    X2 = P[:, [1, 3]]
    j1 = np.searchsorted(t, t[0] + a.align_s)
    Xa, Aa = X2[j[:j1]], A[:j1]; ma, mb = Xa.mean(0), Aa.mean(0)
    Uk, _, Vk = np.linalg.svd((Aa - mb).T @ (Xa - ma)); Dm = np.diag([1, np.sign(np.linalg.det(Uk @ Vk))]); R = Uk @ Dm @ Vk
    B = (R @ (X2[j] - ma).T).T + mb
    # fused: cuVSLAM gives the relative motion between steps; every scan-to-map fix the robot applied re-anchors it,
    # and the camera->farm alignment is re-estimated over the last --anchor-s seconds of fixes (so no one-off
    # alignment error at the dock is carried along the whole route).
    F = np.zeros_like(A); F[0] = A[0]
    n_fix = 0; Xraw = X2[j]; anchors_x, anchors_f, anchors_t = [Xa.mean(0)], [Aa.mean(0)], [t[0]]
    Rk, ma_k, mb_k = R, ma, mb
    for k in range(1, len(S)):
        F[k] = F[k - 1] + (Rk @ (Xraw[k] - Xraw[k - 1]))
        fx = S[k].get("scan_to_map") or {}
        if fx.get("ok") and fx.get("applied_xy") and fx.get("pose"):
            F[k] = np.array(fx["pose"][:2]); n_fix += 1
            anchors_x.append(Xraw[k]); anchors_f.append(F[k]); anchors_t.append(t[k])
            keep_a = [i for i, ta in enumerate(anchors_t) if ta >= t[k] - a.anchor_s]
            if len(keep_a) >= 20:
                Xw = np.array([anchors_x[i] for i in keep_a]); Fw = np.array([anchors_f[i] for i in keep_a])
                if np.linalg.norm(Xw.max(0) - Xw.min(0)) > 1.0:          # enough motion to fix the rotation
                    ma_k, mb_k = Xw.mean(0), Fw.mean(0)
                    Uw, _, Vw = np.linalg.svd((Fw - mb_k).T @ (Xw - ma_k)); Dw = np.diag([1, np.sign(np.linalg.det(Uw @ Vw))]); Rk = Uw @ Dw @ Vw
    eA = np.linalg.norm(A - G, axis=1); eB = np.linalg.norm(B - G, axis=1); eF = np.linalg.norm(F - G, axis=1)
    dis = np.linalg.norm(A - B, axis=1); flag = dis > a.disagree_m
    # slip events recorded by the robot: does the disagreement monitor see them?
    ev = [json.loads(l) for l in open(a.rec / "events.jsonl")] if (a.rec / "events.jsonl").exists() else []
    slips = [e for e in ev if (e.get("type") or e.get("event") or e.get("kind")) == "slip"]
    seen = []
    for e in slips:
        te = float(e.get("t") or e.get("ts") or 0); k = int(np.searchsorted(t, te))
        w = slice(max(0, k - 50), min(len(t), k + 100)); seen.append(bool(flag[w].any()))
    inside = np.array([l in ("drive_in", "drive_out") for l in legs]); outside = ~inside
    res = {"steps": len(S), "route_s": round(float(t[-1] - t[0]), 1),
           "robot_estimate_m": dict(zip(("rmse", "median", "max"), rmse(eA))),
           "cuvslam_dock_anchored_m": dict(zip(("rmse", "median", "max"), rmse(eB))),
           "fused_m": dict(zip(("rmse", "median", "max"), rmse(eF))),
           "inside_aisles_rmse_m": {"robot": round(rmse(eA[inside])[0], 3), "cuvslam": round(rmse(eB[inside])[0], 3), "fused": round(rmse(eF[inside])[0], 3)},
           "outside_rmse_m": {"robot": round(rmse(eA[outside])[0], 3), "cuvslam": round(rmse(eB[outside])[0], 3), "fused": round(rmse(eF[outside])[0], 3)},
           "scan_to_map_fixes_applied": n_fix, "disagreement_m": {"median": round(float(np.median(dis)), 3), "p95": round(float(np.quantile(dis, 0.95)), 3), "max": round(float(dis.max()), 3)},
           "steps_flagged": int(flag.sum()), "slip_events": len(slips), "slip_events_seen_by_monitor": int(sum(seen)),
           "align_window_s": a.align_s}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "fusion.json").write_text(json.dumps(res, indent=1))
        np.save(a.out / "paths.npy", np.c_[t, A, B, F, G, dis])
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(15, 9), dpi=120); fig.patch.set_facecolor("#0e1116")
            m = np.load(a.rec / "prior_map_xy.npy") if (a.rec / "prior_map_xy.npy").exists() else None
            for a_, (lab, X, e) in zip(ax, [("robot: wheels + gyro + LiDAR", A, eA), ("fused: + cuVSLAM stereo", F, eF)]):
                a_.set_facecolor("#0e1116")
                if m is not None:
                    mm = m[(m[:, 0] > -8) & (m[:, 0] < 18) & (m[:, 1] > -42) & (m[:, 1] < 12)]; a_.scatter(mm[:, 0], mm[:, 1], s=0.3, c="#4a545e")
                a_.plot(G[:, 0], G[:, 1], c="#76b900", lw=2.2, label="truth (recorded for evaluation)")
                a_.plot(X[:, 0], X[:, 1], c="#4cc9f0", lw=1.0, label=lab)
                a_.set_aspect("equal"); a_.tick_params(colors="#cfd6dd"); [sp.set_color("#3a424c") for sp in a_.spines.values()]
                a_.set_title(f"{lab}\nRMSE {rmse(e)[0]:.3f} m · median {rmse(e)[1]:.3f} · max {rmse(e)[2]:.2f} m", color="white", fontsize=11)
                a_.legend(loc="lower left", fontsize=8, facecolor="#161b22", labelcolor="white", edgecolor="#3a424c")
            ax2 = fig.add_axes([0.62, 0.74, 0.3, 0.17]); ax2.set_facecolor("#0e1116")
            ax2.plot(t / 60, dis, c="#ff9f1c", lw=0.7); ax2.axhline(a.disagree_m, c="#ff2fd6", lw=0.8, ls="--")
            for e in slips:
                ax2.axvline(float(e.get("t") or e.get("ts") or 0) / 60, c="#f72585", lw=1.0)
            ax2.set_title("observer disagreement |robot - cuVSLAM| (m) · pink = recorded slip", color="white", fontsize=8)
            ax2.tick_params(colors="#cfd6dd", labelsize=7); [sp.set_color("#3a424c") for sp in ax2.spines.values()]
            plt.tight_layout(); plt.savefig(a.out / "fusion.png", facecolor=fig.get_facecolor())
        except Exception as e:                                    # matplotlib is optional
            print("no figure:", e)


if __name__ == "__main__":
    main()
