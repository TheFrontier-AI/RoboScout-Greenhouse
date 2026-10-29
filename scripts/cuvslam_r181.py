"""cuVSLAM on R181-S (ZED 2i) data written by scripts/replay_route.py: stereo, stereo + IMU, or RGB-D.
Scores the camera path against groundtruth.csv (Sim3 for mono-scale safety; the stereo/RGB-D path should need scale 1).

  python3 scripts/cuvslam_r181.py --data /runs/replay --mode stereo|inertial|rgbd [--out result.json]
Needs the cuvslam wheel (github.com/nvidia-isaac/cuVSLAM releases) and its CUDA 12 libraries on LD_LIBRARY_PATH."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from PIL import Image


def sim3_ate(P, G):
    mp, mg = P.mean(0), G.mean(0); U, D, Vt = np.linalg.svd((G - mg).T @ (P - mp) / len(P)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt; s = np.trace(np.diag(D) @ S) / (((P - mp) ** 2).sum() / len(P))
    e = np.linalg.norm((s * (R @ (P - mp).T)).T + mg - G, axis=1)
    return float(np.sqrt((e ** 2).mean())), float(np.median(e)), float(e.max()), float(s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--mode", choices=("stereo", "inertial", "rgbd"), default="inertial"); ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    import cuvslam as v
    cal = json.loads((a.data / "calib.json").read_text()); K = cal["K"]; w, h = cal["size"]
    cam = lambda tx: v.Camera(size=[w, h], principal=[K[0][2], K[1][2]], focal=[K[0][0], K[1][1]],
                              rig_from_camera=v.Pose(rotation=[0, 0, 0, 1], translation=[tx, 0, 0]))
    rig = v.Rig(); cfg = v.Tracker.OdometryConfig(); cfg.use_gpu = True
    if a.mode == "rgbd":
        rig.cameras = [cam(0.0)]; cfg.odometry_mode = v.Tracker.OdometryMode.RGBD
        st = v.Tracker.OdometryRGBDSettings(); st.depth_camera_id = 0; st.depth_scale_factor = 1000.0
        cfg.rgbd_settings = st
    else:
        rig.cameras = [cam(0.0), cam(cal["baseline_m"])]; cfg.rectified_stereo_camera = True
        cfg.odometry_mode = v.Tracker.OdometryMode.Inertial if a.mode == "inertial" else v.Tracker.OdometryMode.Multicamera
        if a.mode == "inertial":
            n = cal["imu_noise"]
            rig.imus = [v.ImuCalibration(rig_from_imu=v.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0]),
                                         gyroscope_noise_density=n["gyro"], gyroscope_random_walk=n["gyro_walk"],
                                         accelerometer_noise_density=n["accel"], accelerometer_random_walk=n["accel_walk"],
                                         frequency=float(cal["imu_hz"]))]
    tr = v.Tracker(rig, cfg, v.Tracker.SlamConfig())
    stamps = sorted(int(p.stem) for p in (a.data / "cam0/data").glob("*.png"))
    imu = np.loadtxt(a.data / "imu0/data.csv", delimiter=",", comments="#") if a.mode == "inertial" else None
    gt = {int(r[0]): r[1:4] for r in np.loadtxt(a.data / "groundtruth.csv", delimiter=",", comments="#")}
    gray = lambda p: np.asarray(Image.open(p).convert("L"))
    frames = []
    for ns in stamps:                                   # load first so only tracking is timed
        imgs = [gray(a.data / "cam0/data" / f"{ns}.png")]
        if a.mode != "rgbd":
            imgs.append(gray(a.data / "cam1/data" / f"{ns}.png"))
        d = np.asarray(Image.open(a.data / "depth0/data" / f"{ns}.png")).astype(np.uint16) if a.mode == "rgbd" else None
        frames.append((ns, imgs, d))
    est, j, t0 = [], 0, time.time()
    for ns, imgs, d in frames:
        if imu is not None:
            while j < len(imu) and imu[j, 0] <= ns:
                r = imu[j]; j += 1
                tr.register_imu_measurement(0, v.ImuMeasurement(timestamp_ns=int(r[0]), angular_velocities=list(r[1:4]),
                                                                linear_accelerations=list(r[4:7])))
        pe, _ = tr.track(ns, imgs, depths=[d] if d is not None else None)
        p = pe.world_from_rig
        if p is not None:
            est.append((ns, [float(x) for x in p.pose.translation], [float(x) for x in p.pose.rotation]))   # rotation = quaternion xyzw
    dt = time.time() - t0
    P = np.array([e for _, e, _ in est]); G = np.array([gt[n] for n, _, _ in est])
    rmse, med, mx, s = sim3_ate(P, G)
    L = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    res = {"method": f"cuVSLAM {a.mode}", "frames": len(frames), "tracked": len(est), "track_s": dt, "fps": len(frames) / dt,
           "path_m": L, "ate_rmse_m": rmse, "ate_median_m": med, "ate_max_m": mx, "sim3_scale": s,
           "poses": [[n, *e] for n, e, _ in est], "quat_xyzw": [q for _, _, q in est]}
    print(json.dumps({k: v_ for k, v_ in res.items() if k != "poses"}, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
