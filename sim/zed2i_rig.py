"""R181-S front sensor: Stereolabs ZED 2i (polarizer) stereo camera with IMU, in the R180 body (looks unchanged).

Added on the session layer at the R180 front camera mount, so the robot files stay untouched. Left/right pinhole
cameras with the ZED 2i baseline; the intrinsics are a typical *rectified* HD1080 calibration (the 110 deg x 70 deg in
the datasheet is the raw lens field of view before rectification). Depth comes from the renderer's
distance_to_image_plane annotator on the left camera (the real camera computes it from the stereo pair).

  from sim.zed2i_rig import add_zed2i, ZED2I
  rig = add_zed2i(stage, resolution=(1920, 1080), depth=True)   # after the robot is in the stage
  rig.rgb("left"), rig.rgb("right"), rig.depth()                  # numpy arrays after a render"""
from __future__ import annotations
from dataclasses import dataclass

FRONT_RGB = "/Robot/Physics_Base/Sensors/front_rgb"


@dataclass(frozen=True)
class Zed2iSpec:
    model: str = "Stereolabs ZED 2i (polarizer, 2.1 mm lens)"
    baseline_m: float = 0.12
    width: int = 1920
    height: int = 1080
    fps: int = 30
    fx: float = 1066.0                 # rectified HD1080, typical factory calibration
    fy: float = 1066.0
    depth_min_m: float = 0.3
    depth_max_m: float = 20.0
    imu_hz: int = 400
    gyro_noise: float = 8.7e-5         # rad/s/sqrt(Hz), BMI085-class IMU used in the ZED 2i (datasheet-level figures)
    accel_noise: float = 1.4e-3        # m/s^2/sqrt(Hz)
    gyro_walk: float = 1.0e-6
    accel_walk: float = 1.0e-4


ZED2I = Zed2iSpec()


class Zed2iRig:
    def __init__(self, stage, resolution=None, depth: bool = True, spec: Zed2iSpec = ZED2I, aperture_mm: float = 20.955):
        import numpy as np
        import omni.replicator.core as rep
        from pxr import Gf, Usd, UsdGeom
        self.spec = spec
        w, h = resolution or (spec.width, spec.height)
        fx = spec.fx * w / spec.width                                  # keep the field of view at any resolution
        self.size, self.K = (w, h), [[fx, 0.0, w / 2], [0.0, fx, h / 2], [0.0, 0.0, 1.0]]
        front = stage.GetPrimAtPath(FRONT_RGB)
        if not front:
            raise RuntimeError(f"{FRONT_RGB} not found: load the robot before adding the ZED 2i")
        base = UsdGeom.Xformable(front).GetLocalTransformation(Usd.TimeCode.Default())
        right_axis = base.GetRow3(0)                                   # camera +X (image right) in the mount frame
        self.cams, self.products, self.ann = {}, {}, {}
        for name, sign in (("left", -0.5), ("right", 0.5)):
            path = f"/Robot/Physics_Base/Sensors/zed2i_{name}"
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateFocalLengthAttr(fx * aperture_mm / w)
            cam.CreateHorizontalApertureAttr(aperture_mm); cam.CreateVerticalApertureAttr(aperture_mm * h / w)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
            m = Gf.Matrix4d(base)
            m.SetTranslateOnly(base.ExtractTranslation() + right_axis * (sign * spec.baseline_m))
            UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(m)
            rp = rep.create.render_product(path, (w, h))
            a = rep.AnnotatorRegistry.get_annotator("rgb"); a.attach([rp])
            self.cams[name], self.products[name], self.ann[name] = path, rp, a
        self.depth_ann = None
        if depth:
            self.depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self.depth_ann.attach([self.products["left"]])
        self._np = np

    @property
    def hydra_textures(self):
        return [rp.hydra_texture for rp in self.products.values()]

    def rgb(self, name: str):
        d = self._np.asarray(self.ann[name].get_data())
        return d[..., :3] if d.ndim == 3 else d

    def depth(self):
        """Metres along the optical axis, left camera; inf/0 outside the ZED 2i's 0.3-20 m range set to 0."""
        d = self._np.asarray(self.depth_ann.get_data(), dtype=self._np.float32)
        d[~self._np.isfinite(d) | (d < self.spec.depth_min_m) | (d > self.spec.depth_max_m)] = 0.0
        return d

    def calibration(self) -> dict:
        """cuVSLAM / EuRoC-style calibration: shared K, right camera offset along the left camera's +X."""
        return {"model": self.spec.model, "size": list(self.size), "K": self.K, "baseline_m": self.spec.baseline_m,
                "rectified": True, "distortion": "none (rectified pinhole)", "fps": self.spec.fps,
                "imu_hz": self.spec.imu_hz, "imu_noise": {"gyro": self.spec.gyro_noise, "accel": self.spec.accel_noise,
                                                          "gyro_walk": self.spec.gyro_walk, "accel_walk": self.spec.accel_walk}}


def add_zed2i(stage, resolution=None, depth: bool = True) -> Zed2iRig:
    return Zed2iRig(stage, resolution, depth)
