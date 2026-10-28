"""Pick a sim backend: "isaac" (R180 env, starts a headless SimulationApp) or "fake" (kinematic, for tests/dry runs).
Both return an object with reset(seed, friction_scale, obstacle_metre, start_offset) / step / side_images / close,
plus .start_metre, .knobs, .has_side_cameras and .backend."""
from __future__ import annotations
from sim.isaac_env import SceneFrame


def make_sim(backend: str, aisle, house_xy, r180_dir: str | None = None, stage: str = "default",
             side_cameras: bool = True, log=print):
    """Returns (sim, app); app is the SimulationApp (None for fake) and must be closed after sim.close()."""
    if backend == "fake":
        from sim.fake_env import FakeR180Sim
        return FakeR180Sim(aisle_length_m=aisle.length_m, bed_half_m=aisle.clearance_m / 2,
                           start_metre=SceneFrame(aisle, house_xy).start_metre, side_cameras=side_cameras), None
    if backend != "isaac":
        raise ValueError(f"unknown backend {backend!r}")
    from isaacsim import SimulationApp                    # must exist before any omni/pxr/isaacsim.* import
    app = SimulationApp({"headless": True, "width": 320, "height": 240})
    from sim.isaac_env import R180Sim
    return R180Sim(app, aisle, house_xy, r180_dir=r180_dir, stage=stage, side_cameras=side_cameras, log=log), app
