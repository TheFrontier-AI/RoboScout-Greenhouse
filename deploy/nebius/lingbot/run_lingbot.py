# Headless runner: reuses demo.py load/inference/postprocess, saves predictions to npz.
import sys, os, time, argparse, json
sys.argv_saved = list(sys.argv)
sys.path.insert(0, "/opt/lingbot-map")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch
import demo
p = argparse.ArgumentParser()
p.add_argument("--image_folder", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model_path", required=True)
p.add_argument("--mode", default="streaming")
p.add_argument("--keyframe_interval", type=int, default=None)
p.add_argument("--window_size", type=int, default=64)
p.add_argument("--overlap_size", type=int, default=16)
p.add_argument("--overlap_keyframes", type=int, default=None)
p.add_argument("--num_scale_frames", type=int, default=8)
a = p.parse_args()
args = argparse.Namespace(image_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
    kv_cache_sliding_window=64, num_scale_frames=a.num_scale_frames, use_sdpa=True,
    camera_num_iterations=4, mode=a.mode, model_path=a.model_path)
dev = torch.device("cuda")
t_all = time.time()
images, paths, _ = demo.load_images(image_folder=a.image_folder, image_size=518, patch_size=14)
model = demo.load_model(args, dev)
dtype = torch.bfloat16
model.aggregator = model.aggregator.to(dtype=dtype)
images = images.to(dev)
N = images.shape[0]
kf = a.keyframe_interval
if kf is None:
    kf = (N + 319) // 320 if (a.mode == "streaming" and N > 320) else 1
print("frames", N, "keyframe_interval", kf, flush=True)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    if a.mode == "streaming":
        pred = model.inference_streaming(images, num_scale_frames=a.num_scale_frames,
                                         keyframe_interval=kf, output_device=torch.device("cpu"))
    else:
        pred = model.inference_windowed(images, window_size=a.window_size, overlap_size=a.overlap_size,
                                        overlap_keyframes=a.overlap_keyframes, num_scale_frames=a.num_scale_frames,
                                        keyframe_interval=kf, output_device=torch.device("cpu"))
torch.cuda.synchronize(); t_inf = time.time() - t0
print(f"inference {t_inf:.1f}s peak_alloc {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
imgs_post = pred["images"]
pred, imgs_cpu = demo.postprocess(pred, imgs_post)
out = {}
for k, v in pred.items():
    if isinstance(v, torch.Tensor):
        v = v.float().numpy()
        if k in ("world_points", "depth"): v = v.astype(np.float16)
        out[k] = v
im = imgs_cpu.float().numpy()
if im.ndim == 5: im = im[0]
out["images"] = (np.clip(im.transpose(0, 2, 3, 1), 0, 1) * 255).astype(np.uint8)
out["paths"] = np.array([os.path.basename(x) for x in paths])
out["t_inference"] = np.array(t_inf); out["t_total"] = np.array(time.time() - t_all)
out["keyframe_interval"] = np.array(kf)
np.savez(a.out, **out)
print({k: getattr(v, "shape", None) for k, v in out.items()})
print(f"total {time.time()-t_all:.1f}s")
