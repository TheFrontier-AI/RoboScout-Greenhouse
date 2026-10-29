#!/usr/bin/env bash
# render8.sh <out_name>: render the recorded run on Astra's permanent scene across the 8 GPUs of one VM.
#   GPU 0-5 presentation cameras (chase, front, CCTV, bystander) at the scene's full profile (64 spp, ISO 100)
#   GPU 6   side cameras at every plant stop (ISO 400 interior profile), then half of the ZED 2i + LiDAR data
#   GPU 7   the other half of the ZED 2i + LiDAR data (16 spp: sensor data, not for display)
set -euo pipefail
NAME=${1:?out name}; O=/home/ubuntu/runs/$NAME; H=/home/passin/Documents/Blender
mkdir -p $O; chmod 777 $O
run() {  # run <container> <gpu> <args...>; each GPU gets its own Kit/shader cache (a shared kvdb cache breaks)
  local C=$1 G=$2; shift 2
  local K=/home/ubuntu/isaac_cache_$G; mkdir -p $K/kit $K/ov $K/glcache $K/computecache; sudo chmod -R 777 $K
  sudo docker rm -f $C >/dev/null 2>&1 || true
  sudo docker run -d --name $C --gpus "device=$G" --network host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 \
    -v /home/ubuntu/TheFarm:$H/TheFarm -v $O:/out -v /home/ubuntu/runs/farm_full3/rec:/rec:ro \
    -v $K/kit:/isaac-sim/kit/cache -v $K/ov:/root/.cache/ov \
    -v $K/glcache:/root/.cache/nvidia/GLCache -v $K/computecache:/root/.nv/ComputeCache \
    -w $H/TheFarm/r1_greenhouse_nav --entrypoint /bin/bash nvcr.io/nvidia/isaac-sim:5.1.0 -c "$*" >/dev/null
  echo "started $C on GPU $G"
}
P="/isaac-sim/python.sh -u scripts/render_route.py --rec /rec --out /out"
# GPU 0 first, alone: it compiles the shaders; its warm cache is then copied to the other GPUs
run rr_view0 0 "$P --streams chase,front,shots --part 0 --parts 6"
until sudo docker logs rr_view0 2>&1 | grep -qE "part 0: 0/|ERROR"; do sleep 10; done
sudo docker logs rr_view0 2>&1 | grep -q ERROR && { echo "rr_view0 failed"; exit 1; }
for g in 1 2 3 4 5 6 7; do sudo rm -rf /home/ubuntu/isaac_cache_$g; sudo cp -a /home/ubuntu/isaac_cache_0 /home/ubuntu/isaac_cache_$g; sudo chmod -R 777 /home/ubuntu/isaac_cache_$g; done
for k in 1 2 3 4 5; do
  run rr_view$k $k "$P --streams chase,front,shots --part $k --parts 6"
  sleep 10
done
run rr_side 6 "$P --streams side --profile interior_inspection && $P --streams zed,lidar --part 0 --parts 2 --spp 16 --zed-every 3"
sleep 20
run rr_zed 7 "$P --streams zed,lidar --part 1 --parts 2 --spp 16 --zed-every 3"
