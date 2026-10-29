#!/usr/bin/env bash
# post_render.sh: after scripts/render_route.py has finished on every GPU (frames merged in ~/runs/render_v2):
#   1. raw per-camera videos   2. cuVSLAM (R181-S stereo / stereo+IMU / RGB-D) vs ground truth
#   3. RT-DETR side-camera detection by category   4. lingbot-map on the front camera
set -uo pipefail
R=/home/ubuntu/runs/render_v2; P=/home/ubuntu/runs/post_v2; REC=/home/ubuntu/runs/farm_full3/rec
NAV=/home/ubuntu/TheFarm/r1_greenhouse_nav
mkdir -p $P
cd $NAV
echo "== videos $(date -u +%H:%M)"
nice -n 5 ~/venv2/bin/python scripts/assemble_route_videos.py --frames $R --out $P/videos --rec $REC > $P/videos.log 2>&1 &
VID=$!

echo "== cuVSLAM $(date -u +%H:%M)"
E=$P/zed_euroc; mkdir -p $E/cam0 $E/cam1 $E/depth0 $E/imu0
ln -sfn $R/zed/cam0 $E/cam0/data; ln -sfn $R/zed/cam1 $E/cam1/data; ln -sfn $R/zed/depth0 $E/depth0/data
cp $R/zed/imu0.csv $E/imu0/data.csv; cp $R/zed/calib.json $E/calib.json
(echo "#timestamp [ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z"; cat $R/zed/groundtruth_part*.csv | sort -n) > $E/groundtruth.csv
SP=$(~/cuvslam_venv/bin/python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr "\n" ":")
for m in stereo inertial rgbd; do
  ~/cuvslam_venv/bin/python scripts/cuvslam_r181.py --data $E --mode $m --out $P/cuvslam_$m.json 2>&1 | grep -v resection | tail -12
done

echo "== detection $(date -u +%H:%M)"
~/venv2/bin/python scripts/detect_side.py --frames $R --rec $REC --model vision_runs/rtdetr_v2_mixed/best --out $P/detection 2>&1 | tail -20

echo "== lingbot-map $(date -u +%H:%M)"
F=$P/front_seq; mkdir -p $F; i=0
for f in $(ls $R/front | sort -n); do cp $R/front/$f $F/$(printf %05d $i).jpg; echo "$i ${f%.jpg}" >> $P/front_seq_index.txt; i=$((i+1)); done
~/lingbot/run.sh $F $P/lingbot/route 0 2>&1 | tail -5

wait $VID; tail -5 $P/videos.log
echo "== done $(date -u +%H:%M)"
