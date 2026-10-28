# Running the mapping job on Nebius (RTX PRO 6000)

Lines marked **UNVERIFIED** are taken from the docs but have not been run by us yet.
Docs used: [Jobs quickstart](https://docs.nebius.com/serverless/quickstart/jobs), [`nebius ai job create` reference](https://docs.nebius.com/cli/reference/ai/job/create), [managing jobs](https://docs.nebius.com/serverless/jobs/manage), [GPU platforms](https://docs.nebius.com/compute/virtual-machines/types), [Container Registry quickstart](https://docs.nebius.com/container-registry/quickstart).

## 0. Facts used

- RTX PRO 6000 platform IDs: `gpu-rtx6000` in **us-central1**, and `gpu-rtx6000-a` in uk-south2 and eu-south1. The 1-GPU preset is `1gpu-24vcpu-218gb`.
- Jobs ("Serverless AI", public preview since Aether 3.5) run one container on a VM that is billed only while it runs.
- Flags: `--volume s3://BUCKET:/path[:ro|rw[:PROFILE]]`, `--env`, `--env-secret`, `--disk-size` (default 250Gi), `--timeout` (1h–168h), `--shm-size` (default 16Gi on GPU).
- **UNVERIFIED**: that Jobs accept `--platform gpu-rtx6000` (the reference lists only H100/L40S as examples). If they don't, use the VM path in section 4.

## 1. Build and push the image (on an x86_64 machine, not the DGX Spark)

```bash
export REGION_ID=us-central1                      # must match the gpu-rtx6000 region
nebius registry configure-helper                  # docker credential helper for cr.<region>.nebius.cloud
export REGISTRY_PATH=$(nebius registry create --name r1-mapping --format json | jq -r ".metadata.id" | cut -d- -f 2)
export IMAGE=cr.$REGION_ID.nebius.cloud/$REGISTRY_PATH/r1-mapping:0.1
docker buildx build --platform linux/amd64 -t $IMAGE cloud/mapping   # ~20 GB with baked weights
docker push $IMAGE
```

A cross-build from the aarch64 Spark (`buildx` + QEMU) should work because nothing is compiled, but it is slow (**UNVERIFIED**).
The weights are baked in at build time, so the job needs no Hugging Face token. Use `--build-arg PREFETCH_WEIGHTS=0` to skip this.

## 2. Object Storage in and out

```bash
nebius storage bucket create --name r1-mapping-io                 # UNVERIFIED subcommand; the web console also works
# S3 access key: create a service account + access key in the console (Object Storage -> access keys).
export AWS_ENDPOINT_URL=https://storage.$REGION_ID.nebius.cloud   # UNVERIFIED host pattern for us-central1
aws s3 cp refs/06_video/drone_flyover_site_2m33s.mp4 s3://r1-mapping-io/in/drone.mp4
```

Store the key pair in SecretStash (MysteryBox) so the volume mount can use it as `PROFILE@SECRET_SELECTOR`. The secret payload format for S3 volumes is **UNVERIFIED**: the reference says only "S3 credentials stored in SecretStash".

## 3. Run as a Serverless Job

```bash
nebius ai job create \
  --name r1-mapping-drone \
  --image $IMAGE \
  --platform gpu-rtx6000 --preset 1gpu-24vcpu-218gb \
  --disk-size 250Gi --timeout 2h --restart-policy never \
  --volume 's3://r1-mapping-io:/data:rw:default@<secret_selector>' \
  --args "--video /data/in/drone.mp4 --out /data/out/drone --fps 2 --max-frames 600 --method primary --up cam-z"
# --container-command is not needed: the image ENTRYPOINT is `python /app/run_job.py`.
# Robot run: --args "--video /data/in/front.mp4 /data/in/side1.mp4 ... --out /data/out/run1 --fisheye"

export JOB_ID=$(nebius ai job get-by-name --name r1-mapping-drone --format jsonpath='{.metadata.id}')
nebius ai job get $JOB_ID          # COMPLETED when done
nebius ai job logs $JOB_ID         # last line: JSON with frames, points, gpu, runtime
aws s3 sync s3://r1-mapping-io/out/drone ./out/drone
```

- If the bucket mount is not available, mount nothing. Pass `--env AWS_ACCESS_KEY_ID=... --env AWS_SECRET_ACCESS_KEY=... --env AWS_ENDPOINT_URL=...`, or better, `--env-secret`, and use `--video s3://r1-mapping-io/in/drone.mp4`, which `run_job.py` downloads with boto3. Outputs then go to local `--out` only; add an upload step or use the VM path.
- If no subnet is set as default, add `--subnet-id`.
- To try the fallback, add `--method fallback`.

## 4. VM fallback (same image)

```bash
# Create a gpu-rtx6000 / 1gpu-24vcpu-218gb VM with an NVIDIA driver + container toolkit image (console is simplest);
# boot disk >= 300 GiB. Then on the VM:
docker pull $IMAGE                                    # after `nebius registry configure-helper` or docker login
aws s3 cp s3://r1-mapping-io/in/drone.mp4 /data/in/   # AWS_ENDPOINT_URL as above
docker run --rm --gpus all --shm-size 16g -v /data:/data $IMAGE \
  --video /data/in/drone.mp4 --out /data/out/drone --fps 2 --max-frames 600
aws s3 sync /data/out/drone s3://r1-mapping-io/out/drone
```

Sanity check on first boot: `docker run --rm --gpus all --entrypoint python $IMAGE -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"` should print `RTX PRO 6000` and include `sm_120`.
