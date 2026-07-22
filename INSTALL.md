# Installation

Setting up the `ml-runner` service on a fresh Ubuntu host. Everything runs
directly on the host (no Docker, no k8s): the Flask webapp as a systemd
service, the GPT-2 scripts inside a separate conda env that the webapp shells
out to.

## One-time host setup

> **About conda env paths.** When you create a conda env without sudo and
> can't write to `<conda_prefix>/envs/`, conda silently falls back to
> `~/.conda/envs/<name>/`. That's where your envs almost certainly live on
> this setup, *not* `/opt/miniconda3/envs/<name>/`. You can confirm with:
>
> ```bash
> conda activate ml-runner && echo $CONDA_PREFIX
> # → /home/<you>/.conda/envs/ml-runner
> ```
>
> The systemd unit's `ExecStart` and `CONDA_BIN` need the real paths — see
> step 5.

### 1. Install the ML conda env (the ML runtime)

The ML scripts use `transformers` + `torch` (not the legacy `gpt_2_simple` +
TF 1.14 — that stack only supported GPUs up to RTX 20xx and has been retired
in favor of torch, which supports modern GPUs natively via its bundled CUDA
runtime).

```bash
conda create -n gpt2 python=3.10 -y
conda activate gpt2
pip install torch transformers accelerate
```

Torch wheels from PyPI bundle their own CUDA 12.x runtime — no `cudatoolkit`,
`cudnn`, or `LD_LIBRARY_PATH` needed. `accelerate` is a runtime dependency of
recent `transformers` versions (the `Trainer` requires it); pip doesn't always
pull it in automatically, so we install it explicitly. Verify the GPU is
visible:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# → CUDA available: True  (on a GPU box)
```

> **⚠️ If `torch.cuda.is_available()` returns `False`** with a warning like
> "The NVIDIA driver on your system is too old (found version 12040)" — the
> default `pip install torch` pulls a wheel compiled against the *latest* CUDA
> (e.g. 12.8), which requires a newer driver than yours. Install a torch build
> matching your driver instead:
>
> ```bash
> # Check your driver's max CUDA version:
> nvidia-smi   # look for "CUDA Version: 12.x" in the header
>
> # Install a torch wheel compiled against an older CUDA your driver supports.
> # cu121 works with driver CUDA ≥ 12.1; cu118 works with driver CUDA ≥ 11.8.
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> # or, for older drivers:
> # pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```
>
> The rule: **driver CUDA version ≥ wheel CUDA version**. Your `nvidia-smi`
> header shows the max CUDA your driver supports.
>
> Also check that no stale `CUDA_VISIBLE_DEVICES=""` env var is hiding the
> GPU (we set this on the old TF 1.14 env for the CPU workaround):
>
> ```bash
> env | grep CUDA_VISIBLE_DEVICES
> conda env config vars list -n gpt2
> # If set, remove it:
> conda env config vars unset CUDA_VISIBLE_DEVICES -n gpt2
> conda deactivate && conda activate gpt2
> ```

On first run, `transformers` will auto-download the GPT-2 117M base model from
HuggingFace (~500 MB, cached under `HF_HOME` — set by the systemd unit to
`data/hf-cache/`). No manual model download needed.

### 2. Clone the repo

The service runs from the `ml-runner/` subdir — cloning to
`/opt/discord_gptbot/` avoids the `ml-runner/ml-runner/` nesting that would
happen if you cloned directly to `/opt/ml-runner`:

```bash
sudo git clone <your-repo-url> /opt/discord_gptbot
sudo chown -R $USER:$USER /opt/discord_gptbot
```

### 3. Install the ml-runner Flask env

The Flask webapp runs in its own conda env (Python 3.11), separate from the
`gpt2` ML env. Conda is already installed from step 1.

```bash
# Create the Flask env
conda create -n ml-runner python=3.11 -y
conda activate ml-runner
pip install -r /opt/discord_gptbot/ml-runner/requirements.txt

# Confirm where the env actually lives (you'll need this path in step 6):
echo $CONDA_PREFIX
```

### 4. Make sure the data dir is writable

```bash
mkdir -p /opt/discord_gptbot/ml-runner/data/{checkpoint,models,datasets,logs,hf-cache}
# config.json is already in data/ from the repo; if not, copy from legacy/.
```

The first time a generation runs against a set whose base model isn't cached,
`transformers` will auto-download the GPT-2 117M base model into
`data/hf-cache/` (configured via `HF_HOME` in the systemd unit). The first
time `/train` runs, same thing.

### 5. Confirm the repo is owned by your user

The service runs as your own user (see step 6), so make sure you own the repo
+ data dir:

```bash
sudo chown -R $USER:$USER /opt/discord_gptbot
```

### 6. Configure and install the systemd service

The committed `ml-runner.service` is a template — **edit it before installing**
to point at the real paths on your box. Three things to verify/change:

1. **`User=` / `Group=`** — set to your own username (the committed default is
   `brody`; check with `id -un`). Running as your own user avoids the
   permission maze of a dedicated service user trying to reach conda envs
   under your home directory.
2. **`CONDA_BIN`** — path to your conda binary. Find it with `which conda`
   after `conda init`, or look for `~/miniconda3/bin/conda`.
3. **`ExecStart`** — path to gunicorn in the `ml-runner` conda env. Find it
   with `conda activate ml-runner && echo $CONDA_PREFIX` → use
   `$CONDA_PREFIX/bin/gunicorn`. On a user-level conda install this is
   typically `/home/<you>/.conda/envs/ml-runner/bin/gunicorn`, *not*
   `/opt/miniconda3/envs/ml-runner/bin/gunicorn`.

```bash
# Edit the unit file in the repo to match your paths:
nano /opt/discord_gptbot/ml-runner/ml-runner.service

# Then install it:
sudo cp /opt/discord_gptbot/ml-runner/ml-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ml-runner
sudo systemctl status ml-runner
# Tail logs:
journalctl -u ml-runner -f
```

Adjust the other paths in `ml-runner.service` if needed. The service expects:
- `User` / `Group` — your username (default `brody`; see step 6)
- `CONDA_BIN` — path to `conda` (default `/home/brody/miniconda3/bin/conda`)
- `CONDA_ENV` — name of the legacy ML env (default `gpt2`)
- `ExecStart` — gunicorn from the Flask conda env
  (default `/home/brody/.conda/envs/ml-runner/bin/gunicorn`)
- `ML_RUNNER_DATA_DIR` — where `checkpoint/`, `hf-cache/`, `jobs.db` live
  (default `/opt/discord_gptbot/ml-runner/data`)
- `ML_RUNNER_PORT` — HTTP port (default `7070`)
- `ML_RUNNER_DEFAULT_STEPS` — default training steps (default `1000`)

## Quick API examples

Once the service is up (verify with `curl localhost:7070/health`):

```bash
# List available sets:
curl localhost:7070/sets | jq

# Generate synchronously (returns plain text):
curl -X POST localhost:7070/generate \
     -H 'Content-Type: application/json' \
     -d '{"set":"trump-tweet","prefix":"Make America"}'

# Train a new set (async — returns a job_id):
curl -X POST localhost:7070/train \
     -F set=trump-tweet \
     -F steps=1000 \
     -F dataset=@/path/to/trump-tweets.txt
# → {"job_id": "...", "status": "queued"}

# Poll a job:
curl localhost:7070/jobs/<job_id>
curl localhost:7070/jobs/<job_id>/log
```

Full API details are in the [root README](README.md).

## Where to go next

- [`README.md`](README.md) — project overview, architecture diagrams, full API table
- [`ml-runner/README.md`](ml-runner/README.md) — the webapp internals (app.py, db.py, job_runner.py, ml-runner.service)
- [`ml-runner/scripts/README.md`](ml-runner/scripts/README.md) — running the GPT-2 scripts by hand for debugging