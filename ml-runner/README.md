# ml-runner

A thin Flask front-end that fronts the GPT-2 scripts
(`scripts/generate_sample.py`, `scripts/train_set.py`) running inside a
separate conda env on the host. The scripts use `transformers` + `torch` for
GPU-accelerated generation and training. Returns generated text over HTTP —
no Discord, no Docker, no k8s.

Whatever app wants to use GPT-2 (Discord bot, web UI, etc.) just talks to this
service over HTTP.

## Architecture

```
┌────────────── host ──────────────────────────────────────┐
│                                                          │
│  conda env "gpt2"  (python 3.10 / torch / transformers)  │
│  └── invoked by ml-runner via `conda run -n gpt2 ...`    │
│                                                          │
│  ml-runner  (Flask, systemd, python 3.11)                │
│  ├── app.py / db.py / job_runner.py                      │
│  ├── scripts/   ← the legacy GPT-2 scripts               │
│  └── data/                                                │
│      ├── checkpoint/<set>/   ← trained models (HF format)  │
│      ├── hf-cache/           ← HuggingFace model cache     │
│      ├── datasets/           ← uploaded training data    │
│      ├── logs/               ← per-job subprocess logs   │
│      ├── jobs.db             ← sqlite job tracking       │
│      └── config.json         ← sets (flat array of objects)  │
│                                                          │
└──────────────────────────────────────────────────────────┘
        ▲
        │ HTTP (default :7070)
        │
   any client (discord bot, web UI, ...)
```

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
pull it in automatically, so we install it explicitly. Verify the GPU is visible:

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

On first run, `transformers` will auto-download the GPT-2 117M base model from
HuggingFace (~500 MB, cached under `HF_HOME` — set by the systemd unit to
`data/hf-cache/`). No manual model download needed.

### 2. Install the ml-runner Flask env

The Flask webapp runs in its own conda env (Python 3.11), separate from the
`gpt2` ML env. Conda is already installed from step 1.

First, clone the repo (the service runs from the `ml-runner/` subdir — cloning
to `/opt/discord_gptbot/` avoids the `ml-runner/ml-runner/` nesting that would
happen if you cloned directly to `/opt/ml-runner`):

```bash
sudo git clone <your-repo-url> /opt/discord_gptbot
sudo chown -R $USER:$USER /opt/discord_gptbot
```

Then create the Flask env and install its deps:

```bash
# Create the Flask env
conda create -n ml-runner python=3.11 -y
conda activate ml-runner
pip install -r /opt/discord_gptbot/ml-runner/requirements.txt

# Confirm where the env actually lives (you'll need this path in step 5):
echo $CONDA_PREFIX
```

### 3. Make sure the data dir is writable

```bash
mkdir -p /opt/discord_gptbot/ml-runner/data/{checkpoint,models,datasets,logs}
# config.json is already in data/ from the repo; if not, copy from legacy/.
```

The first time a generation runs against a set whose base model isn't cached,
`transformers` will auto-download the GPT-2 117M base model into
`data/hf-cache/` (configured via `HF_HOME` in the systemd unit). The first
time `/train` runs, same thing.

### 4. Confirm the repo is owned by your user

The service runs as your own user (see step 5), so make sure you own the repo
+ data dir:

```bash
sudo chown -R $USER:$USER /opt/discord_gptbot
```

### 5. Configure and install the systemd service

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
- `User` / `Group` — your username (default `brody`; see step 5)
- `CONDA_BIN` — path to `conda` (default `/home/brody/miniconda3/bin/conda`)
- `CONDA_ENV` — name of the legacy ML env (default `gpt2`)
- `ExecStart` — gunicorn from the Flask conda env
  (default `/home/brody/.conda/envs/ml-runner/bin/gunicorn`)
- `ML_RUNNER_DATA_DIR` — where `checkpoint/`, `hf-cache/`, `jobs.db` live
  (default `/opt/discord_gptbot/ml-runner/data`)
- `ML_RUNNER_PORT` — HTTP port (default `7070`)
- `ML_RUNNER_DEFAULT_STEPS` — default training steps (default `1000`)

## API

| Method | Path                | Purpose |
|--------|---------------------|---------|
| GET    | `/health`           | liveness + config check |
| GET    | `/sets`             | list known sets (from `config.json` + any trained checkpoints on disk) |
| POST   | `/generate`         | generate text from a set |
| POST   | `/train`            | start a training job (always async) |
| GET    | `/jobs`             | list jobs (`?type=generate|train`, `?status=...`, `?limit=N`) |
| GET    | `/jobs/<id>`        | single job status + result/error |
| GET    | `/jobs/<id>/log`    | raw subprocess log for a job |

### Generate (synchronous, default)

```bash
curl -X POST http://localhost:7070/generate \
     -H 'Content-Type: application/json' \
     -d '{"set":"trump-tweet","prefix":""}'
# -> plain text response
```

Query params also work for quick tests:

```bash
curl "http://localhost:7070/generate?set=trump-tweet&prefix="
```

### Generate (async — for long prefixes or when you don't want to block)

```bash
curl -X POST "http://localhost:7070/generate" \
     -H 'Content-Type: application/json' \
     -d '{"set":"trump-tweet","prefix":"","async":true}'
# -> {"job_id": "...", "status": "queued"}

curl http://localhost:7070/jobs/<job_id>
# -> {"status": "complete", "result": "...", ...}
```

### Train

Either upload a dataset file:

```bash
curl -X POST http://localhost:7070/train \
     -F set=trump-tweet \
     -F steps=1000 \
     -F dataset=@/path/to/trump-tweets.txt
# -> {"job_id": "...", "status": "queued"}
```

…or point at a file already on disk:

```bash
curl -X POST http://localhost:7070/train \
     -H 'Content-Type: application/json' \
     -d '{"set":"trump-tweet","dataset_path":"/opt/discord_gptbot/ml-runner/data/datasets/trump-tweets.txt","steps":1000}'
```

Poll until `status` is `complete` or `failed`:

```bash
curl http://localhost:7070/jobs/<job_id>
curl http://localhost:7070/jobs/<job_id>/log   # tail of the training log
```

The trained checkpoint lands at `data/checkpoint/<set>/` and is immediately
available to `/generate`.

## Job tracking

A small sqlite DB at `data/jobs.db` records every generate/train job with
status (`queued` / `running` / `complete` / `failed`), timestamps, result
text or error message, and a pointer to the subprocess log file.

On `ml-runner` restart, any job still marked `running` is marked `failed
(interrupted)` since we can't reattach to the dead subprocess. Long-running
training jobs will need to be re-submitted.

## Layout

```
ml-runner/
├── app.py                 # Flask routes
├── db.py                  # sqlite job tracking
├── job_runner.py          # subprocess management + poller threads
├── requirements.txt       # Flask only (the ML stack lives in the conda env)
├── ml-runner.service      # systemd unit
├── scripts/
    ├── generate_sample.py # torch + transformers
    │   └── train_set.py       # torch + transformers
└── data/
    ├── config.json        # sets: flat array of set objects (seeded from legacy/)
    ├── checkpoint/        # trained models in HuggingFace format
    ├── hf-cache/          # HuggingFace model cache (GPT-2 117M base model)
    ├── datasets/          # uploaded training data
    ├── logs/              # per-job subprocess logs
    └── jobs.db            # sqlite (created on first run)
```

## Development / running without systemd

```bash
cd ml-runner
conda activate ml-runner          # has flask + gunicorn
export ML_RUNNER_DATA_DIR="$(pwd)/data"
export CONDA_BIN="$(which conda)"
export CONDA_ENV=gpt2
python app.py
# or: gunicorn --workers 4 --bind 0.0.0.0:7070 --timeout 600 app:app
```

## Debugging the scripts directly

The Flask layer is just a wrapper — when something goes wrong with generation
or training, it's almost always easier to run the scripts by hand in the
legacy conda env than to reason about the subprocess plumbing. Both scripts
live under `scripts/` and are designed to be runnable standalone.

**Prerequisites:** activate the legacy env and `cd` into `data/` so that
`checkpoint/` and `models/` resolve the same way they do when ml-runner invokes
them (the scripts expect `cwd=data/`):

```bash
conda activate gpt2
cd /opt/discord_gptbot/ml-runner/data   # or wherever ML_RUNNER_DATA_DIR points
```

### generate_sample.py

```
python ../scripts/generate_sample.py <set> [prefix]
```

- `<set>` — the trained set name (must exist under `checkpoint/<set>/`).
- `[prefix]` — optional. If omitted, the prefix is read from **stdin** (this
  is how ml-runner feeds it in, to avoid shell-escaping issues with
  quotes/newlines). If provided as `argv[2]`, it's used directly — handy for
  quick one-liners.

Generated text is written to **stdout**; diagnostics go to **stderr**.

Examples:

```bash
# Prefix as a command-line arg (fine for simple ASCII prefixes):
python ../scripts/generate_sample.py trump-tweet ""
python ../scripts/generate_sample.py news-headline "Breaking:"

# Prefix via stdin (use this for anything with quotes, newlines, etc.):
echo -n 'Question: what is the meaning of life?' | python ../scripts/generate_sample.py wisdom

# No prefix at all (let the model freewheel):
echo -n "" | python ../scripts/generate_sample.py trump-tweet

# Capture stdout to a file for inspection:
python ../scripts/generate_sample.py seinfeld "JERRY" > /tmp/sample.txt
```

### train_set.py

```
python ../scripts/train_set.py <run_name> <file_name> <steps>
```

- `<run_name>` — the set name; checkpoint is saved to `checkpoint/<run_name>/`.
- `<file_name>` — path to the training dataset (text file). Relative paths
  resolve against `cwd` (i.e. `data/`), so put datasets under `data/` or pass
  an absolute path.
- `<steps>` — integer, number of training steps.

On the first run, `transformers` auto-downloads the GPT-2 117M base model
(`"gpt2"`) from HuggingFace (~500 MB, cached under `HF_HOME` or
`~/.cache/huggingface/`). Training logs stream to stdout/stderr; on success
the script emits a single sample from the freshly trained model so you can
confirm it worked.

Examples:

```bash
# Train a new set from a dataset sitting in data/datasets/:
python ../scripts/train_set.py trump-tweet datasets/trump-tweets.txt 1000

# Train with an absolute dataset path:
python ../scripts/train_set.py my-set /data/my-set.txt 500

# Tee the full training log to a file while watching it:
python ../scripts/train_set.py trump-tweet datasets/trump-tweets.txt 1000 2>&1 | tee /tmp/train.log
```

### Common debugging tips

- **"Couldn't find that set" / load errors** → check that
  `data/checkpoint/<set>/` exists and contains the HuggingFace checkpoint
  files (`config.json`, `pytorch_model.bin`, `tokenizer.json`, etc.).
- **First run is slow** → `transformers` is downloading the 117M base model
  from HuggingFace. This is a one-time ~500 MB download, cached under
  `data/hf-cache/` (or `~/.cache/huggingface/` if `HF_HOME` isn't set).
- **GPU not being used** → confirm `torch.cuda.is_available()` returns `True`
  inside the `gpt2` env. If not, check `nvidia-smi` works and that your torch
  install is the CUDA build (not the CPU-only build — on PyPI the default
  wheel is CUDA-enabled on Linux x86_64).
- **Reproducing an ml-runner job's exact invocation** → check the job's log at
  `data/logs/<job_id>.log` (or `GET /jobs/<id>/log`); it captures the
  subprocess's stderr. The exact command ml-runner runs is:
  `conda run -n gpt2 --no-capture-output python /abs/path/to/scripts/generate_sample.py <set>`
  with the prefix sent to stdin and `cwd=data/`.

## Notes

- The prefix is passed to `generate_sample.py` via **stdin**, not argv, to
  avoid shell-escaping pitfalls with quotes/newlines in user input.
- Generation is synchronous by default (typically a few seconds). Training is
  always async.
- `data/config.json` is the source of truth for "known sets" with metadata
  (description, prefix, embed info for any future Discord UI). It's a **flat
  JSON array of set objects**, each with a `name` field plus its metadata:

  ```json
  [
    {
      "name": "trump-tweet",
      "description": "A sample set of tweets made by Donald Trump",
      "prefix": "<|startoftext|>",
    },
    ...
  ]
  ```

  No separate sets-list — the array order *is* the set order, and `name` is
  the canonical identifier. `/sets` also reports any extra trained checkpoints
  discovered on disk that aren't in the config.
- GPT-2 117M is small and dumb on purpose — that's the charm.