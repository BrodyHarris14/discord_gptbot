# ml-runner

A thin Flask front-end that fronts the legacy GPT-2 scripts
(`scripts/generate_sample.py`, `scripts/train_set.py`) running inside a
separate conda env on the host. Returns generated text over HTTP — no Discord,
no Docker, no k8s.

Whatever app wants to use GPT-2 (Discord bot, web UI, etc.) just talks to this
service over HTTP.

## Architecture

```
┌────────────── host ──────────────────────────────────────┐
│                                                          │
│  conda env "gpt2"  (python 3.6 / TF 1.14 / CUDA)         │
│  └── invoked by ml-runner via `conda run -n gpt2 ...`    │
│                                                          │
│  ml-runner  (Flask, systemd, python 3.11)                │
│  ├── app.py / db.py / job_runner.py                      │
│  ├── scripts/   ← the legacy GPT-2 scripts               │
│  └── data/                                                │
│      ├── checkpoint/<set>/   ← trained models            │
│      ├── models/117M/        ← base GPT-2 model          │
│      ├── datasets/           ← uploaded training data    │
│      ├── logs/               ← per-job subprocess logs   │
│      ├── jobs.db             ← sqlite job tracking       │
│      └── config.json         ← set list/metadata         │
│                                                          │
└──────────────────────────────────────────────────────────┘
        ▲
        │ HTTP (default :7070)
        │
   any client (discord bot, web UI, ...)
```

## One-time host setup

### 1. Install the legacy conda env (the ML runtime)

```bash
conda create -n gpt2 python=3.6 -y
conda activate gpt2
pip install gpt_2_simple==0.7 tensorflow==1.14 discord.py==1.7.3 pillow faker googletrans
# If you want GPU: install CUDA 10.0 + cuDNN 7.4 matching TF 1.14
```

(Only `gpt_2_simple` + `tensorflow` are actually required by the scripts here;
the others are legacy deps listed for completeness.)

### 2. Install the ml-runner Flask env

```bash
sudo mkdir -p /opt/ml-runner
sudo chown $USER:$USER /opt/ml-runner
# Copy the repo contents (or git clone) into /opt/ml-runner

python3.11 -m venv /opt/ml-runner/.venv
/opt/ml-runner/.venv/bin/pip install -r /opt/ml-runner/requirements.txt
```

### 3. Make sure the data dir is writable

```bash
mkdir -p /opt/ml-runner/data/{checkpoint,models,datasets,logs}
# config.json is already in data/ from the repo; if not, copy from legacy/.
```

The first time a generation runs against a set whose base model isn't present,
`gpt_2_simple` will download the 117M model into `data/models/117M/`. The first
time `/train` runs, same thing.

### 4. Install the systemd service

```bash
sudo cp /opt/ml-runner/ml-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ml-runner
sudo systemctl status ml-runner
# Tail logs:
journalctl -u ml-runner -f
```

Adjust paths in `ml-runner.service` if your conda / venv live elsewhere. The
service expects:
- `CONDA_BIN` — path to `conda` (default `/opt/miniconda3/bin/conda`)
- `CONDA_ENV` — name of the legacy env (default `gpt2`)
- `ML_RUNNER_DATA_DIR` — where `checkpoint/`, `models/`, `jobs.db` live
  (default `/opt/ml-runner/data`)
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
     -d '{"set":"trump-tweet","dataset_path":"/opt/ml-runner/data/datasets/trump-tweets.txt","steps":1000}'
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
│   ├── generate_sample.py # ported from legacy/
│   └── train_set.py       # ported from legacy/
└── data/
    ├── config.json        # set list/metadata (seeded from legacy/)
    ├── checkpoint/        # trained models (created on first train)
    ├── models/            # base GPT-2 117M (downloaded on first use)
    ├── datasets/          # uploaded training data
    ├── logs/              # per-job subprocess logs
    └── jobs.db            # sqlite (created on first run)
```

## Development / running without systemd

```bash
cd ml-runner
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export ML_RUNNER_DATA_DIR="$(pwd)/data"
export CONDA_BIN="$(which conda)"
export CONDA_ENV=gpt2
python app.py
# or: gunicorn --workers 4 --bind 0.0.0.0:7070 --timeout 600 app:app
```

## Notes

- The prefix is passed to `generate_sample.py` via **stdin**, not argv, to
  avoid shell-escaping pitfalls with quotes/newlines in user input.
- Generation is synchronous by default (typically a few seconds). Training is
  always async.
- The legacy `config.json` is the source of truth for "known sets" with
  metadata (description, prefix, embed info for any future Discord UI).
  `/sets` also reports any extra trained checkpoints discovered on disk.
- GPT-2 117M is small and dumb on purpose — that's the charm.