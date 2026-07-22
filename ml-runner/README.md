# ml-runner

A thin Flask front-end that fronts the GPT-2 scripts
(`scripts/generate_sample.py`, `scripts/train_set.py`) running inside a
separate conda env on the host. The scripts use `transformers` + `torch` for
GPU-accelerated generation and training. Returns generated text over HTTP —
no Discord, no Docker, no k8s.

Whatever app wants to use GPT-2 (Discord bot, web UI, etc.) just talks to this
service over HTTP.

For setup instructions, see [`../INSTALL.md`](../INSTALL.md). For running the
GPT-2 scripts by hand, see [`scripts/README.md`](scripts/README.md). This doc
covers the webapp internals: how the pieces fit together and how to develop
against them.

## Architecture

```
┌────────────── host ──────────────────────────────────────┐
│                                                          │
│  conda env "gpt2"  (python 3.10 / torch / transformers)  │
│  └── invoked by ml-runner via `conda run -n gpt2 ...`    │
│                                                          │
│  ml-runner  (Flask, systemd, python 3.11)                │
│  ├── app.py / db.py / job_runner.py                      │
│  ├── scripts/   ← the GPT-2 scripts                      │
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

## The pieces

### `app.py` — the Flask routes

The HTTP surface. A small set of routes that all delegate to `job_runner.py`
for the actual ML work:

- **`GET /health`** — liveness + config check. Returns the conda env name,
  conda binary path, data dir, whether `config.json` and the checkpoint dir
  exist. Useful for smoke-testing after install.
- **`GET /sets`** — lists known sets. Merges two sources: the metadata in
  `data/config.json` (with descriptions + prefixes) and any extra trained
  checkpoints discovered under `data/checkpoint/`. Sets in config but not
  on disk show `trained: false`; sets on disk but not in config show up with
  just `{"name": "...", "trained": true}`.
- **`POST /generate`** (also `GET`) — the main endpoint. Takes `set` (the
  set name) and `prefix` (optional text to seed the generation). Synchronous
  by default: blocks while the GPT-2 subprocess runs and returns the
  generated text as `text/plain`. Pass `async=true` to get a `job_id` back
  immediately and poll `/jobs/<id>` instead.
- **`POST /train`** — always async. Takes `set`, `steps` (default 1000),
  and either a multipart `dataset` file upload or a `dataset_path` pointing
  at a file already on disk. Returns a `job_id`.
- **`GET /jobs`** — list jobs, filterable by `type=generate|train` and
  `status=queued|running|complete|failed`.
- **`GET /jobs/<id>`** — single job status + result (or error) + log path.
- **`GET /jobs/<id>/log`** — the raw subprocess log for a job.

The prefix is passed to `generate_sample.py` via **stdin**, not argv, to
avoid shell-escaping pitfalls with quotes/newlines in user input.

`config.json` is read from disk on every `/sets` request (not cached in
memory), and `/generate` doesn't consult it at all — it just passes the set
name straight to the script, which loads `checkpoint/<set>/`.

### `job_runner.py` — subprocess management + poller threads

Where the actual ML work happens. For each `/generate` or `/train` request,
`job_runner.py` spawns a subprocess:

```
conda run -n gpt2 --no-capture-output python /abs/path/to/scripts/<script>.py <args>
```

…with `cwd=data/` so checkpoints and the HF cache resolve correctly. The
script path is computed as an absolute path (relative to `job_runner.py`'s
own location), so it works regardless of cwd.

Two modes:
- **Synchronous generate** (`run_generate_sync`) — blocks the request thread
  while the subprocess runs, captures stdout as the result. Used for the
  default `/generate` path (typically a few seconds on GPU).
- **Async generate / train** (`start_generate` / `start_train`) — launches
  the subprocess and returns immediately. A daemon poller thread waits for
  the process to exit, then updates the sqlite row with the result or error.

The poller threads keep an in-memory map of `job_id -> subprocess.Popen`
handles so they can wait on the right process. On `ml-runner` restart that
map is lost; any row still `running` gets marked `failed (interrupted)` by
`db.mark_interrupted_running()` at startup.

### `db.py` — sqlite job tracking

A small sqlite DB at `data/jobs.db` records every generate/train job. Schema:

```
jobs(
    id          TEXT PRIMARY KEY,    -- uuid
    type        TEXT,                -- 'generate' | 'train'
    set_name    TEXT,
    status      TEXT,                -- 'queued'|'running'|'complete'|'failed'
    created_at  TEXT,                 -- ISO 8601
    started_at  TEXT,
    finished_at TEXT,
    result      TEXT,                 -- generated text (generate) or sample (train)
    error       TEXT,                 -- error message on failure
    log_path    TEXT,                 -- path to the captured subprocess log
    params_json TEXT                  -- JSON blob of the original request params
)
```

A module-level lock serializes access so concurrent Flask threads don't
fight over the connection. The DB is the durable record; the in-memory
process map in `job_runner.py` is the live state.

### `ml-runner.service` — systemd unit

Runs the Flask app under gunicorn as a systemd service. Key env vars it
sets (edit these to match your box — see [`../INSTALL.md`](../INSTALL.md)):

- `CONDA_BIN` — path to the conda binary
- `CONDA_ENV` — name of the ML conda env (`gpt2`)
- `ML_RUNNER_DATA_DIR` — where `checkpoint/`, `hf-cache/`, `jobs.db` live
- `ML_RUNNER_PORT` — HTTP port (default 7070)
- `HF_HOME` — where HuggingFace caches the base GPT-2 model
- `User` / `Group` — the account the service runs as
- `ExecStart` — gunicorn from the `ml-runner` conda env

Gunicorn runs with 4 workers and a 600s timeout (long enough for slow CPU
generations; GPU generations are sub-second).

## Layout

```
ml-runner/
├── app.py                 # Flask routes
├── db.py                  # sqlite job tracking
├── job_runner.py          # subprocess management + poller threads
├── requirements.txt       # Flask only (the ML stack lives in the conda env)
├── ml-runner.service      # systemd unit
├── scripts/
│   ├── generate_sample.py # torch + transformers
│   └── train_set.py       # torch + transformers
└── data/
    ├── config.json        # sets: flat array of set objects (seeded from legacy/)
    ├── checkpoint/        # trained models in HuggingFace format
    ├── hf-cache/          # HuggingFace model cache (GPT-2 117M base model)
    ├── datasets/          # uploaded training data
    ├── logs/              # per-job subprocess logs
    └── jobs.db            # sqlite (created on first run)
```

## Job tracking

A small sqlite DB at `data/jobs.db` records every generate/train job with
status (`queued` / `running` / `complete` / `failed`), timestamps, result
text or error message, and a pointer to the subprocess log file.

On `ml-runner` restart, any job still marked `running` is marked `failed
(interrupted)` since we can't reattach to the dead subprocess. Long-running
training jobs will need to be re-submitted.

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

## Where to go next

- [`../INSTALL.md`](../INSTALL.md) — host setup (conda envs, systemd service, first run)
- [`../README.md`](../README.md) — project overview, architecture diagrams, full API table
- [`scripts/README.md`](scripts/README.md) — running the GPT-2 scripts by hand for debugging