# discord_gptbot

GPT-2 generation + training service, revived. GPT-2 (the 117M model) has a
certain child-like charm — this project keeps that alive.

## What's here

- **`ml-runner/`** — the product. A Flask webservice that fronts the legacy
  GPT-2 scripts running in a separate conda env on the host. Exposes HTTP
  endpoints for generating text from a trained set and for training new sets.
  No Discord, no Docker, no k8s — just a service on the box that anything
  (a Discord bot, a web UI, whatever) can talk to over HTTP. See
  [`ml-runner/README.md`](ml-runner/README.md) for full setup + API docs.

- **`legacy/`** — the original project this revived from. Python 3.6 /
  TensorFlow 1.14 / `gpt_2_simple`, hard-wired to a single Discord server.
  Kept as a reference; not used at runtime. The `config.json` set list and the
  `generate_sample.py` / `train_set.py` scripts were ported from here into
  `ml-runner/` (with `config.json` cleaned up from a redundant dict+list shape
  into a flat array of set objects), then rewritten to use `transformers` +
  `torch` for modern GPU support (the legacy TF 1.14 stack only supported
  GPUs up to RTX 20xx).

## Quick architecture

```
client (discord bot, web UI, ...)
   │ HTTP
   ▼
ml-runner  (Flask, systemd, port 7070)
   │ subprocess: `conda run -n gpt2 python scripts/...`
   ▼
conda env "gpt2"  (python 3.10 / torch / transformers / CUDA 12.x)
   └── reads/writes data/checkpoint/<set>/ and data/hf-cache/
```

The Flask layer owns nothing ML — it just shells out to the scripts in the
ML conda env and tracks jobs in sqlite.

## Pieces involved (generate happy path)

How a single synchronous `/generate` request flows through the pieces on the
host. The Flask process, the ML conda env, the scripts, and the on-disk
model/checkpoint artifacts — and how they connect.

```mermaid
flowchart TB
    Client["Client\n(discord bot / web UI / curl)"]

    subgraph Host["System76 host (Ubuntu + microk8s)"]
        subgraph Flask["ml-runner (systemd, py3.11)"]
            App["app.py\n/generate route"]
            JR["job_runner.py\nsubprocess + poller"]
            DB[("db.py\nsqlite jobs.db")]
        end

        subgraph Conda["conda env 'gpt2' (py3.10)"]
            Script["scripts/generate_sample.py"]
            GPT2["torch + transformers + CUDA 12.x"]
        end

        subgraph Disk["data/ (on disk)"]
            Ckpt[("checkpoint/&lt;set&gt;/")]
            Model[("hf-cache/")]
            Logs[("logs/&lt;job&gt;.log")]
        end
    end

    Client -->|"POST /generate\n{set, prefix}"| App
    App --> JR
    JR --> DB
    JR -->|"conda run -n gpt2\npython scripts/generate_sample.py\n(prefix via stdin)"| Script
    Script --> GPT2
    GPT2 -->|"load checkpoint"| Ckpt
    GPT2 -.->|"first run only:\ncache GPT-2 117M\nfrom HuggingFace"| Model
    Script -->|"stdout = generated text"| JR
    JR -->|"append stderr"| Logs
    App -->|"200 text/plain"| Client
```

The same shape applies to `/train`, except `train_set.py` is invoked, it
*writes* to `checkpoint/<set>/` instead of reading, and the job is async
(`job_runner.py` spawns a poller thread that updates `jobs.db` when training
finishes).

## User flow

```mermaid
flowchart TD
    Start([Caller wants GPT-2 text]) --> QSet{"Set already\ntrained?"}

    QSet -->|"no"| Train["POST /train\n{set, dataset, steps}"]
    Train --> TrainAck["202 {job_id}"]
    TrainAck --> PollTrain{"GET /jobs/&lt;id&gt;\nstatus == complete?"}
    PollTrain -->|"running / queued"| WaitT["wait"] --> PollTrain
    PollTrain -->|"complete"| CkptReady["checkpoint/&lt;set&gt;/ now on disk"]
    CkptReady --> Gen

    QSet -->|"yes"| Gen["POST /generate\n{set, prefix}"]
    Gen --> Sync{"async=true?"}
    Sync -->|"no (default)"| Block["Flask blocks while\nconda subprocess runs"]
    Block --> Text["200 text/plain\n→ generated text"]
    Sync -->|"yes"| Queued["202 {job_id}"]
    Queued --> PollGen{"GET /jobs/&lt;id&gt;\nstatus == complete?"}
    PollGen -->|"running / queued"| WaitG["wait"] --> PollGen
    PollGen -->|"complete"| Text
```

## API (pseudo-swagger)

```mermaid
%% pseudo-swagger: a compact visual summary of the HTTP surface
flowchart TB
    subgraph Meta["Meta"]
        H["GET /health\n→ {status, conda_env, data_dir, ...}"]
        S["GET /sets\n→ {sets:[{name, description, prefix, trained}]}"]
    end

    subgraph Gen["Generate"]
        G["POST /generate  (also GET)\nbody: {set, prefix?, async?}\n→ 200 text/plain  (sync, default)\n→ 202 {job_id}     (async=true)"]
    end

    subgraph Train["Train"]
        T["POST /train\nmultipart: set, dataset, steps?\n  -or-\njson: {set, dataset_path, steps?}\n→ 202 {job_id}"]
    end

    subgraph Jobs["Jobs"]
        JL["GET /jobs?type=&status=&limit=\n→ {jobs:[row,...]}"]
        JG["GET /jobs/&lt;id&gt;\n→ row {status, result?, error?}"]
        JLog["GET /jobs/&lt;id&gt;/log\n→ text/plain (subprocess log)"]
    end

    Meta
    Gen --> JG
    Train --> JG
    JG -.-> JLog
    JL
```

| Endpoint | Method | Body / Params | Returns |
|---|---|---|---|
| `/health` | GET | — | `{status, conda_env, data_dir, config_present, ...}` |
| `/sets` | GET | — | `{sets:[{name, description, prefix, trained}]}` |
| `/generate` | GET/POST | `set`*, `prefix`?, `async`? | sync → `200 text/plain`; async → `202 {job_id}` |
| `/train` | POST | `set`*, `steps`?, `dataset` (file) *or* `dataset_path` | `202 {job_id}` |
| `/jobs` | GET | `type`?, `status`?, `limit`? | `{jobs:[row,...]}` |
| `/jobs/<id>` | GET | — | job row `{status, result?, error?, log_path, ...}` |
| `/jobs/<id>/log` | GET | — | `text/plain` subprocess log |

Job `status` is one of `queued` / `running` / `complete` / `failed`.

See [`ml-runner/README.md`](ml-runner/README.md) for setup and API details.