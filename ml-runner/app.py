"""
ml-runner: a thin Flask front-end that shells out to the legacy GPT-2 scripts
(generate_sample.py / train_set.py) running inside a conda env on the host.

Routes:
    GET  /health            liveness + basic config check
    GET  /sets              list known sets (from config.json + checkpoint dir)
    POST /generate          generate text from a set
    GET  /jobs              list jobs (filter by type, status)
    GET  /jobs/<id>         get a single job
    GET  /jobs/<id>/log     tail of a job's log file

All generation is synchronous by default. Pass `async=true` (or `?async=true`)
to get a job_id back and poll /jobs/<id>. Training is always async.
"""
import json
import os

from flask import Flask, jsonify, request, Response

import db
import job_runner

app = Flask(__name__)


# -------------------------------------------------------------------
# Paths / config
# -------------------------------------------------------------------

DATA_DIR = os.environ.get("ML_RUNNER_DATA_DIR", os.path.join(os.getcwd(), "data"))
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoint")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

DEFAULT_TRAIN_STEPS = int(os.environ.get("ML_RUNNER_DEFAULT_STEPS", "1000"))
MAX_UPLOAD_BYTES = 128 * 1024 * 1024  # 128 MB


def _ensure_dirs():
    for d in (DATA_DIR, CHECKPOINT_DIR, DATASETS_DIR, os.path.join(DATA_DIR, "logs")):
        os.makedirs(d, exist_ok=True)


def _load_config():
    """
    Load config.json. The file is a flat JSON array of set objects, each with
    a `name` field. Returns (configs_by_name, ordered_names) or ({}, []) if
    absent / unparseable.
    """
    if not os.path.isfile(CONFIG_PATH):
        return {}, []
    try:
        with open(CONFIG_PATH) as f:
            sets_list = json.load(f)
        if not isinstance(sets_list, list):
            app.logger.warning("config.json is not a JSON array")
            return {}, []
        by_name = {}
        ordered = []
        for obj in sets_list:
            name = obj.get("name")
            if not name:
                continue
            by_name[name] = obj
            ordered.append(name)
        return by_name, ordered
    except Exception as e:
        app.logger.warning("Could not parse config.json: %s", e)
        return {}, []


def _discovered_sets():
    """Trained sets found under checkpoint/ that may not be in config.json yet."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return sorted(
        name for name in os.listdir(CHECKPOINT_DIR)
        if os.path.isdir(os.path.join(CHECKPOINT_DIR, name))
    )


# -------------------------------------------------------------------
# Health / meta
# -------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "conda_env": job_runner._conda_env(),
        "conda_bin": job_runner._conda_base(),
        "data_dir": DATA_DIR,
        "config_present": os.path.isfile(CONFIG_PATH),
        "checkpoint_dir_present": os.path.isdir(CHECKPOINT_DIR),
    })


@app.route("/sets", methods=["GET"])
def sets():
    configs, ordered = _load_config()
    discovered = _discovered_sets()
    # Merge: configured sets first (in file order, with metadata), then any
    # extras found on disk that aren't in the config.
    extra = [s for s in discovered if s not in configs]
    out = []
    for name in ordered:
        meta = configs.get(name, {})
        out.append({
            "name": name,
            "description": meta.get("description"),
            "prefix": meta.get("prefix"),
            "trained": name in discovered,
        })
    for name in extra:
        out.append({"name": name, "trained": True})
    return jsonify({"sets": out})


# -------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------

@app.route("/generate", methods=["GET", "POST"])
def generate():
    # Accept either query params (GET) or JSON/form (POST).
    if request.method == "POST":
        if request.is_json:
            payload = request.get_json(silent=True) or {}
        else:
            payload = request.form.to_dict()
    else:
        payload = request.args.to_dict()

    set_name = payload.get("set")
    prefix = payload.get("prefix", "")
    async_flag = str(payload.get("async", "false")).lower() in ("1", "true", "yes")

    if not set_name:
        return jsonify({"error": "Missing 'set' parameter"}), 400

    log_path = job_runner._log_path(DATA_DIR, "sync")  # placeholder, replaced below

    if not async_flag:
        # Synchronous: block until the script returns.
        import uuid
        sync_id = str(uuid.uuid4())
        log_path = job_runner._log_path(DATA_DIR, sync_id)
        try:
            rc, text = job_runner.run_generate_sync(DATA_DIR, set_name, prefix, log_path)
        except Exception as e:
            return jsonify({"error": "generate failed: {}".format(e)}), 500
        if rc != 0:
            tail = job_runner._tail(log_path)
            return jsonify({
                "error": "generate_sample.py exited with code {}".format(rc),
                "log_tail": tail,
            }), 500
        # Return JSON with the generated text + embed metadata from config.
        # Embed fields (embed_title, embed_color, embed_image) are optional;
        # missing fields are omitted so the client can fall back to defaults.
        configs, _ = _load_config()
        meta = configs.get(set_name, {})
        resp = {"text": text}
        for field in ("embed_title", "embed_color", "embed_image"):
            val = meta.get(field)
            if val:
                resp[field] = val
        return jsonify(resp)

    # Async: create a job row, launch the subprocess, return the id.
    job_id = db.create_job(
        DATA_DIR, "generate", set_name,
        params={"set": set_name, "prefix": prefix, "async": True},
        log_path=job_runner._log_path(DATA_DIR, "pending"),
    )
    job_runner.start_generate(DATA_DIR, job_id, set_name, prefix)
    return jsonify({"job_id": job_id, "status": "queued"}), 202


# -------------------------------------------------------------------
# Train
# -------------------------------------------------------------------

@app.route("/train", methods=["POST"])
def train():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict()

    set_name = payload.get("set")
    steps = payload.get("steps", DEFAULT_TRAIN_STEPS)
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        return jsonify({"error": "'steps' must be an integer"}), 400

    if not set_name:
        return jsonify({"error": "Missing 'set' parameter"}), 400

    # Dataset: either an uploaded file (multipart) or an explicit dataset_path.
    dataset_path = payload.get("dataset_path")
    uploaded = request.files.get("dataset")

    if uploaded and uploaded.filename:
        os.makedirs(DATASETS_DIR, exist_ok=True)
        save_path = os.path.join(DATASETS_DIR, "{}.txt".format(set_name))
        uploaded.save(save_path)
        dataset_path = save_path
    elif not dataset_path:
        return jsonify({
            "error": "Provide a dataset via multipart 'dataset' upload or 'dataset_path'",
        }), 400

    if not os.path.isfile(dataset_path):
        return jsonify({"error": "Dataset not found: {}".format(dataset_path)}), 400

    job_id = db.create_job(
        DATA_DIR, "train", set_name,
        params={"set": set_name, "steps": steps, "dataset_path": dataset_path},
        log_path=job_runner._log_path(DATA_DIR, "pending"),
    )
    job_runner.start_train(DATA_DIR, job_id, set_name, dataset_path, steps)
    return jsonify({"job_id": job_id, "status": "queued"}), 202


# -------------------------------------------------------------------
# Jobs
# -------------------------------------------------------------------

@app.route("/jobs", methods=["GET"])
def list_jobs():
    job_type = request.args.get("type")
    status = request.args.get("status")
    try:
        limit = int(request.args.get("limit", "100"))
    except ValueError:
        limit = 100
    rows = db.list_jobs(DATA_DIR, job_type=job_type, status=status, limit=limit)
    return jsonify({"jobs": rows})


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    row = db.get_job(DATA_DIR, job_id)
    if not row:
        return jsonify({"error": "job not found"}), 404
    return jsonify(row)


@app.route("/jobs/<job_id>/log", methods=["GET"])
def job_log(job_id):
    row = db.get_job(DATA_DIR, job_id)
    if not row:
        return jsonify({"error": "job not found"}), 404
    log_path = os.path.join(DATA_DIR, "logs", "{}.log".format(job_id))
    if not os.path.isfile(log_path):
        return jsonify({"error": "no log for this job"}), 404
    try:
        with open(log_path) as f:
            data = f.read()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return Response(data, mimetype="text/plain")


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

def _init():
    _ensure_dirs()
    db.init_db(DATA_DIR)
    # Any job left 'running' from a previous process can't be reattached.
    db.mark_interrupted_running(DATA_DIR)


_init()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("ML_RUNNER_PORT", "7070")))