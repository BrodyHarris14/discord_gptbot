"""
Subprocess management for ml-runner.

Pairs each sqlite job row with a live subprocess and a background poller thread
that finalizes the row when the process exits.

Scripts run inside the legacy conda env (`conda run -n gpt2 python ...`) with
cwd set to the data directory so checkpoints and models resolve correctly.

The prefix for generate_sample.py is passed via stdin to avoid shell-escaping
issues with quotes/newlines.
"""
import os
import subprocess
import threading

import db

# In-memory map: job_id -> subprocess.Popen handle.
# Survives only as long as this process lives; sqlite is the durable record.
_procs = {}
_procs_lock = threading.Lock()


def _conda_base():
    """Path to the conda executable. Overridable via CONDA_BIN env var."""
    return os.environ.get("CONDA_BIN", "conda")


def _conda_env():
    """Name of the legacy conda env that has gpt-2-simple + TF 1.14."""
    return os.environ.get("CONDA_ENV", "gpt2")


def _log_path(data_dir, job_id):
    logs_dir = os.path.join(data_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, "{}.log".format(job_id))


def _tail(path, lines=40):
    try:
        with open(path, "r") as f:
            data = f.readlines()
        return "".join(data[-lines:])
    except Exception:
        return "(log unavailable)"


# -------------------------------------------------------------------
# Async paths
# -------------------------------------------------------------------

def start_generate(data_dir, job_id, set_name, prefix):
    """
    Launch generate_sample.py in the background for the given job and return
    immediately. The poller thread feeds the prefix via stdin, waits for the
    process, and finalizes the job row.
    """
    log_path = _log_path(data_dir, job_id)
    cmd = [
        _conda_base(), "run", "-n", _conda_env(), "--no-capture-output",
        "python", "scripts/generate_sample.py", set_name,
    ]
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=data_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log_file,
        text=True,
    )
    log_file.close()  # the OS holds the fd via the child; safe to drop our handle
    with _procs_lock:
        _procs[job_id] = proc
    db.mark_running(data_dir, job_id)
    threading.Thread(
        target=_poll_generate,
        args=(data_dir, job_id, proc, prefix),
        daemon=True,
    ).start()


def start_train(data_dir, job_id, set_name, dataset_path, steps):
    """
    Launch train_set.py in the background for the given job and return
    immediately. stdout/stderr stream to a log file; the poller just waits.
    """
    log_path = _log_path(data_dir, job_id)
    cmd = [
        _conda_base(), "run", "-n", _conda_env(), "--no-capture-output",
        "python", "scripts/train_set.py", set_name, dataset_path, str(steps),
    ]
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=data_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.close()
    with _procs_lock:
        _procs[job_id] = proc
    db.mark_running(data_dir, job_id)
    threading.Thread(
        target=_poll_train,
        args=(data_dir, job_id, proc, log_path),
        daemon=True,
    ).start()


def _poll_generate(data_dir, job_id, proc, prefix):
    """Wait for a generate subprocess, capture stdout as the result."""
    try:
        stdout_data, _ = proc.communicate(input=prefix, timeout=None)
        rc = proc.returncode
    except Exception as e:
        proc.kill()
        db.mark_failed(data_dir, job_id, "generate subprocess error: {}".format(e))
        with _procs_lock:
            _procs.pop(job_id, None)
        return

    if rc == 0:
        db.mark_complete(data_dir, job_id, stdout_data or "")
    else:
        log_path = _log_path(data_dir, job_id)
        db.mark_failed(
            data_dir, job_id,
            "generate_sample.py exited with code {}. Tail of log:\n{}".format(
                rc, _tail(log_path)
            ),
        )
    with _procs_lock:
        _procs.pop(job_id, None)


def _poll_train(data_dir, job_id, proc, log_path):
    """Wait for a train subprocess; on success record a pointer to the checkpoint."""
    rc = proc.wait()
    set_name = _job_set(data_dir, job_id)
    if rc == 0:
        result = "Training complete. See checkpoint/{} for the model.".format(set_name)
        db.mark_complete(data_dir, job_id, result)
    else:
        db.mark_failed(
            data_dir, job_id,
            "train_set.py exited with code {}. Tail of log:\n{}".format(
                rc, _tail(log_path)
            ),
        )
    with _procs_lock:
        _procs.pop(job_id, None)


def _job_set(data_dir, job_id):
    row = db.get_job(data_dir, job_id)
    return row["set_name"] if row else "unknown"


# -------------------------------------------------------------------
# Synchronous generation (used by /generate when async=false)
# -------------------------------------------------------------------

def run_generate_sync(data_dir, set_name, prefix, log_path):
    """
    Run generate_sample.py and block until it returns. Used for the default
    synchronous /generate path. Writes stderr to log_path and returns
    (return_code, stdout_text).
    """
    cmd = [
        _conda_base(), "run", "-n", _conda_env(), "--no-capture-output",
        "python", "scripts/generate_sample.py", set_name,
    ]
    log_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=data_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_file,
            text=True,
        )
        stdout_data, _ = proc.communicate(input=prefix, timeout=None)
        return proc.returncode, stdout_data
    finally:
        log_file.close()