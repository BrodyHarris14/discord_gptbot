# scripts/ — GPT-2 generation and training scripts

The two scripts in this folder do the actual ML work. `ml-runner` shells out
to them via `conda run -n gpt2 python scripts/<script>.py <args>`; you can
also run them by hand for debugging or experimentation.

Both scripts use `transformers` + `torch` (not the legacy `gpt_2_simple` +
TF 1.14 — that stack was retired because it only supported GPUs up to RTX
20xx). The base GPT-2 117M model (`"gpt2"`) is auto-downloaded from
HuggingFace on first run; fine-tuned checkpoints are saved in HuggingFace
format under `../data/checkpoint/<set>/`.

## Prerequisites

Activate the ML conda env and `cd` into `data/` so that `checkpoint/` and the
HuggingFace cache resolve the same way they do when ml-runner invokes the
scripts (the scripts expect `cwd=data/`):

```bash
conda activate gpt2
cd /opt/discord_gptbot/ml-runner/data   # or wherever ML_RUNNER_DATA_DIR points
```

## generate_sample.py

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
python ../scripts/generate_sample.py trump-tweet "Make America"

# Prefix via stdin (use this for anything with quotes, newlines, etc.):
echo -n 'Question: what is the meaning of life?' | python ../scripts/generate_sample.py wisdom

# No prefix at all (let the model freewheel):
echo -n "" | python ../scripts/generate_sample.py trump-tweet

# Capture stdout to a file for inspection:
python ../scripts/generate_sample.py trump-tweet "Crooked Hillary" > /tmp/sample.txt
```

How it works: loads the fine-tuned model from `checkpoint/<set>/` via
`GPT2LMHeadModel.from_pretrained()`, prepends `<|startoftext|>` to the prefix
(same convention the legacy stack used), runs `model.generate()` with
`do_sample=True, temperature=1.0, top_k=50, top_p=0.95` (mirroring
gpt_2_simple's defaults), then strips the `<|startoftext|>` marker and
truncates at the first `` if present.

Device selection is automatic: `cuda` if `torch.cuda.is_available()`, else
`cpu`. No env vars needed.

## train_set.py

```
python ../scripts/train_set.py <run_name> <file_name> <steps>
```

- `<run_name>` — the set name; checkpoint is saved to `checkpoint/<run_name>/`.
- `<file_name>` — path to the training dataset (text file, one document per
  line). Relative paths resolve against `cwd` (i.e. `data/`), so put
  datasets under `data/datasets/` or pass an absolute path.
- `<steps>` — integer, number of optimization steps (matches the semantics
  of the legacy `gpt_2_simple` `steps` argument, not epochs).

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

How it works: loads the base `"gpt2"` model from HuggingFace, tokenizes the
dataset file into 128-token blocks, runs `transformers.Trainer` for the
requested number of steps (using `max_steps` so the count is optimization
steps, not epochs), then saves the model + tokenizer in HuggingFace format
via `model.save_pretrained(checkpoint/<run_name>)` + `tokenizer.save_pretrained()`.
Uses `fp16` on GPU for faster training.

## Common debugging tips

- **"No checkpoint found" / load errors** → check that
  `data/checkpoint/<set>/` exists and contains the HuggingFace checkpoint
  files (`config.json`, `pytorch_model.bin`, `tokenizer.json`, etc.).
- **First run is slow** → `transformers` is downloading the 117M base model
  from HuggingFace. This is a one-time ~500 MB download, cached under
  `data/hf-cache/` (or `~/.cache/huggingface/` if `HF_HOME` isn't set).
- **GPU not being used** → confirm `torch.cuda.is_available()` returns `True`
  inside the `gpt2` env. If not, check `nvidia-smi` works and that your torch
  install is a CUDA build matching your driver (see
  [`../../INSTALL.md`](../../INSTALL.md) step 1 for the CUDA-version gotcha).
- **`CUDA_VISIBLE_DEVICES` is hiding the GPU** → if you ported from the old
  TF 1.14 stack, you may have set `CUDA_VISIBLE_DEVICES=""` on the conda env
  as a CPU workaround. Clear it:
  `conda env config vars unset CUDA_VISIBLE_DEVICES -n gpt2`
- **Reproducing an ml-runner job's exact invocation** → check the job's log
  at `data/logs/<job_id>.log` (or `GET /jobs/<id>/log`); it captures the
  subprocess's stderr. The exact command ml-runner runs is:
  `conda run -n gpt2 --no-capture-output python /abs/path/to/scripts/generate_sample.py <set>`
  with the prefix sent to stdin and `cwd=data/`.

## Where to go next

- [`../README.md`](../README.md) — the webapp internals (app.py, db.py, job_runner.py, ml-runner.service)
- [`../../INSTALL.md`](../../INSTALL.md) — host setup (conda envs, systemd service, first run)
- [`../../README.md`](../../README.md) — project overview, architecture diagrams, full API table