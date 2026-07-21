"""
Finetune GPT-2 against a dataset, producing a checkpoint that generate_sample.py
can load.

Uses `transformers` + `torch` for GPU-accelerated training on modern cards
(RTX 30xx / 40xx) via torch's bundled CUDA 12.x runtime.

Usage:
    python train_set.py <run_name> <file_name> <steps>

    run_name   : name of the trained set (e.g. "trump-tweet"); checkpoint is
                 saved to ./checkpoint/<run_name>/ in HuggingFace format
                 (config.json + pytorch_model.bin + tokenizer files)
    file_name  : path to the training dataset (text file, one document per line)
    steps      : number of optimization steps (matches gpt_2_simple semantics)

The base GPT-2 117M model ("gpt2") is auto-downloaded from HuggingFace on first
run — no manual model download needed.

This script should be run with cwd set to the data directory so that
checkpoint/ resolves correctly.
"""
import os
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)


class LineBlockDataset(Dataset):
    """
    A simple causal-LM dataset: reads a text file, tokenizes it, and splits
    into fixed-length `block_size` token blocks. This replaces the deprecated
    `transformers.TextDataset` with a ~15-line equivalent that doesn't depend
    on removed APIs.

    Each item is a tensor of shape (block_size,) — input_ids for one block.
    The Trainer's default collator stacks them into a batch; labels are
    created by shifting input_ids (the model does this internally for
    GPT2LMHeadModel).
    """

    def __init__(self, tokenizer, file_path, block_size=128):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        # Drop the trailing partial block so every example is exactly block_size.
        n_blocks = len(tokens) // block_size
        tokens = tokens[: n_blocks * block_size]
        self.blocks = [
            torch.tensor(tokens[i : i + block_size], dtype=torch.long)
            for i in range(0, len(tokens), block_size)
        ]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def main():
    if len(sys.argv) != 4:
        sys.stderr.write(
            "Usage: train_set.py <run_name> <file_name> <steps>\n"
        )
        sys.exit(2)

    run_name = sys.argv[1]
    file_name = sys.argv[2]
    try:
        steps = int(sys.argv[3])
    except ValueError:
        sys.stderr.write("steps must be an integer\n")
        sys.exit(2)

    if not os.path.isfile(file_name):
        sys.stderr.write("Dataset not found: {}\n".format(file_name))
        sys.exit(1)

    ckpt_dir = os.path.join("checkpoint", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stderr.write("Using device: {}\n".format(device))

    # Load the base GPT-2 117M model + tokenizer from HuggingFace.
    # On first run this downloads ~500 MB and caches it under HF_HOME
    # (set by the systemd unit) or ~/.cache/huggingface/.
    sys.stderr.write("Loading base GPT-2 117M model...\n")
    model_name = "gpt2"  # 117M
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)

    # GPT-2 has no pad token by default; use EOS as the pad token.
    tokenizer.pad_token = tokenizer.eos_token

    # Build the dataset: tokenize the file and split into 128-token blocks.
    sys.stderr.write("Loading dataset...\n")
    train_dataset = LineBlockDataset(tokenizer, file_name, block_size=128)
    sys.stderr.write(
        "Dataset has {} blocks of 128 tokens\n".format(len(train_dataset))
    )

    # TrainingArguments: max_steps keeps the same semantics as
    # gpt_2_simple's `steps` (optimization steps, not epochs).
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,
        max_steps=steps,
        per_device_train_batch_size=4,
        warmup_steps=10,
        logging_steps=50,
        save_steps=steps,          # save once at the end
        save_total_limit=1,
        report_to="none",          # disable wandb/tensorboard
        fp16=(device == "cuda"),   # mixed precision on GPU
    )

    # Train. GPT2LMHeadModel computes loss internally when given input_ids
    # without explicit labels (it shifts them itself).
    sys.stderr.write("Training for {} steps...\n".format(steps))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save the model + tokenizer in HuggingFace format so generate_sample.py
    # can load it via from_pretrained().
    sys.stderr.write("Saving checkpoint to {}...\n".format(ckpt_dir))
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Emit a single sample so the caller can confirm training worked.
    sys.stderr.write("Generating a test sample...\n")
    model.eval()
    input_ids = tokenizer.encode("<|startoftext|>", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=False)
    if "<|startoftext|>" in text:
        text = text.split("<|startoftext|>", 1)[1]
    if "​" in text:
        text = text.split("​", 1)[0]
    sys.stdout.write(text)


if __name__ == "__main__":
    main()