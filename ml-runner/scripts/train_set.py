"""
Finetune GPT-2 against a dataset, producing a checkpoint that generate_sample.py
can load.

Rewritten to use `transformers` + `torch` instead of `gpt_2_simple` + TF 1.14.
This enables GPU support on modern cards (RTX 30xx / 40xx) via torch's bundled
CUDA 12.x runtime.

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
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


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
    # On first run this downloads ~500 MB and caches it under ~/.cache/huggingface/.
    sys.stderr.write("Loading base GPT-2 117M model...\n")
    model_name = "gpt2"  # 117M
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)

    # GPT-2 has no pad token by default; use EOS as the pad token.
    tokenizer.pad_token = tokenizer.eos_token

    # Build the dataset + data collator for language modeling.
    # TextDataset reads a text file and chunks it into block_size-token blocks.
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_name,
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
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

    # Train.
    sys.stderr.write("Training for {} steps...\n".format(steps))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
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
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>", 1)[0]
    sys.stdout.write(text)


if __name__ == "__main__":
    main()