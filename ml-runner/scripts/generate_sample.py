"""
Generate a single sample from a trained GPT-2 set.

Rewritten to use `transformers` + `torch` instead of `gpt_2_simple` + TF 1.14.
This enables GPU support on modern cards (RTX 30xx / 40xx) via torch's bundled
CUDA 12.x runtime.

Usage:
    python generate_sample.py <set> [prefix]

If no prefix is given on the command line, the prefix is read from stdin.
This avoids shell-escaping issues with quotes/newlines in user input.

The generated text is written to stdout. Errors/diagnostics go to stderr.

Checkpoint resolution: loads a HuggingFace-format model from
./checkpoint/<set>/ (saved by train_set.py via model.save_pretrained()).
This script should be run with cwd set to the data directory so that
checkpoint/ resolves correctly.
"""
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def generate(set_name, prefix):
    ckpt_dir = os.path.join("checkpoint", set_name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            "No checkpoint found at {}. Has '{}' been trained?".format(
                ckpt_dir, set_name
            )
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the fine-tuned model + tokenizer (HuggingFace format).
    tokenizer = GPT2TokenizerFast.from_pretrained(ckpt_dir)
    model = GPT2LMHeadModel.from_pretrained(ckpt_dir)
    model.to(device)
    model.eval()

    # Prepend the <|startoftext|> marker, same as the legacy stack did.
    # GPT-2's tokenizer doesn't have a <|startoftext|> special token, so it
    # gets tokenized as subword pieces — which is fine; the model learned the
    # pattern during training (the dataset used the literal string).
    full_prompt = "<|startoftext|>" + prefix
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    # Generate. Parameters mirror gpt_2_simple's defaults:
    #   temperature=1.0, do_sample=True (sampling, not greedy).
    # Added top_k/top_p for sane sampling — transformers requires explicit
    # flags where gpt_2_simple assumed sampling mode.
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=1024,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=False)

    # Strip the prefix marker and the prompt itself, matching legacy behavior:
    # the old code used include_prefix=True + truncate='' and then
    # removed <|startoftext|>. We replicate that: cut everything up to and
    # including <|startoftext|>, then cut at <|endoftext|> if present.
    if "<|startoftext|>" in text:
        text = text.split("<|startoftext|>", 1)[1]
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>", 1)[0]

    return text


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: generate_sample.py <set> [prefix]\n")
        sys.exit(2)

    set_name = sys.argv[1]

    # Prefix: take from argv[2] if provided, otherwise read from stdin.
    if len(sys.argv) >= 3:
        prefix = sys.argv[2]
    else:
        prefix = sys.stdin.read()

    try:
        text = generate(set_name, prefix)
    except Exception as e:
        sys.stderr.write("generate_sample.py failed: {}\n".format(e))
        sys.exit(1)

    sys.stdout.write(text)


if __name__ == "__main__":
    main()