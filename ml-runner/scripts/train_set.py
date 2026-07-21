"""
Finetune GPT-2 against a dataset, producing a checkpoint that generate_sample.py
can load.

Ported from legacy/train_set.py. Runs under the legacy conda env
(python 3.6 / tensorflow 1.14 / gpt-2-simple).

Usage:
    python train_set.py <run_name> <file_name> <steps>

    run_name   : name of the trained set (e.g. "trump-tweet"); checkpoint is
                 saved to ./checkpoint/<run_name>/
    file_name  : path to the training dataset (text file)
    steps      : number of training steps

On first run the 117M base model is downloaded to ./models/117M/ if missing.

This script should be run with cwd set to the data directory so that
checkpoint/ and models/ resolve correctly.
"""
import gpt_2_simple as gpt2
import os
import sys


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

    model_name = "117M"

    if not os.path.isdir(os.path.join("models", model_name)):
        sys.stderr.write("Downloading {} model...\n".format(model_name))
        gpt2.download_gpt2(model_name=model_name)

    if not os.path.isfile(file_name):
        sys.stderr.write("Dataset not found: {}\n".format(file_name))
        sys.exit(1)

    sess = gpt2.start_tf_sess()
    gpt2.finetune(
        sess,
        file_name,
        run_name=run_name,
        model_name=model_name,
        steps=steps,
    )

    # Emit a single sample so the caller can confirm training worked.
    sample = gpt2.generate(
        sess,
        run_name=run_name,
        return_as_list=True,
    )[0]
    sys.stdout.write(sample)


if __name__ == "__main__":
    main()