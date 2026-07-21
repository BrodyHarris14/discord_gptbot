"""
Generate a single sample from a trained GPT-2 set.

Ported from legacy/generate_sample.py. Runs under the legacy conda env
(python 3.6 / tensorflow 1.14 / gpt-2-simple).

Usage:
    python generate_sample.py <set> [prefix]

If no prefix is given on the command line, the prefix is read from stdin.
This avoids shell-escaping issues with quotes/newlines in user input.

The generated text is written to stdout. Errors/diagnostics go to stderr.

Checkpoint resolution: gpt_2_simple loads from ./checkpoint/<set>/ by default,
so this script should be run with cwd set to the data directory.
"""
import gpt_2_simple as gpt2
import tensorflow as tf
import sys
import os


def generate(set_name, prefix):
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=set_name)
    prefix = "<|startoftext|>" + prefix
    text = gpt2.generate(
        sess,
        run_name=set_name,
        temperature=1.0,
        return_as_list=True,
        prefix=prefix,
        truncate='<|endoftext|>',
        include_prefix=True,
    )[0]
    text = text.replace('<|startoftext|>', '')
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