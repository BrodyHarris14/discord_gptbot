import gpt_2_simple as gpt2
import os
import sys

model_name = "117M"
#model_name = "774M"
if not os.path.isdir(os.path.join("models", model_name)):
    print("Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)
    # model is saved into current directory under /models/124M/

steps = sys.argv[3]
run_name = sys.argv[1]
file_name = sys.argv[2]

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              # prefix='<|startoftext|>',
              # truncate='<|endoftext|>',
              # include_prefix=False,
              run_name=run_name,
              model_name=model_name,
              steps=steps)   # steps is max number of training steps

gpt2.generate(sess,
              run_name=run_name)
