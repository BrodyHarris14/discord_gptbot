import gpt_2_simple as gpt2
import tensorflow as tf


def generate(set, prefix):
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=set)
    prefix = "<|startoftext|>" + prefix
    text = gpt2.generate(sess,
                         run_name=set,
                         temperature=1.0,
                         return_as_list=True,
                         prefix=prefix,
                         truncate='<|endoftext|>',
                         include_prefix=True)[0]
    text = text.replace('<|startoftext|>', '')
    return text
