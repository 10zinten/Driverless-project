"""Tensorflow utility functions for evaluatioin"""

import os

from tqdm import trange
import tensorflow as tf


def evaluate_sess(sess, model_specs, num_steps, writer=None, params=None):
    """Train the model on num_steps batches.

    Args:
        sess: (tf.Session) current session
        model_specs: (dict) contain the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: Hypterparameters
    """

    update_metrics = model_specs['update_metrics']
    eval_metrics = model_specs['metrics']
    global_step = tf.train.get_global_step()

    # load the evaluation dataset into the pipeline and initialize the metrics
    # init op
    sess.run(model_specs['iterator_init_op'])
    sess.run(model_specs['metrics_init_op'])

    # Compute metrics over the dataset
    for _ in range(num_steps):
        sess.run(update_metrics)

    # Get the values of the metrics
    metrics_values = {k: v[0] for k, v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    print("- Eval metrics: " + metrics_string)

    # Add summaries manually to writer at global_step_val
    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag, val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, sample_valiue=val)])
            writer.add_summary(summ, global_step_val)

    return metrics_val
