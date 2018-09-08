import os

from tqdm import trange
import tensorflow as tf


def train_sess(sess, model_specs, num_steps, params):
    """Train the model on num_steps batches

    Args:
        sess: (tf.Session) current session
        model_spces: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
        params: hyperparameters
    """

    # Get relevant graph operations or nodes needed for training
    loss = model_specs['loss']
    train_op = model_specs['train_op']
    update_metrics = model_specs['update_metrics']
    metrics = model_specs['metrics']
    # TODO: Summary op
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics
    # local variables
    sess.run(model_specs['iterator_init_op'])
    sess.run(model_specs['metrics_init_op'])

    t = trange(num_steps)
    for i in t:
        # TODO: Evaluate summaries for tensorboard only once in a while
        _, _, loss_val = sess.run([train_op, update_metrics, loss])
        t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    print(" - Train metrics: " + metrics_string)



def train_and_evaluate(train_model_specs, eval_model_spces, model_dir, params, restor_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_specs: (dict) contains the graph operations or nodes needed for training
        eval_model_specs: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph.
    """

    # TODO: create tf.Saver()

    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model vairables
        sess.run(train_model_specs['variable_init_op'])

        # Load the mobilenet pretrain weights
        train_model_specs['mobilenet_init_op'](sess)

        # TODO:  Reload weights from directory if specified

        # TODO: Create summary writer for train and eval

        best_eval_acc = 0.0
        for epoch in range(begin_at_epoch, begin_at_epoch+params.num_epochs):
            # Run one epoch
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_specs, num_steps, params)

            # TODO: save weights

            # TODO: Evaluate for one epoch on validation set
