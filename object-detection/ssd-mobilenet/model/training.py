import os

from tqdm import trange
import tensorflow as tf

from model.evaluation import evaluate_sess


def train_sess(sess, model_specs, num_steps, params, writer):
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
    summary_op = model_specs['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics
    # local variables
    sess.run(model_specs['iterator_init_op'])
    sess.run(model_specs['metrics_init_op'])

    t = trange(num_steps)
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_step == 0:
            # Perfrom a mini-batch update
            _, _, loss_val, summ, global_step_val = sess.run([train_op, update_metrics, loss,
                                                              summary_op, global_step])

            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    print(" - Train metrics: " + metrics_string)



def train_and_evaluate(train_model_specs, eval_model_specs, model_dir, params, restor_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_specs: (dict) contains the graph operations or nodes needed for training
        eval_model_specs: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph.
    """

    # Initialize tf.Saver() instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1) # only keep 1 best checkpoint (based on eval)

    begin_at_epoch = 0
    with tf.Session() as sess:
        # Initialize model vairables
        sess.run(train_model_specs['variable_init_op'])

        # Load the mobilenet pretrain weights
        train_model_specs['mobilenet_init_op'](sess)

        # TODO:  Reload weights from directory if specified

        # Create summary writer for train and eval
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        best_eval_loss = 0.0
        for epoch in range(begin_at_epoch, begin_at_epoch+params.num_epochs):
            # Run one epoch
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_specs, num_steps, params, train_writer)

            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch+1)

            # Evaluate for one epoch on validation set
            num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
            metrics = evaluate_sess(sess, eval_model_specs, num_steps)

            # If best_loss, best_save_path
            eval_loss = metrics['loss']
            if eval_loss <= best_eval_loss:
                # Store new best loss
                best_eval_loss = eval_loss
                # Save weights
                best_save_path = os.path.join(model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch+1)
                print("- Found new best accuracy, saving in {}".format(best_save_path))
