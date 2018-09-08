"""Set up model's specification"""

import tensorflow as tf

from model.ssdmobilenet import SSDMobileNet

def model_fn(mode, inputs, preset, params, reuse=False):
    """Model functios defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels..)
        params: contains hyperparameters of the model
        reuse: (bool) whether to reuse the weights

    Return:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """

    is_training = (mode == 'train')
    labels = inputs['labels']

    # -------------------------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        ssd = SSDMobileNet(is_training, inputs, preset, params)
        predictions = ssd.result


    # Define loss and accuracy
    loss = ssd.losses['total']
    # TODO: create accuracy


    # Define training step that minimizes the loss with optimizer
    if is_training:
        optimizer = tf.train.MomentumOptimizer(learning_rate=params.learning_rate,
                                               momentum=params.momentum)
        global_step = tf.train.get_or_create_global_step()
        if params.batchnorm_enabled:
            # Add a dependency to update the moving mean and variance of BN
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # -------------------------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'loss': tf.metrics.mean(loss)
            # TODO: add accuracy metrics
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variable used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)


    # -------------------------------------------------------------------------
    # MODEL SPECIFICATION
    model_specs = inputs
    model_specs['mobilenet_init_op'] = ssd.load_mobilenet
    model_specs['variable_init_op'] = tf.group(tf.global_variables_initializer(),
                                               tf.local_variables_initializer())
    model_specs['predictions'] = predictions
    model_specs['loss'] = loss
    # TODO: add accuracy value
    model_specs['metrics_init_op'] = metrics_init_op
    model_specs['metrics'] = metrics
    model_specs['update_metrics'] = update_metrics_op

    if is_training:
        model_specs['train_op'] = train_op

    return model_specs
