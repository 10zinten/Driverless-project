"""
This CNN model1 is based on ../cnn_architecture.txt.
"""

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncate_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def new_conv_layer(input,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=True):

    # Create weights and baises
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = weight_variable(shape=shape)
    biases = bias_variable(shape=num_filters)

    layer = tf.nn.conv2d(input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer


def flatten_layer(layer):
    # Get the shape of input layer
    # - Shape of input layer: [num_images, img_height, img_width, num_channel]
    layer_shape = layer.get_shape()

    # Num. of features: img_height * img_width * num_channels
    # Use TF function to calculate this
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])

    # flatten layer shape: [num_images, img_height * img_width * num_channels]

    return layer_flat, num_features


# Helper function to create FC layer
def new_fc_layer(input, num_input, num_output, use_relu=True):

    weights = weight_variable(shape=[num_input, num_output])
    biases = bias_variable(shape=num_output)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Placeholder
x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
x_image = x
y = tf.placeholder(tf.float32, shape=[None, 1])

# Conlutional Layer 1
layer_conv1 = new_conv_layer(input=x_image,
                             num_input_channels=3,
                             filter_size=3,
                             num_filters=32,
                             use_pooling=True)

# Conlutional Layer 1
layer_conv2 = new_conv_layer(input=layer_conv1,
                             num_input_channels=32,
                             filter_size=3,
                             num_filters=64,
                             use_pooling=True)

# Conlutional Layer 1
layer_conv3 = new_conv_layer(input=layer_conv2,
                             num_input_channels=64,
                             filter_size=3,
                             num_filters=128,
                             use_pooling=True)

# FC Layer 1
layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=100,
                         use_relu=True)

# FC layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=100,
                         num_outputs=50,
                         use_relu=True)

# Out is steering angle
y_pred = new_fc_layer(input=layer_fc2,
                      num_inputs=50,
                      num_outputs=1,
                      use_relu=True)

# Cost per image
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logist=y_pred, labels=y)

# Total Cost of training batch
cost = tf.reduce_mean(cross_entropy)


# Optimization
optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-4).minimize(cost)

# Tranning step
batch_size = 64
num_batch = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch_cost = 0
    costs = []
    for epoch in range(10000):
        for i in range(num_batch):
            batch_x, batch_y = None
            _, cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            epoch_cost += cost / batch_size
            if epoch % 1000 == 0:
                print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            if epoch % 100 == 0:
                costs.append(epoch_cost)
