"""
This module is used to train the CnnModels
"""

import os
import tensorflow as tf
from .model1.data_reader import DataReader
from .model1.cnn_model1 import ConvModel1


BATCH_SIZE = 64
CKPT_DIR = 'tmp/'
CKPT_FILE = 'cnn_model1.ckpt'
LEARNING_RATE = 1e-3
NUM_EPOCH = 10000


def train():
    sess = tf.Session()

    model = ConvModel1()
    data_reader = DataReader()
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    sess.run(tf.initialize_all_variable())
    saver = tf.train.Saver()

    for epoch in range(NUM_EPOCH):
        for train_batch in data_reader.train_batch(BATCH_SIZE):
            train_step.run(feed_dict={model.x: train_batch[0],
                                      model.y_: train_batch[1]})
            train_error = loss.eval(feed_dict={model.x: train_batch[0],
                                    model.y_: train_batch[1]})

        if epoch % 10 == 0:
            cv_set = data_reader.cv_set()
            cv_error = loss.eval(feed_dict={model.x: cv_set[0],
                                            model.y_: cv_set[1]})
            print("Step: %d, train loss: %g, cv loss: " % epoch, train_error,
                  cv_error)

    checkpoint_path = os.path.join(CKPT_DIR, CKPT_FILE)
    filename = saver.save(sess, checkpoint_path)
    print('Model saved in file: %s' % filename)


if __name__ == '__main__':
    train()
