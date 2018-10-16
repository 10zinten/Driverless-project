"""Create the input data pipeline using tf.data"""

import tensorflow as tf

from model.data_gen import TrainingData


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image, label

def _preprocess(image, gt, is_training):
    labels = gt[:, :3]
    bboxes = gt[:, 3:]

    out = preprocess_image(image, labels, bboxes, [160, 160], is_training=is_training,
                     data_format="channel_last", output_rgb=False)

    if is_training:
        image, labels, bboxes = out
        gt = tf.concat([labels, bboxes], axis=-1)
        return image, gt
    else:
        image = out
        return image, gt


def train_preprocess(image, label):
    """Image preprocessing for training

    Apply the following operations:
        - Apply random brightness and saturation
    """

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def input_fn(is_training, image_dir, label_dir, args):
    """Input functions for the Cones dataset."""

    parse_fn = lambda f, gt: _parse_function(f, gt)
    train_fn = lambda f, gt: train_preprocess(f, gt)

    td = TrainingData(image_dir, label_dir, args)

    if is_training:
        dataset = tf.data.Dataset().batch(args.batch_size).from_generator(
                        td.train_generator,
                        (tf.float32, tf.float32),
                        (tf.TensorShape([None, 160, 160, 3]), tf.TensorShape([None, 8540, 7]))
                  )

    else:
        dataset = tf.data.Dataset().batch(args.batch_size).from_generator(
                        td.val_generator,
                        (tf.float32, tf.float32),
                        (tf.TensorShape([None, 160, 160, 3]), tf.TensorShape([None, 8540, 7]))
                   )


    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}

    return inputs
