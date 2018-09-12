"""Create the input data pipeline using tf.data"""

import tensorflow as tf

from model.ssd_preprocessing import preprocess_image


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



def input_fn(is_training, filenames, labels, args):
    """Input functions for the Cones dataset."""

    num_samples = len(filenames)
    assert len(filenames) > 0, 'Datapoint not found'

    parse_fn = lambda f, gt: _parse_function(f, gt)
    train_preprocess_fn = lambda image, gt: _preprocess(image, gt, True)
    eval_preprocess_fn = lambda image, gt: _preprocess(image, gt, False)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)
            .map(parse_fn, num_parallel_calls=4)
            .map(train_preprocess_fn, num_parallel_calls=4)
            .batch(args.batch_size)
            .prefetch(1)
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
               .map(parse_fn)
               .map(eval_preprocess_fn, num_parallel_calls=4)
               .batch(args.batch_size)
               .prefetch(1)
        )


    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}

    return inputs
