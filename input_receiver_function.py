import tensorflow as tf


def serving_input_receiver_fn():
    feature_spec = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
    }

    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[batch_size],
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=6,
    predict_batch_size=8)
estimator._export_to_tpu = False  # !!important to add this
estimator.export_saved_model(
    export_dir_base=EXPORT_PATH,
    serving_input_receiver_fn=serving_input_receiver_fn)
