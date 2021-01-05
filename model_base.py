import tensorflow as tf
import numpy as np
import glob


def get_raw_keras_model():
    """
    returns a keras model object.
    You can customise the layers here to build your own
    custom model per your needs
    """

    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])


def get_compiled_model():
    """
    returns a compiled model.
    You can customise is with the loss function and the metrics
    """

    model = get_raw_keras_model()
    optimizer = 'adam'
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def load_model_from_file(file_path):
    """
    Just loads h5 models and returns the model object
    """

    return tf.keras.models.load_model(file_path)


def load_dataset(batch_size):
    """
    This function can again be customised according to how your dataset is built.
    Similar function is used on the client script that runs on the providers,
    but only loads the requested training data.
    """

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_length = len(X_train)
    test_length = len(X_test)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(batch_size)
    return train_dataset, test_dataset, train_length, test_length


def federated_avg_weights(all_client_weights, weights=None):
    """
    Federated Averaging is one of the many algorithms used to combine
    distributed trained data. You can read more about it here -
    https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2
    and if you prefer a research paper - https://arxiv.org/pdf/1602.05629
    For values `v_1, ..., v_k`, and weights `w_1, ..., w_k`, this means
    `sum_{i=1}^k (w_i * v_i) / sum_{i=1}^k w_i`.
    """
    return np.average(np.array(all_client_weights), axis=0, weights=weights)


def get_client_model_weights(worker_model_folder, round_num):
    """
    Returns a list of all downloaded weights for a given training round
    """
    client_weights = []
    for model_weights in glob.glob(f'{worker_model_folder}/round_{round_num}_worker_*[0-9].h5'):
        temp_model = load_model_from_file(f'{model_weights}')
        client_weights.append(temp_model.get_weights())
    return client_weights
