import tensorflow as tf
import json
from collections import namedtuple


DATASET_PATH = '/golem/dataset/mnist.npz'
SPECS_FILE = '/golem/work/specs.json'


def get_train_dataset(start, end, batch_size):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(DATASET_PATH)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train[start:end], y_train[start:end])).batch(batch_size)
    return train_dataset


def load_model_from_file(file_path):
    return tf.keras.models.load_model(file_path)


def main():
    specs = json.load(open(SPECS_FILE, 'r'))
    specs = namedtuple('RoundSpecs', specs.keys())(*specs.values())
    training_dataset = get_train_dataset(
        specs.start, specs.end, specs.batch_size)
    model = load_model_from_file(specs.model_path)
    train_history = model.fit(training_dataset, epochs=specs.epochs)
    model.save(f'/golem/output/model_round_{specs.global_round}_{specs.node_number}.h5')
    with open(f'/golem/output/log_round_{specs.global_round}_{specs.node_number}.json', 'w') as log_file:
        log_file.write(json.dumps(train_history.history))


if __name__ == "__main__":
    main()
