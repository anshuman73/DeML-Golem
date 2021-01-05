import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_federated as tff


BATCH_SIZE = 20
abs_path = os.path.join(os.getcwd(), 'docker/mnist.npz')
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(abs_path)
X_train, X_test = X_train / 255.0, X_test / 255.0

train_dataset = [tf.data.Dataset.from_tensor_slices(
    (X_train[x:x+10000], y_train[x:x+10000])).batch(BATCH_SIZE) for x in range(0, len(X_train), 10000)]
test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(BATCH_SIZE)


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])


def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=train_dataset[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )


@tf.function
def server_update(model, mean_client_weights):
    """Updates the server model weights as the average of the client model weights."""
    model_weights = model.trainable_variables
    # Assign the mean client weights to the server model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          model_weights, mean_client_weights)
    return model_weights


@tff.tf_computation
def server_init():
    model = model_fn()
    return model.trainable_variables


@tff.federated_computation
def initialize_fn():
    return tff.federated_value(server_init(), tff.SERVER)


dummy_model = model_fn()
tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
model_weights_type = server_init.type_signature.result


@tff.tf_computation(model_weights_type)
def server_update_fn(mean_client_weights):
    model = model_fn()
    return server_update(model, mean_client_weights)


federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)


@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
    print('sw', dir(server_weights))
    # Broadcast the server weights to the clients.
    server_weights_at_client = tff.federated_broadcast(server_weights)
    print(server_weights_at_client)
    # Each client computes their updated weights.
    client_weights = tff.federated_map(
        client_update_fn, (federated_dataset, server_weights_at_client))
    print('weights', client_weights)
    # The server averages these updates.
    mean_client_weights = tff.federated_mean(client_weights)

    # The server updates its model.
    server_weights = tff.federated_map(server_update_fn, mean_client_weights)

    return server_weights


federated_algorithm = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn,
    next_fn=next_fn
)

def evaluate(server_state):
    keras_model = create_keras_model()
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    keras_model.set_weights(server_state)
    keras_model.evaluate(test_dataset)


server_state = federated_algorithm.initialize()
evaluate(server_state)
print(len(server_state))
print(server_state)
# for round in range(100):
#     server_state = federated_algorithm.next(server_state, train_dataset)
#     print(round)
#     evaluate(server_state)
