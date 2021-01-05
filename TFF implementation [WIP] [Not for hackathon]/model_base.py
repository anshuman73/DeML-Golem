import collections
import attr
import tensorflow as tf
import tensorflow_federated as tff

ModelWeights = collections.namedtuple(
    'ModelWeights', 'trainable non_trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')


class KerasModelWrapper(object):
    """A standalone keras wrapper to be used in TFF."""

    def __init__(self, keras_model, input_spec, loss):
        """A wrapper class that provides necessary API handles for TFF.

        Args:
          keras_model: A `tf.keras.Model` to be trained.
          input_spec: Metadata of dataset that desribes the input tensors, which
            will be converted to `tff.Type` specifying the expected type of input
            and output of the model.
          loss: A `tf.keras.losses.Loss` instance to be used for training.
        """
        self.keras_model = keras_model
        self.input_spec = input_spec
        self.loss = loss

    def forward_pass(self, batch_input, training=True):
        """Forward pass of the model to get loss for a batch of data.

        Args:
          batch_input: A `collections.abc.Mapping` with two keys, `x` for inputs and
            `y` for labels.
          training: Boolean scalar indicating training or inference mode.

        Returns:
          A scalar tf.float32 `tf.Tensor` loss for current batch input.
        """
        preds = self.keras_model(batch_input['x'], training=training)
        loss = self.loss(batch_input['y'], preds)
        return ModelOutputs(loss=loss)

    @property
    def weights(self):
        return ModelWeights(
            trainable=self.keras_model.trainable_variables,
            non_trainable=self.keras_model.non_trainable_variables)

    def from_weights(self, model_weights):
        tff.utils.assign(self.keras_model.trainable_variables,
                         list(model_weights.trainable))
        tff.utils.assign(self.keras_model.non_trainable_variables,
                         list(model_weights.non_trainable))


def keras_evaluate(model, test_data, metric):
    metric.reset_states()
    for batch in test_data:
        preds = model(batch['x'], training=False)
        metric.update_state(y_true=batch['y'], y_pred=preds)
    return metric.result()


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
    """Structure for outputs returned from clients during federated optimization.

    Fields:
    -   `weights_delta`: A dictionary of updates to the model's trainable
        variables.
    -   `client_weight`: Weight to be used in a weighted mean when
        aggregating `weights_delta`.
    -   `model_output`: A structure matching
        `tff.learning.Model.report_local_outputs`, reflecting the results of
        training on the input dataset.
    """
    weights_delta = attr.ib()
    client_weight = attr.ib()
    model_output = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
    """Structure for state on the server.

    Fields:
    -   `model_weights`: A dictionary of model's trainable variables.
    -   `optimizer_state`: Variables of optimizer.
    -   'round_num': Current round index
    """
    model_weights = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
    """Structure for tensors broadcasted by server during federated optimization.

    Fields:
    -   `model_weights`: A dictionary of model's trainable tensors.
    -   `round_num`: Round index to broadcast. We use `round_num` as an example to
            show how to broadcast auxiliary information that can be helpful on
            clients. It is not explicitly used, but can be applied to enable
            learning rate scheduling.
    """
    model_weights = attr.ib()
    round_num = attr.ib()


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
    """Updates `server_state` based on `weights_delta`.

    Args:
      model: A `KerasModelWrapper` or `tff.learning.Model`.
      server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
        creates variables, they must have already been created.
      server_state: A `ServerState`, the state to be updated.
      weights_delta: A nested structure of tensors holding the updates to the
        trainable variables of the model.

    Returns:
      An updated `ServerState`.
    """
    # Initialize the model with the current state.
    model_weights = model.weights
    tff.utils.assign(model_weights, server_state.model_weights)
    tff.utils.assign(server_optimizer.variables(),
                     server_state.optimizer_state)

    # Apply the update to the model.
    grads_and_vars = tf.nest.map_structure(
        lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
        tf.nest.flatten(model_weights.trainable))
    server_optimizer.apply_gradients(grads_and_vars, name='server_update')

    # Create a new state based on the updated model.
    return tff.utils.update_state(
        server_state,
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=server_state.round_num + 1)


@tf.function
def build_server_broadcast_message(server_state):
    """Builds `BroadcastMessage` for broadcasting.

    This method can be used to post-process `ServerState` before broadcasting.
    For example, perform model compression on `ServerState` to obtain a compressed
    state that is sent in a `BroadcastMessage`.

    Args:
      server_state: A `ServerState`.

    Returns:
      A `BroadcastMessage`.
    """
    return BroadcastMessage(
        model_weights=server_state.model_weights,
        round_num=server_state.round_num)


@tf.function
def client_update(model, dataset, server_message, client_optimizer):
    """Performans client local training of `model` on `dataset`.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      server_message: A `BroadcastMessage` from server.
      client_optimizer: A `tf.keras.optimizers.Optimizer`.

    Returns:
      A 'ClientOutput`.
    """
    model_weights = model.weights
    initial_weights = server_message.model_weights
    tff.utils.assign(model_weights, initial_weights)

    num_examples = tf.constant(0, dtype=tf.int32)
    loss_sum = tf.constant(0, dtype=tf.float32)
    # Explicit use `iter` for dataset is a trick that makes TFF more robust in
    # GPU simulation and slightly more performant in the unconventional usage
    # of large number of small datasets.
    for batch in iter(dataset):
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)
        grads = tape.gradient(outputs.loss, model_weights.trainable)
        grads_and_vars = zip(grads, model_weights.trainable)
        client_optimizer.apply_gradients(grads_and_vars)
        batch_size = tf.shape(batch['x'])[0]
        num_examples += batch_size
        loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    client_weight = tf.cast(num_examples, tf.float32)
    return ClientOutput(weights_delta, client_weight, loss_sum / client_weight)


def _initialize_optimizer_vars(model, optimizer):
    """Creates optimizer variables to assign the optimizer's state."""
    model_weights = model.weights
    model_delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)
    # Create zero gradients to force an update that doesn't modify.
    # Force eagerly constructing the optimizer variables. Normally Keras lazily
    # creates the variables on first usage of the optimizer. Optimizers such as
    # Adam, Adagrad, or using momentum need to create a new set of variables shape
    # like the model weights.
    grads_and_vars = tf.nest.map_structure(
        lambda x, v: (tf.zeros_like(x), v), tf.nest.flatten(model_delta),
        tf.nest.flatten(model_weights.trainable))
    optimizer.apply_gradients(grads_and_vars)
    assert optimizer.variables()


def build_federated_averaging_process(
        model_fn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)):
    """Builds the TFF computations for optimization using federated averaging.

    Args:
      model_fn: A no-arg function that returns a
        `simple_fedavg_tf.KerasModelWrapper`.
      server_optimizer_fn: A no-arg function that returns a
        `tf.keras.optimizers.Optimizer` for server update.
      client_optimizer_fn: A no-arg function that returns a
        `tf.keras.optimizers.Optimizer` for client update.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    dummy_model = model_fn()

    @tff.tf_computation
    def server_init_tf():
        model = model_fn()
        server_optimizer = server_optimizer_fn()
        _initialize_optimizer_vars(model, server_optimizer)
        return ServerState(
            model_weights=model.weights,
            optimizer_state=server_optimizer.variables(),
            round_num=0)

    server_state_type = server_init_tf.type_signature.result

    model_weights_type = server_state_type.model_weights

    @tff.tf_computation(server_state_type, model_weights_type.trainable)
    def server_update_fn(server_state, model_delta):
        model = model_fn()
        server_optimizer = server_optimizer_fn()
        _initialize_optimizer_vars(model, server_optimizer)
        return server_update(model, server_optimizer, server_state, model_delta)

    @tff.tf_computation(server_state_type)
    def server_message_fn(server_state):
        return build_server_broadcast_message(server_state)

    server_message_type = server_message_fn.type_signature.result
    tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

    @tff.tf_computation(tf_dataset_type, server_message_type)
    def client_update_fn(tf_dataset, server_message):
        model = model_fn()
        client_optimizer = client_optimizer_fn()
        return client_update(model, tf_dataset, server_message, client_optimizer)

    federated_server_state_type = tff.type_at_server(server_state_type)
    federated_dataset_type = tff.type_at_clients(tf_dataset_type)

    @tff.federated_computation(federated_server_state_type,
                               federated_dataset_type)
    def run_one_round(server_state, federated_dataset):
        """Orchestration logic for one round of computation.

        Args:
          server_state: A `ServerState`.
          federated_dataset: A federated `tf.data.Dataset` with placement
            `tff.CLIENTS`.

        Returns:
          A tuple of updated `ServerState` and `tf.Tensor` of average loss.
        """
        server_message = tff.federated_map(server_message_fn, server_state)
        server_message_at_client = tff.federated_broadcast(server_message)

        client_outputs = tff.federated_map(
            client_update_fn, (federated_dataset, server_message_at_client))

        weight_denom = client_outputs.client_weight
        round_model_delta = tff.federated_mean(
            client_outputs.weights_delta, weight=weight_denom)

        server_state = tff.federated_map(server_update_fn,
                                         (server_state, round_model_delta))
        round_loss_metric = tff.federated_mean(
            client_outputs.model_output, weight=weight_denom)

        return server_state, round_loss_metric

    @tff.federated_computation
    def server_init_tff():
        """Orchestration logic for server model initialization."""
        return tff.federated_value(server_init_tf(), tff.SERVER)

    return tff.templates.IterativeProcess(
        initialize_fn=server_init_tff, next_fn=run_one_round)
