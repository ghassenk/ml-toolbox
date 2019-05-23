import tensorflow as tf


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def predict_input_fn(features, batch_size):
    """An input function for training"""

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features)))
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def tensorflow_nn(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

    return model


def tensorflow_dnn(tf_feature_columns, train_x, train_y, x_test, y_test, batch_size=10, train_steps=100):
    # Check if dnn can be improved by changing architecture
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=tf_feature_columns,
        # ? hidden layers of ? nodes each.
        hidden_units=[10, 10],
        # The model must choose between 2 classes.
        n_classes=2)
    # Train
    dnn_classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=train_steps)
    # Eval
    eval_dnn = dnn_classifier.evaluate(input_fn=lambda: eval_input_fn(x_test, y_test, batch_size))
    print('\nDNNClassifier Test set accuracy: {accuracy:0.3f}\n'.format(**eval_dnn))

    return dnn_classifier


def tensorflow_linear(tf_feature_columns, train_x, train_y, x_test, y_test, batch_size=10, train_steps=100):
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=tf_feature_columns)

    # Train
    linear_classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=train_steps)

    # Eval
    eval_linear = linear_classifier.evaluate(input_fn=lambda: eval_input_fn(x_test, y_test, batch_size))
    print('LinearClassifier Test set accuracy: {accuracy:0.3f}\n'.format(**eval_linear))

    return linear_classifier


def predict_y(classifier, predict_x, batch_size=10):
    predictions = classifier.predict(input_fn=lambda: predict_input_fn(predict_x, batch_size))

    predictions_y = []
    predictions_y_probabilities = []
    for p in predictions:
        predictions_y_probabilities.append(p['probabilities'].max())
        predictions_y.append(p['class_ids'][0])

    return predictions_y, predictions_y_probabilities


def aggregate_predictions(predictions_y_1, predictions_y_2, probabilities_1, probabilities_2):
    i = 0
    agg_predictions = []
    agg_probabilities = []
    for _ in predictions_y_1:
        if probabilities_1[i] >= probabilities_2[i]:
            agg_predictions.append(predictions_y_1[i])
            agg_probabilities.append(probabilities_1[i])
        else:
            agg_predictions.append(predictions_y_2[i])
            agg_probabilities.append(probabilities_2[i])
        i += 1

    print('dnn: \t', probabilities_1)
    print('lin: \t', probabilities_2)
    print('agg: \t', agg_probabilities)
    print('agg: \t', agg_predictions)

    return aggregate_predictions, agg_probabilities


# **************************** future *************************** #
class MLearner:
    def train(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass


class LinearRegressionLearner(MLearner):
    def __init__(self):
        pass


class KnnLearner(MLearner):
    k = 1

    def __init__(self, k):
        self.k = k


class DecisionTreeLearner(MLearner):
    def __init__(self):
        pass


class SvmLearner(MLearner):
    def __init__(self):
        pass


class MixedEnsembleLearner(MLearner):
    """ Create instances of some learners (different kinds and/or same kind with different params) and use their
    averaged predictions"""

    def __init__(self):
        pass


class BaggingEnsembleLearner(MLearner):
    """ Use bagging (also called bootstrap aggregating) to create an ensemble of learners """

    def __init__(self):
        pass


class AdaBoostEnsembleLearner(MLearner):
    """ Use bagging choosing bags according to previous errors """

    def __init__(self):
        pass
