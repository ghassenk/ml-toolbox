import tensorflow as tf


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
