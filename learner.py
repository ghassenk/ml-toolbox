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
