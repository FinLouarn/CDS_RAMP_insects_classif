import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, updates, init, objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import EarlyStopping
import numpy as np

''' class EarlyStopping(object):

    def __init__(self, patience=100, criterion='valid_loss',
                 criterion_smaller_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.criterion = criterion
        self.criterion_smaller_is_better = criterion_smaller_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.criterion]
        current_epoch = train_history[-1]['epoch']
        if self.criterion_smaller_is_better:
            cond = current_valid < self.best_valid
        else:
            cond = current_valid > self.best_valid
        if cond:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if nn.verbose:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            if nn.verbose:
                print("Weights set.")
            raise StopIteration()

    def load_best_weights(self, nn, train_history):
        nn.load_weights_from(self.best_weights)
'''


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
#            ('pool1', layers.MaxPool2DLayer),
#            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
#            ('pool3', layers.MaxPool2DLayer),
#            ('conv4', layers.Conv2DLayer),
            ('pool4', layers.MaxPool2DLayer),
            ('conv5', layers.Conv2DLayer),
            ('pool5', layers.MaxPool2DLayer),
            ('hidden6', layers.DenseLayer),
            ('dropout6', layers.DropoutLayer),
            ('hidden7', layers.DenseLayer),
            ('dropout7', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44), # 64 to 44 because cropping input images with 10:54 x 10:54
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=32, conv1_filter_size=(5, 5),
#    pool1_pool_size=(1, 1),
#    conv2_num_filters=32, conv2_filter_size=(5, 5),
    pool2_pool_size=(2, 2),
    conv3_num_filters=64, conv3_filter_size=(3, 3),
#    pool3_pool_size=(1, 1),
#    conv4_num_filters=64, conv4_filter_size=(3, 3),
    pool4_pool_size=(2, 2),
    conv5_num_filters=128, conv5_filter_size=(2, 2),
    pool5_pool_size=(2, 2),
    hidden6_num_units=300, dropout6_p=0.5, hidden6_nonlinearity = nonlinearities.leaky_rectify,
    #hidden6_regularization = regularization.l1,
    hidden7_num_units=300, dropout7_p=0.5, hidden7_nonlinearity = nonlinearities.leaky_rectify,
    #hidden7_regularization = regularization.l2,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    #update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=30,

    # handlers
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_accuracy', criterion_smaller_is_better=False)]
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / X.max())
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X

    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
