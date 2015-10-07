import os
os.environ["THEANO_FLAGS"] = "device=gpu"
import numpy as np
from sklearn.base import BaseEstimator
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers, nonlinearities, updates, init, objectives
from nolearn.lasagne.handlers import EarlyStopping
from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
from lasagne.regularization import regularize_layer_params, l2, l1

def build_model(hyper_parameters):
    net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 64, 64), # 3 = depth of input layer (color), 64x64 image
    use_label_encoder=True,
    verbose=1,
    **hyper_parameters
    )  
    return net
 
hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(5, 5), pool2_pool_size=(2, 2),
    hidden3_num_units=200,
    dropout3_p=0.5,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    max_epochs=200,
    on_epoch_finished = [EarlyStopping(patience=10, criterion='valid_accuracy', 
                                       criterion_smaller_is_better=False)]
)
 
 
class Classifier(BaseEstimator):
 
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, y)
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)
 
    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)