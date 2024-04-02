import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Subtract, Multiply, Add, Dot, Concatenate
from keras.models import Sequential, Model


class Deep_Hedge:
    steps = 100  # time discretization
    m = 1  # dimension of price
    d = 3  # number of layers
    n = 32  # nodes in the first but last layer
    xtrain = None
    ytrain = None
    xtest = None
    ytest = None
    activation = "tanh"  # activation function for layers
    model_hedge = None  # neural network model

    def __init__(self, steps, m, d, n, actf, xtrain, ytrain, xtest, ytest):
        # Initializer
        self.steps = steps
        self.m = m
        self.d = d
        self.n = n
        self.activation = actf
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        # Neural Network Architecture:
        layers = []
        for j in range(self.steps):
            for i in range(self.d):
                if i < d - 1:
                    layer = Dense(n, activation=self.activation, trainable=True)
                else:
                    layer = Dense(m, activation="linear", trainable=True)
                layers = layers + [layer]
        # Loss Function:
        price = Input(shape=(self.m,))
        inputs = [price]
        strategy = None
        for j in range(self.steps):
            strategy = price
            for k in range(self.d):
                strategy = layers[k + (j) * d](strategy)
        outputs = strategy
        self.model_hedge = Model(inputs=inputs, outputs=outputs)

    def model_summary(self):
        self.model_hedge.summary()

    def fit(self, lf, opt, epochs, batch_size):
        self.model_hedge.compile(optimizer=opt, loss=lf)
        self.model_hedge.fit(x=self.xtrain, y=self.ytrain, epochs=epochs, batch_size=batch_size)

    def predict(self):
        self.model_hedge.predict(self.xtest)