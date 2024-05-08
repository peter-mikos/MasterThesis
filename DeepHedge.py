import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, Subtract, Multiply, Add, Dot, Concatenate, TimeDistributed, Lambda
from keras.models import Sequential, Model


class Deep_Hedge:
    steps = 100  # time discretization
    o = 0  # dimension of observations - 1 for the time dimension
    m = 1  # dimension of price
    d = 2  # number of layers
    n = 200  # nodes in the first but last layer
    T = 1
    initial_wealth = None
    xtrain = None
    ytrain = None
    xtest = None
    ytest = None
    activation = "tanh"  # activation function for layers
    model_wealth = None  # wealth neural network
    model_hedge = None  # hedge neural network

    def __init__(self, train_pathes, ytrain, test_pathes, ytest, initial_wealth, other_train=None, other_test=None,
                 steps=100, m=1,
                 d=2, n=200, actf="tanh", T=1):
        # Initializer
        self.steps = steps
        self.m = m
        self.T = T
        if type(other_train) != type(None):
            self.o = len(other_train)
        self.d = d
        self.n = n
        self.other_train = other_train
        self.other_test = other_test
        self.activation = actf
        self.train_pathes = train_pathes
        self.ytrain = ytrain
        self.test_pathes = test_pathes
        self.ytest = ytest
        self.initial_wealth = initial_wealth
        self.prepare_inputs()
        self.create_model()

    def prepare_inputs(self):
        def shape_inputs(pathes, other):
            def TimeConv(t):
                return self.T - t

            K, N = np.shape(pathes)
            N = N - 1
            TimePoints = np.linspace(0, self.T, N + 1)
            x = [np.zeros((K, N, self.o + 1 + self.m))] + [np.zeros((K, N, self.m))]
            for i in range(N):  # time points
                x[0][:, i, 0] = np.repeat(TimeConv(TimePoints[i]), K)
            for j in range(self.o):  # additional inputs
                x[0][:, :, j + 1] = other[j].transpose()[:, 0:N]
            x[0][:, :, self.o + 1] = pathes[:, 0:N]
            x[1][:, :, 0] = pathes[:, 1:(N + 1)] - pathes[:, 0:N]
            return x

        # transposing inputs necessary
        self.xtrain = shape_inputs(self.train_pathes.transpose(), self.other_train)
        self.xtest = shape_inputs(self.test_pathes.transpose(), self.other_test)

    def create_model(self):
        # If no initial wealth is provided the NN will figure it out itself:
        trainable = False
        if type(None) == type(self.initial_wealth):
            trainable = True

        # neural network for initial wealth
        d_V0 = 0  ## Number of hidden layers
        price0 = Input(shape=(self.m,))
        V0 = price0
        for i in range(d_V0):
            V0 = Dense(1, activation=self.activation)(V0)
        V0 = Dense(1, activation='linear', trainable=trainable)(V0)
        pi = Model(inputs=price0, outputs=V0)

        # architecture of the hedging network
        time_price = Input(shape=(self.o + 1 + self.m,))
        output = time_price
        for i in range(self.d):
            output = Dense(self.n, activation=self.activation)(output)
        output = Dense(self.m, activation='linear')(output)
        hedge = Model(inputs=time_price, outputs=output)
        self.model_hedge = hedge

        # architecture of the wealth network
        Obs = Input(shape=(self.steps, self.o + 1 + self.m))
        Incr = Input(shape=(self.steps, self.m))
        inputs = [Obs, Incr]
        V0 = pi(Obs[:, 0, 1])
        H = TimeDistributed(Lambda(self.model_hedge))(Obs)
        H = Flatten()(H)
        Incr = Flatten()(Incr)
        Gain = Dot(axes=1)([H, Incr])
        wealth = Add()([V0, Gain])

        # Defining the model: inputs -> outputs
        model_wealth = Model(inputs=inputs, outputs=wealth)

        # Set pi network to mean payoff
        V0_train = self.initial_wealth
        weights_new = [np.array([[0]]), np.array([V0_train])]
        pi.set_weights(weights_new)
        self.model_wealth = model_wealth

    def summary(self, tp="w"):
        if tp == "w":
            self.model_wealth.summary()
        elif tp == "h":
            self.model_hedge.summary()

    def train(self, batch_size=200, epochs=20, optimizer="adam", loss='mean_squared_error'):
        self.model_wealth.compile(optimizer=optimizer, loss=loss)
        self.model_wealth.fit(x=self.xtrain, y=self.ytrain, batch_size=batch_size, epochs=epochs)

    def loss_test(self):
        pr = self.model_wealth.predict(x=self.xtest)
        self.test_loss = np.mean((self.ytest - pr) ** 2)
        self.std_err = np.std(self.ytest - pr)
        print("Loss: " + str(self.test_loss) + "\n" +
              "Standard Error: " + str(self.std_err))
