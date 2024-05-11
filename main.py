import scipy as sp
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Subtract, Multiply, Add, Dot, Concatenate
from keras.models import Sequential, Model
from SABR import SABR_model as path
from DeepHedge import Deep_Hedge as dh
from Parameters import get_parameters

seed_train = 10538  # seed for training
seed_test = 711  # seed for testing
params = get_parameters()

SABR_train = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r"],
                  params["steps"], 10000, params["T"], seed_train)
SABR_test = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r"],
                 params["steps"], 10000, params["T"], seed_test)

SABR_EC_hedge = dh(train_pathes=SABR_train.futures_paths, ytrain=SABR_train.payoff(K=150),
                   test_pathes=SABR_test.futures_paths, ytest=SABR_test.payoff(K=150),
                   initial_wealth=np.mean(SABR_train.get_price(K=150, step=0)))
SABR_EC_hedge.train()
SABR_EC_hedge.loss_test()

SABR_EC_hedge.save("EC_wealth.keras", "EC_hedge.keras")
