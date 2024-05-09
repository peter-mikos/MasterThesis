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

# TODO: determine these dynamically from data
seed = 10538  # seed for replicability
F0 = 150  # initial futures value
alpha = 0.2  # initial volatility value
beta = 0.5  # shape parameter
rho = 0.5  # correlation between BMs
nu = 0.2  # volvol
r = 0.03  # interest rate
steps = 100  # number of time steps
T = 2  # time of maturity

SABR_train = path(F0, alpha, beta, rho, nu, r, steps, 10000, T, seed)
SABR_test = path(F0, alpha, beta, rho, nu, r, steps, 1000, T, seed)

SABR_EC_hedge = dh(train_pathes=SABR_train.futures_paths, ytrain=SABR_train.payoff(K=150), test_pathes=SABR_test.futures_paths, ytest=SABR_test.payoff(K=150), initial_wealth=np.mean(SABR_train.get_price(K=150, step=0)))
SABR_EC_hedge.train()
SABR_EC_hedge.loss_test()

SABR_EC_hedge.save("EC_wealth.keras", "EC_hedge.keras")