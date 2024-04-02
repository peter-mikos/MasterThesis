import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense, Subtract, Multiply, Add, Dot, Concatenate
from keras.models import Sequential, Model
from SABR import SABR_model as path

# TODO: determine these dynamically from data
seed = 10538  # seed for replicability
F0 = 150  # initial futures value
alpha = 0.2  # initial volatility value
beta = 0.5  # shape parameter
rho = 0.5  # correlation between BMs
nu = 0.2  # volvol
r = 0.03  # interest rate
steps = 100  # number of time steps
N = 10000  # number of simulated paths
T = 1  # time of maturity

paths_SABR = path(F0, alpha, beta, rho, nu, r, steps, N, T, seed)
print(paths_SABR.get_price(150, 50))