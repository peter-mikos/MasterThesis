import numpy as np
from SABR import SABR_model as path
from DeepHedge import Deep_Hedge as dh
from Parameters import get_parameters
import matplotlib.pyplot as plt

seed_train = 10538  # seed for training
seed_test = 711  # seed for testing
params = get_parameters()

# plot showing the futures path
plt.plot(params["data"].t, params["data"].F)
plt.show()

# plot shwowing the volatility path
plt.plot(params["data"].t, params["data"].Sigma)
plt.show()

SABR_train = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r_tar"],
                  params["r_base"], params["steps"], 10000, params["T"], seed_train, voltype="daily")
SABR_test = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r_tar"],
                 params["r_base"], params["steps"], 1000, params["T"], seed_test, voltype="daily")

SABR_EC_hedge = dh(train_pathes=SABR_train.futures_paths, other_train=[SABR_train.vol_paths], ytrain=SABR_train.payoff(K=params["F0"]),
                   test_pathes=SABR_test.futures_paths, other_test=[SABR_test.vol_paths], ytest=SABR_test.payoff(K=params["F0"]),
                   initial_wealth=np.mean(SABR_train.get_price(K=params["F0"], step=0)))
SABR_EC_hedge.train()
SABR_EC_hedge.loss_test()

SABR_EC_hedge.save("EC_wealth.keras", "EC_hedge.keras")
