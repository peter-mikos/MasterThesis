import numpy as np
import pandas as pd
from SABR import SABR_model as path
from DeepHedge import Deep_Hedge as dh
from Parameters import get_parameters
import matplotlib.pyplot as plt

seed_train = 10538  # seed for training
seed_test = 420  # seed for testing
params = get_parameters()

# plot showing the futures path
fig, axs = plt.subplots(2, 1)
axs[0].plot(params["data"].index, params["data"].F)
axs[0].set(ylabel="Futures Price USD/EUR")

# plot shwowing the volatility path
axs[1].plot(params["data"].index, params["data"].Sigma*100)
axs[1].set(ylabel="Volatility in %")
plt.show()

SABR_train = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r_tar"],
                  params["r_base"], params["steps"], 30000, params["T"], seed_train, voltype="daily")
SABR_test = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r_tar"],
                 params["r_base"], params["steps"], 1000, params["T"], seed_test, voltype="daily")

SABR_EC_hedge = dh(train_pathes=SABR_train.futures_paths, other_train=[SABR_train.vol_paths],
                   ytrain=SABR_train.payoff(K=params["F0"]),
                   test_pathes=SABR_test.futures_paths, other_test=[SABR_test.vol_paths],
                   ytest=SABR_test.payoff(K=params["F0"]),
                   initial_wealth=np.mean(SABR_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
                   n=300, d=3)

SABR_EC_hedge.summary("h")

# uncomment if you want to retrain the model
# SABR_EC_hedge.train(batch_size=500, epochs=20, learning_rate=0.0001)

SABR_EC_hedge.load_weights()
SABR_EC_hedge.loss_test()
SABR_test.performance(K=params["F0"])

payoffs = SABR_test.payoff(K=params["F0"])
wealth_NN = SABR_EC_hedge.model_wealth.predict(x=SABR_EC_hedge.xtest)[:, 0]
wealth_BS = SABR_test.wealth_BS
wealth_SABR = SABR_test.wealth_SABR
wealth_nothing = SABR_test.wealth_nothing

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(wealth_nothing - payoffs)
axs[0, 0].set_title("Nothing")
axs[0, 1].hist(wealth_NN - payoffs)
axs[0, 1].set_title("NN")
axs[1, 0].hist(wealth_BS - payoffs)
axs[1, 0].set_title("BS")
axs[1, 1].hist(wealth_SABR - payoffs)
axs[1, 1].set_title("SABR")

for ax in axs.flat:
    ax.set(xlabel="terminal wealth - payoffs")

plt.show()

quantiles = np.array([0.001, 0.01, 0.025, 0.05])
VaRs = {
    "Nothing": np.quantile(wealth_nothing - payoffs, quantiles),
    "NN": np.quantile(wealth_NN - payoffs, quantiles),
    "BS": np.quantile(wealth_BS - payoffs, quantiles),
    "SABR": np.quantile(wealth_SABR - payoffs, quantiles)
}


def cvar(wealth, payoff, VaRs):
    cvars = np.zeros(len(VaRs))
    tw = wealth - payoff
    for i in range(len(VaRs)):
        cvars[i] = tw[tw < VaRs[i]].mean()
    return cvars


CVaRs = {
    "Nothing": cvar(wealth_nothing, payoffs, VaRs["Nothing"]),
    "NN": cvar(wealth_NN, payoffs, VaRs["NN"]),
    "BS": cvar(wealth_BS, payoffs, VaRs["BS"]),
    "SABR": cvar(wealth_SABR, payoffs, VaRs["SABR"])
}

VaRs = pd.concat([pd.DataFrame(VaRs), pd.Series(1-quantiles).rename("Quantiles")], axis=1).set_index("Quantiles")
CVaRs = pd.concat([pd.DataFrame(CVaRs), pd.Series(1-quantiles).rename("Quantiles")], axis=1).set_index("Quantiles")

print(VaRs)
print(CVaRs)
