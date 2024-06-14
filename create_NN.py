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
axs[0].set(ylabel="Futures Price USD/EUR", title="Futures Price")

# plot shwowing the volatility path
axs[1].plot(params["data"].index, params["data"].Sigma * 100, color="darkorange")
axs[1].set(ylabel="Volatility in %", title="Volatility")
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

SABR_EC_hedge_OTM = dh(train_pathes=SABR_train.futures_paths, other_train=[SABR_train.vol_paths],
                       ytrain=SABR_train.payoff(K=params["F0"] * 1.1),
                       test_pathes=SABR_test.futures_paths, other_test=[SABR_test.vol_paths],
                       ytest=SABR_test.payoff(K=params["F0"] * 1.1),
                       initial_wealth=np.mean(SABR_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
                       n=300, d=3)

SABR_EC_hedge_OTM2 = dh(train_pathes=SABR_train.futures_paths, other_train=[SABR_train.vol_paths],
                        ytrain=SABR_train.payoff(K=params["F0"] * 1.2),
                        test_pathes=SABR_test.futures_paths, other_test=[SABR_test.vol_paths],
                        ytest=SABR_test.payoff(K=params["F0"] * 1.2),
                        initial_wealth=np.mean(SABR_train.get_price(K=params["F0"], step=0)), actf="tanh",
                        drop_out=None,
                        n=300, d=3)

SABR_EC_hedge_ITM = dh(train_pathes=SABR_train.futures_paths, other_train=[SABR_train.vol_paths],
                       ytrain=SABR_train.payoff(K=params["F0"] * 0.9),
                       test_pathes=SABR_test.futures_paths, other_test=[SABR_test.vol_paths],
                       ytest=SABR_test.payoff(K=params["F0"] * 0.9),
                       initial_wealth=np.mean(SABR_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
                       n=300, d=3)

SABR_EC_hedge_ITM2 = dh(train_pathes=SABR_train.futures_paths, other_train=[SABR_train.vol_paths],
                        ytrain=SABR_train.payoff(K=params["F0"] * 0.8),
                        test_pathes=SABR_test.futures_paths, other_test=[SABR_test.vol_paths],
                        ytest=SABR_test.payoff(K=params["F0"] * 0.8),
                        initial_wealth=np.mean(SABR_train.get_price(K=params["F0"], step=0)), actf="tanh",
                        drop_out=None,
                        n=300, d=3)

SABR_EC_hedge.summary("h")

# uncomment if you want to retrain the model
# SABR_EC_hedge.train(batch_size=500, epochs=20, learning_rate=0.0001)
# SABR_EC_hedge_OTM.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp_otm.weights.h5")
# SABR_EC_hedge_OTM2.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp_otm2.weights.h5")
# SABR_EC_hedge_ITM.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp_itm.weights.h5")
# SABR_EC_hedge_ITM2.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp_itm2.weights.h5")

SABR_EC_hedge.load_weights()
SABR_EC_hedge_OTM.load_weights(cp_path="cp_otm.weights.h5")
SABR_EC_hedge_OTM2.load_weights(cp_path="cp_otm2.weights.h5")
SABR_EC_hedge_ITM.load_weights(cp_path="cp_itm.weights.h5")
SABR_EC_hedge_ITM2.load_weights(cp_path="cp_itm2.weights.h5")


def performance_summary(NN, test_paths, strike):
    # Loss Statistics
    NN.loss_test()
    test_paths.performance(K=strike)

    # Terminal wealth - payoffs
    payoffs = test_paths.payoff(K=strike)
    wealth_NN = NN.model_wealth.predict(x=NN.xtest)[:, 0] - payoffs
    wealth_BS = test_paths.wealth_BS - payoffs
    wealth_SABR = test_paths.wealth_SABR - payoffs
    wealth_nothing = test_paths.wealth_nothing - payoffs

    # plotting wealth distributions:
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(wealth_nothing)
    axs[0, 0].set_title("Nothing")
    axs[0, 1].hist(wealth_NN)
    axs[0, 1].set_title("NN")
    axs[1, 0].hist(wealth_BS)
    axs[1, 0].set_title("BS")
    axs[1, 1].hist(wealth_SABR)
    axs[1, 1].set_title("SABR")

    for ax in axs.flat:
        ax.set(xlabel="terminal wealth - payoffs")

    plt.show()

    # Risk measures
    quantiles = np.array([0.001, 0.01, 0.025, 0.05])
    VaRs = {
        "Nothing": np.quantile(wealth_nothing, quantiles),
        "BS": np.quantile(wealth_BS, quantiles),
        "SABR": np.quantile(wealth_SABR, quantiles),
        "NN": np.quantile(wealth_NN, quantiles)
    }

    def cvar(wealth, VaRs):
        cvars = np.zeros(len(VaRs))
        tw = wealth
        for i in range(len(VaRs)):
            cvars[i] = tw[tw < VaRs[i]].mean()
        return cvars

    CVaRs = {
        "Nothing": cvar(wealth_nothing, VaRs["Nothing"]),
        "BS": cvar(wealth_BS, VaRs["BS"]),
        "SABR": cvar(wealth_SABR, VaRs["SABR"]),
        "NN": cvar(wealth_NN, VaRs["NN"])
    }

    VaRs = pd.concat([pd.DataFrame(VaRs), pd.Series(1 - quantiles).rename("Quantiles")], axis=1).set_index("Quantiles")
    CVaRs = pd.concat([pd.DataFrame(CVaRs), pd.Series(1 - quantiles).rename("Quantiles")], axis=1).set_index(
        "Quantiles")

    print(VaRs)
    print(CVaRs)

    VaRs.to_latex()
    CVaRs.to_latex()


# ATM
print("ATM")
performance_summary(SABR_EC_hedge, SABR_test, params["F0"])

# OTM
# print("OTM")
# performance_summary(SABR_EC_hedge_OTM, SABR_test, params["F0"] * 1.1)

# OTM 2
print("OTM 2")
performance_summary(SABR_EC_hedge_OTM2, SABR_test, params["F0"] * 1.2)

# ITM
# print("ITM")
# performance_summary(SABR_EC_hedge_ITM, SABR_test, params["F0"] * 0.9)

# ITM 2
# print("ITM 2")
# performance_summary(SABR_EC_hedge_ITM2, SABR_test, params["F0"] * 0.8)
