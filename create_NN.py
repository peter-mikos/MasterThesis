import numpy as np
import pandas as pd
from SABR import SABR_model as path
from DeepHedge import Deep_Hedge as dh
from Parameters import get_parameters
import matplotlib.pyplot as plt

seed_train = 10538  # seed for training
seed_test = 420  # seed for testing
load = False  # should NN-weights be loaded or should they be retrained


def train_networks(params, name, strikes=[0.6, 0.8, 1, 1.2, 1.4], load=True):
    paths_train = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r_tar"],
                       params["r_base"], params["steps"], 30000, params["T"], seed_train, voltype="daily")
    paths_test = path(params["F0"], params["alpha"], params["beta"], params["rho"], params["nu"], params["r_tar"],
                      params["r_base"], params["steps"], 1000, params["T"], seed_train, voltype="daily")

    itm2 = dh(train_pathes=paths_train.futures_paths, other_train=[paths_train.vol_paths],
              ytrain=paths_train.payoff(K=params["F0"] * strikes[0]),
              test_pathes=paths_test.futures_paths, other_test=[paths_test.vol_paths],
              ytest=paths_test.payoff(K=params["F0"] * strikes[0]),
              initial_wealth=np.mean(paths_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
              n=300, d=3)
    itm = dh(train_pathes=paths_train.futures_paths, other_train=[paths_train.vol_paths],
             ytrain=paths_train.payoff(K=params["F0"] * strikes[1]),
             test_pathes=paths_test.futures_paths, other_test=[paths_test.vol_paths],
             ytest=paths_test.payoff(K=params["F0"] * strikes[1]),
             initial_wealth=np.mean(paths_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
             n=300, d=3)
    atm = dh(train_pathes=paths_train.futures_paths, other_train=[paths_train.vol_paths],
             ytrain=paths_train.payoff(K=params["F0"] * strikes[2]),
             test_pathes=paths_test.futures_paths, other_test=[paths_test.vol_paths],
             ytest=paths_test.payoff(K=params["F0"] ** strikes[2]),
             initial_wealth=np.mean(paths_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
             n=300, d=3)
    otm = dh(train_pathes=paths_train.futures_paths, other_train=[paths_train.vol_paths],
             ytrain=paths_train.payoff(K=params["F0"] * strikes[3]),
             test_pathes=paths_test.futures_paths, other_test=[paths_test.vol_paths],
             ytest=paths_test.payoff(K=params["F0"] * strikes[3]),
             initial_wealth=np.mean(paths_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
             n=300, d=3)
    otm2 = dh(train_pathes=paths_train.futures_paths, other_train=[paths_train.vol_paths],
              ytrain=paths_train.payoff(K=params["F0"] * strikes[4]),
              test_pathes=paths_test.futures_paths, other_test=[paths_test.vol_paths],
              ytest=paths_test.payoff(K=params["F0"] * strikes[4]),
              initial_wealth=np.mean(paths_train.get_price(K=params["F0"], step=0)), actf="tanh", drop_out=None,
              n=300, d=3)
    if load:
        atm.load_weights(cp_path="cp/cp_" + name + "_atm" + ".weights.h5")
        otm.load_weights(cp_path="cp/cp_" + name + "_otm" + ".weights.h5")
        otm2.load_weights(cp_path="cp/cp_" + name + "_otm2" + ".weights.h5")
        itm.load_weights(cp_path="cp/cp_" + name + "_itm" + ".weights.h5")
        itm2.load_weights(cp_path="cp/cp_" + name + "_itm2" + ".weights.h5")
    else:
        atm.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp/cp_" + name + "_atm" + ".weights.h5")
        otm.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp/cp_" + name + "_otm" + ".weights.h5")
        otm2.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp/cp_" + name + "_otm2" + ".weights.h5")
        itm.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp/cp_" + name + "_itm" + ".weights.h5")
        itm2.train(batch_size=500, epochs=20, learning_rate=0.0001, cp_path="cp/cp_" + name + "_itm2" + ".weights.h5")

    return {
        "atm": atm,
        "otm": otm,
        "otm2": otm2,
        "itm": itm,
        "itm2": itm2,
        "train": paths_train,
        "test": paths_test,
        "strikes": {
            "itm2": params["F0"] * strikes[0],
            "itm": params["F0"] * strikes[1],
            "atm": params["F0"] * strikes[2],
            "otm": params["F0"] * strikes[3],
            "otm2": params["F0"] * strikes[4]
        }
    }


def performance_summary(NN, test_paths, strike):
    # Loss Statistics
    nn_loss = NN.loss_test()
    model_losses = test_paths.performance(K=strike)

    losses = {
        "Nothing": model_losses["loss_nothing"],
        "BS": model_losses["loss_BS"],
        "SABR": model_losses["loss_SABR"],
        "NN": nn_loss["loss_NN"]
    }

    std_errs = {
        "Nothing": model_losses["std_err_nothing"],
        "BS": model_losses["std_err_BS"],
        "SABR": model_losses["std_err_SABR"],
        "NN": nn_loss["std_err_NN"]
    }

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

    print("VaR:")
    print(VaRs)
    print(VaRs.to_latex())
    print("CVaR:")
    print(CVaRs)
    print(CVaRs.to_latex())

    return {
        "Loss": losses,
        "Standard Error": std_errs,
        "VaR": VaRs,
        "CVaR": CVaRs
    }


def performance_summaries(NNs):
    return {
        "ATM": performance_summary(NNs["atm"], NNs["test"], NNs["strikes"]["atm"]),
        "OTM": performance_summary(NNs["otm"], NNs["test"], NNs["strikes"]["otm"]),
        "OTM2": performance_summary(NNs["otm2"], NNs["test"], NNs["strikes"]["otm2"]),
        "ITM": performance_summary(NNs["itm"], NNs["test"], NNs["strikes"]["itm"]),
        "ITM2": performance_summary(NNs["itm2"], NNs["test"], NNs["strikes"]["itm2"])
    }


################################################################################
##### PARAMETERS ###############################################################
################################################################################

##### USD/EUR ##################################################################

# Yield curve spot rate, 1-year maturity - Government bond, nominal, all issuers whose rating is triple A - Euro area
# source: https://data.ecb.europa.eu/data/datasets/YC/YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y
r_eur = 0.02518973
# Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity, Quoted on an Investment Basis (DGS1)
# source: https://fred.stlouisfed.org/series/DGS1
r_usd = 0.0472

params_usd_eur = get_parameters(
    r_tar=r_usd,
    r_base=r_eur,
    zips=["HISTDATA_COM_ASCII_EURUSD_M12023.zip", "HISTDATA_COM_ASCII_EURUSD_M1202401.zip"],
    files=["DAT_ASCII_EURUSD_M1_2023.csv", "DAT_ASCII_EURUSD_M1_202401.csv"]
)

################################################################################
##### NETWORKS #################################################################
################################################################################

##### USD/EUR ##################################################################

# training/loading neural networks
nns_usd_eur = train_networks(
    params=params_usd_eur,
    name="USD_EUR",
    load=load
)

################################################################################
##### NETWORKS #################################################################
################################################################################

##### USD/EUR ##################################################################

performance_usd_eur = performance_summaries(nns_usd_eur)
