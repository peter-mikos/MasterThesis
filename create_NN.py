import json
import numpy as np
import pandas as pd
from SABR import SABR_model as path
from DeepHedge import Deep_Hedge as dh
from Parameters import get_parameters
import matplotlib.pyplot as plt

seed_train = 10538  # seed for training
seed_test = 420  # seed for testing
load = True  # should NN-weights be loaded or should they be retrained


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
        atm.load_weights(cp_path="weights/cp_" + name + "_atm" + ".weights.h5")
        otm.load_weights(cp_path="weights/cp_" + name + "_otm" + ".weights.h5")
        otm2.load_weights(cp_path="weights/cp_" + name + "_otm2" + ".weights.h5")
        itm.load_weights(cp_path="weights/cp_" + name + "_itm" + ".weights.h5")
        itm2.load_weights(cp_path="weights/cp_" + name + "_itm2" + ".weights.h5")
    else:
        atm.train(batch_size=500, epochs=20, learning_rate=0.0001,
                  cp_path="weights/cp_" + name + "_atm" + ".weights.h5")
        otm.train(batch_size=500, epochs=20, learning_rate=0.0001,
                  cp_path="weights/cp_" + name + "_otm" + ".weights.h5")
        otm2.train(batch_size=500, epochs=20, learning_rate=0.0001,
                   cp_path="weights/cp_" + name + "_otm2" + ".weights.h5")
        itm.train(batch_size=500, epochs=20, learning_rate=0.0001,
                  cp_path="weights/cp_" + name + "_itm" + ".weights.h5")
        itm2.train(batch_size=500, epochs=20, learning_rate=0.0001,
                   cp_path="weights/cp_" + name + "_itm2" + ".weights.h5")

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

# Yield curve spot rate, 1-year maturity - Government bond, nominal, all issuers whose rating is triple A - Euro area
# source: https://data.ecb.europa.eu/data/datasets/YC/YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y
r_eur = 0.02518973
# Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity, Quoted on an Investment Basis (DGS1)
# source: https://fred.stlouisfed.org/series/DGS1
r_usd = 0.0472
# Australia 1 Year Government Bond
# source: https://www.marketwatch.com/investing/bond/tmbmkau-01y?countrycode=bx
r_aud = 0.03361
# U.K. 1 Year Gilt
# source: https://www.marketwatch.com/investing/bond/tmbmkgb-01y?countrycode=bx
r_gbp = 0.03326
# New Zealand - 1 Year Government Bond Yield
# source: https://www.worldgovernmentbonds.com/bond-historical-data/new-zealand/1-year/#title-historical-data
r_nzd = 0.0515
# Canada 1 Year Treasury Bill Yield
# source: https://ycharts.com/indicators/canada_1_year_treasury_bill_yield#:~:text=Canada%201%20Year%20Treasury%20Bill%20Yield%20is%20at%204.37%25%2C%20compared,a%20maturity%20of%201%20year.
r_cad = 0.0448
# Switzerland - 1 Year Government Bond Yield
# source: https://www.worldgovernmentbonds.com/bond-historical-data/switzerland/1-year/#title-historical-data
r_chf = 0.01503
# Switzerland - 1 Year Government Bond Yield
# source: https://www.worldgovernmentbonds.com/bond-historical-data/hong-kong/1-year/#title-historical-data
r_hkd = 0.04151
# Japan 1 Year Government Bond
# source: https://www.marketwatch.com/investing/bond/tmbmkjp-01y?countrycode=jp
r_jpy = -0.0001
# Switzerland - 1 Year Government Bond Yield
# source: https://www.worldgovernmentbonds.com/bond-historical-data/norway/1-year/
r_nok = 0.03243

params_usd_eur = get_parameters(
    r_tar=r_usd,
    r_base=r_eur,
    zips=["HISTDATA_COM_ASCII_EURUSD_M12023.zip", "HISTDATA_COM_ASCII_EURUSD_M1202401.zip"],
    files=["DAT_ASCII_EURUSD_M1_2023.csv", "DAT_ASCII_EURUSD_M1_202401.csv"]
)
params_usd_aud = get_parameters(
    r_tar=r_usd,
    r_base=r_eur,
    zips=["HISTDATA_COM_ASCII_AUDUSD_M12023.zip", "HISTDATA_COM_ASCII_AUDUSD_M1202401.zip"],
    files=["DAT_ASCII_AUDUSD_M1_2023.csv", "DAT_ASCII_AUDUSD_M1_202401.csv"]
)
params_usd_gbp = get_parameters(
    r_tar=r_usd,
    r_base=r_gbp,
    zips=["HISTDATA_COM_ASCII_GBPUSD_M12023.zip", "HISTDATA_COM_ASCII_GBPUSD_M1202401.zip"],
    files=["DAT_ASCII_GBPUSD_M1_2023.csv", "DAT_ASCII_GBPUSD_M1_202401.csv"]
)
params_usd_nzd = get_parameters(
    r_tar=r_usd,
    r_base=r_nzd,
    zips=["HISTDATA_COM_ASCII_NZDUSD_M12023.zip", "HISTDATA_COM_ASCII_NZDUSD_M1202401.zip"],
    files=["DAT_ASCII_NZDUSD_M1_2023.csv", "DAT_ASCII_NZDUSD_M1_202401.csv"]
)
params_cad_usd = get_parameters(
    r_tar=r_cad,
    r_base=r_usd,
    zips=["HISTDATA_COM_ASCII_USDCAD_M12023.zip", "HISTDATA_COM_ASCII_USDCAD_M1202401.zip"],
    files=["DAT_ASCII_USDCAD_M1_2023.csv", "DAT_ASCII_USDCAD_M1_202401.csv"]
)
params_chf_usd = get_parameters(
    r_tar=r_chf,
    r_base=r_usd,
    zips=["HISTDATA_COM_ASCII_USDCHF_M12023.zip", "HISTDATA_COM_ASCII_USDCHF_M1202401.zip"],
    files=["DAT_ASCII_USDCHF_M1_2023.csv", "DAT_ASCII_USDCHF_M1_202401.csv"]
)
params_hkd_usd = get_parameters(
    r_tar=r_hkd,
    r_base=r_usd,
    zips=["HISTDATA_COM_ASCII_USDHKD_M12023.zip", "HISTDATA_COM_ASCII_USDHKD_M1202401.zip"],
    files=["DAT_ASCII_USDHKD_M1_2023.csv", "DAT_ASCII_USDHKD_M1_202401.csv"]
)
params_jpy_usd = get_parameters(
    r_tar=r_jpy,
    r_base=r_usd,
    zips=["HISTDATA_COM_ASCII_USDJPY_M12023.zip", "HISTDATA_COM_ASCII_USDJPY_M1202401.zip"],
    files=["DAT_ASCII_USDJPY_M1_2023.csv", "DAT_ASCII_USDJPY_M1_202401.csv"]
)
params_nok_usd = get_parameters(
    r_tar=r_nok,
    r_base=r_usd,
    zips=["HISTDATA_COM_ASCII_USDNOK_M12023.zip", "HISTDATA_COM_ASCII_USDNOK_M1202401.zip"],
    files=["DAT_ASCII_USDNOK_M1_2023.csv", "DAT_ASCII_USDNOK_M1_202401.csv"]
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

print("USD_EUR Done!")

##### USD/AUD ##################################################################

# training/loading neural networks
nns_usd_aud = train_networks(
    params=params_usd_aud,
    name="USD_AUD",
    load=load
)

print("USD_AUD Done!")

##### USD/GBP ##################################################################

# training/loading neural networks
nns_usd_gbp = train_networks(
    params=params_usd_gbp,
    name="USD_GBP",
    load=load
)

print("USD_GBP Done!")

##### USD/NZD ##################################################################

# training/loading neural networks
nns_usd_nzd = train_networks(
    params=params_usd_nzd,
    name="USD_NZD",
    load=load
)

print("USD_NZD Done!")

##### CAD/USD ##################################################################

# training/loading neural networks
nns_cad_usd = train_networks(
    params=params_cad_usd,
    name="CAD_USD",
    load=load
)

print("CAD_USD Done!")

##### CHF/USD ##################################################################

# training/loading neural networks
nns_chf_usd = train_networks(
    params=params_chf_usd,
    name="CHF_USD",
    load=load
)

print("CHF_USD Done!")

##### HKD/USD ##################################################################

# training/loading neural networks
nns_hkd_usd = train_networks(
    params=params_hkd_usd,
    name="HKD_USD",
    load=load
)

print("HKD_USD Done!")

##### JPY/USD ##################################################################

# training/loading neural networks
# nns_jpy_usd = train_networks(
#     params=params_jpy_usd,
#     name="JPY_USD",
#     load=False
# )

# print("JPY_USD Done!")   # TODO

##### NOK/USD ##################################################################

# training/loading neural networks
nns_nok_usd = train_networks(
    params=params_nok_usd,
    name="NOK_USD",
    load=load
)

print("NOK_USD Done!")

################################################################################
##### NETWORKS #################################################################
################################################################################

performance_usd_eur = performance_summaries(nns_usd_eur)
performance_usd_aud = performance_summaries(nns_usd_aud)
performance_usd_gbp = performance_summaries(nns_usd_gbp)
performance_usd_nzd = performance_summaries(nns_usd_nzd)
performance_cad_usd = performance_summaries(nns_cad_usd)
performance_chf_usd = performance_summaries(nns_chf_usd)
performance_hkd_usd = performance_summaries(nns_hkd_usd)
# performance_jpy_usd = performance_summaries(nns_jpy_usd)
performance_nok_usd = performance_summaries(nns_nok_usd)

performance = {
    "EUR": performance_usd_eur,
    "AUD": performance_usd_aud,
    "GBP": performance_usd_gbp,
    "NZD": performance_usd_nzd,
    "CAD": performance_cad_usd,
    "CHF": performance_chf_usd,
    "HKD": performance_hkd_usd,
    "NOK": performance_nok_usd
}
