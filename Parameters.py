import numpy as np
import zipfile
import datetime as dt
import pandas as pd


def get_parameters():
    # Yield curve spot rate, 1-year maturity - Government bond, nominal, all issuers whose rating is triple A - Euro area
    # source: https://data.ecb.europa.eu/data/datasets/YC/YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y
    r_eur = 0.02518973

    # Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity, Quoted on an Investment Basis (DGS1)
    # source: https://fred.stlouisfed.org/series/DGS1
    r_usd = 0.0472

    # Day Count Convention: ACT/ACT
    # 2023 had 365 days
    def year_frac(start, end, basis=365):
        return (end - start).days / basis

    def discount_factor(YF, r):
        return 1 / ((1 + r) ** YF)

    # Read in tick-data from zip file
    eurusd = pd.concat(
        [
            pd.read_csv(zipfile.ZipFile("HISTDATA_COM_ASCII_EURUSD_M12023.zip").open("DAT_ASCII_EURUSD_M1_2023.csv"),
                        header=None, delimiter=";"),
            pd.read_csv(
                zipfile.ZipFile("HISTDATA_COM_ASCII_EURUSD_M1202401.zip").open("DAT_ASCII_EURUSD_M1_202401.csv"),
                header=None, delimiter=";")
        ]
    )

    # Data transformation:
    eurusd.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    eurusd.DateTime = eurusd.DateTime.apply(lambda x: dt.datetime.strptime(x, '%Y%m%d %H%M%S'))
    eurusd = eurusd.set_index(eurusd.DateTime)

    # forward data
    eurusd = eurusd.loc[dt.datetime(2023, 1, 2, 0, 0):dt.datetime(2024, 1, 2, 23, 59), "DateTime":"Close"]
    eurusd["Date"] = eurusd.DateTime.apply(lambda x: x.date())
    eurusd["YF"] = eurusd.DateTime.apply(lambda x: year_frac(x, dt.datetime(2024, 1, 2, 23, 59)))  # year fraction
    eurusd["D_EUR"] = eurusd.YF.apply(lambda x: discount_factor(x, r_eur))
    eurusd["D_USD"] = eurusd.YF.apply(lambda x: discount_factor(x, r_usd))
    # EUR is the base currency and USD the target currency so the forward price is:
    # Forward = Spot * D_USD / D_EUR
    eurusd["Close_Forward"] = eurusd.Close * eurusd.D_USD / eurusd.D_EUR
    eurusd_daily = pd.concat(
        [
            eurusd.groupby(["Date"])["Close_Forward"].apply(lambda x: x[-1]),
            eurusd.groupby(["Date"])["Close_Forward"].apply(lambda x: x.std() * np.sqrt(24 * 60)),
            eurusd.groupby(["Date"])["YF"].apply(lambda x: 1 - x.max())
        ],
        axis=1
    )
    eurusd_daily.columns = ["F", "Sigma", "t"]

    F0 = eurusd_daily.F[0]  # initial futures value
    alpha = eurusd_daily.Sigma[0]  # initial volatility value
    beta = 1  # shape parameter
    rho = eurusd_daily[["F", "Sigma"]].corr().loc["F", "Sigma"]  # correlation between BMs
    nu = eurusd_daily.Sigma.std()  # volvol
    # r = (eurusd.D_EUR[0] / eurusd.D_USD[0]) - 1  # interest rate
    steps = 365  # number of time steps
    T = 1  # time of maturity

    return {
        "F0": F0,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "nu": nu,
        "r_tar": r_usd,
        "r_base": r_eur,
        "steps": steps,
        "T": T,
        "data": eurusd_daily
    }
