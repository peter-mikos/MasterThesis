import numpy as np
import zipfile
import datetime as dt
import pandas as pd


def get_parameters(r_tar, r_base, zips, files):
    # Day Count Convention: ACT/ACT
    # 2023 had 365 days
    def year_frac(start, end, basis=365):
        return (end - start).days / basis

    def discount_factor(YF, r):
        return 1 / ((1 + r) ** YF)

    # Read in tick-data from zip file
    basetar = pd.concat(
        [
            pd.read_csv(zipfile.ZipFile("data/" + zips[0]).open(files[0]),
                        header=None, delimiter=";"),
            pd.read_csv(
                zipfile.ZipFile("data/" + zips[1]).open(files[1]),
                header=None, delimiter=";")
        ]
    )

    # Data transformation:
    basetar.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
    basetar.DateTime = basetar.DateTime.apply(lambda x: dt.datetime.strptime(x, '%Y%m%d %H%M%S'))
    ind = (basetar.DateTime > dt.datetime(2023, 1, 2, 0, 0)) & (basetar.DateTime < dt.datetime(2024, 1, 2, 23, 59))
    basetar = basetar.set_index(basetar.DateTime)

    # forward data
    basetar = basetar.loc[ind, "DateTime":"Close"]
    basetar["Date"] = basetar.DateTime.apply(lambda x: x.date())
    basetar["YF"] = basetar.DateTime.apply(lambda x: year_frac(x, dt.datetime(2024, 1, 2, 23, 59)))  # year fraction
    basetar["D_EUR"] = basetar.YF.apply(lambda x: discount_factor(x, r_base))
    basetar["D_USD"] = basetar.YF.apply(lambda x: discount_factor(x, r_tar))
    # EUR is the base currency and USD the target currency so the forward price is:
    # Forward = Spot * D_USD / D_EUR
    basetar["Close_Forward"] = basetar.Close * basetar.D_USD / basetar.D_EUR
    basetar_daily = pd.concat(
        [
            basetar.groupby(["Date"])["Close_Forward"].apply(lambda x: x[-1]),
            basetar.groupby(["Date"])["Close_Forward"].apply(lambda x: x.std() * np.sqrt(24 * 60)),
            basetar.groupby(["Date"])["YF"].apply(lambda x: 1 - x.max())
        ],
        axis=1
    )
    basetar_daily.columns = ["F", "Sigma", "t"]

    basetar_daily["Sigma"] = basetar_daily["Sigma"] / basetar_daily.reset_index().set_index("Date", drop=False)[
        "Date"].diff().transform(lambda x: np.sqrt(x.days)).fillna(1)

    F0 = basetar_daily.F[0]  # initial futures value
    alpha = basetar_daily.Sigma[0]  # initial volatility value
    beta = 1  # shape parameter
    rho = basetar_daily[["F", "Sigma"]].corr().loc["F", "Sigma"]  # correlation between BMs
    nu = basetar_daily.Sigma.std()  # volvol
    # r = (eurusd.D_EUR[0] / eurusd.D_USD[0]) - 1  # interest rate
    steps = 365  # number of time steps
    T = 1  # time of maturity

    return {
        "F0": F0,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "nu": nu,
        "r_tar": r_tar,
        "r_base": r_base,
        "steps": steps,
        "T": T,
        "data": basetar_daily
    }
