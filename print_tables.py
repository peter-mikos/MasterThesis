import pickle
import numpy as np
import pandas as pd

performance = pickle.load(open("performance/performance.p", "rb"))
params = pickle.load(open("performance/parameters.p", "rb"))

for CCY in performance.keys():
    for moneyness in performance[CCY].keys():
        table = pd.DataFrame(
            np.array([
                "Loss on real Path",
                "Mean",
                "Standard Error",
                "99.9%-VaR",
                "99%-VaR",
                "97.5%-VaR",
                "95%-VaR",
                "99.9%-CVaR",
                "99%-CVaR",
                "97.5%-CVaR",
                "95%-CVaR"
            ]),
            columns=[CCY + " " + moneyness]
        )
        for name in ["Nothing", "BS", "SABR"]:
            table[name] = pd.Series(np.array([
                -performance[CCY][moneyness]["Real_Loss"][name][0],
                -performance[CCY][moneyness]["Mean"][name],
                performance[CCY][moneyness]["Standard Error"][name],
                -performance[CCY][moneyness]["VaR"][name].iloc[0],
                -performance[CCY][moneyness]["VaR"][name].iloc[1],
                -performance[CCY][moneyness]["VaR"][name].iloc[2],
                -performance[CCY][moneyness]["VaR"][name].iloc[3],
                -performance[CCY][moneyness]["CVaR"][name].iloc[0],
                -performance[CCY][moneyness]["CVaR"][name].iloc[1],
                -performance[CCY][moneyness]["CVaR"][name].iloc[2],
                -performance[CCY][moneyness]["CVaR"][name].iloc[3],
            ]))
        table["NN"] = pd.Series(np.array([
            -performance[CCY][moneyness]["Real_Loss"]["NN"][0, 0],
            -performance[CCY][moneyness]["Mean"]["NN"],
            performance[CCY][moneyness]["Standard Error"]["NN"],
            -performance[CCY][moneyness]["VaR"]["NN"].iloc[0],
            -performance[CCY][moneyness]["VaR"]["NN"].iloc[1],
            -performance[CCY][moneyness]["VaR"]["NN"].iloc[2],
            -performance[CCY][moneyness]["VaR"]["NN"].iloc[3],
            -performance[CCY][moneyness]["CVaR"]["NN"].iloc[0],
            -performance[CCY][moneyness]["CVaR"]["NN"].iloc[1],
            -performance[CCY][moneyness]["CVaR"]["NN"].iloc[2],
            -performance[CCY][moneyness]["CVaR"]["NN"].iloc[3],
        ]))
        tabular = table.to_latex(index=False).replace("%", "\%").replace("99.9\%-VaR", "\midrule\n99.9\%-VaR").replace(
            "99.9\%-CVaR", "\midrule\n99.9\%-CVaR")
        mon = None
        mon_mult = 1
        if moneyness == "OTM2":
            mon = " out of the money "
            mon_mult = 1.4
        elif moneyness == "OTM":
            mon = " out of the money "
            mon_mult = 1.2
        elif moneyness == "ATM":
            mon = " at the money "
        elif moneyness == "ITM2":
            mon = " in the money "
            mon_mult = 0.8
        elif moneyness == "ITM":
            mon = " in the money "
            mon_mult = 0.6
        caption = ("Risk Metrics of the loss distributions from hedging " + CCY +
                   mon + "call options with initial underlying price: " +
                   str(round(params[CCY]["F0"], 4)) + " and strike price: " +
                   str(round(params[CCY]["F0"] * mon_mult, 4)) + ', using the hedging strategies "Nothing"-strategy (Nothing), Black Scholes delta hedging (BS), approximate SABR delta hedging (SABR) and deep hedging (NN).')
        caption = "\n\\caption{" + caption + "}\n"
        label = "\\label{Tab:" + CCY + "_" + moneyness + "}"
        table_latex = "\\begin{table}[h!]\n\\centering\n" + tabular + caption + label + "\n\\end{table}\n\n"
        print(table_latex)
