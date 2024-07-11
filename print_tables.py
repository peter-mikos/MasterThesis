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
                "Test Loss",
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
                performance[CCY][moneyness]["Real_Loss"][name][0],
                performance[CCY][moneyness]["Loss"][name],
                performance[CCY][moneyness]["Standard Error"][name],
                performance[CCY][moneyness]["VaR"][name].iloc[0],
                performance[CCY][moneyness]["VaR"][name].iloc[1],
                performance[CCY][moneyness]["VaR"][name].iloc[2],
                performance[CCY][moneyness]["VaR"][name].iloc[3],
                performance[CCY][moneyness]["CVaR"][name].iloc[0],
                performance[CCY][moneyness]["CVaR"][name].iloc[1],
                performance[CCY][moneyness]["CVaR"][name].iloc[2],
                performance[CCY][moneyness]["CVaR"][name].iloc[3],
            ]))
        table["NN"] = pd.Series(np.array([
            performance[CCY][moneyness]["Real_Loss"]["NN"][0,0],
            performance[CCY][moneyness]["Loss"]["NN"],
            performance[CCY][moneyness]["Standard Error"]["NN"],
            performance[CCY][moneyness]["VaR"]["NN"].iloc[0],
            performance[CCY][moneyness]["VaR"]["NN"].iloc[1],
            performance[CCY][moneyness]["VaR"]["NN"].iloc[2],
            performance[CCY][moneyness]["VaR"]["NN"].iloc[3],
            performance[CCY][moneyness]["CVaR"]["NN"].iloc[0],
            performance[CCY][moneyness]["CVaR"]["NN"].iloc[1],
            performance[CCY][moneyness]["CVaR"]["NN"].iloc[2],
            performance[CCY][moneyness]["CVaR"]["NN"].iloc[3],
        ]))
        print(table.to_latex(index=False).replace("%", "\%").replace("99.9\%-VaR", "\midrule\n99.9\%-VaR").replace("99.9\%-CVaR", "\midrule\n99.9\%-CVaR"))
