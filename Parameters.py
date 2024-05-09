import zipfile
from datetime import datetime
import pandas as pd

# Yield curve spot rate, 1-year maturity - Government bond, nominal, all issuers whose rating is triple A - Euro area
# source: https://data.ecb.europa.eu/data/datasets/YC/YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y
r_eur = 0.02518973
d_eur = 1 / (1 + r_eur)  # discount factor EUR
# Market Yield on U.S. Treasury Securities at 1-Year Constant Maturity, Quoted on an Investment Basis (DGS1)
# source: https://fred.stlouisfed.org/series/DGS1
r_usd = 0.0472
d_usd = 1 / (1 + r_usd)  # discount factor USD

# Read in tick-data from zip file
eurusd = pd.read_csv(zipfile.ZipFile("HISTDATA_COM_ASCII_EURUSD_M12023.zip").open("DAT_ASCII_EURUSD_M1_2023.csv"),
                     header=None, delimiter=";")

# Data transformation:
eurusd.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
eurusd.DateTime = eurusd.DateTime.apply(lambda x: datetime.strptime(x, '%Y%m%d %H%M%S'))

# forward data
eurusd_fw = eurusd.loc[:, "DateTime":"Close"]
# TODO: figure out discounting ;)
