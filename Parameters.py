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

# Day Count Convention: ACT/ACT
days_basis = 365  # 2023 had 365 days

# Read in tick-data from zip file
eurusd = pd.concat(
    [
        pd.read_csv(zipfile.ZipFile("HISTDATA_COM_ASCII_EURUSD_M12023.zip").open("DAT_ASCII_EURUSD_M1_2023.csv"),
                    header=None, delimiter=";"),
        pd.read_csv(zipfile.ZipFile("HISTDATA_COM_ASCII_EURUSD_M1202401.zip").open("DAT_ASCII_EURUSD_M1_202401.csv"),
                    header=None, delimiter=";")
    ]
)

# Data transformation:
eurusd.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
date_format = '%Y%m%d %H%M%S'
eurusd.DateTime = eurusd.DateTime.apply(lambda x: datetime.strptime(x, date_format))
eurusd = eurusd.set_index(eurusd.DateTime)

# forward data
start_time = datetime.strptime("20230102 0000", date_format)
end_time = datetime.strptime("20240102 0000", date_format)
eurusd_fw = eurusd.loc[start_time:end_time, "Close"]

