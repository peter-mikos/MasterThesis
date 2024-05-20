import tensorflow as tf
from SABR import SABR_model as path
from DeepHedge import Deep_Hedge as dh
from Parameters import get_parameters

# load European Call NN
wealth_NN = tf.keras.models.load_model("EC_wealth.keras")
hedge_NN = tf.keras.models.load_model("EC_hedge.keras")
