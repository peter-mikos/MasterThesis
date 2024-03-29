import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Subtract, Multiply, Add, Dot, Concatenate

from keras.models import Model

import matplotlib.pyplot as plt

# Trajectories of the Black scholes model
# Let it run to initialize the following parameters, the trajectories
# are not needed afterwards

N = 100  # time disrectization
F0 = np.exp(5)  # initial value of the asset in multiples of np.exp(5)
strike = 150  # strike for the call option
T = 2.0  # maturity
sigma0 = 0.15
nu = 0.1
beta = 1
rho = -0.5
R = 3  # number of shown Trajectories --- not used for training/testing
activator = "tanh"

Ktrain = 20000  # Size of training data
Ktest = 1000  # Size of test data
epochs = 10

## Normalising initial value of S0 to 1 (1 unit of money = price of S0)
# strike = strike/S0
# S0 = 1

#### Time points ####
## If a different vector of length N+1 is used, then the model is generated
## along the given vector
## This could be intresting if one expects that more frequent rebalancing
## is required near (or far away) from maturity
TimePoints = np.linspace(0, T, N + 1)


#### Converter of the time information that is fed into the NN
def TimeConv(t):
    # return np.sqrt(T-t)   ## Works better as input variable in case of diffusion models!
    return T - t  ## The NN expects a time to maturity as information
    # return t  ## The NN expects actual time as information. The results are the same as 'time to maturity'
    ## but the information will be stored differently in the NN.


#### Defining the price model ####
def path_SABR(F0, sigma0, nu, beta, rho, Timepoints, R):
    N = len(Timepoints) - 1
    volpath = np.zeros((N + 1, R)) + sigma0
    futpath = np.zeros((N + 1, R)) + F0

    for i in range(N):
        dt = Timepoints[i + 1] - Timepoints[i]
        dZ = np.random.multivariate_normal(
            mean=np.array([0, 0]),
            cov=np.matrix([[dt, rho * dt], [rho * dt, dt]]),
            size=R
        )
        volpath[i + 1, :] = volpath[i, :] + nu * volpath[i, :] * dZ[:, 0]
        futpath[i + 1, :] = futpath[i, :] + volpath[i, :] * (futpath[i, :] ** beta) * dZ[:, 1]
    return futpath


#### Creating R sample paths from the model ####
F = path_SABR(F0, sigma0, nu, beta, rho, TimePoints, R)
for i in range(R):
    plt.plot(TimePoints, F[:, i])
plt.title("SABR")
plt.show()

#### Defining the payoff as well as implementing the BS-formula for it
import scipy.stats as scipy


# from scipy.stats import norm

def sigma_B(F0, strike, T, alpha, beta, nu, rho):
    z = (nu / alpha) * np.log(F0 / strike) * (F0 * strike) ** ((1 - beta) / 2)

    def x(y):
        log((sqrt(1 - 2 * rho * y + y ** 2) + y - rho) / (1 - rho))

    term1 = alpha / (((F0 * strike) ** ((1 - beta) / 2)) *
                     (1 + T * ((((1 - beta) ** 2) / 24) * (np.log(F0 / strike) ** 2)
                               + (((1 - beta) ** 4) / 1920) * (np.log(F0 / strike) ** 4))))
    term2 = T * (1 + ((((1 - beta) ** 2) * (alpha ** 2)) / (24 * ((F0 * strike) ** (1 - beta)))
                      + 0.25 * rho * beta * nu * alpha / ((F0 * strike) ** ((1 - beta) / 2))
                      + (2 - 3 * (rho ** 2) * (nu ** 2)) / 24))
    return term1 * (z / x(z)) * term2


# Black Scholes price formula (European call)
def BS(F0, strike, T, sigma, call=True):
    d1 = (np.log(F0 / strike) + 0.5 * T * (sigma ** 2)) / (sigma * np.sqrt(T))
    d2 = (np.log(F0 / strike) - 0.5 * T * (sigma ** 2)) / (sigma * np.sqrt(T))
    if (call):
        return np.exp(-T) * (F0 * scipy.norm.cdf(d1) - strike * scipy.norm.cdf(d2))
    else:
        return np.exp(-T) * (strike * scipy.norm.cdf(-d2) - F0 * scipy.norm.cdf(-d1))

def solve_alpha():
    pass #TODO

# SABR price formula
def SABR(F0, strike, T, alpha, beta, nu, rho, call=True):
    return BS(F0, strike, T, sigma_B(F0, strike, T, alpha, beta, nu, rho), call)


def delta_SABR(F0, sigma0, alpha, beta, nu, rho, call=True):
    d1 = (np.log(F0 / strike) + 0.5 * T * (sigma ** 2)) / (sigma * np.sqrt(T))
    # TODO:
    # sigma derivative wrt F0
    # sigma derivative wrt alpha
    # alpha derivative wrt F0

    if (call):
        return scipy.norm.cdf(d1) + F0 * scipy.norm.pdf(d1) * np.sqrt(T)
    else:
        return -scipy.norm.cdf(-d1) + F0 * scipy.norm.pdf(d1) * np.sqrt(T)


# Black Scholes Delta hedge
def deltaBS(F0, strike, T, sigma, call=True):
    d1 = (np.log(F0 / strike) + 0.5 * T * (sigma ** 2)) / (sigma * np.sqrt(T))
    if (call):
        return scipy.norm.cdf(d1)
    else:
        return -scipy.norm.cdf(-d1)


priceBS = BS(F0, strike, T, sigma0)
print('Price of a Call option in the Black scholes model with initial price', F0, 'strike', strike, 'maturity', T,
      'and volatility', sigma0, 'is equal to', BS(F0, strike, T, sigma0))


## payoff [European call, ie  f_T = max( S(T)-strike, 0) ]
def f(S):
    return (np.maximum(S[N, :] - strike, 0))


##
m = 1  # dimension of price

### Definition of neural networks for initial wealth ####
d = 1  ## Number of layers
price0 = Input(shape=(m,))
V0 = price0
for i in range(d):
    V0 = Dense(1, activation='linear')(V0)

pi = Model(inputs=price0, outputs=V0)
pi.summary()

#### Architercure of the network --- Expecting (timeToMaturity,price) vector ####
d = 2  # number of layers in strategy
n = 128  # nodes in the first but last layers

timeprice = Input(shape=(1 + m,))

output = timeprice
for i in range(d - 1):
    output = Dense(n, activation=activator)(output)
output = Dense(m, activation=activator)(output)

hedge = Model(inputs=timeprice, outputs=output)
hedge.summary()

#### Architercure of the wealth network --- expecting a price path
# Reading initial price of the risky asset. 'price' stands for current price.
time = Input(shape=(1,))  ## time is time-to-maturity
price = Input(shape=(m,))  ## price is the vector of prices accross assets
# Recording the initial price to the input struture 'inputs'
inputs = [time, price]

# The starting wealth
wealth = pi(price)

## Creating position size and new wealth from current wealth and new asset price
for j in range(N):
    timenew = Input(shape=(1,))
    pricenew = Input(shape=(m,))  # Reading new price
    inputs = inputs + [timenew, pricenew]  # Writing new price to the inputs record
    priceshift = Subtract()([pricenew, price])  # Price shift

    strategy = hedge(Concatenate()([time, price]))

    ## Calculating trading gains
    if (m == 1):
        gains = Multiply()([strategy, priceshift])  # For multi-asset setting:
    if (m > 1):
        gains = Dot()([strategy, priceshift])

    ## Calculating new wealth
    wealth = Add()([wealth, gains])  ## current wealth + gains = newwealth
    ## Setting the next price to be the current price for the next iteration
    price = pricenew
    time = timenew

## Defining the model: inputs -> outputs
model_wealth = Model(inputs=inputs, outputs=wealth)
model_wealth.compile(optimizer='adam', loss='mean_squared_error')

#### Training the model ####
# xtrain consists of the price flow of the risky asset
trainpathes = path_SABR(F0, sigma0, nu, beta, rho, TimePoints, Ktrain)

## Shaping to input architecture
xtrain = []
for i in range(N + 1):
    TimeToMaturity = np.repeat(TimeConv(TimePoints[i]), Ktrain)
    xtrain = xtrain + [TimeToMaturity, trainpathes[i, :]]

# ytrain is filled with zeros
ytrain = f(trainpathes)
print(ytrain[0:10, ])

maxL = 10
maxiter = 5
for i in range(maxiter):
    model_wealth.fit(x=xtrain, y=ytrain, epochs=epochs, verbose=True, batch_size=100 * (i + 1))  ##
    if model_wealth.evaluate(xtrain, ytrain, batch_size=1000) < maxL:
        break

#### Creating test pathes for evaluation ####
testpathes = path_SABR(F0, sigma0, nu, beta, rho, TimePoints, Ktest)
## Building test pathes to input architecture
xtest = []
for i in range(N + 1):
    TimeToMaturity = np.repeat(TimeConv(TimePoints[i]), Ktest)
    xtest = xtest + [TimeToMaturity, testpathes[i, :]]

#### Visualisation of results ####
# ytrain is filled with zeros
ytest = f(testpathes)  ## Option payoff
model_wealth.evaluate(x=xtest, y=ytest)
ztest = model_wealth.predict(xtest)[:, 0]  ## Terminal wealth NN
difftest = ztest - ytest  ## Error in terminal wealth

V0test = pi.predict(testpathes[0, :])[:, 0]  ## Initial wealth NN
V0correct = BS(testpathes[0, :], strike, T, sigma0)  ## Initial wealth BS
V0diff = np.mean(V0test - V0correct)  ## mean error of initial wealth

print("\n\n[Below]: Scatter plot for payoff/terminal wealth with a diagonal for orientation")
plt.plot(ytest, ztest, 'o')
plt.plot(ytest, ytest)
plt.show()

print("\n\n[Below]: Option payoffs (in blue) vs terminal wealth (in orange))")
plt.plot(testpathes[N,], ytest, 'o')
plt.plot(testpathes[N,], ztest, 'o')
plt.show()

print("\n\n[Below]: Performance of learned hedge (Realised wealth minus option payoff).")
plt.hist(difftest)
plt.show()
print("Standard error (learned):", np.std(difftest))
print("Mean sample error (learned):", np.mean(difftest))

print("\n\nSetup cost (BS):", V0correct[0])
print("Setup cost (NN):", V0test[0])
print("Setup cost (MC-EMM):", np.mean(ytest))
print("Setup cost (NN-BS):", V0diff)


## Comparrison of correct BS-hedge at time 1 and NN hedge at time 1
def Comparehedge(t=0, showBS=True, sameArea=False):
    for i in range(N):
        if TimePoints[i] <= t:
            k = i
    t = TimePoints[k]
    Svals = testpathes[k,]
    if sameArea:
        a = np.min(testpathes)
        b = np.max(testpathes)
        Svals = np.linspace(a, b, Ktest)
    h_BS = deltaBS(Svals, strike, T - t, sigma0)
    timeprice = np.concatenate(
        (np.reshape(np.repeat(TimeConv(TimePoints[k]), Ktest), (Ktest, 1)), np.reshape(Svals, (Ktest, 1))), axis=1)
    h_NN = hedge.predict(timeprice)[:, 0]
    print("\n\n[Below]: BS hedging position blue vs NN hedging position orange time:", t)
    if showBS:
        plt.plot(Svals, h_BS, 'o')
    plt.plot(Svals, h_NN, 'o')
    plt.show()


tshow = np.array(T) * np.array((0.1, 0.25, 0.5, 0.75, 0.9))
for t in tshow:
    Comparehedge(t, True, False)
