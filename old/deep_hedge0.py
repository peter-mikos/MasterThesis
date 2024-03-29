import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Subtract, Multiply, Add, Dot, Concatenate

from keras.models import Model

import matplotlib.pyplot as plt

# Trajectories of the Black scholes model
# Let it run to initialize the following parameters, the trajectories
# are not needed afterwards

N = 100  # time disrectization
S0 = np.exp(5)  # initial value of the asset in multiples of np.exp(5)
strike = 150  # strike for the call option
T = 2.0  # maturity
sigma = 0.2  # volatility in Black Scholes
mu = -sigma ** 2 / 2  ## Q-dynamics
R = 3  # number of shown Trajectories --- not used for training/testing
activator = "tanh"

Ktrain = 20000  # Size of training data
Ktest = 1000  # Size of test data
epochs = 10

## Normalising initial value to of S0 to 1 (1 unit of money = price of S0)
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
def path(S0, mu, sigma, Timepoints, R):
    N = len(Timepoints) - 1
    X = np.zeros((N + 1, R)) + np.log(S0)
    for j in range(N):
        dt = Timepoints[j + 1] - Timepoints[j]
        dZ = np.random.normal(0, 1, R)
        increment = mu * dt + sigma * dZ * np.sqrt(dt)
        X[j + 1, :] = X[j, :] + increment
    return np.exp(X)


#### Creating R sample paths from the model ####
S = path(S0, mu, sigma, TimePoints, R)

for i in range(R):
    plt.plot(TimePoints, S[:, i])
plt.show()

#### Defining the payoff as well as implementing the BS-formula for it
import scipy.stats as scipy


# from scipy.stats import norm

# Black Scholes price formula (European call)
def BS(S0, strike, T, sigma):
    return S0 * scipy.norm.cdf(
        (np.log(S0 / strike) + 0.5 * T * sigma ** 2) / (np.sqrt(T) * sigma)) - strike * scipy.norm.cdf(
        (np.log(S0 / strike) - 0.5 * T * sigma ** 2) / (np.sqrt(T) * sigma))


# Black Scholes Delta hedge
def deltaBS(S0, strike, T, sigma):
    return scipy.norm.cdf((np.log(S0 / strike) + 0.5 * T * sigma ** 2) / (np.sqrt(T) * sigma))


priceBS = BS(S0, strike, T, sigma)
print('Price of a Call option in the Black scholes model with initial price', S0, 'strike', strike, 'maturity', T,
      'and volatility', sigma, 'is equal to', BS(S0, strike, T, sigma))


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

## Creating position size and new weath from current wealth and new asset price
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
trainpathes = path(S0, mu, sigma, TimePoints, Ktrain)

## Shaping to input architecture
xtrain = []
for i in range(N + 1):
    TimeToMaturity = np.repeat(TimeConv(TimePoints[i]), Ktrain)
    xtrain = xtrain + [TimeToMaturity, trainpathes[i, :]]

# ytrain is filled with zeros
ytrain = f(trainpathes)

maxL = 10
maxiter = 5
for i in range(maxiter):
    model_wealth.fit(x=xtrain, y=ytrain, epochs=epochs, verbose=True, batch_size=100 * (i + 1))  ##
    if model_wealth.evaluate(xtrain, ytrain, batch_size=1000) < maxL:
        break

#### Creating test pathes for evaluation ####
testpathes = path(S0, mu, sigma, TimePoints, Ktest)
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
V0correct = BS(testpathes[0, :], strike, T, sigma)  ## Initial wealth BS
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
    h_BS = deltaBS(Svals, strike, T - t, sigma)
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

