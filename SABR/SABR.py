import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


class SABR_model:
    seed = None  # seed for replicability
    F0 = None  # initial futures value
    alpha = None  # initial volatility value
    beta = None  # shape parameter
    rho = None  # correlation between BMs
    nu = None  # volvol
    steps = None  # number of time steps
    N = None  # number of simulated paths
    T = None  # time of maturity
    vol_paths = None  # N*steps array of volatility paths
    futures_paths = None  # N*steps array of futures paths
    time_points = None  # list of time points between 0 and T

    def __init__(self, F0, alpha, beta, rho, nu, steps, N, T=1, seed=None):
        # Initializer
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.steps = steps
        self.N = N
        self.T = T
        self.seed = seed
        self.create_paths()
        self.time_points = [self.T * x / self.steps for x in range(0, self.steps + 1)]

    def get_parameters(self):
        # returns parameters as dictionary
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
            "f": self.F0,
            "steps": self.steps,
            "N": self.N,
            "T": self.T
        }

    def set_parameters(self, F0, alpha, beta, rho, nu, steps, N, T=1, seed=None):
        # alter parameters of object and create new paths
        self.seed = seed
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.steps = steps
        self.N = N
        self.T = T
        self.create_paths(self)

    def create_paths(self):
        # creates paths given the model parameters
        if type(self.seed) != type(None):
            np.random.seed(self.seed)
        futures = np.zeros((self.steps + 1, self.N))
        volas = np.zeros((self.steps + 1, self.N))
        futures[0,] = self.F0;
        volas[0,] = self.alpha
        dt = self.T / self.steps
        for i in range(self.steps):
            dW = np.random.multivariate_normal(
                mean=np.array([0, 0]),
                cov=np.array([[dt, self.rho * dt], [self.rho * dt, dt]]),
                size=self.N
            )
            volas[i + 1] = volas[i, :] + self.nu * volas[i, :] * dW[:, 0]
            futures[i + 1] = futures[i, :] + volas[i + 1] * (futures[i, :] ** self.beta) * dW[:, 1]
        self.vol_paths = volas
        self.futures_paths = futures

    def plot_paths(self, i=5, vol=False):
        # plots i futures/volatility paths
        if self.N < i:
            i = self.N
        for j in range(i):
            if vol:
                plt.plot(
                    self.time_points,
                    self.vol_paths[:, j]
                )
            else:
                plt.plot(
                    self.time_points,
                    self.futures_paths[:, j]
                )
        if vol:
            plt.title((str(i) + " Volatility Paths"))
        else:
            plt.title((str(i) + " Futures Paths"))

        plt.show()

    def x(self, z):
        # helper function for the SABR sigma formula
        return np.log((np.sqrt(1 - 2 * self.rho * z + (z ** 2)) + z - self.rho) / (1 - self.rho))

    def sigma(self, F, K, tau):
        # SABR sigma formula
        Z = (self.nu / self.alpha) * ((F * K) ** ((1 - self.beta) / 2)) * np.log(F / K)
        f1 = self.alpha / (
                ((F * K) ** ((1 - self.beta) / 2)) * (
                1 + (((1 - self.beta) ** 2) / 24) * (np.log(F / K) ** 2) + (((1 - self.beta) ** 4) / 1920) * (
                np.log(F / K) ** 4)))
        f2 = Z / self.x(Z)
        f3 = tau * (1 + (((1 - self.beta) ** 2) / 24) * ((self.alpha ** 2) / ((F * K) ** (1 - self.beta))) + 0.25 * (
                self.rho * self.beta * self.nu * self.alpha) / ((F * K) ** ((1 - self.beta) / 2)) + (self.nu ** 2) * (
                            2 - 3 * (self.rho ** 2)) / 24)
        return f1 * f2 * f3

    def BS_pricer(self, F, K, t, r, call=True, sigma=None):
        # BS pricer for European call or put options
        tau = self.T - t
        if type(sigma) == type(None):
            sigma = self.sigma(F, K, tau)
        d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        D = np.exp(-r * tau)  # TODO: discounting method might change
        if call:
            return D * (sp.stats.norm.cdf(d1) * F - sp.stats.norm.cdf(d2) * K)
        elif not call:
            return D * (sp.stats.norm.cdf(-d2) * K - sp.stats.norm.cdf(-d1) * F)
        else:
            raise ValueError("You must specify 'call' correctly! True or False")

    def get_price(self, K, step, r, call=True, sigma=False):
        # gets European call/put price given "true" volatility or SABR_model volatility
        if (step > self.steps or step < 0) or type(step) != type(1):
            raise ValueError("You must specify 'step' correctly! Integer between 0 and " + str(self.steps))
        if sigma:
            return self.BS_pricer(self.futures_paths[step, :], K, self.time_points[step], r, call,
                                  sigma=self.vol_paths[step, :])
        return self.BS_pricer(self.futures_paths[step, :], K, self.time_points[step], r, call)
    def get_delta(self):
        # gets delta hedging ratio according to SABR-model
        pass # TODO


model = SABR_model(150, 0.4, 1, 0.5, 0.05, 100, 10, seed=10538)
print(model.get_price(150, 50, 0.04))  # here we use the SABR pricing formula
print(model.get_price(150, 50, 0.04, sigma=True))  # here we use the "true" volatility
model.plot_paths()
