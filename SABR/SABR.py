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
    r = None  # interest rate
    steps = None  # number of time steps
    N = None  # number of simulated paths
    T = None  # time of maturity
    vol_paths = None  # N*steps array of volatility paths
    futures_paths = None  # N*steps array of futures paths
    time_points = None  # list of time points between 0 and T

    def __init__(self, F0, alpha, beta, rho, nu, r, steps, N, T=1, seed=None):
        # Initializer
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.r = r
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
            "r": self.r,
            "steps": self.steps,
            "N": self.N,
            "T": self.T
        }

    def set_parameters(self, F0, alpha, beta, rho, nu, r, steps, N, T=1, seed=None):
        # alter parameters of object and create new paths
        self.seed = seed
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.r = r
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

    def sigma(self, F, K, tau, alpha_new):
        # SABR sigma formula
        Z = (self.nu / alpha_new) * ((F * K) ** ((1 - self.beta) / 2)) * np.log(F / K)
        f1 = alpha_new / (
                ((F * K) ** ((1 - self.beta) / 2)) * (
                1 + (((1 - self.beta) ** 2) / 24) * (np.log(F / K) ** 2) + (((1 - self.beta) ** 4) / 1920) * (
                np.log(F / K) ** 4)))
        f2 = Z / self.x(Z)
        f3 = tau * (1 + (((1 - self.beta) ** 2) / 24) * ((alpha_new ** 2) / ((F * K) ** (1 - self.beta))) + 0.25 * (
                self.rho * self.beta * self.nu * alpha_new) / ((F * K) ** ((1 - self.beta) / 2)) + (self.nu ** 2) * (
                            2 - 3 * (self.rho ** 2)) / 24)
        return f1 * f2 * f3

    def sigma_prime(self, F, K, tau, alpha_new, increment=1 / 10000):
        # numerical derivative of SABR sigma formula
        eps = F * increment
        return (self.sigma(F + eps, K, tau, alpha_new) - self.sigma(F - eps, K, tau, alpha_new)) / (2 * eps)

    def BS_pricer(self, F, K, t, alpha_new, call=True, sigma=None):
        # BS pricer for European call or put options
        tau = self.T - t
        if type(sigma) == type(None):
            sigma = self.sigma(F, K, tau, alpha_new)
        d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        D = np.exp(-self.r * tau)  # TODO: discounting method might change
        if call:
            return D * (sp.stats.norm.cdf(d1) * F - sp.stats.norm.cdf(d2) * K)
        elif not call:
            return D * (sp.stats.norm.cdf(-d2) * K - sp.stats.norm.cdf(-d1) * F)
        else:
            raise ValueError("You must specify 'call' correctly! True or False")

    def get_price(self, K, step, call=True, sigma=False):
        # gets European call/put price given "true" volatility or SABR_model volatility
        if (step > self.steps or step < 0) or type(step) != type(1):
            raise ValueError("You must specify 'step' correctly! Integer between 0 and " + str(self.steps))
        if sigma:
            return self.BS_pricer(self.futures_paths[step, :], K, self.time_points[step], self.vol_paths[step, :], call,
                                  sigma=self.vol_paths[step, :])
        return self.BS_pricer(self.futures_paths[step, :], K, self.time_points[step], self.vol_paths[step, :], call)

    def BS_delta(self, d1, tau, call):
        # BS delta for European call or put options
        if call:
            return np.exp(-self.r * tau) * sp.stats.norm.cdf(d1)
        elif not call:
            return np.exp(-self.r * tau) * (-sp.stats.norm.cdf(-d1))

    def BS_vega(self, F, d1, tau):
        # BS vega for European call or put options
        return F * np.exp(-self.r * tau) * sp.stats.norm.pdf(d1) * np.sqrt(tau)

    def get_delta(self, K, step, call=True, sigma=False):
        # gets delta hedging ratio according to "true" volatility or SABR-model
        F = self.futures_paths[step, :]
        tau = self.T - self.time_points[step]
        if sigma:
            sigma = self.vol_paths[step, :]
            d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
            return self.BS_delta(d1, tau, call)
        else:
            sigma = self.sigma(F, K, tau, self.vol_paths[step, :])
        d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        BSd = self.BS_delta(d1, tau, call)
        BSv = self.BS_vega(F, d1, tau)
        sigma_pr = self.sigma_prime(F, K, tau, self.vol_paths[step, :])
        return BSd + BSv * sigma_pr


model = SABR_model(150, 0.4, 1, 0.5, 0.05, 0.04, 100, 10, seed=10538)
print(model.get_price(150, 50))  # here we use the SABR pricing formula
print(model.get_price(150, 50, sigma=True))  # "naive" BS price

print(model.get_delta(150, 50))  # here we use the SABR delta formula
print(model.get_delta(150, 50, sigma=True))  # "naive" BS delta
model.plot_paths()

# Note:
# one can create a BS model by specifying nu=0 --> constant volatility
# and using sigma=True for pricing and delta
