import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


class SABR_model:
    seed = None
    F0 = None
    alpha = None
    beta = None
    rho = None
    nu = None
    steps = None
    N = None
    T = None
    vol_paths = None
    futures_paths = None
    time_points = None

    def __init__(self, F0, alpha, beta, rho, nu, steps, N, T=1, seed=None):
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
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
            "f": self.F0,
            "steps": self.steps,
            "N": self.steps,
            "T": self.T
        }

    def set_parameters(self, F0, alpha, beta, rho, nu, steps, N, T=1, seed=None):
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

    def BS_pricer(self, F, K, sigma, t, r, call=True):
        tau = self.T - t
        d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        D = np.exp(-r * tau)  # TODO: discounting method might change
        if call:
            return D * (sp.stats.norm.cdf(d1) * F - sp.stats.norm.cdf(d2) * K)
        elif not call:
            return D * (sp.stats.norm.cdf(-d2) * K - sp.stats.norm.cdf(-d1) * F)
        else:
            raise ValueError("You must specify 'call' correctly! True or False")

    def get_price(self, K, step, r, call=True):
        if (step > self.steps or step < 0) or type(step) != type(1):
            raise ValueError("You must specify 'step' correctly! Integer between 0 and " + str(self.steps))
        return self.BS_pricer(self.futures_paths[step,:], K, self.vol_paths[step,:], self.time_points[step], r, call)


model = SABR_model(150, 0.4, 1, 0.5, 0.05, 100, 10)
print(model.get_price(150, 50, 0.04))

model.plot_paths()
