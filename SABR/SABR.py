import numpy as np
import matplotlib.pyplot as plt


class SABR_path:
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
                    [self.T * x / self.steps for x in range(0, self.steps + 1)],
                    self.vol_paths[:, j]
                )
            else:
                plt.plot(
                    [self.T * x / self.steps for x in range(0, self.steps + 1)],
                    self.futures_paths[:, j]
                )
        if vol:
            plt.title((str(i) + " Volatility Paths"))
        else:
            plt.title((str(i) + " Futures Paths"))

        plt.show()

        def get_price():
            pass


path = SABR_path(150, 0.4, 1, 0.5, 0.05, 100, 10)
path.plot_paths(vol=True)
path.plot_paths()