import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


class SABR_model:
    def __init__(self, F0, alpha, beta, rho, nu, r_tar, r_base, steps, N, T=1, seed=None, voltype="yearly"):
        # Initializer
        self.voltype = voltype
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.r_tar = r_tar
        self.r_base = r_base
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
            "r_tar": self.r_tar,
            "r_base": self.r_base,
            "steps": self.steps,
            "N": self.N,
            "T": self.T
        }

    def set_parameters(self, F0, alpha, beta, rho, nu, r_tar, r_base, steps, N, T=1, seed=None):
        # alter parameters of object and create new paths
        self.seed = seed
        self.F0 = F0
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.r_tar = r_tar
        self.r_base = r_base
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
        futures[0,] = self.F0
        volas[0,] = self.alpha
        dt = self.T / self.steps
        if self.voltype == "daily":
            dt = 1
        for i in range(self.steps):
            dW = np.random.multivariate_normal(
                mean=np.array([0, 0]),
                cov=np.array([[dt, self.rho * dt], [self.rho * dt, dt]]),
                size=self.N
            )
            volas[i + 1] = volas[i, :] + self.nu * volas[i, :] * dW[:, 0]
            volas[i + 1] = np.abs(volas[i + 1])
            futures[i + 1] = futures[i, :] + volas[i] * (futures[i, :] ** self.beta) * dW[:, 1]
            futures[i + 1] = np.abs(futures[i + 1])
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

    def plot_delta(self, step, K):
        delta = self.get_delta(step=step, K=K)
        price = self.futures_paths[step]
        plt.scatter(price, delta)
        plt.title(("Hedge Ratio at time t=" + str(self.time_points[step])))
        plt.show()

    def plot_wealth(self, step, K):
        delta = self.get_delta(step=step, K=K)
        price = self.futures_paths[step]
        plt.scatter(price * delta, delta)
        plt.title(("Wealth at time t=" + str(self.time_points[step])))
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
        f3 = 1 + (tau * ((((1 - self.beta) ** 2) / 24) * ((alpha_new ** 2) / ((F * K) ** (1 - self.beta))) + 0.25 * (
                self.rho * self.beta * self.nu * alpha_new) / ((F * K) ** ((1 - self.beta) / 2)) + (
                            self.nu ** 2) * (2 - 3 * (self.rho ** 2)) / 24))
        res = f1 * f2 * f3
        if np.any(np.isnan(res)):
            ind = np.isnan(res)
            res[ind] = (alpha_new[ind] / (F[ind] ** (1 - self.beta))) * f3[ind]
        return res

    def sigma_prime(self, F, K, tau, alpha_new, eps=1 / 10000):
        # numerical derivative of SABR sigma formula
        return (self.sigma(F + eps, K, tau, alpha_new) - self.sigma(F - eps, K, tau, alpha_new)) / (2 * eps)

    def payoff(self, K, tp="european", call=True):
        if tp == "european":
            if call:
                return np.maximum(self.futures_paths[-1, :] - K, 0)
            else:
                return np.maximum(K - self.futures_paths[-1, :], 0)

    def discount_factor(self, t, tau=False):
        if not tau:
            tau = self.T - t
        D_tar = 1 / ((1 + self.r_tar) ** tau)
        D_base = 1 / ((1 + self.r_base) ** tau)
        return D_base / D_tar

    def BS_pricer(self, F, K, t, alpha_new, call=True, sigma=None):
        # BS pricer for European call or put options
        tau = self.T - t
        if type(sigma) == type(None):
            sigma = self.sigma(F, K, tau, alpha_new)
        d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        D = self.discount_factor(t)
        if call:
            return D * (sp.stats.norm.cdf(d1) * F - sp.stats.norm.cdf(d2) * K)
        elif not call:
            return D * (sp.stats.norm.cdf(-d2) * K - sp.stats.norm.cdf(-d1) * F)
        else:
            raise ValueError("You must specify 'call' correctly! True or False")

    def get_price(self, K, step, call=True, BS=False):
        # gets European call/put price given initial volatility or SABR_model volatility
        if (step > self.steps or step < 0) or type(step) != type(1):
            raise ValueError("You must specify 'step' correctly! Integer between 0 and " + str(self.steps))
        if BS:
            return self.BS_pricer(F=self.futures_paths[step, :], K=K, t=self.time_points[step], alpha_new=self.vol_paths[0, :], call=call,
                                  sigma=self.vol_paths[0, :])
        return self.BS_pricer(F=self.futures_paths[step, :], K=K, t=self.time_points[step], alpha_new=self.vol_paths[step, :], call=call)

    def BS_delta(self, d1, tau, call):
        # BS delta for European call or put options
        if call:
            return self.discount_factor(tau, True) * sp.stats.norm.cdf(d1)
        elif not call:
            return self.discount_factor(tau, True) * (-sp.stats.norm.cdf(-d1))

    def BS_vega(self, F, d1, tau):
        # BS vega for European call or put options
        return F * self.discount_factor(tau, True) * sp.stats.norm.pdf(d1) * np.sqrt(tau)

    def get_delta(self, K, step, call=True, BS=False):
        # gets delta hedging ratio according to initial volatility or SABR-model
        F = self.futures_paths[step, :]
        tau = self.T - self.time_points[step]
        if BS:
            sigma = self.vol_paths[0, :]  # just taking initial volatility
            d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
            return self.BS_delta(d1, tau, call)
        else:
            sigma = self.sigma(F, K, tau, self.vol_paths[step, :])
        d1 = (np.log(F / K) + 0.5 * (sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        BSd = self.BS_delta(d1, tau, call)
        BSv = self.BS_vega(F, d1, tau)
        sigma_pr = self.sigma_prime(F, K, tau, self.vol_paths[step, :])
        return BSd + BSv * sigma_pr

    def performance(self, K, real_path=None):
        if type(None) == type(real_path):
            payoffs = self.payoff(K=K)
            wealth_SABR = self.get_price(step=0, K=K)
            wealth_BS = self.get_price(step=0, K=K, BS=True)
            for i in range(self.steps):
                wealth_SABR = wealth_SABR + self.get_delta(step=i, K=K) * (
                        self.futures_paths[i + 1, :] - self.futures_paths[i, :])
                wealth_BS = wealth_BS + self.get_delta(step=i, K=K, BS=True) * (
                        self.futures_paths[i + 1, :] - self.futures_paths[i, :])
            self.wealth_SABR = wealth_SABR
            self.wealth_BS = wealth_BS
            self.wealth_nothing = self.get_price(step=0, K=K) / self.discount_factor(t=0)
            loss = np.mean((wealth_SABR - payoffs) ** 2)
            std_err = np.std(wealth_SABR - payoffs)
            mean = np.mean(wealth_SABR - payoffs)
            loss_BS = np.mean((wealth_BS - payoffs) ** 2)
            std_err_BS = np.std(wealth_BS - payoffs)
            mean_BS = np.mean(wealth_BS - payoffs)
            loss_nothing = np.mean((self.wealth_nothing - payoffs) ** 2)
            std_err_nothing = np.std(self.wealth_nothing - payoffs)
            mean_nothing = np.mean(self.wealth_nothing - payoffs)
            print("BS-Model-Hedge:\n" + "Loss (MSE): " + str(loss_BS) + "\n" +
                  "Standard Error: " + str(std_err_BS))
            print("SABR-Model-Hedge:\n" + "Loss (MSE): " + str(loss) + "\n" +
                  "Standard Error: " + str(std_err))
            print("This happens if we do nothing:\n" +
                  "Loss (MSE): " + str(loss_nothing) + "\n" +
                  "Standard Error: " + str(std_err_nothing))
            return {
                "loss_BS": loss_BS,
                "std_err_BS": std_err_BS,
                "mean_BS": mean_BS,
                "loss_SABR": loss,
                "std_err_SABR": std_err,
                "mean_SABR": mean,
                "loss_nothing": loss_nothing,
                "std_err_nothing": std_err_nothing,
                "mean_nothing": mean_nothing
            }
        else:
            sabr = SABR_model(
                F0=self.F0,
                alpha=self.alpha,
                beta=self.beta,
                rho=self.rho,
                nu=self.nu,
                r_tar=self.r_tar,
                r_base=self.r_base,
                steps=self.steps,
                N=1
            )
            sabr.futures_paths[:, 0] = np.array(real_path["F"])
            sabr.vol_paths[:, 0] = np.array(real_path["Sigma"])
            payoff = sabr.payoff(K=K)
            wealth_SABR = sabr.get_price(step=0, K=K)
            wealth_BS = sabr.get_price(step=0, K=K, BS=True)
            wealth_nothing = sabr.get_price(step=0, K=K) / sabr.discount_factor(t=0)
            for i in range(sabr.steps):
                wealth_SABR = wealth_SABR + sabr.get_delta(step=i, K=K) * (
                        sabr.futures_paths[i + 1, 0] - sabr.futures_paths[i, 0])
                wealth_BS = wealth_BS + sabr.get_delta(step=i, K=K, BS=True) * (
                        sabr.futures_paths[i + 1, 0] - sabr.futures_paths[i, 0])
            return {
                "BS": wealth_BS - payoff,
                "SABR": wealth_SABR - payoff,
                "Nothing": wealth_nothing - payoff,
                "payoff": payoff,
                "futp": sabr.futures_paths,
                "volp": sabr.vol_paths
            }
