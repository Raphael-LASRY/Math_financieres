import numpy as np
from scipy.stats import *

################################################################################


class European_Option(object):
    """Compute European option value.

    Parameters
    ==========
    S0 : Initial stock price
    T : Expiration time (fraction of a year or a century)
    r : Risk free interest rate
    sigma : Volatility
    K_c : Strike price of call options
    mu_p : Number of "put" options in the portfolio (default = 1)
    K_p : Strike price of put options
    kind : Type of option ({'call', 'put'}, default 'call')
    """

    def __init__(self, S0, T, r, sigma, K_c, K_p, kind="call", mu_p=1):

        self.kind = kind
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.K_c = K_c
        self.K_p = K_p
        self.mu_p = mu_p

    def d_values(self, S0, K, r, T, sigma):
        """Compute the values of d1 and d2."""
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def phi(self, d):
        """Compute the value of phi."""
        return norm.cdf(d, 0.0, 1.0)

    def mu_c(self, d_p, d_c):
        """Compute the value of mu_c with the Theorem 3.1."""
        return self.mu_p * (1 - self.phi(d_p)) / self.phi(d_c)

    def call_value(self, S0, K, r, T, sigma):
        """Compute call value."""
        d1, d2 = self.d_values(self, S0, K, r, T, sigma)
        return S0 * self.phi(d1) - K * np.exp(-r * T) * self.phi(d2)

    def put_value(self, S0, K, r, T, sigma):
        """Compute put value."""
        return self.call_value(S0, K, r, T, sigma) - S0 + K * np.exp(-r * T)

    def initial_value(self):
        """Compute the initial value X0."""
        d_p, _ = d_values(self, self.S0, self.K_p, self.r, self.T, self.sigma)
        d_c, _ = d_values(self, self.S0, self.K_c, self.r, self.T, self.sigma)
        mu_c = mu_c(self, d_p, d_c)
        return self.mu_p * put_value(
            self.S0, self.K_p, self.r, self.T, self.sigma
        ) + mu_c * call_value(self.S0, self.K_c, self.r, self.T, self.sigma)

    def final_value(self, a):
        """Compute the final value."""
        S_T = self.S0 * np.exp(
            a * self.T - 0.5 * self.T * self.sigma ** 2 + self.sigma * W_T
        )  # CALCULER W_T
        return self.mu_p * max(0, self.K_p - S_T) + self.mu_c * max(0, S_T - self.K_c)


################################################################################

S0 = 30  # Initial stock price
K_c, K_p = 25, 25  # Strike price
T = 0.25  # Time in years
r = 0.05  # Risk-free interest rate
sigma = 0.45  # Volatility in market


option = European_Option(S0, T, r, sigma, K_p, K_c)
d_p = option.d_values(S0, K_p, r, T, sigma)[0]
d_c = option.d_values(S0, K_c, r, T, sigma)[0]
print(
    "The values of d_p and d_c are:",
    round(d_p, 3),
    round(d_c, 3),
    "and we expected to have:",
    0.978,
    0.978,
)
print(
    "The value of phi(d) is:",
    round(option.phi(d_c), 3),
    "and we espected to have:",
    0.836,
)
print(
    "The value of the ratio mu_c/mu_p is:",
    round(option.mu_c(d_p, d_c), 3),
    "and we espected to have:",
    round(164 / 836, 3),
    "\n",
)

################################################################################

S0 = 30  # Initial stock price
K_c, K_p = 30, 30  # Strike price
T = 0.25  # Time in years
r = 0.05  # Risk-free interest rate
sigma = 0.45  # Volatility in market


option = European_Option(S0, T, r, sigma, K_p, K_c)
d_p = option.d_values(S0, K_p, r, T, sigma)[0]
d_c = option.d_values(S0, K_c, r, T, sigma)[0]
print(
    "The values of d_p and d_c are:",
    round(d_p, 3),
    round(d_c, 3),
    "and we espected to have:",
    0.168,
    0.168,
)
print(
    "The value of phi(d) is:",
    round(option.phi(d_c), 3),
    "and we espected to have:",
    0.567,
)
print(
    "The value of the ratio mu_c/mu_p is:",
    round(option.mu_c(d_p, d_c), 3),
    "and we espected to have:",
    round(433 / 567, 3),
    "\n",
)
