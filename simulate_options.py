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

    def d_values(self, K):
        """Compute the values of d1 and d2."""
        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def phi(self, d):
        """Compute the value of phi."""
        return norm.cdf(d, 0.0, 1.0)

    def mu_c(self, d_p, d_c):
        """Compute the value of mu_c with the Theorem 3.1."""
        return self.mu_p * (1 - self.phi(d_p)) / self.phi(d_c)

    def call_value(self, K):
        """Compute call value."""
        d1, d2 = self.d_values(K)
        return S0 * self.phi(d1) - K * np.exp(-r * T) * self.phi(d2)

    def put_value(self, K):
        """Compute put value."""
        return self.call_value(K) - S0 + K * np.exp(-r * T)

    def initial_value(self):
        """Compute the initial value X0."""
        d_p, _ = self.d_values(self.K_p)
        d_c, _ = self.d_values(self.K_c)
        return self.mu_p * self.put_value(self.K_p) + self.mu_c(
            d_p, d_c
        ) * self.call_value(self.K_c)

    def final_value(self, a):
        """Compute the final value."""
        t_sqrt = np.sqrt(1 / self.T)
        Z = np.random.randn(t_sqrt)  # Hypothesis : one point/day
        Z[0] = 0
        W = np.cumsum(t_sqrt * Z)
        S_T = self.S0 * np.exp(
            a * self.T - 0.5 * self.T * self.sigma ** 2 + self.sigma * W[-1]
        )
        d_p = self.d_values(self.K_p)[0]
        d_c = self.d_values(self.K_c)[0]
        return self.mu_p * max(0, self.K_p - S_T) + self.mu_c(d_p, d_c) * max(
            0, S_T - self.K_c
        )

    def average_gain(self, a):
        risk_gain = 0
        for i in range(1000):  # Approximation of the expectation
            risk_gain += self.final_value(a)
        risk_gain /= 1000
        return risk_gain - np.exp(self.r * self.T) * self.initial_value()


################################################################################

S0 = 30  # Initial stock price
K_c, K_p = 25, 25  # Strike price
T = 0.25  # Time in years
r = 0.05  # Risk-free interest rate
sigma = 0.45  # Volatility in market


option = European_Option(S0, T, r, sigma, K_p, K_c)
d_p = option.d_values(K_p)[0]
d_c = option.d_values(K_c)[0]

################################################################################

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
d_p = option.d_values(K_p)[0]
d_c = option.d_values(K_c)[0]

################################################################################

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

################################################################################

S0 = 30  # Initial stock price
K_c, K_p = 30, 30  # Strike price
T = 1  # Time in years
r = 0.05  # Risk-free interest rate
sigma = 0.05  # Volatility in market


option = European_Option(S0, T, r, sigma, K_p, K_c)

################################################################################

print(option.average_gain(0.05))
