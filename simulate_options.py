"""
METEF ENPC course project: A paradox of diffusion market model related with existence of winning combinations of options
@authors: Florentin POUCIN & Raphaël LASRY
"""

################################################################################

import numpy as np
from scipy.stats import norm as norme
import matplotlib.pyplot as plt
from pylab import MultipleLocator

################################################################################


class EuropeanOption(object):
    """Compute European option value.

    Parameters
    ==========
    S_0 : Initial stock price
    T : Expiration time (fraction of a year or a century)
    r : Risk free interest rate
    sigma : Volatility
    K_c : Strike price of call options
    mu_p : Number of "put" options in the portfolio (default = 1)
    K_p : Strike price of put options
    kind : Type of option ({'call', 'put'}, default 'call')
    """

    def __init__(self, S_0, T, r, sigma, K_c, K_p, kind="call", mu_p=1):

        self.kind = kind
        self.S_0 = S_0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.K_c = K_c
        self.K_p = K_p
        self.mu_p = mu_p

    def d_values(self, K):
        """Compute the values of d1 and d2."""
        d1 = (np.log(self.S_0 / K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def phi(self, d):
        """Compute the value of phi."""
        return norme.cdf(d, 0.0, 1.0)

    def mu_c(self, d_p, d_c):
        """Compute the value of mu_c with the Theorem 3.1."""
        return self.mu_p * (1 - self.phi(d_p)) / self.phi(d_c)

    def call_value(self, K):
        """Compute call value."""
        d1, d2 = self.d_values(K)
        return S_0 * self.phi(d1) - K * np.exp(-r * T) * self.phi(d2)

    def put_value(self, K):
        """Compute put value."""
        return self.call_value(K) - S_0 + K * np.exp(-r * T)

    def initial_value(self):
        """Compute the initial value X0."""
        d_p, _ = self.d_values(self.K_p)
        d_c, _ = self.d_values(self.K_c)
        return self.mu_p * self.put_value(self.K_p) + self.mu_c(
            d_p, d_c
        ) * self.call_value(self.K_c)

    def final_value(self, a):
        """Compute the final value."""
        Z = np.random.randn(100)
        Z[0] = 0
        W = np.sqrt(T / 100) * np.cumsum(Z)
        S_T = self.S_0 * np.exp(
            a * self.T - 0.5 * self.T * self.sigma ** 2 + self.sigma * W[-1]
        )
        d_p = self.d_values(self.K_p)[0]
        d_c = self.d_values(self.K_c)[0]
        return self.mu_p * max(0, self.K_p - S_T) + self.mu_c(d_p, d_c) * max(
            0, S_T - self.K_c
        )

    def average_gain(self, a):
        """Compute the average gain: E(X_T) - exp(rX_0)."""
        risk_gain = 0
        for i in range(1000):  # Approximation of the expectation
            risk_gain += self.final_value(a)
        risk_gain /= 1000
        return risk_gain - np.exp(self.r * self.T) * self.initial_value()


################################################################################

S_0 = 30  # Initial stock price
K_c, K_p = 25, 25  # Strike price
T = 0.25  # Time in years
r = 0.05  # Risk-free interest rate
sigma = 0.45  # Volatility in market


option = EuropeanOption(S_0, T, r, sigma, K_p, K_c)
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

S_0 = 30  # Initial stock price
K_c, K_p = 30, 30  # Strike price
T = 0.25  # Time in years
r = 0.05  # Risk-free interest rate
sigma = 0.45  # Volatility in market


option = EuropeanOption(S_0, T, r, sigma, K_p, K_c)
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

S_0 = 30  # Initial stock price
K_c, K_p = 30, 30  # Strike price
T = 1  # Time in years
r = 0.03  # Risk-free interest rate
sigma = 0.05  # Volatility in market




fig = plt.figure()

def graph(r, position):
    """Plot the average gain for different values of a.
    It should be a strict convex where the global minimum is at a = r,
    And the average gain should be non positive."""

    option = EuropeanOption(S_0, T, r, sigma, K_p, K_c)
    axes = fig.add_subplot(position)
    A = np.linspace(r - 0.1, r + 0.1, 100)
    E = []
    for a in A:
        E.append(option.average_gain(a))
    
    axes = plt.gca()
    axes.plot(A, E)
    axes.set_xlim(r - 0.15, r + 0.15)
    axes.set_title("r = " + str(r))
    
    axes.xaxis.set_major_locator(MultipleLocator(1.0))
    axes.xaxis.set_minor_locator(MultipleLocator(0.01))
    axes.yaxis.set_major_locator(MultipleLocator(1.0))
    axes.yaxis.set_minor_locator(MultipleLocator(0.1))
    axes.grid(which="major", axis="x", linewidth=0.75, linestyle="-", color="0.75")
    axes.grid(which="minor", axis="x", linewidth=0.25, linestyle="-", color="0.75")
    axes.grid(which="major", axis="y", linewidth=0.75, linestyle="-", color="0.75")
    axes.grid(which="minor", axis="y", linewidth=0.25, linestyle="-", color="0.75")

graph(0.03, 211)
graph(-0.05, 212)

################################################################################
