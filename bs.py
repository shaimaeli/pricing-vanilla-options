import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(self, k, s0, t, r, sigma, q=0.0):
        self.k = k
        self.s0 = s0
        self.t = t
        self.r = r
        self.sigma = sigma
        self.q = q          # rendement continu en dividendes

    def _d1(self):
        return (np.log(self.s0 / self.k) + (self.r - self.q + self.sigma**2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))

    def pricing_bs(self, type_option="call"):
        d1 = self._d1()
        d2 = d1 - self.sigma * np.sqrt(self.t)
        if type_option == 'call':
            return (self.s0 * np.exp(-self.q * self.t) * norm.cdf(d1)
                    - self.k * np.exp(-self.r * self.t) * norm.cdf(d2))
        else:
            return (self.k * np.exp(-self.r * self.t) * norm.cdf(-d2)
                    - self.s0 * np.exp(-self.q * self.t) * norm.cdf(-d1))


class Iv(BlackScholes):
    def __init__(self, prix_marche, k, s0, t, r, sigma, type_option="call", q=0.0):
        super().__init__(k, s0, t, r, sigma, q)
        self.prix_marche = prix_marche
        self.type_option = type_option

    def iv(self, tol_err=1e-10, max_iter=100):
        for _ in range(max_iter):
            d1 = self._d1()
            prix_bs = self.pricing_bs(type_option=self.type_option)
            erreur = prix_bs - self.prix_marche
            if abs(erreur) < tol_err:
                return self.sigma
            vega = self.s0 * np.exp(-self.q * self.t) * norm.pdf(d1) * np.sqrt(self.t)
            if vega < 1e-12:
                break
            self.sigma = max(self.sigma - erreur / vega, 1e-4)
        raise ValueError("La volatilité implicite n'a pas convergé (prix de marché incohérent ?)")