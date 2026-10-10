from moteur import remonter


class Boyle:
    def __init__(self, k, s0, r, t, sigma, date_initiale=None, q=0.0):
        self.k = k
        self.s0 = s0
        self.r = r
        self.t = t
        self.sigma = sigma
        self.q = q
        self.date_initiale = date_initiale
        self.resultat = None
        self.exercices = []

    def boyle(self, N, option_type='call', option_style='Eu', n_affiche=0):
        self.resultat = remonter("boyle", self.k, self.s0, self.r, self.t, self.sigma, self.q,
                                 N, option_type, option_style, n_affiche, self.date_initiale)
        self.exercices = self.resultat["exercices"]
        return self.resultat["prix"]