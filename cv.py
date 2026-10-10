from datetime import date
from bs import BlackScholes
from crr import Crr
from boyle import Boyle


class Cv:
    """Variable de contrôle : américain (arbre) + [BS - européen (arbre)]."""

    def __init__(self, k, s0, r, t, sigma, date_initiale=None, q=0.0):
        self.k = k
        self.s0 = s0
        self.r = r
        self.t = t
        self.sigma = sigma
        self.q = q
        self.date_initiale = date_initiale or date.today().isoformat()

    def prix_arbre(self, modele, N, type_option, style):
        if modele == 'crr':
            arbre = Crr(self.k, self.s0, self.r, self.t, self.sigma, self.date_initiale, self.q)
            return arbre.binomial_crr(N, type_option, style)
        if modele == 'boyle':
            arbre = Boyle(self.k, self.s0, self.r, self.t, self.sigma, self.date_initiale, self.q)
            return arbre.boyle(N, type_option, style)
        raise ValueError("Modèle inconnu")

    def cv(self, N, model='crr', option_type='call', ref_model='crr', N_ref=2000):
        bs = BlackScholes(self.k, self.s0, self.t, self.r, self.sigma, self.q)
        self.valeur_bs = bs.pricing_bs(option_type)
        self.valeur_eu = self.prix_arbre(model, N, option_type, 'Eu')
        self.valeur_am = self.prix_arbre(model, N, option_type, 'Am')
        self.valeur_cv = self.valeur_am + (self.valeur_bs - self.valeur_eu)

        # référence : arbre américain choisi par l'utilisateur (modèle et nombre de pas)
        self.valeur_ref = self.prix_arbre(ref_model, N_ref, option_type, 'Am')
        return self.valeur_cv, self.valeur_am