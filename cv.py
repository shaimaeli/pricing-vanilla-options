from datetime import date
from bs import BlackScholes
from crr import Crr
from boyle import Boyle


class Cv:
    """Variable de contrôle : américain(arbre) + [BS - européen(arbre)]."""

    def __init__(self, k, s0, r, t, sigma, date_initiale=None, q=0.0):
        self.k = k
        self.s0 = s0
        self.r = r
        self.t = t
        self.sigma = sigma
        self.q = q
        self.date_initiale = date_initiale or date.today().isoformat()
        self.exercices = []

    def prix_arbre(self, modele, N, type_option, style):
        """Prix d'un arbre ('crr' ou 'boyle'), et l'objet pour récupérer les exercices."""
        if modele == 'crr':
            m = Crr(self.k, self.s0, self.r, self.t, self.sigma, self.date_initiale, self.q)
            return m.binomial_crr(N, type_option, style), m
        if modele == 'boyle':
            m = Boyle(self.k, self.s0, self.r, self.t, self.sigma, self.date_initiale, self.q)
            return m.boyle(N, type_option, style), m
        raise ValueError("Modèle inconnu")

    def prix_bs(self, type_option):
        return BlackScholes(self.k, self.s0, self.t, self.r, self.sigma, self.q).pricing_bs(type_option)

    def cv(self, N, model='crr', option_type='call', ref_model='crr', N_ref=2000):
        self.valeur_bs = self.prix_bs(option_type)
        self.valeur_eu, _ = self.prix_arbre(model, N, option_type, 'Eu')
        self.valeur_am, m_am = self.prix_arbre(model, N, option_type, 'Am')
        self.exercices = m_am.exercices
        self.valeur_cv = self.valeur_am + (self.valeur_bs - self.valeur_eu)
        # référence : arbre américain choisi par l'utilisateur (modèle et nombre de pas)
        self.valeur_ref, _ = self.prix_arbre(ref_model, N_ref, option_type, 'Am')
        return self.valeur_cv, self.valeur_am

    def convergence(self, liste_N, model, option_type):
        """Eu / Am / CV pour plusieurs N (tableau de convergence)."""
        bs = self.prix_bs(option_type)
        lignes = []
        for n in liste_N:
            try:
                eu, _ = self.prix_arbre(model, n, option_type, 'Eu')
                am, _ = self.prix_arbre(model, n, option_type, 'Am')
                lignes.append({"N": n, "eu": eu, "am": am, "cv": am + (bs - eu)})
            except ValueError:
                lignes.append({"N": n, "eu": None, "am": None, "cv": None})
        return lignes