import math
from datetime import datetime, timedelta


class Crr:
    def __init__(self, k, s0, r, t, sigma, date_initiale=None, q=0.0):
        self.k = k
        self.s0 = s0
        self.r = r
        self.t = t
        self.sigma = sigma
        self.q = q
        if date_initiale:
            self.date_initiale = datetime.strptime(date_initiale, "%Y-%m-%d")
        else:
            self.date_initiale = datetime.today()
        self.resultat = None

    def payoff(self, s, type_option):
        if type_option == 'call':
            return max(s - self.k, 0)
        return max(self.k - s, 0)

    def date_du_pas(self, i, dt):
        jours = round(i * dt * 365)
        date = self.date_initiale + timedelta(days=jours)
        return date.strftime("%Y-%m-%d")

    def binomial_crr(self, N, type_option='call', option_style='Eu', n_affiche=0):
        dt = self.t / N
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((self.r - self.q) * dt) - d) / (u - d)
        actu = math.exp(-self.r * dt)
        if p < 0 or p > 1:
            raise ValueError("CRR : p n'est pas entre 0 et 1, augmente N")
        n_affiche = min(n_affiche, N)

        # valeurs à l'échéance
        V = []
        niveaux = [None] * (n_affiche + 1)
        if n_affiche == N:
            niveaux[N] = []
        for j in range(N + 1):
            s = self.s0 * u ** (2 * j - N)
            V.append(self.payoff(s, type_option))
            if n_affiche == N:
                niveaux[N].append({"s": round(s, 6), "v": round(V[j], 6), "ex": False})

        # on remonte l'arbre
        exercices = []
        for i in range(N - 1, -1, -1):
            nouveau_V = []
            niveau = []
            plus_petite_valeur = None
            for j in range(i + 1):
                s = self.s0 * u ** (2 * j - i)
                continuation = actu * (p * V[j + 1] + (1 - p) * V[j])
                exercice = self.payoff(s, type_option)
                # petite marge pour ignorer les erreurs d'arrondi
                if option_style == 'Am' and exercice > continuation + 1e-10:
                    valeur = exercice
                    exerce = True
                    if plus_petite_valeur is None or exercice < plus_petite_valeur:
                        plus_petite_valeur = exercice
                else:
                    valeur = continuation
                    exerce = False
                nouveau_V.append(valeur)
                if i <= n_affiche:
                    niveau.append({"s": round(s, 6), "v": round(valeur, 6), "ex": exerce})
            V = nouveau_V
            if i <= n_affiche:
                niveaux[i] = niveau

            # exercice anticipé à ce pas (on regroupe les pas qui ont la même date)
            if plus_petite_valeur is not None:
                date = self.date_du_pas(i, dt)
                if len(exercices) > 0 and exercices[-1]["date"] == date:
                    exercices[-1]["pas_debut"] = i
                    exercices[-1]["valeur"] = min(exercices[-1]["valeur"], plus_petite_valeur)
                else:
                    exercices.append({"date": date, "pas_debut": i, "pas_fin": i,
                                      "valeur": plus_petite_valeur})
        exercices.reverse()

        self.resultat = {
            "prix": V[0],
            "niveaux": niveaux,
            "dates_niveaux": [self.date_du_pas(i, dt) for i in range(n_affiche + 1)],
            "exercices": exercices,
            "params": {"u": u, "d": d, "p": p},
        }
        return V[0]