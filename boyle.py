import math
from datetime import datetime, timedelta


class Boyle:
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

    def payoff(self, s, option_type):
        if option_type == 'call':
            return max(s - self.k, 0)
        return max(self.k - s, 0)

    def date_du_pas(self, i, dt):
        jours = round(i * dt * 365)
        date = self.date_initiale + timedelta(days=jours)
        return date.strftime("%Y-%m-%d")

    def boyle(self, N, option_type='call', option_style='Eu', n_affiche=0):
        dt = self.t / N
        u = math.exp(self.sigma * math.sqrt(2 * dt))
        d = 1 / u
        a = self.r - self.q - self.sigma ** 2 / 2
        actu = math.exp(-self.r * dt)

        terme1 = (self.sigma ** 2 * dt + a ** 2 * dt ** 2) / (2 * self.sigma ** 2 * dt)
        terme2 = (a * dt) / (self.sigma * math.sqrt(2 * dt))
        pu = 0.5 * (terme1 + terme2)
        pd = 0.5 * (terme1 - terme2)
        pm = 1 - pu - pd
        if pu < 0 or pd < 0 or pm < 0:
            raise ValueError("Boyle : une probabilité est négative, augmente N")
        n_affiche = min(n_affiche, N)

        # valeurs à l'échéance
        V = []
        niveaux = [None] * (n_affiche + 1)
        if n_affiche == N:
            niveaux[N] = []
        for j in range(-N, N + 1):
            s = self.s0 * u ** j
            V.append(self.payoff(s, option_type))
            if n_affiche == N:
                niveaux[N].append({"s": round(s, 6), "v": round(V[-1], 6), "ex": False})

        # on remonte l'arbre
        exercices = []
        for i in range(N - 1, -1, -1):
            nouveau_V = []
            niveau = []
            plus_petite_valeur = None
            for j in range(2 * i + 1):
                s = self.s0 * u ** (j - i)
                continuation = actu * (pd * V[j] + pm * V[j + 1] + pu * V[j + 2])
                exercice = self.payoff(s, option_type)
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
            "params": {"u": u, "d": d, "pu": pu, "pm": pm, "pd": pd},
        }
        return V[0]