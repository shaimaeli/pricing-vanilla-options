import numpy as np
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
        self.exercices = []

    def payoff(self, S, option_type):
        if option_type == 'call':
            return np.maximum(S - self.k, 0)
        return np.maximum(self.k - S, 0)

    def date_du_pas(self, i, dt):
        jours = int(round(i * dt * 365))
        return (self.date_initiale + timedelta(days=jours)).strftime("%Y-%m-%d")

    def boyle(self, N, option_type='call', option_style='Eu', n_affiche=0):
        if N < 1:
            raise ValueError("N doit être au moins 1")
        n_affiche = min(max(int(n_affiche), 0), N)

        # paramètres de l'arbre
        dt = self.t / N
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u
        a = self.r - self.q - self.sigma ** 2 / 2
        actu = np.exp(-self.r * dt)

        terme1 = (self.sigma ** 2 * dt + a ** 2 * dt ** 2) / (2 * self.sigma ** 2 * dt)
        terme2 = (a * dt) / (self.sigma * np.sqrt(2 * dt))
        pu = 0.5 * (terme1 + terme2)
        pd = 0.5 * (terme1 - terme2)
        pm = 1 - pu - pd
        if pu < 0 or pd < 0 or pm < 0:
            raise ValueError("Boyle : une probabilité est négative. Augmente N.")

        # niveau N : au niveau i, le nœud j (de -i à +i) vaut S0 * u^j
        S = self.s0 * u ** np.arange(-N, N + 1)
        V = self.payoff(S, option_type)

        gardes = {}                       # niveaux qu'on veut afficher
        if n_affiche == N:
            gardes[N] = (S, V, np.zeros(2 * N + 1, dtype=bool))

        liste_exercices = []              # (pas, plus petite valeur d'exercice)

        # remontée de l'arbre
        for i in range(N - 1, -1, -1):
            S = self.s0 * u ** np.arange(-i, i + 1)
            continuation = actu * (pd * V[:-2] + pm * V[1:-1] + pu * V[2:])

            if option_style == 'Am':
                exercice = self.payoff(S, option_type)
                en_exercice = exercice > continuation + 1e-12
                V = np.where(en_exercice, exercice, continuation)
                if en_exercice.any():
                    liste_exercices.append((i, float(exercice[en_exercice].min())))
            else:
                en_exercice = np.zeros(2 * i + 1, dtype=bool)
                V = continuation

            if i <= n_affiche:
                gardes[i] = (S, V, en_exercice)

        # dates d'exercice anticipé (un pas par ligne, regroupés si même date)
        exercices = []
        for i, valeur in reversed(liste_exercices):
            date = self.date_du_pas(i, dt)
            if len(exercices) > 0 and exercices[-1]["date"] == date:
                exercices[-1]["pas_fin"] = i
                exercices[-1]["valeur"] = min(exercices[-1]["valeur"], valeur)
            else:
                exercices.append({"date": date, "pas_debut": i, "pas_fin": i, "valeur": valeur})

        # niveaux à afficher
        niveaux = []
        for i in range(n_affiche + 1):
            S_i, V_i, ex_i = gardes[i]
            niveau = []
            for j in range(len(S_i)):
                niveau.append({"s": round(float(S_i[j]), 6),
                               "v": round(float(V_i[j]), 6),
                               "ex": bool(ex_i[j])})
            niveaux.append(niveau)

        self.exercices = exercices
        self.resultat = {
            "prix": float(V[0]),
            "niveaux": niveaux,
            "dates_niveaux": [self.date_du_pas(i, dt) for i in range(n_affiche + 1)],
            "exercices": exercices,
            "params": {"u": float(u), "d": float(d), "pu": float(pu), "pm": float(pm), "pd": float(pd)},
        }
        return float(V[0])