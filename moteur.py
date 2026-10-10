"""Moteur de valorisation par arbre (CRR binomial et Boyle trinomial), vectorisé avec numpy.

Un seul code calcule : le prix, les P premières périodes de l'arbre (pour l'affichage),
et les exercices anticipés sur tous les pas de l'arbre à N pas.
"""
import numpy as np
from datetime import datetime, timedelta

TOL = 1e-12


def remonter(modele, k, s0, r, t, sigma, q, N, type_option="call", style="Eu",
             n_affiche=0, date_initiale=None):
    if N < 1:
        raise ValueError("N doit être au moins 1")
    dt = t / N
    actu = np.exp(-r * dt)
    call = type_option == "call"
    americain = style == "Am"

    if isinstance(date_initiale, str):
        base = datetime.strptime(date_initiale, "%Y-%m-%d")
    else:
        base = date_initiale or datetime.today()

    def payoff(S):
        return np.maximum(S - k, 0.0) if call else np.maximum(k - S, 0.0)

    def date_pas(i):
        return (base + timedelta(days=int(round(float(i * dt * 365))))).strftime("%Y-%m-%d")

    # ----- paramètres de l'arbre -----
    if modele == "crr":
        h = sigma * np.sqrt(dt)
        u = np.exp(h)
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        if not 0 <= p <= 1:
            raise ValueError(f"CRR : la probabilité p = {p:.4f} n'est pas dans [0, 1]. Augmente N.")
        params = {"u": u, "d": d, "p": p}

        def expo(i):                      # exposants des prix au niveau i (i+1 nœuds)
            return h * (2 * np.arange(i + 1) - i)

        def suivant(V):
            return actu * (p * V[1:] + (1 - p) * V[:-1])
    else:
        h = sigma * np.sqrt(2 * dt)
        a = r - q - sigma**2 / 2
        terme1 = (sigma**2 * dt + a**2 * dt**2) / (2 * sigma**2 * dt)
        terme2 = (a * dt) / (sigma * np.sqrt(2 * dt))
        pu = 0.5 * (terme1 + terme2)
        pd = 0.5 * (terme1 - terme2)
        pm = 1 - pu - pd
        if min(pu, pd, pm) < 0:
            raise ValueError("Boyle : une probabilité est négative. Augmente N.")
        u = np.exp(h)
        params = {"u": u, "d": 1 / u, "pu": pu, "pm": pm, "pd": pd}

        def expo(i):                      # 2i+1 nœuds : exposants -i ... +i
            return h * np.arange(-i, i + 1)

        def suivant(V):
            return actu * (pd * V[:-2] + pm * V[1:-1] + pu * V[2:])

    P = min(max(int(n_affiche), 0), N)
    garde = {}

    # ----- échéance -----
    S = s0 * np.exp(expo(N))
    V = payoff(S)
    if P == N:
        garde[N] = (S, V, np.zeros(len(S), dtype=bool))

    # ----- remontée -----
    rec = []          # (pas, nb de nœuds, S min, S max, S frontière)
    for i in range(N - 1, -1, -1):
        S = s0 * np.exp(expo(i))
        vc = suivant(V)
        if americain:
            pay = payoff(S)
            masque = pay > vc + TOL
            V = np.where(masque, pay, vc)
            if masque.any():
                Sx = S[masque]
                front = Sx.min() if call else Sx.max()
                rec.append((i, int(masque.sum()), float(Sx.min()), float(Sx.max()), float(front)))
        else:
            masque = np.zeros(len(S), dtype=bool)
            V = vc
        if i <= P:
            garde[i] = (S, V, masque)

    # ----- regroupement des exercices par date -----
    groupes = []
    for (i, nb, smin, smax, front) in reversed(rec):
        dte = date_pas(i)
        if groupes and groupes[-1]["date"] == dte:
            g = groupes[-1]
            g["pas_fin"] = i
            g["nb"] += nb
            g["smin"] = min(g["smin"], smin)
            g["smax"] = max(g["smax"], smax)
            g["front"] = min(g["front"], front) if call else max(g["front"], front)
        else:
            groupes.append({"date": dte, "pas_debut": i, "pas_fin": i, "t": i * dt,
                            "nb": nb, "smin": smin, "smax": smax, "front": front})
    for g in groupes:
        g["valeur"] = float(max(g["front"] - k, 0) if call else max(k - g["front"], 0))

    niveaux = []
    for i in range(P + 1):
        Si, Vi, Ei = garde[i]
        niveaux.append([{"s": round(float(s), 6), "v": round(float(v), 6), "ex": bool(e)}
                        for s, v, e in zip(Si, Vi, Ei)])

    return {
        "prix": float(V[0]),
        "niveaux": niveaux,
        "dates_niveaux": [date_pas(i) for i in range(P + 1)],
        "exercices": groupes,
        "nb_noeuds_ex": sum(g["nb"] for g in groupes),
        "P": P, "N": N, "dt": float(dt),
        "params": {kk: float(vv) for kk, vv in params.items()},
    }