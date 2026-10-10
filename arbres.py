import numpy as np


def _payoff(s, k, type_option):
    if type_option == "call":
        return max(s - k, 0)
    return max(k - s, 0)


def _remonter(prix, k, type_option, style, continuation):
    # prix[i][j] = prix du sous-jacent au niveau i, nœud j
    N = len(prix) - 1
    niveaux = [None] * (N + 1)

    niveaux[N] = [
        {"s": round(float(s), 4), "v": round(float(_payoff(s, k, type_option)), 4), "ex": False}
        for s in prix[N]
    ]

    for i in range(N - 1, -1, -1):
        V = [n["v"] for n in niveaux[i + 1]]
        niveau = []
        for j in range(len(prix[i])):
            s = prix[i][j]
            vc = continuation(V, j)
            ex = _payoff(s, k, type_option)
            if style == "Am" and ex > vc:
                niveau.append({"s": round(float(s), 4), "v": round(float(ex), 4), "ex": True})
            else:
                niveau.append({"s": round(float(s), 4), "v": round(float(vc), 4), "ex": False})
        niveaux[i] = niveau

    return niveaux


def arbre_crr(k, s0, r, t, sigma, N, type_option="call", style="Eu", q=0.0):
    dt = t / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    actu = np.exp(-r * dt)

    prix = [[s0 * u**j * d**(i - j) for j in range(i + 1)] for i in range(N + 1)]

    def continuation(V, j):
        return actu * (p * V[j + 1] + (1 - p) * V[j])

    return _remonter(prix, k, type_option, style, continuation)


def arbre_boyle(k, s0, r, t, sigma, N, type_option="call", style="Eu", q=0.0):
    dt = t / N
    u = np.exp(sigma * np.sqrt(2 * dt))
    a = r - q - sigma**2 / 2

    pu = 0.5 * ((sigma**2 * dt + a**2 * dt**2) / (2 * sigma**2 * dt) + (a * dt) / (sigma * np.sqrt(2 * dt)))
    pd = 0.5 * ((sigma**2 * dt + a**2 * dt**2) / (2 * sigma**2 * dt) - (a * dt) / (sigma * np.sqrt(2 * dt)))
    pm = 1 - pu - pd
    actu = np.exp(-r * dt)

    prix = [[s0 * u**j for j in range(-i, i + 1)] for i in range(N + 1)]

    def continuation(V, j):
        return actu * (pd * V[j] + pm * V[j + 1] + pu * V[j + 2])

    return _remonter(prix, k, type_option, style, continuation)