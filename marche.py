import math
import time
from datetime import date, datetime

import numpy as np
import yfinance as yf

from bs import Iv

# ---------- cache : évite de re-solliciter Yahoo à chaque clic ----------
_cache = {}


def en_cache(cle, fonction, duree=300):
    maintenant = time.time()
    if cle in _cache and maintenant - _cache[cle][0] < duree:
        return _cache[cle][1]
    valeur = fonction()
    _cache[cle] = (maintenant, valeur)
    return valeur


def propre(x):
    # transforme les NaN de pandas en None (sinon le JSON est invalide)
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def donnees_ticker(ticker):
    """Un seul appel d'historique : spot, vol historique, rendement en dividendes, échéances."""
    def charger():
        action = yf.Ticker(ticker)
        h = action.history(period="1y")
        if h.empty:
            raise ValueError(
                f"Yahoo ne renvoie aucune donnée pour {ticker}. Ticker faux, ou Yahoo limite "
                "temporairement ton IP : attends quelques minutes ou change de réseau."
            )
        spot = float(h["Close"].iloc[-1])

        rendements = np.log(h["Close"] / h["Close"].shift(1)).dropna().tail(252)
        vol_hist = float(rendements.std() * np.sqrt(252)) if len(rendements) >= 30 else None

        # dividendes versés sur les 12 derniers mois / spot
        q = float(h["Dividends"].sum()) / spot if "Dividends" in h.columns else 0.0

        expirations = list(action.options)
        if not expirations:
            raise ValueError("Aucune option cotée pour ce ticker")
        return spot, vol_hist, q, expirations

    return en_cache(("ticker", ticker), charger)


def taux_sans_risque():
    # Taux des bons du Trésor américain à 13 semaines (^IRX, coté en %)
    def charger():
        historique = yf.Ticker("^IRX").history(period="5d")
        if historique.empty:
            return None
        return float(historique["Close"].iloc[-1]) / 100
    return en_cache(("taux",), charger, duree=3600)


def chaine_en_cache(ticker, expiration):
    return en_cache(("chaine", ticker, expiration),
                    lambda: yf.Ticker(ticker).option_chain(expiration))


def maturite(expiration):
    jours = (datetime.strptime(expiration, "%Y-%m-%d").date() - date.today()).days
    return max(jours, 1) / 365


def liste_strikes(ticker, expiration):
    chaine = chaine_en_cache(ticker, expiration)
    return sorted(set(float(s) for s in chaine.calls["strike"]))


def prix_marche(o):
    """Prix de marché d'une ligne de la chaîne : milieu bid/ask, sinon dernier prix."""
    bid, ask, dernier = propre(o["bid"]), propre(o["ask"]), propre(o["lastPrice"])
    return (bid + ask) / 2 if (bid and ask) else dernier


def vol_pour_strike(ticker, expiration, k, spot, r, type_option, vol_hist, q=0.0):
    """Vol implicite du strike choisi ; vol historique si Newton ne converge pas."""
    t = maturite(expiration)
    chaine = chaine_en_cache(ticker, expiration)
    df = chaine.calls if type_option == "call" else chaine.puts
    ligne = df[(df["strike"] - k).abs() < 1e-6]

    prix = None
    if not ligne.empty:
        prix = prix_marche(ligne.iloc[0])

    if prix:
        try:
            iv = Iv(prix, k, spot, t, r, 0.3, type_option, q).iv()
            if 0.01 < iv < 3:
                return t, prix, iv, "implicite (Newton)"
        except Exception:
            pass
        raison = "Newton n'a pas convergé"
    elif ligne.empty:
        raison = "strike non coté"
    else:
        raison = "pas de prix de marché"

    if vol_hist:
        return t, prix, vol_hist, f"historique ({raison})"
    return t, prix, None, f"indisponible ({raison})"


def smile(ticker, expiration, spot, r, q=0.0):
    """Smile de volatilité : vol implicite de chaque strike de l'échéance.

    On utilise les options hors de la monnaie (puts sous le spot, calls au-dessus),
    car les options très dans la monnaie ont peu de valeur temps et donnent une vol bruitée.
    """
    def charger():
        t = maturite(expiration)
        chaine = chaine_en_cache(ticker, expiration)

        par_strike = {}
        for nom, df in (("call", chaine.calls), ("put", chaine.puts)):
            for _, o in df.iterrows():
                k = float(o["strike"])
                if not (0.6 * spot <= k <= 1.4 * spot):
                    continue
                prix = prix_marche(o)
                if not prix:
                    continue
                try:
                    iv = Iv(prix, k, spot, t, r, 0.3, nom, q).iv()
                except Exception:
                    continue
                if 0.01 < iv < 3:
                    par_strike.setdefault(k, {})[nom] = iv

        points = []
        for k in sorted(par_strike):
            v = par_strike[k]
            prefere, autre = ("put", "call") if k < spot else ("call", "put")
            iv = v.get(prefere) or v.get(autre)
            if iv:
                points.append({"strike": k, "iv": round(iv, 4)})
        return points

    cle = ("smile", ticker, expiration, round(spot, 2), round(r, 4), round(q, 4))
    return en_cache(cle, charger)