import traceback
from datetime import date
import numpy as np
from scipy.stats import norm
from flask import Flask, render_template, request, jsonify

from bs import BlackScholes
from crr import Crr
from boyle import Boyle
from cv import Cv
from moteur import remonter
from marche import donnees_ticker, taux_sans_risque, liste_strikes, vol_pour_strike, smile

app = Flask(__name__)


# ---------- outils ----------

def erreur(e):
    if not isinstance(e, ValueError):
        traceback.print_exc()
    return jsonify(erreur=str(e)), 400


def f(x):
    return f"{x:.4f}"


def f6(x):
    return f"{x:.6f}"


def carte(label, valeur, ton=""):
    return {"label": label, "valeur": valeur, "ton": ton}


def lire_q(d):
    q = d.get("q")
    q = 0.0 if q in (None, "") else float(q)
    if q < 0 or q >= 1:
        raise ValueError("q doit être entre 0 et 1")
    return q


def verifier_params(d):
    for cle in ("s0", "k", "t", "sigma"):
        if not d.get(cle) or d[cle] <= 0:
            raise ValueError("Charge un ticker, choisis l'échéance et le strike.")
    lire_q(d)


def entier(d, cle, mini, maxi, nom):
    n = int(d[cle])
    if n < mini or n > maxi:
        raise ValueError(f"{nom} doit être compris entre {mini} et {maxi}")
    return n


def nom_modele(m):
    return "CRR" if m == "crr" else "Boyle"


def tableau(titre, entetes, lignes):
    return {"type": "tableau", "titre": titre, "entetes": entetes, "lignes": lignes}


def ligne(cells, ton=""):
    return {"cells": cells, "ton": ton}


@app.route("/")
def index():
    return render_template("index.html")


# ---------- Marché ----------

@app.route("/api/ticker", methods=["POST"])
def api_ticker():
    d = request.get_json()
    try:
        ticker = d["ticker"].strip().upper()
        spot, vol_hist, q, expirations = donnees_ticker(ticker)
        r = taux_sans_risque()
        return jsonify(
            ticker=ticker,
            spot=round(spot, 2),
            r=round(r, 4) if r is not None else None,
            q=round(q, 4),
            vol_hist=round(vol_hist, 4) if vol_hist is not None else None,
            expirations=expirations,
        )
    except Exception as e:
        return erreur(e)


@app.route("/api/strikes", methods=["POST"])
def api_strikes():
    d = request.get_json()
    try:
        return jsonify(strikes=liste_strikes(d["ticker"], d["expiration"]))
    except Exception as e:
        return erreur(e)


@app.route("/api/vol", methods=["POST"])
def api_vol():
    d = request.get_json()
    try:
        if not d.get("k") or d["k"] <= 0:
            raise ValueError("Choisis un strike")
        if d.get("r") is None or d["r"] == "":
            raise ValueError("Saisis le taux sans risque r")
        q = lire_q(d)
        t, prix, sigma, source = vol_pour_strike(
            d["ticker"], d["expiration"], d["k"], d["s0"], d["r"], d["type"], d.get("vol_hist"), q
        )
        return jsonify(t=round(t, 4), prix=prix, sigma=sigma, source=source)
    except Exception as e:
        return erreur(e)


# ---------- Black-Scholes ----------

@app.route("/api/bs", methods=["POST"])
def api_bs():
    d = request.get_json()
    try:
        verifier_params(d)
        q = lire_q(d)
        bs = BlackScholes(d["k"], d["s0"], d["t"], d["r"], d["sigma"], q)
        call, put = bs.pricing_bs("call"), bs.pricing_bs("put")
        droite = d["s0"] * np.exp(-q * d["t"]) - d["k"] * np.exp(-d["r"] * d["t"])
        ecart = call - put - droite

        # smile : si Yahoo échoue, les prix s'affichent quand même
        points, note = [], ""
        try:
            points = smile(d["ticker"], d["expiration"], d["s0"], d["r"], q)
            if not points:
                note = "Aucune volatilité implicite exploitable pour cette échéance."
        except Exception as e:
            note = f"Smile indisponible : {e}"

        blocs = [{"type": "cartes", "titre": "Prix", "cartes": [
            carte("Call", f(call)),
            carte("Put", f(put)),
        ]}]
        if points:
            # k et sigma servent à placer le point du strike choisi sur le graphe
            blocs.append({"type": "smile", "points": points, "echeance": d["expiration"],
                          "k": d["k"], "sigma": d["sigma"]})
        else:
            blocs.append({"type": "info", "texte": note})
        return jsonify(blocs=blocs)
    except Exception as e:
        return erreur(e)


# ---------- CRR et Boyle ----------

@app.route("/api/arbre", methods=["POST"])
def api_arbre():
    d = request.get_json()
    try:
        verifier_params(d)
        q = lire_q(d)
        n = entier(d, "n", 1, 2000, "N")
        p = entier(d, "n_arbre", 1, 10, "Le nombre de périodes affichées")
        modele, style, typ = d["model"], d["style"], d["type"]
        aujourdhui = date.today().isoformat()

        if modele == "crr":
            m = Crr(d["k"], d["s0"], d["r"], d["t"], d["sigma"], aujourdhui, q)
            prix = m.binomial_crr(n, typ, style, p)
        else:
            m = Boyle(d["k"], d["s0"], d["r"], d["t"], d["sigma"], aujourdhui, q)
            prix = m.boyle(n, typ, style, p)
        res = m.resultat
        P = res["P"]

        ref = BlackScholes(d["k"], d["s0"], d["t"], d["r"], d["sigma"], q).pricing_bs(typ)
        ecart = prix - ref

        # prix + paramètres de l'arbre (u, d et probabilités risque-neutre)
        pr = res["params"]
        cartes = [carte(f"Prix {nom_modele(modele)}", f(prix)),
                  carte("u (hausse)", f6(pr["u"])),
                  carte("d (baisse)", f6(pr["d"]))]
        if modele == "crr":
            cartes.append(carte("p (proba risque-neutre)", f6(pr["p"])))
        else:
            cartes.append(carte("pu (proba risque-neutre, hausse)", f6(pr["pu"])))
            cartes.append(carte("pm (proba risque-neutre, milieu)", f6(pr["pm"])))
            cartes.append(carte("pd (proba risque-neutre, baisse)", f6(pr["pd"])))

        blocs = [{"type": "cartes", "titre": "Prix et paramètres de l'arbre", "cartes": cartes},
                 {"type": "arbre", "titre": f"Arbre {nom_modele(modele)}", "niveaux": res["niveaux"],
                  "dates": res["dates_niveaux"], "crr": modele == "crr"}]

        if style == "Am":
            ex = res["exercices"]
            if not ex:
                blocs.append({"type": "cartes", "titre": "Exercice anticipé",
                              "cartes": [carte("Dates d'exercice", "Aucune")]})
            else:
                lignes = []
                for g in ex:
                    cond = f"S ≥ {g['front']:.2f}" if typ == "call" else f"S ≤ {g['front']:.2f}"
                    pas = str(g["pas_debut"]) if g["pas_debut"] == g["pas_fin"] else f"{g['pas_debut']} – {g['pas_fin']}"
                    lignes.append(ligne([g["date"], pas, str(g["nb"]), cond, f(g["valeur"])]))
                blocs.append(tableau("Exercice anticipé",
                                     ["Date", "Pas", "Nœuds", "Frontière", "Valeur d'exercice"], lignes))
        return jsonify(blocs=blocs)
    except Exception as e:
        return erreur(e)


# ---------- Variable de contrôle ----------

# en dessous de ce seuil (en prix), on considère l'erreur comme quasi nulle
SEUIL_PARFAIT = 1e-3


@app.route("/api/cv", methods=["POST"])
def api_cv():
    d = request.get_json()
    try:
        verifier_params(d)
        q = lire_q(d)
        n = entier(d, "n", 1, 2000, "N")
        n_ref = entier(d, "n_ref", 1, 5000, "N de la référence")
        modele, ref_modele, typ = d["model"], d["ref_model"], d["type"]

        c = Cv(d["k"], d["s0"], d["r"], d["t"], d["sigma"], date.today().isoformat(), q)
        valeur_cv, valeur_am = c.cv(n, modele, typ, ref_modele, n_ref)
        bs, eu, ref = c.valeur_bs, c.valeur_eu, c.valeur_ref

        err_am = abs(valeur_am - ref)     # erreur de l'arbre américain brut
        err_cv = abs(valeur_cv - ref)     # erreur après correction

        # conclusion courte
        if err_cv < SEUIL_PARFAIT and err_am < SEUIL_PARFAIT:
            texte, ton = f"Arbre déjà exact : erreur {f(err_am)}", "ok"
        elif err_cv < SEUIL_PARFAIT:
            texte, ton = f"Amélioration parfaite : erreur passe de {f(err_am)} à {f(err_cv)}", "ok"
        elif err_cv < err_am:
            texte, ton = f"Amélioration partielle : erreur passe de {f(err_am)} à {f(err_cv)}", "hl"
        else:
            texte, ton = f"Pas d'amélioration : erreur passe de {f(err_am)} à {f(err_cv)}", "bad"

        nom, nom_ref = nom_modele(modele), nom_modele(ref_modele)
        lignes = [
            ligne(["Black-Scholes (européen)", f(bs)]),
            ligne([f"Arbre {nom} européen (N = {n})", f(eu)]),
            ligne([f"Arbre {nom} américain (N = {n})", f(valeur_am)]),
            ligne(["Américain corrigé (CV)", f(valeur_cv)], "hl"),
            ligne([f"Référence {nom_ref} américain (N = {n_ref})", f(ref)]),
            ligne(["|Américain corrigé − Vréf|", f(err_cv)], ton),
            ligne(["|Américain arbre − Vréf|", f(err_am)]),
            ligne(["Conclusion", texte], ton + " note"),
        ]
        t_res = tableau(f"Control Variate · {nom} · N = {n}", ["", "Valeur"], lignes)
        return jsonify(blocs=[t_res])
    except Exception as e:
        return erreur(e)


# ---------- Convergence ----------

@app.route("/api/convergence", methods=["POST"])
def api_convergence():
    d = request.get_json()
    try:
        verifier_params(d)
        q = lire_q(d)
        n_max = entier(d, "n", 10, 300, "N max")
        typ = d["type"]
        bs = BlackScholes(d["k"], d["s0"], d["t"], d["r"], d["sigma"], q).pricing_bs(typ)

        ns = list(range(5, n_max + 1))
        series = {"crr": [], "boyle": []}
        for n in ns:
            for modele in series:
                try:
                    series[modele].append(remonter(modele, d["k"], d["s0"], d["r"], d["t"], d["sigma"],
                                                   q, n, typ, "Eu")["prix"])
                except ValueError:
                    series[modele].append(None)

        dernier = lambda v: next((x for x in reversed(v) if x is not None), None)
        cartes = []
        for modele in ("crr", "boyle"):
            x = dernier(series[modele])
            cartes.append(carte(f"Erreur {nom_modele(modele)} (N = {n_max})",
                                "—" if x is None else f"{abs(x - bs):.5f}"))
        return jsonify(blocs=[
            {"type": "convergence", "ns": ns, "crr": series["crr"], "boyle": series["boyle"], "bs": bs},
            {"type": "cartes", "titre": "Erreur vs Black-Scholes", "cartes": cartes},
        ])
    except Exception as e:
        return erreur(e)



if __name__ == "__main__":
    app.run(debug=True)