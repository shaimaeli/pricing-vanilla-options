import io
import json
import traceback
from datetime import date

import numpy as np
from scipy.stats import norm
from flask import Flask, render_template, request, jsonify, send_file

import matplotlib
matplotlib.use("Agg")                      # pas de fenêtre : les figures sont envoyées au navigateur
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch
import mpld3
from mpld3 import plugins, urls
from mpld3._display import NumpyEncoder

from bs import BlackScholes
from crr import Crr
from boyle import Boyle
from cv import Cv
from moteur import remonter
from marche import donnees_ticker, taux_sans_risque, liste_strikes, vol_pour_strike, smile

app = Flask(__name__)

# couleurs de l'interface
MARINE, GRIS, GRILLE, AXE = "#0b2a5b", "#5b6b86", "#e3eaf5", "#9db3d8"
ROUGE, ORANGE = "#c0392b", "#e08a00"


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


# ---------- graphes matplotlib (interactifs avec mpld3) ----------

def bloc_graphe(titre, fig, classe="graphe-large"):
    figure = json.loads(json.dumps(mpld3.fig_to_dict(fig), cls=NumpyEncoder))
    return {"type": "graphe", "titre": titre, "figure": figure, "classe": classe}


def figure_axes(largeur, hauteur):
    fig = Figure(figsize=(largeur, hauteur), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_facecolor("none")
    ax.grid(color=GRILLE)
    ax.tick_params(colors=GRIS, labelsize=9)
    for cote in ax.spines.values():
        cote.set_color(AXE)
    return fig, ax


def graphe_smile(points, k, sigma):
    fig, ax = figure_axes(8.6, 3.8)
    ks = [p["strike"] for p in points]
    ivs = [p["iv"] * 100 for p in points]
    v0 = sigma * 100

    tous = ivs + [v0]
    marge = (max(tous) - min(tous)) * 0.15 or 1
    bas, haut = max(min(tous) - marge, 0), max(tous) + marge
    gauche = min(ks + [k])

    # courbe du smile + info-bulle au survol
    ax.plot(ks, ivs, color=MARINE, lw=2.2)
    pts = ax.plot(ks, ivs, "o", color=MARINE, ms=5)[0]
    plugins.connect(fig, plugins.PointLabelTooltip(
        pts, labels=[f"K = {x:g} · vol implicite {y:.2f} %" for x, y in zip(ks, ivs)]))

    # intersection de K et σ : point rouge relié aux deux axes
    ax.plot([k, k], [bas, v0], color=ROUGE, lw=1.5, ls="--")
    ax.plot([gauche, k], [v0, v0], color=ROUGE, lw=1.5, ls="--")
    rouge = ax.plot([k], [v0], "o", color=ROUGE, ms=11, mec="white", mew=2)[0]
    plugins.connect(fig, plugins.PointLabelTooltip(
        rouge, labels=[f"Strike choisi : K = {k:g} · σ = {v0:.2f} %"]))

    ax.set_ylim(bas, haut)
    ax.set_xlabel("Strike K", color=GRIS)
    ax.set_ylabel("Volatilité implicite (%)", color=GRIS)
    return fig


def graphe_convergence(ns, crr, boyle, bs):
    fig, ax = figure_axes(8.6, 4.0)
    for valeurs, couleur, nom in ((crr, MARINE, "CRR"), (boyle, ORANGE, "Boyle")):
        xs = [n for n, v in zip(ns, valeurs) if v is not None]
        ys = [v for v in valeurs if v is not None]
        ax.plot(xs, ys, color=couleur, lw=2, label=nom)
    ax.plot([ns[0], ns[-1]], [bs, bs], color=ROUGE, lw=2, ls="--", label="Black-Scholes")
    ax.set_xlabel("N (nombre de pas)", color=GRIS)
    ax.legend(loc="upper right")
    return fig


def graphe_arbre(niveaux, dates, crr):
    # les coordonnées sont en pouces : 1 pouce = 100 pixels
    P = len(niveaux) - 1
    dx, dy, l, h, marge, haut = 1.25, (0.5 if crr else 0.46), 0.9, 0.38, 0.2, 0.46
    largeur = P * dx + l + 2 * marge
    contenu = (P if crr else 2 * P) * dy + h
    hauteur = contenu + haut + marge
    centre = marge + contenu / 2

    fig = Figure(figsize=(largeur, hauteur), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, largeur)
    ax.set_ylim(0, hauteur)
    ax.axis("off")
    ax.set_facecolor("none")

    def X(i):
        return marge + i * dx

    def Y(i, j):
        return centre - ((i / 2 - j) if crr else (i - j)) * dy

    # en-têtes des niveaux
    for i in range(P + 1):
        ax.text(X(i) + l / 2, hauteur - 0.18, f"i = {i}", ha="center", va="center", color=MARINE, fontsize=9)
        ax.text(X(i) + l / 2, hauteur - 0.33, dates[i], ha="center", va="center", color=GRIS, fontsize=7)

    # traits entre un nœud et ses enfants
    for i in range(P):
        for j in range(len(niveaux[i])):
            for c in ((j, j + 1) if crr else (j, j + 1, j + 2)):
                ax.plot([X(i) + l, X(i + 1)], [Y(i, j), Y(i + 1, c)], color=AXE, lw=1.2)

    # nœuds (plein bleu marine = exercice anticipé)
    for i, niveau in enumerate(niveaux):
        for j, nd in enumerate(niveau):
            x, y = X(i), Y(i, j)
            plein = nd["ex"]
            ax.add_patch(FancyBboxPatch((x, y - h / 2), l, h, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=MARINE if plein else "none", ec=MARINE, lw=3 if plein else 1.5))
            couleur = "white" if plein else MARINE
            ax.text(x + l / 2, y + 0.07, f"S {nd['s']:.2f}", ha="center", va="center", color=couleur, fontsize=8)
            ax.text(x + l / 2, y - 0.08, f"V {nd['v']:.2f}", ha="center", va="center", color=couleur, fontsize=8.5)

    plugins.clear(fig)          # pas de zoom sur l'arbre
    return fig


# ---------- pages et fichiers JS de mpld3 (servis depuis la bibliothèque installée) ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mpld3/d3.js")
def js_d3():
    return send_file(urls.D3_LOCAL, mimetype="text/javascript")


@app.route("/mpld3/mpld3.js")
def js_mpld3():
    return send_file(urls.MPLD3_LOCAL, mimetype="text/javascript")


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
            titre = f"Smile de volatilité implicite · échéance {d['expiration']}"
            blocs.append(bloc_graphe(titre, graphe_smile(points, d["k"], d["sigma"])))
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

        fig = graphe_arbre(res["niveaux"], res["dates_niveaux"], modele == "crr")
        blocs = [{"type": "cartes", "titre": "Prix et paramètres de l'arbre", "cartes": cartes},
                 bloc_graphe(f"Arbre {nom_modele(modele)}", fig, "arbre")]

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
        fig = graphe_convergence(ns, series["crr"], series["boyle"], bs)
        return jsonify(blocs=[
            bloc_graphe("Prix européen en fonction de N", fig),
            {"type": "cartes", "titre": "Erreur vs Black-Scholes", "cartes": cartes},
        ])
    except Exception as e:
        return erreur(e)


if __name__ == "__main__":
    app.run(debug=True)