import json
import traceback
from datetime import date

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import HTTPException

import matplotlib
matplotlib.use("Agg")
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

MARINE = "#0b2a5b"
GRIS = "#5b6b86"
AXE = "#9db3d8"
ROUGE = "#c0392b"
ORANGE = "#e08a00"


@app.errorhandler(Exception)
def gerer_erreur(e):
    if isinstance(e, HTTPException):
        return e
    if not isinstance(e, ValueError):
        traceback.print_exc()
    return jsonify(erreur=str(e)), 400



def carte(label, valeur, ton=""):
    return {"label": label, "valeur": valeur, "ton": ton}


def ligne(cells, ton=""):
    return {"cells": cells, "ton": ton}


def tableau(titre, entetes, lignes):
    return {"type": "tableau", "titre": titre, "entetes": entetes, "lignes": lignes}


def verifier(d):
    """Vérifie les paramètres et renvoie q."""
    for cle in ["s0", "k", "t", "sigma"]:
        if not d.get(cle) or d[cle] <= 0:
            raise ValueError("Charge un ticker, choisis l'échéance et le strike.")
    q = d.get("q") or 0
    if q < 0 or q >= 1:
        raise ValueError("q doit être entre 0 et 1")
    return q


def lire_entier(d, cle, mini, maxi, nom):
    n = int(d[cle])
    if n < mini or n > maxi:
        raise ValueError(f"{nom} doit être compris entre {mini} et {maxi}")
    return n


def nom_modele(modele):
    if modele == "crr":
        return "CRR"
    return "Boyle"



def bloc_graphe(titre, fig, classe="graphe-large"):
    figure = json.loads(json.dumps(mpld3.fig_to_dict(fig), cls=NumpyEncoder))
    return {"type": "graphe", "titre": titre, "figure": figure, "classe": classe}


def nouvelle_figure(largeur, hauteur):
    fig = Figure(figsize=(largeur, hauteur), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_facecolor("none")
    ax.grid(False)
    ax.tick_params(colors=GRIS, labelsize=9)
    for cote in ax.spines.values():
        cote.set_color(AXE)
    return fig, ax


def graphe_smile(points, k, sigma):
    fig, ax = nouvelle_figure(8.6, 3.8)
    strikes = [p["strike"] for p in points]
    vols = [p["iv"] * 100 for p in points]
    vol_choisie = sigma * 100

    toutes = vols + [vol_choisie]
    marge = (max(toutes) - min(toutes)) * 0.15 or 1
    bas = max(min(toutes) - marge, 0)
    haut = max(toutes) + marge

    ax.plot(strikes, vols, color=MARINE, lw=2.2)
    points_smile = ax.plot(strikes, vols, "o", color=MARINE, ms=5)[0]
    textes = [f"K = {x:g} · vol implicite {y:.2f} %" for x, y in zip(strikes, vols)]
    plugins.connect(fig, plugins.PointLabelTooltip(points_smile, labels=textes))

    ax.plot([k, k], [bas, vol_choisie], color=ROUGE, lw=1.5, ls="--")
    ax.plot([min(strikes + [k]), k], [vol_choisie, vol_choisie], color=ROUGE, lw=1.5, ls="--")
    point_rouge = ax.plot([k], [vol_choisie], "o", color=ROUGE, ms=11, mec="white", mew=2)[0]
    texte = f"Strike choisi : K = {k:g} · σ = {vol_choisie:.2f} %"
    plugins.connect(fig, plugins.PointLabelTooltip(point_rouge, labels=[texte]))

    ax.set_ylim(bas, haut)
    ax.set_xlabel("Strike K", color=GRIS)
    ax.set_ylabel("Volatilité implicite (%)", color=GRIS)
    return fig


def graphe_convergence(ns, prix_crr, prix_boyle, prix_bs):
    fig, ax = nouvelle_figure(8.6, 4.0)
    for prix, couleur, nom in [(prix_crr, MARINE, "CRR"), (prix_boyle, ORANGE, "Boyle")]:
        xs = [n for n, p in zip(ns, prix) if p is not None]
        ys = [p for p in prix if p is not None]
        ax.plot(xs, ys, color=couleur, lw=2, label=nom)
    ax.plot([ns[0], ns[-1]], [prix_bs, prix_bs], color=ROUGE, lw=2, ls="--", label="Black-Scholes")
    ax.set_xlabel("N (nombre de pas)", color=GRIS)
    ax.legend(loc="upper right")
    return fig


def graphe_arbre(niveaux, dates, est_crr):
    nb = len(niveaux) - 1                    
    ecart_x = 1.25                           
    ecart_y = 0.5 if est_crr else 0.46       
    larg_noeud, haut_noeud = 0.9, 0.38
    marge, haut_titre = 0.2, 0.46

    largeur = nb * ecart_x + larg_noeud + 2 * marge
    if est_crr:
        contenu = nb * ecart_y + haut_noeud
    else:
        contenu = 2 * nb * ecart_y + haut_noeud
    hauteur = contenu + haut_titre + marge
    centre = marge + contenu / 2

    fig = Figure(figsize=(largeur, hauteur), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, largeur)
    ax.set_ylim(0, hauteur)
    ax.axis("off")
    ax.set_facecolor("none")

    def x_noeud(i):
        return marge + i * ecart_x

    def y_noeud(i, j):
        if est_crr:
            return centre - (i / 2 - j) * ecart_y
        return centre - (i - j) * ecart_y

    for i in range(nb + 1):
        milieu = x_noeud(i) + larg_noeud / 2
        ax.text(milieu, hauteur - 0.18, f"i = {i}", ha="center", va="center", color=MARINE, fontsize=9)
        ax.text(milieu, hauteur - 0.33, dates[i], ha="center", va="center", color=GRIS, fontsize=7)

    for i in range(nb):
        for j in range(len(niveaux[i])):
            if est_crr:
                enfants = [j, j + 1]
            else:
                enfants = [j, j + 1, j + 2]
            for c in enfants:
                ax.plot([x_noeud(i) + larg_noeud, x_noeud(i + 1)], [y_noeud(i, j), y_noeud(i + 1, c)],
                        color=AXE, lw=1.2)

    for i in range(nb + 1):
        for j in range(len(niveaux[i])):
            noeud = niveaux[i][j]
            x, y = x_noeud(i), y_noeud(i, j)
            exercice = noeud["ex"]
            forme = FancyBboxPatch((x, y - haut_noeud / 2), larg_noeud, haut_noeud,
                                   boxstyle="round,pad=0,rounding_size=0.06",
                                   fc=MARINE if exercice else "none", ec=MARINE,
                                   lw=3 if exercice else 1.5)
            ax.add_patch(forme)
            couleur = "white" if exercice else MARINE
            ax.text(x + larg_noeud / 2, y + 0.07, f"S {noeud['s']:.2f}", ha="center", va="center",
                    color=couleur, fontsize=8)
            ax.text(x + larg_noeud / 2, y - 0.08, f"V {noeud['v']:.2f}", ha="center", va="center",
                    color=couleur, fontsize=8.5)

    plugins.clear(fig)
    return fig



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/mpld3/d3.js")
def js_d3():
    return send_file(urls.D3_LOCAL, mimetype="text/javascript")


@app.route("/mpld3/mpld3.js")
def js_mpld3():
    return send_file(urls.MPLD3_LOCAL, mimetype="text/javascript")



@app.route("/api/ticker", methods=["POST"])
def api_ticker():
    d = request.get_json()
    ticker = d["ticker"].strip().upper()
    spot, vol_hist, q, expirations = donnees_ticker(ticker)
    r = taux_sans_risque()
    if r is not None:
        r = round(r, 4)
    if vol_hist is not None:
        vol_hist = round(vol_hist, 4)
    return jsonify(ticker=ticker, spot=round(spot, 2), r=r, q=round(q, 4),
                   vol_hist=vol_hist, expirations=expirations)


@app.route("/api/strikes", methods=["POST"])
def api_strikes():
    d = request.get_json()
    return jsonify(strikes=liste_strikes(d["ticker"], d["expiration"]))


@app.route("/api/vol", methods=["POST"])
def api_vol():
    d = request.get_json()
    if not d.get("k") or d["k"] <= 0:
        raise ValueError("Choisis un strike")
    if d.get("r") in (None, ""):
        raise ValueError("Saisis le taux sans risque r")
    q = d.get("q") or 0
    t, prix, sigma, source = vol_pour_strike(d["ticker"], d["expiration"], d["k"], d["s0"],
                                             d["r"], d["type"], d.get("vol_hist"), q)
    return jsonify(t=round(t, 4), prix=prix, sigma=sigma, source=source)



@app.route("/api/bs", methods=["POST"])
def api_bs():
    d = request.get_json()
    q = verifier(d)
    bs = BlackScholes(d["k"], d["s0"], d["t"], d["r"], d["sigma"], q)
    call = bs.pricing_bs("call")
    put = bs.pricing_bs("put")
    blocs = [{"type": "cartes", "titre": "Prix",
              "cartes": [carte("Call", f"{call:.4f}"), carte("Put", f"{put:.4f}")]}]

    try:
        points = smile(d["ticker"], d["expiration"], d["s0"], d["r"], q)
        message = "Aucune volatilité implicite exploitable pour cette échéance."
    except Exception as e:
        points = []
        message = f"Smile indisponible : {e}"

    if points:
        titre = f"Smile de volatilité implicite · échéance {d['expiration']}"
        blocs.append(bloc_graphe(titre, graphe_smile(points, d["k"], d["sigma"])))
    else:
        blocs.append({"type": "info", "texte": message})
    return jsonify(blocs=blocs)



@app.route("/api/arbre", methods=["POST"])
def api_arbre():
    d = request.get_json()
    q = verifier(d)
    n = lire_entier(d, "n", 1, 2000, "N")
    nb_affiche = lire_entier(d, "n_arbre", 1, 10, "Le nombre de périodes affichées")
    modele, style, typ = d["model"], d["style"], d["type"]
    aujourdhui = date.today().isoformat()

    if modele == "crr":
        arbre = Crr(d["k"], d["s0"], d["r"], d["t"], d["sigma"], aujourdhui, q)
        prix = arbre.binomial_crr(n, typ, style, nb_affiche)
    else:
        arbre = Boyle(d["k"], d["s0"], d["r"], d["t"], d["sigma"], aujourdhui, q)
        prix = arbre.boyle(n, typ, style, nb_affiche)
    res = arbre.resultat

    p = res["params"]
    cartes = [carte("Prix " + nom_modele(modele), f"{prix:.4f}"),
              carte("u (hausse)", f"{p['u']:.6f}"),
              carte("d (baisse)", f"{p['d']:.6f}")]
    if modele == "crr":
        cartes.append(carte("p (proba risque-neutre)", f"{p['p']:.6f}"))
    else:
        cartes.append(carte("pu (proba risque-neutre, hausse)", f"{p['pu']:.6f}"))
        cartes.append(carte("pm (proba risque-neutre, milieu)", f"{p['pm']:.6f}"))
        cartes.append(carte("pd (proba risque-neutre, baisse)", f"{p['pd']:.6f}"))

    fig = graphe_arbre(res["niveaux"], res["dates_niveaux"], modele == "crr")
    blocs = [{"type": "cartes", "titre": "Prix et paramètres de l'arbre", "cartes": cartes},
             bloc_graphe("Arbre " + nom_modele(modele), fig, "arbre")]

    if style == "Am":
        exercices = res["exercices"]
        if len(exercices) == 0:
            blocs.append({"type": "cartes", "titre": "Exercice anticipé",
                          "cartes": [carte("Dates d'exercice", "Aucune")]})
        else:
            lignes = []
            for g in exercices:
                if g["pas_debut"] == g["pas_fin"]:
                    pas = str(g["pas_debut"])
                else:
                    pas = f"{g['pas_debut']} – {g['pas_fin']}"
                lignes.append(ligne([g["date"], pas, f"{g['valeur']:.4f}"]))
            blocs.append(tableau("Exercice anticipé", ["Date", "Pas", "Valeur d'exercice"], lignes))
    return jsonify(blocs=blocs)



@app.route("/api/cv", methods=["POST"])
def api_cv():
    d = request.get_json()
    q = verifier(d)
    n = lire_entier(d, "n", 1, 2000, "N")
    n_ref = lire_entier(d, "n_ref", 1, 5000, "N de la référence")
    modele, modele_ref, typ = d["model"], d["ref_model"], d["type"]

    cv = Cv(d["k"], d["s0"], d["r"], d["t"], d["sigma"], date.today().isoformat(), q)
    prix_cv, prix_am = cv.cv(n, modele, typ, modele_ref, n_ref)
    prix_bs, prix_eu, prix_ref = cv.valeur_bs, cv.valeur_eu, cv.valeur_ref

    erreur_arbre = abs(prix_am - prix_ref)
    erreur_cv = abs(prix_cv - prix_ref)

    if erreur_cv < 0.001 and erreur_arbre < 0.001:
        conclusion = f"Arbre déjà exact : erreur {erreur_arbre:.4f}"
    elif erreur_cv < 0.001:
        conclusion = f"Amélioration parfaite : erreur passe de {erreur_arbre:.4f} à {erreur_cv:.4f}"
    elif erreur_cv < erreur_arbre:
        conclusion = f"Amélioration partielle : erreur passe de {erreur_arbre:.4f} à {erreur_cv:.4f}"
    else:
        conclusion = f"Pas d'amélioration : erreur passe de {erreur_arbre:.4f} à {erreur_cv:.4f}"

    nom, nom_ref = nom_modele(modele), nom_modele(modele_ref)
    lignes = [
        ligne(["Black-Scholes (européen)", f"{prix_bs:.4f}"]),
        ligne([f"Arbre {nom} européen (N = {n})", f"{prix_eu:.4f}"]),
        ligne([f"Arbre {nom} américain (N = {n})", f"{prix_am:.4f}"]),
        ligne(["Américain corrigé (CV)", f"{prix_cv:.4f}"]),
        ligne([f"Référence {nom_ref} américain (N = {n_ref})", f"{prix_ref:.4f}"]),
        ligne(["|Américain corrigé − Vréf|", f"{erreur_cv:.4f}"]),
        ligne(["|Américain arbre − Vréf|", f"{erreur_arbre:.4f}"]),
        ligne(["Conclusion", conclusion], "hl note"),      # seule ligne en bleu
    ]
    return jsonify(blocs=[tableau("Control Variate Analyse", ["", "Valeur"], lignes)])


# ---------- Convergence ----------

@app.route("/api/convergence", methods=["POST"])
def api_convergence():
    d = request.get_json()
    q = verifier(d)
    n_max = lire_entier(d, "n", 10, 300, "N max")
    typ = d["type"]
    prix_bs = BlackScholes(d["k"], d["s0"], d["t"], d["r"], d["sigma"], q).pricing_bs(typ)

    def prix_europeen(modele, n):
        try:
            return remonter(modele, d["k"], d["s0"], d["r"], d["t"], d["sigma"], q, n, typ, "Eu")["prix"]
        except ValueError:       
            return None

    ns = list(range(5, n_max + 1))
    prix_crr = [prix_europeen("crr", n) for n in ns]
    prix_boyle = [prix_europeen("boyle", n) for n in ns]

    cartes = []
    for nom, prix in [("CRR", prix_crr), ("Boyle", prix_boyle)]:
        valeurs = [p for p in prix if p is not None]
        if valeurs:
            erreur = f"{abs(valeurs[-1] - prix_bs):.5f}"
        else:
            erreur = "—"
        cartes.append(carte(f"Erreur {nom} (N = {n_max})", erreur))

    fig = graphe_convergence(ns, prix_crr, prix_boyle, prix_bs)
    return jsonify(blocs=[bloc_graphe("Prix européen en fonction de N", fig),
                          {"type": "cartes", "titre": "Erreur vs Black-Scholes", "cartes": cartes}])


if __name__ == "__main__":
    app.run(debug=True)