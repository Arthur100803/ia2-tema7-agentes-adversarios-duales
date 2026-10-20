"""
analyze.py — Paso 3 del experimento: análisis estadístico, tablas y figuras.

Lee los resultados de experiment.py (results_runs_*.csv y window_scores_*.pkl) y las
salidas de prep.py, y genera en OUT_DIR:

    tabla_descriptiva.csv                    media y DE de cada métrica (Tabla 4 del artículo)
    tabla_pre_post.csv                       pre-prueba (IF) frente a post-prueba (LSTM-AE, LSTM-GAN)
    tabla_inferencial.csv                    Shapiro-Wilk, Wilcoxon, t pareada y tamaño del efecto (n = 30)
    tabla_inferencial_por_subconjunto.csv    Wilcoxon por subconjunto (n = 10)
    tabla_bootstrap_auc.csv                  IC 95 % de la diferencia de AUC (bootstrap)
    tabla_sensibilidad_umbral.csv            F1 y FPR con umbrales P95, P99 y P99.9
    tabla_dataset.csv                        descripción de los datos (Tabla 1 del artículo)
    RESULTADOS.md                            todas las tablas en un solo archivo
    figs/fig1_indicadores.png                barras de F1, FPR y AUC
    figs/fig2_roc.png                        curvas ROC
    figs/fig3_serie_puntuacion.png           puntuación del LSTM-AE en el tiempo

Uso
---
    python analyze.py <OUT_DIR>
    python analyze.py out            # después de correr prep.py y experiment.py
    python analyze.py resultados     # regenera las tablas con los resultados ya guardados

    En PyCharm se puede ejecutar sin parámetros: usa <repo>/resultados.

Nota: este archivo solo tiene comentarios y orden nuevos; los cálculos son los mismos
que produjeron resultados/.
"""
import sys, os, json, pickle, glob
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates

# =====================================================================================
# 0. Entrada, estilo de las figuras y carga de resultados
# =====================================================================================
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "resultados")
FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)

COL = {"IsolationForest": "#2a78d6", "LSTM-AE": "#eb6834", "LSTM-GAN": "#1baf7a"}   # color de cada modelo
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

# Todas las corridas (una fila por subconjunto, semilla y modelo)
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/results_runs_*.csv"))], ignore_index=True)
SUBS = [x for x in ["logdb", "loggit", "logsapp"] if x in set(R.subset)]
MODELS = ["IsolationForest", "LSTM-AE", "LSTM-GAN"]
IND = ["F1", "FPR", "AUC"]                     # indicadores de las tres dimensiones de la VD
md = []                                         # texto de RESULTADOS.md

# =====================================================================================
# 1. Estadística descriptiva por subconjunto y modelo
# =====================================================================================
desc = R.groupby(["subset", "model"])[["F1", "FPR", "AUC", "P", "R", "MCC", "epochs"]].agg(["mean", "std"]).round(3)
desc.to_csv(f"{OUT}/tabla_descriptiva.csv"); md += ["## Tabla descriptiva (media y DE sobre 10 semillas)", desc.to_markdown(), ""]

# =====================================================================================
# 2. Pre-prueba (IF) frente a post-prueba (LSTM-AE y LSTM-GAN)
#    diferencia absoluta (post - pre) y variación relativa (%)
# =====================================================================================
rows = []
for post_m in ["LSTM-AE", "LSTM-GAN"]:
    for sub in SUBS + ["consolidado"]:
        q = R if sub == "consolidado" else R[R.subset == sub]
        for ind in IND:
            pre = q[q.model == "IsolationForest"][ind].mean(); post = q[q.model == post_m][ind].mean()
            rows.append(dict(modelo_post=post_m, subconjunto=sub, indicador=ind, pre=round(pre, 2), post=round(post, 2),
                             delta=round(post - pre, 2), delta_pct=round(100 * (post - pre) / pre, 2)))
dd = pd.DataFrame(rows); dd.to_csv(f"{OUT}/tabla_pre_post.csv", index=False); md += ["## Pre-prueba (IF) vs post-prueba (LSTM-AE, LSTM-GAN)", dd.to_markdown(index=False), ""]


# =====================================================================================
# 3. Contraste de hipótesis con pruebas pareadas
# =====================================================================================
def paired(a_model, b_model, label):
    """Compara b_model contra a_model sobre los 30 pares (10 semillas x 3 subconjuntos).

    - Shapiro-Wilk: ¿las diferencias son normales? (si no, la prueba principal es Wilcoxon)
    - Wilcoxon de rangos con signo (prueba principal) y t pareada (referencia)
    - Tamaño del efecto r = |Z| / sqrt(n), con Z por aproximación normal
    """
    out = []
    for ind in IND + ["MCC"]:
        a = R[R.model == a_model].sort_values(["subset", "seed"])[ind].values
        b = R[R.model == b_model].sort_values(["subset", "seed"])[ind].values
        d = b - a; sw = stats.shapiro(d)
        w = stats.wilcoxon(b, a, zero_method="wilcox"); t = stats.ttest_rel(b, a)
        nz = int(np.sum(d != 0)); mu = nz * (nz + 1) / 4; sg = np.sqrt(nz * (nz + 1) * (2 * nz + 1) / 24); r_eff = abs(w.statistic - mu) / sg / np.sqrt(len(d))
        out.append(dict(comparacion=label, indicador=ind, n_pares=len(d), media_dif=round(d.mean(), 3), de_dif=round(d.std(ddof=1), 3),
                        shapiro_p=round(sw.pvalue, 4), wilcoxon_W=w.statistic, wilcoxon_p=w.pvalue, r_efecto=round(r_eff, 3),
                        t=round(t.statistic, 3), t_p=t.pvalue))
    return out


# H1 y H2: LSTM-AE frente a IF; H3: LSTM-GAN frente a LSTM-AE; complementaria: LSTM-GAN frente a IF
inf = pd.DataFrame(paired("IsolationForest", "LSTM-AE", "LSTM-AE vs IF (H1, H2)") + paired("LSTM-AE", "LSTM-GAN", "LSTM-GAN vs LSTM-AE (H3)") + paired("IsolationForest", "LSTM-GAN", "LSTM-GAN vs IF"))

# Las mismas comparaciones por subconjunto (10 pares cada una)
ps = []
for am, bm in [("IsolationForest", "LSTM-AE"), ("LSTM-AE", "LSTM-GAN"), ("IsolationForest", "LSTM-GAN")]:
    for sub in SUBS:
        for ind in IND:
            a = R[(R.subset == sub) & (R.model == am)].sort_values("seed")[ind].values
            b = R[(R.subset == sub) & (R.model == bm)].sort_values("seed")[ind].values
            w = stats.wilcoxon(b, a) if np.any(b != a) else None
            ps.append(dict(comparacion=f"{bm} vs {am}", subconjunto=sub, indicador=ind, media_a=round(a.mean(), 2), media_b=round(b.mean(), 2),
                           wilcoxon_W=None if w is None else w.statistic, p=None if w is None else round(w.pvalue, 4)))
ps = pd.DataFrame(ps); ps.to_csv(f"{OUT}/tabla_inferencial_por_subconjunto.csv", index=False)
inf.to_csv(f"{OUT}/tabla_inferencial.csv", index=False); md += ["## Pruebas pareadas", inf.to_markdown(index=False), "", "## Wilcoxon por subconjunto", ps.to_markdown(index=False), ""]

# =====================================================================================
# 4. Bootstrap del AUC y sensibilidad al umbral
#    (usa las puntuaciones de cada ventana guardadas por experiment.py)
# =====================================================================================
S = {}
for f in sorted(glob.glob(f"{OUT}/window_scores_*.pkl")): S.update(pickle.load(open(f, "rb")))
boot = []; roc_data = {}
rng = np.random.default_rng(0)                  # semilla fija: el bootstrap siempre da lo mismo
sens = []
for sub in SUBS:
    lab = pd.read_csv(f"{OUT}/{sub}_windows.csv", parse_dates=["win"]).sort_values("win").reset_index(drop=True)
    n = len(lab); te = np.arange(n) >= int(n * 0.8); va = (np.arange(n) >= int(n * 0.6)) & ~te; y = lab.y.values   # mismos tramos que experiment.py
    seeds = sorted({k[1] for k in S if k[0] == sub})
    # puntuación promedio de las 10 semillas (en rangos, para que todas pesen igual)
    avg = {m: np.mean([stats.rankdata(S[(sub, s, m)]) / n for s in seeds], 0) for m in MODELS}
    roc_data[sub] = (y[te], {m: avg[m][te] for m in MODELS}, lab[te], {m: (avg[m][te], np.percentile(avg[m][va], 95)) for m in MODELS})

    # --- 4.1 sensibilidad: mismas puntuaciones con umbral en P95, P99 y P99.9 de validación ---
    for m in MODELS:
        for pc in [95, 99, 99.9]:
            f1s, fprs, rs = [], [], []
            for sd in seeds:
                sc = S[(sub, sd, m)]; thr = np.percentile(sc[va], pc); p = sc[te] > thr; yy = y[te]
                f1s.append(100 * f1_score(yy, p, zero_division=0)); fprs.append(100 * (p & (yy == 0)).sum() / (yy == 0).sum()); rs.append(100 * (p & (yy == 1)).sum() / max(1, yy.sum()))
            sens.append(dict(subconjunto=sub, modelo=m, percentil=pc, F1=round(np.mean(f1s), 2), FPR=round(np.mean(fprs), 2), Recall=round(np.mean(rs), 2)))

    # --- 4.2 bootstrap: 2000 remuestreos de las ventanas de prueba -> IC 95 % de la diferencia de AUC ---
    yt = y[te]; idx = np.arange(te.sum())
    for am, bm in [("IsolationForest", "LSTM-AE"), ("LSTM-AE", "LSTM-GAN")]:
        diffs = []
        for _ in range(2000):
            bi = rng.choice(idx, len(idx))
            if yt[bi].min() == yt[bi].max(): continue          # la muestra necesita ambas clases
            diffs.append(roc_auc_score(yt[bi], avg[bm][te][bi]) - roc_auc_score(yt[bi], avg[am][te][bi]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        boot.append(dict(comparacion=f"{bm} - {am}", subconjunto=sub, auc_a=round(100 * roc_auc_score(yt, avg[am][te]), 2),
                         auc_b=round(100 * roc_auc_score(yt, avg[bm][te]), 2), dif=round(100 * np.mean(diffs), 2),
                         IC95_inf=round(100 * lo, 2), IC95_sup=round(100 * hi, 2)))
sn = pd.DataFrame(sens); sn.to_csv(f"{OUT}/tabla_sensibilidad_umbral.csv", index=False); md += ["## Sensibilidad al umbral (media de semillas)", sn.to_markdown(index=False), ""]
bt = pd.DataFrame(boot); bt.to_csv(f"{OUT}/tabla_bootstrap_auc.csv", index=False); md += ["## Bootstrap dif. AUC", bt.to_markdown(index=False), ""]

# =====================================================================================
# 5. Descripción del dataset
# =====================================================================================
dsr = []
for sub in SUBS:
    info = json.load(open(f"{OUT}/{sub}_info.json")); ev = pd.read_parquet(f"{OUT}/{sub}_events.parquet")
    lab = pd.read_csv(f"{OUT}/{sub}_windows.csv")
    rules = {c: int((lab[c] > 0).sum()) for c in lab.columns if c not in ("win", "y", "n_events")}   # ventanas que activan cada regla
    dsr.append(dict(subconjunto=sub, eventos=len(ev), plantillas=info["n_templates"], ventanas=info["n_win"],
                    desde=str(ev.ts.min())[:16], hasta=str(ev.ts.max())[:16], **{f"ventanas_{k}": v for k, v in info["split"].items()},
                    **{f"maliciosas_{k}": v for k, v in info["mal"].items()}, reglas_activadas=json.dumps(rules)))
pd.DataFrame(dsr).to_csv(f"{OUT}/tabla_dataset.csv", index=False); md += ["## Dataset", pd.DataFrame(dsr).T.to_markdown(), ""]
open(f"{OUT}/RESULTADOS.md", "w", encoding="utf-8").write("\n".join(md))

# =====================================================================================
# 6. Figuras
# =====================================================================================
# Figura 1: F1, FPR y AUC por modelo y subconjunto (barras = media, líneas = DE)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, ind in zip(axs, IND):
    for j, sub in enumerate(SUBS):
        for k, m in enumerate(MODELS):
            v = R[(R.subset == sub) & (R.model == m)][ind]; x = j + (k - 1) * 0.26
            ax.bar(x, v.mean(), 0.24, color=COL[m], label=m if j == 0 else None, edgecolor="white", linewidth=1)
            ax.errorbar(x, v.mean(), v.std(), color=INK2, capsize=2, lw=1)
            ax.text(x, v.mean() + v.std() + (0.15 if ind == "FPR" else 1.2), f"{v.mean():.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(SUBS)), SUBS); ax.set_title(f"{ind} (%)" + (" — menor es mejor" if ind == "FPR" else ""), color=INK, fontsize=9)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_indicadores.png", dpi=200); plt.close()

# Figura 2: curvas ROC con las puntuaciones promediadas sobre las semillas
fig, axs = plt.subplots(1, len(SUBS), figsize=(4 * len(SUBS), 3.6))
for ax, sub in zip(axs, SUBS):
    yt, sc, _, _ = roc_data[sub]
    for m in MODELS:
        fpr, tpr, _ = roc_curve(yt, sc[m]); ax.plot(fpr, tpr, color=COL[m], lw=2, label=f"{m} (AUC={100*roc_auc_score(yt, sc[m]):.1f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRID); ax.set_title(sub, color=INK); ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_roc.png", dpi=200); plt.close()

# Figura 3: puntuación del LSTM-AE en el tramo de prueba, umbral rho y ventanas maliciosas
fig, axs = plt.subplots(len(SUBS), 1, figsize=(10, 2.5 * len(SUBS)))
for ax, sub in zip(axs, SUBS):
    yt, _, labt, thr = roc_data[sub]; s, rho = thr["LSTM-AE"]
    t = labt.win.reset_index(drop=True); seg = (t.diff() > pd.Timedelta("30min")).cumsum()   # no unir huecos sin registros
    for i, g in enumerate(seg.unique()):
        k = (seg == g).values; ax.plot(t[k], s[k], color=COL["LSTM-AE"], lw=0.8, label="Puntuación LSTM-AE" if i == 0 else None)
    ax.axhline(rho, color=INK2, ls="--", lw=1, label="Umbral ρ (P95 validación)")
    ax.scatter(labt.win[yt == 1], s[yt == 1], color="#d03b3b", s=10, zorder=3, marker="x", label="Ventana maliciosa (etiqueta débil)")
    ax.set_title(sub, color=INK, fontsize=9); ax.set_ylabel("Puntuación"); loc = mdates.AutoDateLocator(maxticks=8); ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc)); ax.set_ylim(0, 1.25); ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_serie_puntuacion.png", dpi=200); plt.close()
print("\n".join(md))
