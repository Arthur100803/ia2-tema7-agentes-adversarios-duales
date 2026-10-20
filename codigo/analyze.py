"""Analisis estadistico y figuras. Uso: python -I analyze.py <OUT_DIR>"""
import sys, json, pickle, glob
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates

OUT = sys.argv[1]; FIG = f"{OUT}/figs"; import os; os.makedirs(FIG, exist_ok=True)
COL = {"IsolationForest": "#2a78d6", "LSTM-AE": "#eb6834", "LSTM-GAN": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/results_runs_*.csv"))], ignore_index=True)
SUBS = [x for x in ["logdb", "loggit", "logsapp"] if x in set(R.subset)]; MODELS = ["IsolationForest", "LSTM-AE", "LSTM-GAN"]; IND = ["F1", "FPR", "AUC"]
md = []

# 1. Descriptivos por subconjunto y modelo
desc = R.groupby(["subset", "model"])[["F1", "FPR", "AUC", "P", "R", "MCC", "epochs"]].agg(["mean", "std"]).round(3)
desc.to_csv(f"{OUT}/tabla_descriptiva.csv"); md += ["## Tabla descriptiva (media y DE sobre 10 semillas)", desc.to_markdown(), ""]

# 2. Pre (IF) vs Post (LSTM-GAN): diferencias absolutas y relativas
rows = []
for sub in SUBS + ["consolidado"]:
    q = R if sub == "consolidado" else R[R.subset == sub]
    for ind in IND:
        pre = q[q.model == "IsolationForest"][ind].mean(); post = q[q.model == "LSTM-GAN"][ind].mean()
        rows.append(dict(subconjunto=sub, indicador=ind, pre=round(pre, 2), post=round(post, 2),
                         delta=round(post - pre, 2), delta_pct=round(100 * (post - pre) / pre, 2)))
dd = pd.DataFrame(rows); dd.to_csv(f"{OUT}/tabla_pre_post.csv", index=False); md += ["## Pre-prueba (IF) vs post-prueba (LSTM-GAN)", dd.to_markdown(index=False), ""]

# 3. Pruebas inferenciales pareadas (10 semillas x n subconjuntos)
def paired(a_model, b_model, label):
    out = []
    for ind in IND + ["MCC"]:
        a = R[R.model == a_model].sort_values(["subset", "seed"])[ind].values
        b = R[R.model == b_model].sort_values(["subset", "seed"])[ind].values
        d = b - a; sw = stats.shapiro(d)
        w = stats.wilcoxon(b, a, zero_method="wilcox"); t = stats.ttest_rel(b, a)
        z = stats.norm.isf(w.pvalue / 2); r_eff = z / np.sqrt(len(d))
        out.append(dict(comparacion=label, indicador=ind, n_pares=len(d), media_dif=round(d.mean(), 3), de_dif=round(d.std(ddof=1), 3),
                        shapiro_p=round(sw.pvalue, 4), wilcoxon_W=w.statistic, wilcoxon_p=w.pvalue, r_efecto=round(r_eff, 3),
                        t=round(t.statistic, 3), t_p=t.pvalue))
    return out
inf = pd.DataFrame(paired("IsolationForest", "LSTM-GAN", "LSTM-GAN vs IF") + paired("LSTM-AE", "LSTM-GAN", "LSTM-GAN vs LSTM-AE (ablacion)"))
ps = []                                                         # por subconjunto (10 pares cada uno)
for sub in SUBS:
    for ind in IND:
        a = R[(R.subset == sub) & (R.model == "IsolationForest")].sort_values("seed")[ind].values
        b = R[(R.subset == sub) & (R.model == "LSTM-GAN")].sort_values("seed")[ind].values
        w = stats.wilcoxon(b, a) if np.any(b != a) else None
        ps.append(dict(subconjunto=sub, indicador=ind, media_IF=round(a.mean(), 2), media_GAN=round(b.mean(), 2),
                       wilcoxon_W=None if w is None else w.statistic, p=None if w is None else round(w.pvalue, 4)))
ps = pd.DataFrame(ps); ps.to_csv(f"{OUT}/tabla_inferencial_por_subconjunto.csv", index=False)
inf.to_csv(f"{OUT}/tabla_inferencial.csv", index=False); md += ["## Pruebas pareadas", inf.to_markdown(index=False), "", "## Wilcoxon por subconjunto (IF vs LSTM-GAN)", ps.to_markdown(index=False), ""]

# 4. Bootstrap de la diferencia de AUC a nivel de ventana (puntuaciones promediadas sobre semillas)
S = {}
for f in sorted(glob.glob(f"{OUT}/window_scores_*.pkl")): S.update(pickle.load(open(f, "rb")))
boot = []; roc_data = {}
rng = np.random.default_rng(0)
sens = []
for sub in SUBS:
    lab = pd.read_csv(f"{OUT}/{sub}_windows.csv", parse_dates=["win"]).sort_values("win").reset_index(drop=True)
    n = len(lab); te = np.arange(n) >= int(n * 0.8); va = (np.arange(n) >= int(n * 0.6)) & ~te; y = lab.y.values
    seeds = sorted({k[1] for k in S if k[0] == sub})
    avg = {m: np.mean([stats.rankdata(S[(sub, s, m)]) / n for s in seeds], 0) for m in MODELS}
    roc_data[sub] = (y[te], {m: avg[m][te] for m in MODELS}, lab[te], {m: (avg[m][te], np.percentile(avg[m][va], 95)) for m in MODELS})
    from sklearn.metrics import f1_score
    for m in MODELS:                                            # sensibilidad al percentil del umbral (post hoc, mismas puntuaciones)
        for pc in [95, 99, 99.9]:
            f1s, fprs, rs = [], [], []
            for sd in seeds:
                sc = S[(sub, sd, m)]; thr = np.percentile(sc[va], pc); p = sc[te] > thr; yy = y[te]
                f1s.append(100 * f1_score(yy, p, zero_division=0)); fprs.append(100 * (p & (yy == 0)).sum() / (yy == 0).sum()); rs.append(100 * (p & (yy == 1)).sum() / max(1, yy.sum()))
            sens.append(dict(subconjunto=sub, modelo=m, percentil=pc, F1=round(np.mean(f1s), 2), FPR=round(np.mean(fprs), 2), Recall=round(np.mean(rs), 2)))
    yt = y[te]; idx = np.arange(te.sum()); diffs = []
    for _ in range(2000):
        bi = rng.choice(idx, len(idx));
        if yt[bi].min() == yt[bi].max(): continue
        diffs.append(roc_auc_score(yt[bi], avg["LSTM-GAN"][te][bi]) - roc_auc_score(yt[bi], avg["IsolationForest"][te][bi]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    boot.append(dict(subconjunto=sub, auc_IF=round(100 * roc_auc_score(yt, avg["IsolationForest"][te]), 2),
                     auc_GAN=round(100 * roc_auc_score(yt, avg["LSTM-GAN"][te]), 2), dif=round(100 * np.mean(diffs), 2),
                     IC95_inf=round(100 * lo, 2), IC95_sup=round(100 * hi, 2)))
sn = pd.DataFrame(sens); sn.to_csv(f"{OUT}/tabla_sensibilidad_umbral.csv", index=False); md += ["## Sensibilidad al umbral (media de semillas)", sn.to_markdown(index=False), ""]
bt = pd.DataFrame(boot); bt.to_csv(f"{OUT}/tabla_bootstrap_auc.csv", index=False); md += ["## Bootstrap dif. AUC (GAN - IF)", bt.to_markdown(index=False), ""]

# 5. Descripcion del dataset
dsr = []
for sub in SUBS:
    info = json.load(open(f"{OUT}/{sub}_info.json")); ev = pd.read_parquet(f"{OUT}/{sub}_events.parquet")
    lab = pd.read_csv(f"{OUT}/{sub}_windows.csv")
    rules = {c: int((lab[c] > 0).sum()) for c in lab.columns if c not in ("win", "y", "n_events")}
    dsr.append(dict(subconjunto=sub, eventos=len(ev), plantillas=info["n_templates"], ventanas=info["n_win"],
                    desde=str(ev.ts.min())[:16], hasta=str(ev.ts.max())[:16], **{f"ventanas_{k}": v for k, v in info["split"].items()},
                    **{f"maliciosas_{k}": v for k, v in info["mal"].items()}, reglas_activadas=json.dumps(rules)))
pd.DataFrame(dsr).to_csv(f"{OUT}/tabla_dataset.csv", index=False); md += ["## Dataset", pd.DataFrame(dsr).T.to_markdown(), ""]
open(f"{OUT}/RESULTADOS.md", "w").write("\n".join(md))

# ---------------- Figuras
# Fig. 1: F1, FPR, AUC por modelo y subconjunto (media +- DE)
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

# Fig. 2: curvas ROC (puntuaciones promediadas sobre semillas)
fig, axs = plt.subplots(1, len(SUBS), figsize=(4 * len(SUBS), 3.6))
for ax, sub in zip(axs, SUBS):
    yt, sc, _, _ = roc_data[sub]
    for m in MODELS:
        fpr, tpr, _ = roc_curve(yt, sc[m]); ax.plot(fpr, tpr, color=COL[m], lw=2, label=f"{m} (AUC={100*roc_auc_score(yt, sc[m]):.1f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GRID); ax.set_title(sub, color=INK); ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_roc.png", dpi=200); plt.close()

# Fig. 3: puntuacion S(X) del LSTM-GAN en el tramo de prueba, umbral y ventanas etiquetadas como maliciosas
fig, axs = plt.subplots(len(SUBS), 1, figsize=(10, 2.5 * len(SUBS)))
for ax, sub in zip(axs, SUBS):
    yt, _, labt, thr = roc_data[sub]; s, rho = thr["LSTM-GAN"]
    t = labt.win.reset_index(drop=True); seg = (t.diff() > pd.Timedelta("30min")).cumsum()   # no unir huecos sin registros
    for i, g in enumerate(seg.unique()):
        k = (seg == g).values; ax.plot(t[k], s[k], color=COL["LSTM-GAN"], lw=0.8, label="S(X) LSTM-GAN" if i == 0 else None)
    ax.axhline(rho, color=INK2, ls="--", lw=1, label="Umbral ρ (P95 validación)")
    ax.scatter(labt.win[yt == 1], s[yt == 1], color="#d03b3b", s=10, zorder=3, marker="x", label="Ventana maliciosa (etiqueta débil)")
    ax.set_title(sub, color=INK, fontsize=9); ax.set_ylabel("Puntuación"); loc = mdates.AutoDateLocator(maxticks=8); ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc)); ax.set_ylim(0, 1.25); ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_serie_puntuacion.png", dpi=200); plt.close()
print("\n".join(md))
