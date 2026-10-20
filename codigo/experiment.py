"""
experiment.py — Paso 2 del experimento: entrenamiento y evaluación de los modelos.

Compara tres modelos no supervisados sobre las ventanas de 5 minutos creadas por prep.py:
    - Isolation Forest   línea base tradicional (pre-prueba)
    - LSTM-AE            reconstrucción temporal latente (post-prueba)
    - LSTM-GAN           LSTM-AE + discriminador adversarial (post-prueba)

Flujo por subconjunto y por semilla
-----------------------------------
1. Partición cronológica de las ventanas: 60 % entrenamiento, 20 % validación, 20 % prueba.
2. Isolation Forest sobre el vector log(1 + conteo de plantillas) de cada ventana.
3. Word2Vec sobre el texto de las plantillas -> un vector de 300 números por plantilla.
4. Serie semántica: cada ventana es la suma ponderada de los vectores de sus plantillas;
   el modelo ve series de W = 20 ventanas (100 minutos).
5. Entrenamiento del LSTM-AE y del LSTM-GAN (sin etiquetas).
6. Puntuación de anomalía de cada ventana, umbral rho = percentil 95 de validación y
   métricas en el tramo de prueba (F1, FPR, AUC, precisión, exhaustividad, MCC).

Salidas en OUT_DIR
------------------
    results_runs_<subconjuntos>.csv    una fila por subconjunto, semilla y modelo
    window_scores_<subconjuntos>.pkl   puntuación de todas las ventanas (para analyze.py)
    <sub>_info.json                    tamaño de cada tramo y ventanas maliciosas

Uso
---
    python experiment.py <OUT_DIR> <n_semillas> [max_epocas] [subconjuntos]
    python experiment.py out 10 30 logdb,loggit      # ~35 min en 2 núcleos
    python experiment.py out 10 30 logsapp           # ~70 min
    python experiment.py out 1 30 logdb              # prueba rápida (1 semilla)

    En PyCharm se puede ejecutar sin parámetros: usa <repo>/out, 10 semillas,
    30 épocas y los subconjuntos logdb y loggit.

Nota: este archivo solo tiene comentarios y orden nuevos; la lógica, los
hiperparámetros y las semillas son los mismos que produjeron resultados/.
"""
import sys, os, time, json, re, pickle
import numpy as np, pandas as pd, torch, torch.nn as nn
from gensim.models import Word2Vec
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix

# =====================================================================================
# 0. Parámetros de entrada e hiperparámetros
# =====================================================================================
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "out")   # carpeta con las salidas de prep.py
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 10                    # número de semillas (0 .. NSEED-1)
MAXEP = int(sys.argv[3]) if len(sys.argv) > 3 else 30                    # épocas máximas de entrenamiento
SUBS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["logdb", "loggit"]
TAG = "_".join(SUBS)                                                     # sufijo de los archivos de salida

torch.set_num_threads(os.cpu_count())      # usa todos los núcleos de la CPU

W, D, H = 20, 300, 128                     # ventanas por serie (100 min), dim. Word2Vec, estado oculto del LSTM
BATCH, LR, PATIENCE = 1024, 1e-3, 5        # tamaño de lote, tasa de aprendizaje (Adam), paciencia de parada temprana
MAX_PER_EPOCH = 8192                       # series muestreadas por época (solo limita a logsapp, el más grande)
ALPHA = 0.8                                # peso de L_rec en la pérdida del generador: 0.8 L_rec + 0.2 L_GAN
LAMBDA = 0.7                               # peso de la reconstrucción en la puntuación dual S(X)
PCTL = 95                                  # percentil de validación que define el umbral rho
SPLIT = (0.6, 0.2, 0.2)                    # entrenamiento / validación / prueba (cronológico)


# =====================================================================================
# 1. Datos: partición, Word2Vec y series de ventanas
# =====================================================================================
def split_windows(win):
    """Marca cada ventana como 'train', 'val' o 'test' según su posición en el tiempo."""
    n = len(win); a, b = int(n*SPLIT[0]), int(n*(SPLIT[0]+SPLIT[1]))
    part = np.array(["train"]*a + ["val"]*(b-a) + ["test"]*(n-b)); return part


def w2v_embeddings(templates, train_tids, seed):
    """Vector semántico (300 dim.) de cada plantilla de Drain3.

    Se parte el texto de cada plantilla en palabras, se entrena Word2Vec (skip-gram)
    sobre el vocabulario de plantillas (sin etiquetas) y el vector de la plantilla es
    el promedio normalizado de sus palabras. (train_tids no se usa; se mantiene la firma.)
    """
    tok = lambda s: [t for t in re.split(r"[\s:=,()\[\]]+", s.lower()) if t and t != "<*>"] or ["<empty>"]
    corpus = [tok(templates[t]) for t in sorted(templates)]   # vocabulario completo de plantillas (sin etiquetas)
    m = Word2Vec(corpus * 20, vector_size=D, window=5, min_count=1, sg=1, seed=seed, workers=1, epochs=5)
    E = np.zeros((max(templates) + 1, D), dtype=np.float32)          # fila 0 = relleno (PAD)
    for t, s in templates.items():
        vs = [m.wv[w] for w in tok(s) if w in m.wv]
        if vs:
            v = np.mean(vs, 0); E[t] = v / (np.linalg.norm(v) + 1e-8)
    return E


def window_series(C, E, tids, train_mask):
    """Construye la serie temporal semántica de cada ventana.

    x_t = sum_k log(1 + c_tk) * e_k   (c_tk = conteo de la plantilla k en la ventana t)
    Se estandariza con la media y la desviación del tramo de entrenamiento.
    Para cada ventana t se arma la serie de las W ventanas que terminan en t
    (con relleno a la izquierda al inicio de los datos). M marca las posiciones reales.
    """
    F = C @ E[tids]                                            # (n_ventanas, D)
    mu, sd = F[train_mask].mean(0), F[train_mask].std(0) + 1e-6
    F = ((F - mu) / sd).astype(np.float32)
    n = len(F); X = np.zeros((n, W, D), np.float32); M = np.zeros((n, W), np.float32)
    for t in range(n):
        lo = max(0, t - W + 1); X[t, W-(t-lo+1):] = F[lo:t+1]; M[t, W-(t-lo+1):] = 1   # relleno a la izquierda
    return X, M


# =====================================================================================
# 2. Modelos: autocodificador LSTM (generador) y discriminador
# =====================================================================================
class Gen(nn.Module):
    """Autocodificador LSTM (generador G).

    El codificador resume la serie de W ventanas en un estado oculto de H números;
    el decodificador recibe ese resumen en cada paso y reconstruye la serie completa.
    """
    def __init__(s): super().__init__(); s.enc = nn.LSTM(D, H, batch_first=True); s.dec = nn.LSTM(H, H, batch_first=True); s.out = nn.Linear(H, D)
    def encode(s, x): hs, (h, _) = s.enc(x); return hs, h[-1]                    # estados ocultos y último estado
    def forward(s, x):
        hs, hT = s.encode(x); y, _ = s.dec(hT.unsqueeze(1).repeat(1, x.size(1), 1)); return s.out(y), hs


class Disc(nn.Module):
    """Discriminador secuencial: LSTM de 64 unidades sobre las codificaciones latentes.

    Devuelve la probabilidad de que la secuencia de estados ocultos venga de la entrada
    real (1) y no de la recodificación de la reconstrucción (0).
    """
    def __init__(s): super().__init__(); s.rnn = nn.LSTM(H, 64, batch_first=True); s.fc = nn.Linear(64, 1)
    def forward(s, hs): _, (h, _) = s.rnn(hs); return torch.sigmoid(s.fc(h[-1])).squeeze(1)


def rec_err(x, xh, m):
    """Error cuadrático medio de reconstrucción por serie (solo en posiciones reales, m = 1)."""
    return (((x - xh) ** 2).sum(-1) * m).sum(1) / m.sum(1).clamp(min=1)


# =====================================================================================
# 3. Entrenamiento
# =====================================================================================
def train_model(Xtr, Mtr, Xva, Mva, seed, gan=True):
    """Entrena el LSTM-AE (gan=False) o el LSTM-GAN (gan=True) sin etiquetas.

    LSTM-AE : minimiza L_rec.
    LSTM-GAN: alterna (a) paso del discriminador, que aprende a distinguir lo real de lo
              reconstruido, y (b) paso del generador, que minimiza ALPHA*L_rec + (1-ALPHA)*L_GAN.
    Parada temprana: se detiene si el error de validación no mejora en PATIENCE épocas
    y se queda con el mejor modelo.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    G = Gen(); Dn = Disc()
    og, od = torch.optim.Adam(G.parameters(), LR), torch.optim.Adam(Dn.parameters(), LR)
    bce = nn.BCELoss(); best, bad, state, hist = 1e9, 0, None, []
    for ep in range(MAXEP):
        G.train(); perm = np.random.permutation(len(Xtr))[:MAX_PER_EPOCH]     # series de esta época
        for i in range(0, len(perm), BATCH):
            x, m = torch.tensor(Xtr[perm[i:i+BATCH]]), torch.tensor(Mtr[perm[i:i+BATCH]])
            xh, hs = G(x)
            if gan:                                            # (a) paso del discriminador: maximiza V(D,G)
                hs_f = G.encode(xh.detach() * m.unsqueeze(-1))[0]
                d_r, d_f = Dn(hs.detach()), Dn(hs_f.detach())
                ld = bce(d_r, torch.ones_like(d_r)) + bce(d_f, torch.zeros_like(d_f))
                od.zero_grad(); ld.backward(); od.step()
            lrec = rec_err(x, xh, m).mean()                    # (b) paso del generador
            if gan:
                d_f = Dn(G.encode(xh * m.unsqueeze(-1))[0]); lg = bce(d_f, torch.ones_like(d_f))
                loss = ALPHA * lrec + (1 - ALPHA) * lg         # alfa*L_rec + (1-alfa)*L_GAN (forma no saturante)
            else: loss = lrec
            og.zero_grad(); loss.backward(); og.step()
        # --- validación: error medio de reconstrucción ---
        G.eval()
        with torch.no_grad():
            vl = float(np.mean(np.concatenate([rec_err(torch.tensor(Xva[i:i+2048]), G(torch.tensor(Xva[i:i+2048]))[0], torch.tensor(Mva[i:i+2048])).numpy() for i in range(0, len(Xva), 2048)])))
        hist.append(vl)
        if vl < best - 1e-5: best, bad, state = vl, 0, ({k: v.clone() for k, v in G.state_dict().items()}, {k: v.clone() for k, v in Dn.state_dict().items()})
        else:
            bad += 1
            if bad >= PATIENCE: break                          # parada temprana
    G.load_state_dict(state[0]); Dn.load_state_dict(state[1]); return G, Dn, hist


# =====================================================================================
# 4. Puntuación y métricas
# =====================================================================================
def seq_scores(G, Dn, X, M, bs=2048):
    """Para cada ventana t: error de reconstrucción de la última ventana de su serie
    y probabilidad del discriminador para la serie que termina en t."""
    r, d = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, m = torch.tensor(X[i:i+bs]), torch.tensor(M[i:i+bs]); xh, _ = G(x)
            r.append(((x[:, -1] - xh[:, -1]) ** 2).sum(-1).numpy())
            d.append(Dn(G.encode(xh * m.unsqueeze(-1))[0]).numpy())
    return np.concatenate(r), np.concatenate(d)


def metrics(y, s, thr):
    """Métricas con el umbral thr: una ventana es alerta si su puntuación s > thr."""
    p = (s > thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(F1=100*f1_score(y, p, zero_division=0), FPR=100*fp/(fp+tn), AUC=100*roc_auc_score(y, s),
                P=100*precision_score(y, p, zero_division=0), R=100*recall_score(y, p, zero_division=0),
                MCC=matthews_corrcoef(y, p), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


# =====================================================================================
# 5. Programa principal
# =====================================================================================
rows, wscores = [], {}
for name in SUBS:
    # --- 5.1 cargar las salidas de prep.py ---
    ev = pd.read_parquet(f"{OUT}/{name}_events.parquet"); lab = pd.read_csv(f"{OUT}/{name}_windows.csv", parse_dates=["win"])
    templates = pickle.load(open(f"{OUT}/{name}_templates.pkl", "rb"))
    lab = lab.sort_values("win").reset_index(drop=True); part = split_windows(lab); y = lab.y.values
    tr_tids = set(ev[ev.win.isin(lab.win[part == "train"])].tid)          # plantillas vistas en entrenamiento

    # --- 5.2 matriz de conteos: ventanas x plantillas, en escala log(1 + conteo) ---
    Ct = pd.crosstab(ev.win, ev.tid).reindex(lab.win).fillna(0); tids = Ct.columns.values; C = np.log1p(Ct.values)
    info = dict(n_win=len(lab), split={k: int((part == k).sum()) for k in ["train", "val", "test"]},
                mal={k: int(y[part == k].sum()) for k in ["train", "val", "test"]}, n_templates=len(templates))
    print(name, info, flush=True); json.dump(info, open(f"{OUT}/{name}_info.json", "w"))

    for seed in range(NSEED):
        t0 = time.time()
        # --- 5.3 línea base: Isolation Forest (200 árboles) ---
        iso = IsolationForest(n_estimators=200, random_state=seed).fit(C[part == "train"])
        s_if = -iso.score_samples(C)                          # signo invertido: mayor = más anómalo

        # --- 5.4 representación semántica y series de 20 ventanas ---
        E = w2v_embeddings(templates, tr_tids, seed)
        X, M = window_series(C, E, tids, part == "train")
        res = {"IsolationForest": s_if}

        # --- 5.5 LSTM-AE y LSTM-GAN ---
        for mname, gan in [("LSTM-AE", False), ("LSTM-GAN", True)]:
            tr, va = part == "train", part == "val"
            G, Dn, hist = train_model(X[tr], M[tr], X[va], M[va], seed, gan)
            r, d = seq_scores(G, Dn, X, M)
            rv = np.sort(r[va]); rn = np.searchsorted(rv, r) / len(rv)               # normalización [0,1] con la FDA empírica de validación
            res[mname] = rn if not gan else LAMBDA * rn + (1 - LAMBDA) * (1 - d)       # LSTM-AE: rn ; LSTM-GAN: puntuación dual S(X)
            res[mname + "_D"] = d
            res[mname + "_epochs"] = len(hist)

        # --- 5.6 umbral rho (solo validación) y métricas en prueba ---
        for mname in ["IsolationForest", "LSTM-AE", "LSTM-GAN"]:
            s = res[mname]; thr = np.percentile(s[part == "val"], PCTL)
            mt = metrics(y[part == "test"], s[part == "test"], thr)
            rows.append(dict(subset=name, seed=seed, model=mname, epochs=res.get(mname + "_epochs", 0), **mt))
            wscores[(name, seed, mname)] = s
        print(name, seed, f"{time.time()-t0:.0f}s", {r["model"]: (round(r["F1"], 1), round(r["FPR"], 1), round(r["AUC"], 1)) for r in rows[-3:]}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/results_runs_{TAG}.csv", index=False)       # se guarda al terminar cada subconjunto
pickle.dump(wscores, open(f"{OUT}/window_scores_{TAG}.pkl", "wb"))
