"""Experimento: Isolation Forest (pre-prueba) vs LSTM-AE (ablacion) vs LSTM-GAN dual (post-prueba).
Uso: python -I experiment.py <OUT_DIR> <n_semillas> [max_epocas]"""
import sys, os, time, json, re, pickle
import numpy as np, pandas as pd, torch, torch.nn as nn
from gensim.models import Word2Vec
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix

OUT, NSEED = sys.argv[1], int(sys.argv[2]); MAXEP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
torch.set_num_threads(os.cpu_count())
W, S, D, H = 20, 4, 300, 128             # ventana de eventos, paso, dim. Word2Vec, estado oculto LSTM
BATCH, LR, PATIENCE = 1024, 1e-3, 5
MAX_PER_EPOCH = 8192                      # secuencias muestreadas por epoca (no afecta tramos menores)
ALPHA, LAMBDA, PCTL = 0.8, 0.7, 95        # peso L_rec en L_total, peso reconstruccion en S(X), percentil del umbral rho
SPLIT = (0.6, 0.2, 0.2)                   # entrenamiento / validacion / prueba (cronologico)

def split_windows(win):
    n = len(win); a, b = int(n*SPLIT[0]), int(n*(SPLIT[0]+SPLIT[1]))
    part = np.array(["train"]*a + ["val"]*(b-a) + ["test"]*(n-b)); return part

def w2v_embeddings(templates, train_tids, seed):
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
    """Serie temporal de ventanas de 5 min: x_t = sum_k log(1+c_tk) * e_k (embedding semantico ponderado),
    estandarizada con media/desv. del tramo de entrenamiento. Devuelve secuencias de W ventanas que terminan en t."""
    F = C @ E[tids]                                            # (n_ventanas, D)
    mu, sd = F[train_mask].mean(0), F[train_mask].std(0) + 1e-6
    F = ((F - mu) / sd).astype(np.float32)
    n = len(F); X = np.zeros((n, W, D), np.float32); M = np.zeros((n, W), np.float32)
    for t in range(n):
        lo = max(0, t - W + 1); X[t, W-(t-lo+1):] = F[lo:t+1]; M[t, W-(t-lo+1):] = 1   # relleno a la izquierda
    return X, M

class Gen(nn.Module):                     # autocodificador LSTM (generador G)
    def __init__(s): super().__init__(); s.enc = nn.LSTM(D, H, batch_first=True); s.dec = nn.LSTM(H, H, batch_first=True); s.out = nn.Linear(H, D)
    def encode(s, x): hs, (h, _) = s.enc(x); return hs, h[-1]
    def forward(s, x):
        hs, hT = s.encode(x); y, _ = s.dec(hT.unsqueeze(1).repeat(1, x.size(1), 1)); return s.out(y), hs
class Disc(nn.Module):                    # discriminador secuencial sobre el espacio latente
    def __init__(s): super().__init__(); s.rnn = nn.LSTM(H, 64, batch_first=True); s.fc = nn.Linear(64, 1)
    def forward(s, hs): _, (h, _) = s.rnn(hs); return torch.sigmoid(s.fc(h[-1])).squeeze(1)

def rec_err(x, xh, m): return (((x - xh) ** 2).sum(-1) * m).sum(1) / m.sum(1).clamp(min=1)

def train_model(Xtr, Mtr, Xva, Mva, seed, gan=True):
    torch.manual_seed(seed); np.random.seed(seed)
    G = Gen(); Dn = Disc()
    og, od = torch.optim.Adam(G.parameters(), LR), torch.optim.Adam(Dn.parameters(), LR)
    bce = nn.BCELoss(); best, bad, state, hist = 1e9, 0, None, []
    for ep in range(MAXEP):
        G.train(); perm = np.random.permutation(len(Xtr))[:MAX_PER_EPOCH]
        for i in range(0, len(perm), BATCH):
            x, m = torch.tensor(Xtr[perm[i:i+BATCH]]), torch.tensor(Mtr[perm[i:i+BATCH]])
            xh, hs = G(x)
            if gan:                                            # paso del discriminador: maximiza V(D,G)
                hs_f = G.encode(xh.detach() * m.unsqueeze(-1))[0]
                d_r, d_f = Dn(hs.detach()), Dn(hs_f.detach())
                ld = bce(d_r, torch.ones_like(d_r)) + bce(d_f, torch.zeros_like(d_f))
                od.zero_grad(); ld.backward(); od.step()
            lrec = rec_err(x, xh, m).mean()                    # paso del generador: alfa*L_rec + (1-alfa)*L_GAN
            if gan:
                d_f = Dn(G.encode(xh * m.unsqueeze(-1))[0]); lg = bce(d_f, torch.ones_like(d_f))
                loss = ALPHA * lrec + (1 - ALPHA) * lg
            else: loss = lrec
            og.zero_grad(); loss.backward(); og.step()
        G.eval()
        with torch.no_grad():
            vl = float(np.mean(np.concatenate([rec_err(torch.tensor(Xva[i:i+2048]), G(torch.tensor(Xva[i:i+2048]))[0], torch.tensor(Mva[i:i+2048])).numpy() for i in range(0, len(Xva), 2048)])))
        hist.append(vl)
        if vl < best - 1e-5: best, bad, state = vl, 0, ({k: v.clone() for k, v in G.state_dict().items()}, {k: v.clone() for k, v in Dn.state_dict().items()})
        else:
            bad += 1
            if bad >= PATIENCE: break
    G.load_state_dict(state[0]); Dn.load_state_dict(state[1]); return G, Dn, hist

def seq_scores(G, Dn, X, M, bs=2048):
    """Error de reconstruccion de la ultima ventana (t) y probabilidad del discriminador para la secuencia que termina en t."""
    r, d = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x, m = torch.tensor(X[i:i+bs]), torch.tensor(M[i:i+bs]); xh, _ = G(x)
            r.append(((x[:, -1] - xh[:, -1]) ** 2).sum(-1).numpy())
            d.append(Dn(G.encode(xh * m.unsqueeze(-1))[0]).numpy())
    return np.concatenate(r), np.concatenate(d)

def metrics(y, s, thr):
    p = (s > thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(F1=100*f1_score(y, p, zero_division=0), FPR=100*fp/(fp+tn), AUC=100*roc_auc_score(y, s),
                P=100*precision_score(y, p, zero_division=0), R=100*recall_score(y, p, zero_division=0),
                MCC=matthews_corrcoef(y, p), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))

rows, wscores = [], {}
SUBS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["logdb", "loggit"]; TAG = "_".join(SUBS)
for name in SUBS:
    ev = pd.read_parquet(f"{OUT}/{name}_events.parquet"); lab = pd.read_csv(f"{OUT}/{name}_windows.csv", parse_dates=["win"])
    templates = pickle.load(open(f"{OUT}/{name}_templates.pkl", "rb"))
    lab = lab.sort_values("win").reset_index(drop=True); part = split_windows(lab); y = lab.y.values
    tr_tids = set(ev[ev.win.isin(lab.win[part == "train"])].tid)
    # Isolation Forest: vector de conteo de plantillas por ventana (log1p)
    Ct = pd.crosstab(ev.win, ev.tid).reindex(lab.win).fillna(0); tids = Ct.columns.values; C = np.log1p(Ct.values)
    info = dict(n_win=len(lab), split={k: int((part == k).sum()) for k in ["train", "val", "test"]},
                mal={k: int(y[part == k].sum()) for k in ["train", "val", "test"]}, n_templates=len(templates))
    print(name, info, flush=True); json.dump(info, open(f"{OUT}/{name}_info.json", "w"))
    for seed in range(NSEED):
        t0 = time.time()
        iso = IsolationForest(n_estimators=200, random_state=seed).fit(C[part == "train"])
        s_if = -iso.score_samples(C)
        E = w2v_embeddings(templates, tr_tids, seed)
        X, M = window_series(C, E, tids, part == "train")
        res = {"IsolationForest": s_if}
        for mname, gan in [("LSTM-AE", False), ("LSTM-GAN", True)]:
            tr, va = part == "train", part == "val"
            G, Dn, hist = train_model(X[tr], M[tr], X[va], M[va], seed, gan)
            r, d = seq_scores(G, Dn, X, M)
            rv = np.sort(r[va]); rn = np.searchsorted(rv, r) / len(rv)               # normalizacion [0,1] con FDA empirica de validacion
            res[mname] = rn if not gan else LAMBDA * rn + (1 - LAMBDA) * (1 - d)       # puntuacion dual S(X)
            res[mname + "_D"] = d
            res[mname + "_epochs"] = len(hist)
        for mname in ["IsolationForest", "LSTM-AE", "LSTM-GAN"]:
            s = res[mname]; thr = np.percentile(s[part == "val"], PCTL)               # umbral rho solo con validacion
            mt = metrics(y[part == "test"], s[part == "test"], thr)
            rows.append(dict(subset=name, seed=seed, model=mname, epochs=res.get(mname + "_epochs", 0), **mt))
            wscores[(name, seed, mname)] = s
        print(name, seed, f"{time.time()-t0:.0f}s", {r["model"]: (round(r["F1"], 1), round(r["FPR"], 1), round(r["AUC"], 1)) for r in rows[-3:]}, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/results_runs_{TAG}.csv", index=False)
pickle.dump(wscores, open(f"{OUT}/window_scores_{TAG}.pkl", "wb"))
