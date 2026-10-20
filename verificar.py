"""
verificar.py — Reproduce y comprueba los resultados del artículo con un solo comando.

    python verificar.py            # verificación rápida (≈5-8 min en una laptop, sin GPU)
    python verificar.py completo   # repite el experimento entero de logdb y loggit (≈35-60 min)

Qué hace la verificación rápida
  1. Revisa que estén instaladas las librerías de codigo/requirements.txt.
  2. Extrae datos/crudos/logdb.tar.gz en data/ (solo la primera vez).
  3. Corre codigo/prep.py sobre logdb y compara las ventanas con resultados/logdb_windows.csv.
  4. Corre codigo/experiment.py con 1 semilla sobre logdb y compara F1, FPR y AUC
     con la semilla 0 guardada en resultados/results_runs_logdb_loggit.csv.
  5. Corre codigo/analyze.py sobre una copia de las 10 semillas guardadas y compara
     las tablas que genera con las de resultados/ (son las que están en el artículo).

No modifica nada de resultados/: todo lo nuevo se escribe en out/.
Se puede ejecutar desde cualquier carpeta (en PyCharm: clic derecho → Run 'verificar').
"""
import sys, os, glob, shutil, subprocess, tarfile, time, importlib.util

REPO = os.path.dirname(os.path.abspath(__file__))
COD, RES = os.path.join(REPO, "codigo"), os.path.join(REPO, "resultados")
DATA, OUT = os.path.join(REPO, "data"), os.path.join(REPO, "out")
MODO = sys.argv[1] if len(sys.argv) > 1 else "rapido"
TOL = 0.5          # tolerancia (puntos porcentuales) por diferencias de procesador / número de hilos
ok_total = True


def titulo(t): print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)


def check(cond, msg):
    global ok_total
    print(("  [OK]    " if cond else "  [FALLA] ") + msg, flush=True); ok_total &= bool(cond)


def run(script, *args):
    """Ejecuta un script de codigo/ con el mismo Python (mismo entorno virtual) que este archivo."""
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(COD, script), *args], cwd=REPO)
    if r.returncode != 0: sys.exit(f"\n{script} terminó con error (código {r.returncode}).")
    print(f"  ({script} tardó {time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------ 1. librerías
titulo("1. Librerías")
faltan = [p for m, p in [("torch", "torch"), ("drain3", "drain3"), ("gensim", "gensim"), ("sklearn", "scikit-learn"),
                         ("scipy", "scipy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
                         ("matplotlib", "matplotlib"), ("tabulate", "tabulate")] if importlib.util.find_spec(m) is None]
if faltan:
    sys.exit(f"  Faltan: {', '.join(faltan)}\n  Instálalas con:\n    \"{sys.executable}\" -m pip install -r \"{os.path.join(COD, 'requirements.txt')}\"")
print("  Todas instaladas.")
import pandas as pd, numpy as np

# ------------------------------------------------------------------ 2. datos
titulo("2. Datos crudos")
subs = ["logdb"] if MODO == "rapido" else ["logdb", "loggit"]
carpeta = {"logdb": "logpostgres", "loggit": "loggitlab"}
for s in subs:
    if os.path.isdir(os.path.join(DATA, carpeta[s])):
        print(f"  {s}: ya extraído en data/{carpeta[s]}"); continue
    tgz = os.path.join(REPO, "datos", "crudos", f"{s}.tar.gz")
    print(f"  Extrayendo {os.path.relpath(tgz, REPO)} ...", flush=True)
    with tarfile.open(tgz) as t:
        try: t.extractall(DATA, filter="data")          # Python >= 3.12 (y parches de 3.10/3.11)
        except TypeError: t.extractall(DATA)
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ 3. preprocesamiento
titulo("3. Preprocesamiento (prep.py)")
run("prep.py", DATA, OUT, ",".join(subs))
for s in subs:
    a = pd.read_csv(os.path.join(OUT, f"{s}_windows.csv")); b = pd.read_csv(os.path.join(RES, f"{s}_windows.csv"))
    check(a.equals(b), f"{s}: {len(a)} ventanas y {int(a.y.sum())} maliciosas, iguales a resultados/{s}_windows.csv")

# ------------------------------------------------------------------ 4. experimento
titulo("4. Experimento (experiment.py)")
nseed = "1" if MODO == "rapido" else "10"
run("experiment.py", OUT, nseed, "30", ",".join(subs))
nuevo = pd.read_csv(os.path.join(OUT, f"results_runs_{'_'.join(subs)}.csv"))
guard = pd.read_csv(os.path.join(RES, "results_runs_logdb_loggit.csv"))
m = nuevo.merge(guard, on=["subset", "seed", "model"], suffixes=("", "_g"))
for _, r in m.iterrows():
    d = max(abs(r[k] - r[k + "_g"]) for k in ["F1", "FPR", "AUC"])
    check(d <= TOL, f"{r.subset} semilla {r.seed} {r.model:15s} F1={r.F1:6.2f} FPR={r.FPR:5.2f} AUC={r.AUC:6.2f}"
                    f"   (guardado {r.F1_g:6.2f} {r.FPR_g:5.2f} {r.AUC_g:6.2f})")

# ------------------------------------------------------------------ 5. análisis estadístico
titulo("5. Análisis estadístico (analyze.py) sobre las 10 semillas guardadas")
A = os.path.join(OUT, "analisis"); os.makedirs(A, exist_ok=True)
for pat in ["results_runs_*.csv", "window_scores_*.pkl", "*_windows.csv", "*_info.json", "*_events.parquet"]:
    for f in glob.glob(os.path.join(RES, pat)): shutil.copy(f, A)
run("analyze.py", A)
for t in ["tabla_descriptiva", "tabla_pre_post", "tabla_inferencial", "tabla_sensibilidad_umbral", "tabla_bootstrap_auc"]:
    a = pd.read_csv(os.path.join(A, f"{t}.csv")); b = pd.read_csv(os.path.join(RES, f"{t}.csv"))
    num = a.select_dtypes("number").columns
    igual = a.shape == b.shape and np.allclose(a[num].values, b[num].values, atol=1e-6, equal_nan=True)
    check(igual, f"{t}.csv igual a resultados/")

# ------------------------------------------------------------------ resumen
titulo("RESUMEN")
print("  Todo coincide con los resultados del artículo." if ok_total else
      "  Hubo diferencias (ver [FALLA] arriba). Diferencias pequeñas pueden deberse al procesador.")
print(f"  Archivos nuevos en: {OUT}")
