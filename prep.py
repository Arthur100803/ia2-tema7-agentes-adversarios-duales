"""
prep.py — Paso 1 del experimento: preprocesamiento de los logs crudos.

Qué hace
--------
1. Lee los logs crudos de cada subconjunto del repositorio uni-logs:
     - logdb   : PostgreSQL 12
     - loggit  : GitLab (nginx, sshd, gitlab-shell)
     - logsapp : ERP CubicERP / Odoo (werkzeug, gevent, gunicorn)
2. Aplica reglas de seguridad a cada evento (inyección SQL, sondeos, escaneos web,
   errores 4xx...). Con ellas se arman las ETIQUETAS DÉBILES, que solo se usan para
   evaluar los modelos, nunca para entrenarlos.
3. Agrupa los eventos en ventanas fijas de 5 minutos y etiqueta cada ventana como
   normal (y = 0) o maliciosa (y = 1).
4. Parsea los mensajes con Drain3 (en orden cronológico) para asignar a cada evento
   el número de su plantilla.
5. Guarda, por subconjunto, tres archivos en la carpeta de salida:
     <sub>_events.parquet   eventos con hora, ventana, plantilla y reglas
     <sub>_windows.csv      ventanas con conteos de reglas y etiqueta y
     <sub>_templates.pkl    texto de cada plantilla de Drain3

Uso
---
    python prep.py <DATA_ROOT> <OUT_DIR> [subconjuntos]
    python prep.py data out                      # los tres subconjuntos
    python prep.py data out logdb                # solo logdb

    En PyCharm se puede ejecutar sin parámetros: usa <repo>/data y <repo>/out.

DATA_ROOT debe contener:
    logpostgres/postgresql/postgresql-12-main.log*      (de logdb.tar.gz)
    loggitlab/gitlab/{nginx,sshd,gitlab-shell}/...      (de loggit.tar.gz)
    logsapp/logecubic/{cubicerp-server.log, cubicerp-gevent.log, gunicorn.log}  (de logsaap.zip)

Nota: este archivo solo tiene comentarios y orden nuevos; la lógica es la misma que
produjo los resultados de la carpeta resultados/.
"""
import sys, os, re, gzip, glob, json, pickle
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# =====================================================================================
# 0. Parámetros de entrada
# =====================================================================================
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # carpeta raíz del repositorio
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data")   # logs crudos extraídos
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "out")     # carpeta de salida
os.makedirs(OUT, exist_ok=True)
WIN = "5min"                                   # tamaño de la ventana temporal (unidad de análisis)


def open_any(p):
    """Abre un log de texto, esté comprimido (.gz) o no.
    Se fija UTF-8 para que en Windows se lea igual que en Linux (los caracteres inválidos se reemplazan)."""
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.endswith(".gz") else open(p, encoding="utf-8", errors="replace")


def drain(sim=0.5):
    """Crea un analizador Drain3.

    Drain3 agrupa los mensajes que comparten la misma estructura en una PLANTILLA.
    Antes de comparar, se reemplazan ("enmascaran") las partes que siempre cambian,
    para que, p. ej., dos conexiones desde IPs distintas caigan en la misma plantilla.
    sim = umbral de similitud (más alto = plantillas más finas).
    """
    c = TemplateMinerConfig()
    c.drain_sim_th, c.drain_depth, c.parametrize_numeric_tokens = sim, 4, True
    from drain3.masking import MaskingInstruction as M
    c.masking_instructions = [M(r"\b[0-9a-f]{16,}\b", "HEX"),     # identificadores hexadecimales
                              M(r"\d+\.\d+\.\d+\.\d+", "IP"),     # direcciones IPv4
                              M(r"\b\d+\b", "NUM"),                # números
                              M(r"'[^']*'", "STR"),                # textos entre comillas simples
                              M(r'"[^"]*"', "STR")]                # textos entre comillas dobles
    return TemplateMiner(config=c)


# =====================================================================================
# 1. logdb — PostgreSQL 12
# =====================================================================================
# Cabecera de cada línea: fecha, zona horaria, PID, usuario@base (opcional), nivel y mensaje.
PG_HDR = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (-\d\d) \[(\d+)\] (\S*?@\S*? )?(\w+):\s+(.*)$")
# Regla de seguridad 1: firmas de inyección SQL (blind SQLi por tiempo, XOR, OR n=n, UNION SELECT, ...).
SQLI = re.compile(r"(sleep\s*\(|pg_sleep|\bxor\s*\(|\bor\s+\d+\s*[=+-]|union\s+(all\s+)?select|sysdate\(|benchmark\(|waitfor\s+delay|'\s*--|\";)", re.I)
# Regla de seguridad 2: sondeos (bases o roles inexistentes, autenticación fallida).
PG_PROBE = re.compile(r'(database "[^"]*" does not exist|role "[^"]*" does not exist|password authentication failed|no pg_hba\.conf entry)', re.I)


def read_logdb():
    """Lee los logs de PostgreSQL y devuelve un DataFrame con un evento por fila."""
    files = sorted(glob.glob(f"{ROOT}/logpostgres/postgresql/postgresql-12-main.log*"))
    files = [f for f in files if not f.endswith(".bk")]          # .bk = arranque de 2021, fuera del periodo
    recs = []
    for f in files:
        cur = None
        for line in open_any(f):
            m = PG_HDR.match(line.rstrip("\n"))
            if m:                                                 # línea nueva con cabecera
                if cur: recs.append(cur)
                # la hora del servidor está en UTC-5 (Lima); se pasa a UTC
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone(timedelta(hours=-5)))
                cur = dict(ts=ts.astimezone(timezone.utc), lvl=m.group(5), msg=f"{m.group(5)}: {m.group(6)}")
            elif cur is not None:                                 # continuación (p. ej. sentencia SQL partida)
                cur["msg"] += " " + line.strip()
        if cur: recs.append(cur)
    df = pd.DataFrame(recs)
    df["msg"] = df.msg.str.slice(0, 400)                          # se recortan mensajes muy largos
    df["r_sqli"] = df.msg.apply(lambda x: bool(SQLI.search(x)))   # ¿el evento tiene firma de SQLi?
    df["r_probe"] = df.msg.apply(lambda x: bool(PG_PROBE.search(x)))  # ¿es un sondeo?
    return df


# =====================================================================================
# 2. loggit — GitLab (nginx + sshd + gitlab-shell)
# =====================================================================================
# Línea de acceso de nginx: IP, fecha, método, URL, código HTTP y agente de usuario.
NGX = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+)[^"]*" (\d{3}) \S+ "[^"]*" "([^"]*)"')
# Regla: URL con carga de ataque (SQLi, XSS, path traversal, archivos sensibles, PHP/WordPress...).
WEB_ATTACK = re.compile(r"(union.{0,20}select|sleep\(|%27|'|<script|%3cscript|\.\./|%2e%2e|/etc/passwd|\.env\b|\.git/config|"
                        r"wp-login|wp-content|wp-admin|xmlrpc|phpmyadmin|cgi-bin|\.php\b|\.asp|\.jsp)", re.I)
# Regla: herramientas de escaneo conocidas en el agente de usuario.
SCANNER = re.compile(r"(sqlmap|nikto|nmap|zgrab|masscan|nuclei|dirbuster|gobuster|wpscan|acunetix|nessus)", re.I)
# Regla: fallos de autenticación SSH.
SSH_BAD = re.compile(r"(invalid user|failed password|authentication failure|did not receive identification|bad protocol)", re.I)


def read_loggit():
    """Lee nginx, sshd y gitlab-shell de GitLab y devuelve un DataFrame de eventos."""
    G = f"{ROOT}/loggitlab/gitlab"; recs = []

    # --- 2.1 accesos web (nginx) -----------------------------------------------------
    for f in sorted(glob.glob(f"{G}/nginx/gitlab_access.log*")):
        for line in open_any(f):
            m = NGX.match(line)
            if not m: continue
            ts = datetime.strptime(m.group(2), "%d/%b/%Y:%H:%M:%S %z").astimezone(timezone.utc)
            # sin parámetros de consulta, salvo que la URL tenga carga de ataque
            path = m.group(4).split("?")[0] if not WEB_ATTACK.search(m.group(4)) else m.group(4)
            ua = (SCANNER.search(m.group(6)) or re.search(r"(bot|curl|mozilla|git|python|go-http)", m.group(6), re.I))
            uaf = ua.group(1).lower() if ua else "other"                          # familia del cliente
            segs = [p for p in path.split("/") if p]
            seg1 = re.sub(r"\d", "", segs[0])[:25] if segs else "root"          # primer segmento de la ruta
            last = segs[-1] if segs else ""
            ext = last.rsplit(".", 1)[1][:6].lower() if "." in last else "none"   # extensión del recurso
            # CODIFICACIÓN de la petición: "nginx GET seg_users ext_php st404 ua_curl".
            # Así Drain3 conserva la señal del ataque (ruta, extensión y código HTTP).
            recs.append(dict(ts=ts, src="nginx", status=int(m.group(5)),
                             msg=f"nginx {m.group(3)} seg_{seg1} ext_{ext} st{m.group(5)} ua_{uaf}",
                             r_web=bool(WEB_ATTACK.search(m.group(4))), r_scan=bool(SCANNER.search(m.group(6)))))

    # --- 2.2 sshd y gitlab-shell ---------------------------------------------------------
    for f in sorted(glob.glob(f"{G}/sshd/*")) + sorted(glob.glob(f"{G}/gitlab-shell/*")):
        src = "sshd" if "sshd" in os.path.normpath(f).split(os.sep) else "shell"   # funciona con / y con \ (Windows)
        for line in open_any(f):
            line = line.strip()
            if src == "sshd":                                    # formato: 2026-05-01_12:00:00.123 mensaje
                m = re.match(r"^(\d{4}-\d\d-\d\d)_(\d\d:\d\d:\d\d\.\d+) (.*)$", line)
                if not m: continue
                ts = datetime.strptime(m.group(1)+" "+m.group(2), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc); msg = m.group(3)
            else:                                                # gitlab-shell escribe JSON
                try: j = json.loads(line)
                except Exception: continue
                if "time" not in j: continue
                ts = pd.Timestamp(j["time"]).to_pydatetime().astimezone(timezone.utc)
                msg = f'{j.get("level","")} {j.get("msg","")} {j.get("method","")} {j.get("status","")} {str(j.get("url","")).split("?")[0]}'
            recs.append(dict(ts=ts, src=src, status=0, msg=f"{src} {msg}"[:400],
                             r_web=False, r_scan=False, r_ssh=bool(SSH_BAD.search(msg))))
    df = pd.DataFrame(recs); df["r_ssh"] = df.get("r_ssh", False); df["r_ssh"] = df.r_ssh.fillna(False).astype(bool)
    df["is4xx"] = df.status.between(400, 498)   # 499 = el cliente cerró la conexión (no es error de petición)
    return df


# =====================================================================================
# 3. logsapp — CubicERP / Odoo (werkzeug + gevent + gunicorn)
# =====================================================================================
ODOO = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \d+ ([A-Z]+) (\S+) ([\w.]+): (.*)$")   # línea de Odoo
WZ = re.compile(r'"(\S+) (\S+)[^"]*" (\d{3})')                                                  # petición werkzeug
GUN = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([+-]\d{4})\] \[\d+\] \[(\w+)\] (.*)$")  # cabecera gunicorn


def web_msg(src, method, url, status):
    """Codifica una petición web igual que en nginx: 'odoo GET seg_web ext_none st200'."""
    path = url.split("?")[0]; segs = [p for p in path.split("/") if p]
    seg1 = re.sub(r"\d", "", segs[0])[:25] if segs else "root"
    last = segs[-1] if segs else ""
    ext = last.rsplit(".", 1)[1][:6].lower() if "." in last else "none"
    return f"{src} {method} seg_{seg1} ext_{ext} st{status}"


def read_logsapp():
    """Lee los tres logs del ERP y devuelve un DataFrame de eventos."""
    A = f"{ROOT}/logsapp/logecubic"; recs = []

    # --- 3.1 servidor principal y servidor gevent (formato Odoo) -----------------------
    for fn, src in [("cubicerp-server.log", "odoo"), ("cubicerp-gevent.log", "gevent")]:
        cur = None
        for line in open(f"{A}/{fn}", encoding="utf-8", errors="replace"):
            m = ODOO.match(line.rstrip("\n"))
            if m:
                if cur: recs.append(cur)
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                lvl, logger, msg = m.group(2), m.group(4), m.group(5)
                w = WZ.search(msg) if logger == "werkzeug" else None
                if w:                                             # petición HTTP: se codifica
                    st = int(w.group(3))
                    cur = dict(ts=ts, status=st, msg=web_msg(src, w.group(1), w.group(2), st), r_web=bool(WEB_ATTACK.search(w.group(2))))
                else:                                             # otro mensaje del ERP
                    cur = dict(ts=ts, status=0, msg=f"{src} {lvl} {logger}: {msg}"[:300], r_web=False)
            elif cur is not None and len(cur["msg"]) < 300 and line.strip():
                cur["msg"] = (cur["msg"] + " | " + line.strip())[:300]     # traza: se conserva el inicio
        if cur: recs.append(cur)

    # --- 3.2 gunicorn: cabecera del error + última línea de la traza (la excepción) ------
    cur = None
    for line in open(f"{A}/gunicorn.log", encoding="utf-8", errors="replace"):
        m = GUN.match(line.rstrip("\n"))
        if m:
            if cur: recs.append(cur)
            ts = datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M:%S %z").astimezone(timezone.utc)
            cur = dict(ts=ts, status=0, msg=f"gunicorn {m.group(3)} {m.group(4)}"[:200], r_web=False, last="")
        elif cur is not None and line.strip() and not line.startswith(" "):
            cur["last"] = line.strip()[:150]                                   # excepción final de la traza
    if cur: recs.append(cur)
    df = pd.DataFrame(recs)
    if "last" in df: df["msg"] = np.where(df["last"].fillna("") != "", df.msg + " | " + df["last"].fillna(""), df.msg); df = df.drop(columns="last")
    df["is4xx"] = df.status.between(400, 498)
    return df


# =====================================================================================
# 4. Ventanas de 5 minutos y etiquetas débiles
# =====================================================================================
def window_labels(df, name):
    """Asigna cada evento a su ventana de 5 min y etiqueta cada ventana.

    Reglas de etiquetado (una ventana es maliciosa, y = 1, si...):
      logdb   : >= 1 inyección SQL  o  >= 3 sondeos
      loggit  : >= 1 URL de ataque  o  >= 1 escáner  o  >= 3 fallos SSH  o  >= 10 errores 4xx
      logsapp : >= 1 URL de ataque  o  >= 10 errores 4xx
    """
    df = df.sort_values("ts").reset_index(drop=True)
    df["win"] = df.ts.dt.floor(WIN)                       # inicio de la ventana de 5 min
    g = df.groupby("win")
    if name == "logdb":
        lab = pd.DataFrame({"sqli": g.r_sqli.sum(), "probe": g.r_probe.sum()})
        lab["y"] = ((lab.sqli >= 1) | (lab.probe >= 3)).astype(int)
    elif name == "logsapp":
        lab = pd.DataFrame({"web": g.r_web.sum(), "c4xx": g.is4xx.sum()})
        lab["y"] = ((lab.web >= 1) | (lab.c4xx >= 10)).astype(int)
    else:
        lab = pd.DataFrame({"web": g.r_web.sum(), "scan": g.r_scan.sum(), "ssh": g.r_ssh.sum(), "c4xx": g.is4xx.sum()})
        lab["y"] = ((lab.web >= 1) | (lab.scan >= 1) | (lab.ssh >= 3) | (lab.c4xx >= 10)).astype(int)
    lab["n_events"] = g.size()                            # eventos por ventana
    return df, lab


# =====================================================================================
# 5. Programa principal
# =====================================================================================
READERS = {"logdb": read_logdb, "loggit": read_loggit, "logsapp": read_logsapp}
DRAIN_SIM = {"logdb": 0.5, "loggit": 0.7, "logsapp": 0.6}     # umbral de similitud de Drain3 por fuente

NEEDED = {"logdb": "logpostgres/postgresql", "loggit": "loggitlab/gitlab", "logsapp": "logsapp/logecubic"}  # carpeta que debe existir

print("Datos crudos :", ROOT, "\nSalida       :", OUT, flush=True)
for name in (sys.argv[3].split(",") if len(sys.argv) > 3 else list(READERS)):
    # Comprobación previa: si no están los logs, se avisa en vez de fallar con un error confuso.
    if not os.path.isdir(os.path.join(ROOT, NEEDED[name])):
        sys.exit(f"[{name}] No encuentro la carpeta {os.path.join(ROOT, NEEDED[name])}\n"
                 f"  Extrae los datos en {ROOT} o pasa la ruta como primer parámetro: python prep.py <DATA_ROOT> <OUT_DIR>")
    reader = READERS[name]
    df = reader()                                      # 1) leer y aplicar reglas
    df, lab = window_labels(df, name)                  # 2) ventanas y etiquetas débiles
    tm = drain(DRAIN_SIM[name]); tids = []
    for m in df.msg:                                   # 3) parseo en línea, en orden cronológico
        tids.append(tm.add_log_message(m)["cluster_id"])
    df["tid"] = tids                                   #    número de plantilla de cada evento
    templates = {c.cluster_id: c.get_template() for c in tm.drain.clusters}
    # 4) guardar resultados
    df[["ts", "win", "tid"] + [c for c in df.columns if c.startswith("r_")]].to_parquet(f"{OUT}/{name}_events.parquet")
    lab.to_csv(f"{OUT}/{name}_windows.csv")
    pickle.dump(templates, open(f"{OUT}/{name}_templates.pkl", "wb"))
    print(name, "eventos", len(df), "plantillas", len(templates), "ventanas", len(lab),
          "maliciosas", int(lab.y.sum()), f"({lab.y.mean():.2%})", "periodo", df.ts.min(), "->", df.ts.max())
