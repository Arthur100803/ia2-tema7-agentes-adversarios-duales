"""Preprocesamiento: lectura de logs crudos, parseo con Drain3, ventanas de 5 min
y etiquetas debiles por reglas de seguridad.  Uso:  python -I prep.py <DATA_ROOT> <OUT_DIR>
DATA_ROOT contiene logpostgres/ (logdb) y loggitlab/ (loggit) extraidos de los .tar.gz."""
import sys, os, re, gzip, glob, json, pickle
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
WIN = "5min"

def open_any(p):
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")

def drain(sim=0.5):
    c = TemplateMinerConfig()
    c.drain_sim_th, c.drain_depth, c.parametrize_numeric_tokens = sim, 4, True
    from drain3.masking import MaskingInstruction as M
    c.masking_instructions = [M(r"\b[0-9a-f]{16,}\b", "HEX"), M(r"\d+\.\d+\.\d+\.\d+", "IP"),
                              M(r"\b\d+\b", "NUM"), M(r"'[^']*'", "STR"), M(r'"[^"]*"', "STR")]
    return TemplateMiner(config=c)

# ------------------------------------------------------------------ logdb (PostgreSQL)
PG_HDR = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) (-\d\d) \[(\d+)\] (\S*?@\S*? )?(\w+):\s+(.*)$")
SQLI = re.compile(r"(sleep\s*\(|pg_sleep|\bxor\s*\(|\bor\s+\d+\s*[=+-]|union\s+(all\s+)?select|sysdate\(|benchmark\(|waitfor\s+delay|'\s*--|\";)", re.I)
PG_PROBE = re.compile(r'(database "[^"]*" does not exist|role "[^"]*" does not exist|password authentication failed|no pg_hba\.conf entry)', re.I)

def read_logdb():
    files = sorted(glob.glob(f"{ROOT}/logpostgres/postgresql/postgresql-12-main.log*"))
    files = [f for f in files if not f.endswith(".bk")]          # .bk = arranque de 2021, fuera del periodo
    recs = []
    for f in files:
        cur = None
        for line in open_any(f):
            m = PG_HDR.match(line.rstrip("\n"))
            if m:
                if cur: recs.append(cur)
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone(timedelta(hours=-5)))
                cur = dict(ts=ts.astimezone(timezone.utc), lvl=m.group(5), msg=f"{m.group(5)}: {m.group(6)}")
            elif cur is not None:
                cur["msg"] += " " + line.strip()
        if cur: recs.append(cur)
    df = pd.DataFrame(recs)
    df["msg"] = df.msg.str.slice(0, 400)
    df["r_sqli"] = df.msg.apply(lambda x: bool(SQLI.search(x)))
    df["r_probe"] = df.msg.apply(lambda x: bool(PG_PROBE.search(x)))
    return df

# ------------------------------------------------------------------ loggit (GitLab)
NGX = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+)[^"]*" (\d{3}) \S+ "[^"]*" "([^"]*)"')
WEB_ATTACK = re.compile(r"(union.{0,20}select|sleep\(|%27|'|<script|%3cscript|\.\./|%2e%2e|/etc/passwd|\.env\b|\.git/config|"
                        r"wp-login|wp-content|wp-admin|xmlrpc|phpmyadmin|cgi-bin|\.php\b|\.asp|\.jsp)", re.I)
SCANNER = re.compile(r"(sqlmap|nikto|nmap|zgrab|masscan|nuclei|dirbuster|gobuster|wpscan|acunetix|nessus)", re.I)
SSH_BAD = re.compile(r"(invalid user|failed password|authentication failure|did not receive identification|bad protocol)", re.I)

def read_loggit():
    G = f"{ROOT}/loggitlab/gitlab"; recs = []
    for f in sorted(glob.glob(f"{G}/nginx/gitlab_access.log*")):
        for line in open_any(f):
            m = NGX.match(line)
            if not m: continue
            ts = datetime.strptime(m.group(2), "%d/%b/%Y:%H:%M:%S %z").astimezone(timezone.utc)
            path = m.group(4).split("?")[0] if not WEB_ATTACK.search(m.group(4)) else m.group(4)
            ua = (SCANNER.search(m.group(6)) or re.search(r"(bot|curl|mozilla|git|python|go-http)", m.group(6), re.I))
            uaf = ua.group(1).lower() if ua else "other"
            segs = [p for p in path.split("/") if p]
            seg1 = re.sub(r"\d", "", segs[0])[:25] if segs else "root"          # primer segmento de la ruta
            last = segs[-1] if segs else ""
            ext = last.rsplit(".", 1)[1][:6].lower() if "." in last else "none"   # extension del recurso
            recs.append(dict(ts=ts, src="nginx", status=int(m.group(5)),
                             msg=f"nginx {m.group(3)} seg_{seg1} ext_{ext} st{m.group(5)} ua_{uaf}",
                             r_web=bool(WEB_ATTACK.search(m.group(4))), r_scan=bool(SCANNER.search(m.group(6)))))
    for f in sorted(glob.glob(f"{G}/sshd/*")) + sorted(glob.glob(f"{G}/gitlab-shell/*")):
        src = "sshd" if "/sshd/" in f else "shell"
        for line in open_any(f):
            line = line.strip()
            if src == "sshd":
                m = re.match(r"^(\d{4}-\d\d-\d\d)_(\d\d:\d\d:\d\d\.\d+) (.*)$", line)
                if not m: continue
                ts = datetime.strptime(m.group(1)+" "+m.group(2), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc); msg = m.group(3)
            else:
                try: j = json.loads(line)
                except Exception: continue
                if "time" not in j: continue
                ts = pd.Timestamp(j["time"]).to_pydatetime().astimezone(timezone.utc)
                msg = f'{j.get("level","")} {j.get("msg","")} {j.get("method","")} {j.get("status","")} {str(j.get("url","")).split("?")[0]}'
            recs.append(dict(ts=ts, src=src, status=0, msg=f"{src} {msg}"[:400],
                             r_web=False, r_scan=False, r_ssh=bool(SSH_BAD.search(msg))))
    df = pd.DataFrame(recs); df["r_ssh"] = df.get("r_ssh", False); df["r_ssh"] = df.r_ssh.fillna(False).astype(bool)
    df["is4xx"] = df.status.between(400, 498)   # 499 = cliente cerro conexion (no es error de peticion)
    return df


# ------------------------------------------------------------------ logsapp (CubicERP / Odoo: werkzeug + gunicorn)
ODOO = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \d+ ([A-Z]+) (\S+) ([\w.]+): (.*)$")
WZ = re.compile(r'"(\S+) (\S+)[^"]*" (\d{3})')
GUN = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([+-]\d{4})\] \[\d+\] \[(\w+)\] (.*)$")

def web_msg(src, method, url, status):
    path = url.split("?")[0]; segs = [p for p in path.split("/") if p]
    seg1 = re.sub(r"\d", "", segs[0])[:25] if segs else "root"
    last = segs[-1] if segs else ""
    ext = last.rsplit(".", 1)[1][:6].lower() if "." in last else "none"
    return f"{src} {method} seg_{seg1} ext_{ext} st{status}"

def read_logsapp():
    A = f"{ROOT}/logsapp/logecubic"; recs = []
    for fn, src in [("cubicerp-server.log", "odoo"), ("cubicerp-gevent.log", "gevent")]:
        cur = None
        for line in open(f"{A}/{fn}", errors="replace"):
            m = ODOO.match(line.rstrip("\n"))
            if m:
                if cur: recs.append(cur)
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                lvl, logger, msg = m.group(2), m.group(4), m.group(5)
                w = WZ.search(msg) if logger == "werkzeug" else None
                if w:
                    st = int(w.group(3))
                    cur = dict(ts=ts, status=st, msg=web_msg(src, w.group(1), w.group(2), st), r_web=bool(WEB_ATTACK.search(w.group(2))))
                else:
                    cur = dict(ts=ts, status=0, msg=f"{src} {lvl} {logger}: {msg}"[:300], r_web=False)
            elif cur is not None and len(cur["msg"]) < 300 and line.strip():
                cur["msg"] = (cur["msg"] + " | " + line.strip())[:300]     # traza: se conserva el inicio
        if cur: recs.append(cur)
    cur = None
    for line in open(f"{A}/gunicorn.log", errors="replace"):                 # gunicorn: cabecera + ultima linea de la traza
        m = GUN.match(line.rstrip("\n"))
        if m:
            if cur: recs.append(cur)
            ts = datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M:%S %z").astimezone(timezone.utc)
            cur = dict(ts=ts, status=0, msg=f"gunicorn {m.group(3)} {m.group(4)}"[:200], r_web=False, last="")
        elif cur is not None and line.strip() and not line.startswith(" "):
            cur["last"] = line.strip()[:150]                                   # excepcion final de la traza
    if cur: recs.append(cur)
    df = pd.DataFrame(recs)
    if "last" in df: df["msg"] = np.where(df["last"].fillna("") != "", df.msg + " | " + df["last"].fillna(""), df.msg); df = df.drop(columns="last")
    df["is4xx"] = df.status.between(400, 498)
    return df

def window_labels(df, name):
    df = df.sort_values("ts").reset_index(drop=True)
    df["win"] = df.ts.dt.floor(WIN)
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
    lab["n_events"] = g.size()
    return df, lab

READERS = {"logdb": read_logdb, "loggit": read_loggit, "logsapp": read_logsapp}
for name in (sys.argv[3].split(",") if len(sys.argv) > 3 else list(READERS)):
    reader = READERS[name]
    df = reader()
    df, lab = window_labels(df, name)
    tm = drain({"logdb": 0.5, "loggit": 0.7, "logsapp": 0.6}[name]); tids = []
    for m in df.msg:                                   # parseo en linea, en orden cronologico
        tids.append(tm.add_log_message(m)["cluster_id"])
    df["tid"] = tids
    templates = {c.cluster_id: c.get_template() for c in tm.drain.clusters}
    df[["ts", "win", "tid"] + [c for c in df.columns if c.startswith("r_")]].to_parquet(f"{OUT}/{name}_events.parquet")
    lab.to_csv(f"{OUT}/{name}_windows.csv")
    pickle.dump(templates, open(f"{OUT}/{name}_templates.pkl", "wb"))
    print(name, "eventos", len(df), "plantillas", len(templates), "ventanas", len(lab),
          "maliciosas", int(lab.y.sum()), f"({lab.y.mean():.2%})", "periodo", df.ts.min(), "->", df.ts.max())
