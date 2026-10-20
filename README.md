# Reconstrucción temporal latente (LSTM-AE) y discriminación adversarial (LSTM–GAN) en la detección sin etiquetas de comportamientos maliciosos en series temporales de logs de servidores universitarios

**Autor:** Arturo Ruiz Tarazona — Escuela Profesional de Ingeniería de Ciberseguridad, FIEE, Universidad Nacional de Ingeniería (UNI), Lima, Perú
**Curso:** Inteligencia Artificial II (CBS06), semestre 2026-2 — Examen Parcial, tema 7
**Artículo:** [`articulo/articulo.pdf`](articulo/articulo.pdf) — *Reconstrucción temporal latente (LSTM-AE) y discriminación adversarial (LSTM–GAN) en la detección sin etiquetas de comportamientos maliciosos en series temporales de logs de servidores universitarios*

Repositorio del experimento que compara una técnica no supervisada tradicional (Isolation Forest, pre-prueba)
con un autocodificador LSTM de reconstrucción temporal (LSTM-AE) y su variante adversarial con discriminador (LSTM-GAN),
entrenados **sin etiquetas** sobre registros crudos reales.

## Resultados principales (tramo de prueba, 10 semillas × 3 subconjuntos)

| Modelo | F1 (%) | FPR (%) | AUC-ROC (%) | MCC |
|---|---:|---:|---:|---:|
| Isolation Forest (pre-prueba) | 25.06 | 6.27 | 86.52 | .24 |
| **LSTM-AE (post-prueba)** | **33.56** | 3.28 | **89.04** | **.35** |
| LSTM-GAN (post-prueba) | 32.26 | **2.88** | 85.68 | .32 |

Wilcoxon pareado (n = 30): LSTM-AE frente a IF, F1 p < .001 (H1 aceptada) y FPR p < .001 (H2 aceptada);
LSTM-GAN frente a LSTM-AE, F1 p = .129 y AUC p < .001 a favor del LSTM-AE (H3 aceptada).
Detalle completo en [`resultados/RESULTADOS.md`](resultados/RESULTADOS.md).

## Estructura

```
├── articulo/          Artículo en LaTeX (APA 7, clase apa7 + biblatex-apa), PDF y figuras
├── codigo/            prep.py, experiment.py, analyze.py, requirements.txt
├── datos/crudos/      Registros crudos: logdb.tar.gz, loggit.tar.gz (logsapp: ver abajo)
├── resultados/        Datos procesados, corridas, puntuaciones, tablas y figuras
└── fuentes/           PDF de los artículos citados (acceso abierto)
```

## Datos

Origen: repositorio *uni-logs* — https://uni-logs.us-mia-1.linodeobjects.com/list.html

| Subconjunto | Fuente | Eventos | Plantillas Drain3 | Ventanas de 5 min |
|---|---|---:|---:|---:|
| `logdb` | PostgreSQL 12 | 267 896 | 186 | 5 719 |
| `loggit` | GitLab (nginx, sshd, gitlab-shell) | 56 135 | 121 | 3 398 |
| `logsapp` | CubicERP / Odoo (server, gevent, gunicorn) | 2 893 513 | 361 | 44 299 |

`logsaap.zip` (118 MB) supera el límite de 100 MB por archivo de GitHub, por lo que se publica como
archivo adjunto de la **Release v1.0** de este repositorio (pestaña *Releases*).

## Reproducción

Requiere Python 3.10+ (CPU suficiente; ~105 min en 2 núcleos).

```bash
pip install -r codigo/requirements.txt
mkdir -p data/logsapp
tar -xzf datos/crudos/logdb.tar.gz  -C data
tar -xzf datos/crudos/loggit.tar.gz -C data
unzip logsaap.zip -d data/logsapp           # descargado de la Release v1.0

python -I codigo/prep.py data out                       # Drain3 + ventanas + etiquetas débiles
python -I codigo/experiment.py out 10 30 logdb,loggit   # ~35 min
python -I codigo/experiment.py out 10 30 logsapp        # ~70 min
python -I codigo/analyze.py out                         # tablas, pruebas estadísticas y figuras
```

Para regenerar solo tablas y figuras a partir de los resultados ya incluidos:

```bash
python -I codigo/analyze.py resultados
```

## Metodología resumida

1. **Parseo** en línea con Drain3; las líneas web se codifican como `método · segmento · extensión · estado HTTP · agente`.
2. **Ventanas** fijas de 5 min; partición cronológica 60/20/20; entrenamiento sin depurar (contiene ataques).
3. **Etiquetas débiles** por reglas de seguridad (inyección SQL, sondeos, escaneos web, 4xx), usadas **solo para evaluar**.
4. **Representación**: Word2Vec skip-gram (d = 300); `x_t = Σ_k log(1+c_tk)·e_k`; series de W = 20 ventanas.
5. **Modelos**: Isolation Forest (200 árboles); LSTM-AE (128); LSTM–GAN = LSTM-AE + discriminador LSTM (64),
   `L = 0.8·L_rec + 0.2·L_GAN`, puntuación `S = 0.7·F(e) + 0.3·(1−D)`.
6. **Umbral**: percentil 95 de validación. **Estadística**: Wilcoxon pareado, bootstrap del AUC, sensibilidad del umbral.

## Compilar el artículo

Subir la carpeta `articulo/` a Overleaf (pdfLaTeX + Biber) o compilar localmente:
`pdflatex articulo && biber articulo && pdflatex articulo && pdflatex articulo`.
