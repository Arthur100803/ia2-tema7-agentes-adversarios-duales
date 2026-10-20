# Código del experimento

Son tres scripts que se ejecutan en orden:

| Paso | Script | Qué hace | Entrada | Salida |
|---|---|---|---|---|
| 1 | `prep.py` | Lee los logs crudos, aplica las reglas de seguridad, arma ventanas de 5 min y parsea con Drain3 | `data/` | `out/<sub>_events.parquet`, `out/<sub>_windows.csv`, `out/<sub>_templates.pkl` |
| 2 | `experiment.py` | Entrena Isolation Forest, LSTM-AE y LSTM-GAN con 10 semillas y los evalúa | `out/` | `out/results_runs_*.csv`, `out/window_scores_*.pkl` |
| 3 | `analyze.py` | Pruebas estadísticas, tablas y figuras | `out/` o `resultados/` | tablas `.csv`, `RESULTADOS.md` y `figs/` |

Los resultados que uso en el artículo están en `resultados/`. Si corres los tres scripts en orden, en `out/` salen los mismos números.

## Correrlo en PyCharm

1. **Abrir el proyecto.** *File → Open* y elige la carpeta del repositorio (la que tiene `codigo/`, `datos/` y `resultados/`).
2. **Intérprete.** *File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv* (Python 3.10, 3.11 o 3.12).
3. **Librerías.** En la pestaña *Terminal* de PyCharm:
   ```
   pip install -r codigo/requirements.txt
   ```
   Si `drain3` da error: `pip install --no-deps drain3==0.9.11` y luego `pip install jsonpickle cachetools`.
4. **Datos.** En la misma terminal (Windows 10/11 trae `tar`):
   ```
   mkdir data
   mkdir data\logsapp
   tar -xzf datos/crudos/logdb.tar.gz -C data
   tar -xzf datos/crudos/loggit.tar.gz -C data
   tar -xf RUTA\logsaap.zip -C data/logsapp
   ```
   `logsaap.zip` se descarga de la Release v1.0 del repositorio. Debe quedar `data/logpostgres`, `data/loggitlab` y `data/logsapp/logecubic`.
5. **Ejecutar.** Clic derecho sobre el script → *Run*. Sin parámetros cada script usa valores por defecto:

   | Script | Sin parámetros hace | Para otro caso, en *Run → Edit Configurations → Parameters* |
   |---|---|---|
   | `prep.py` | `data` → `out`, los tres subconjuntos (≈10 min) | `data out logdb` (solo logdb) |
   | `experiment.py` | `out`, 10 semillas, 30 épocas, logdb y loggit (≈35 min) | `out 10 30 logsapp` (≈70 min) · `out 1 30 logdb` (prueba de ≈4 min) |
   | `analyze.py` | regenera las tablas de `resultados/` (≈2 min) | `out` (para analizar tu propia corrida) |

## Verificación rápida

- **Lo más fácil:** clic derecho sobre `verificar.py` (en la raíz del repositorio) → *Run*. Hace todo solo en unos 5 minutos y marca `[OK]` o `[FALLA]`.

- **2 minutos:** corre `analyze.py` sin parámetros y compara `resultados/RESULTADOS.md` con el artículo.
- **5 minutos:** `prep.py` con `data out logdb` y luego `experiment.py` con `out 1 30 logdb`. Debe imprimir para la semilla 0:
  `IsolationForest (26.5, 5.1, 78.8)  LSTM-AE (32.5, 4.2, 82.9)  LSTM-GAN (33.8, 4.1, 80.1)`  (F1, FPR, AUC),
  igual que la semilla 0 de logdb en `resultados/results_runs_logdb_loggit.csv`.

Pueden aparecer diferencias en el último decimal si tu procesador tiene otro número de núcleos.
