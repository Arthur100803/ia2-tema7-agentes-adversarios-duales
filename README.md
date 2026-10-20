# Reconstrucción temporal latente (LSTM-AE) y discriminación adversarial (LSTM–GAN) en la detección sin etiquetas de comportamientos maliciosos en series temporales de logs de servidores universitarios

**Arturo Ruiz Tarazona**
Escuela Profesional de Ingeniería de Ciberseguridad – FIEE, Universidad Nacional de Ingeniería (UNI)
Inteligencia Artificial II (CBS06) · 2026-2 · Examen parcial, tema 7: *Agentes adversarios duales para detección sin etiquetas*

📄 Artículo: [`articulo/articulo.pdf`](articulo/articulo.pdf)

---

## ¿De qué trata?

Los servidores de una universidad generan millones de líneas de logs y nadie las etiqueta, así que no se puede entrenar un clasificador supervisado para encontrar ataques. En este trabajo probé si un **autocodificador LSTM (LSTM-AE)** puede aprender cómo se ve el tráfico "normal" y marcar como sospechoso lo que no logra reconstruir, y si agregarle un **discriminador adversarial (LSTM–GAN)** lo mejora todavía más. Como línea base usé **Isolation Forest**.

Trabajé con los logs crudos del repositorio del curso (*uni-logs*): la base de datos PostgreSQL, el GitLab y el ERP en Odoo.

## Resultados

Promedio de 10 semillas en cada uno de los 3 subconjuntos, evaluado en el tramo de prueba:

| Modelo | F1 (%) | FPR (%) | AUC-ROC (%) | MCC |
|---|---:|---:|---:|---:|
| Isolation Forest (línea base) | 25.06 | 6.27 | 86.52 | .24 |
| **LSTM-AE** | **33.56** | 3.28 | **89.04** | **.35** |
| LSTM–GAN | 32.26 | **2.88** | 85.68 | .32 |

Lo más importante:

- El **LSTM-AE** sube el F1 en 8.5 puntos y baja las falsas alarmas casi a la mitad frente a Isolation Forest (Wilcoxon, p < .001 en ambos casos).
- El **discriminador no ayudó**: con el LSTM–GAN el F1 no cambia de forma significativa (p = .129), el AUC baja 3.4 puntos (p < .001) y los resultados varían mucho entre semillas.
- Sin usar etiquetas, el modelo detectó inyecciones SQL con `sleep()` contra PostgreSQL, escaneos de `wp-login.php` y webshells `.php` contra GitLab, y sondeos de `.env` contra el ERP.
- No detectó bien una enumeración lenta de bases de datos (unos 630 intentos repartidos en 5 horas).

Las tablas completas y las pruebas estadísticas están en [`resultados/RESULTADOS.md`](resultados/RESULTADOS.md).

## Verificación rápida (≈5 minutos)

Para comprobar que el código da los mismos números del artículo basta con tres comandos. No hace falta descargar nada aparte ni descomprimir a mano:

```bash
git clone https://github.com/Arthur100803/ia2-tema7-agentes-adversarios-duales.git
cd ia2-tema7-agentes-adversarios-duales
pip install -r codigo/requirements.txt
python verificar.py
```

En PyCharm: *File → Open* la carpeta del repositorio, instalar `requirements.txt` y luego clic derecho sobre `verificar.py` → *Run*.

`verificar.py` extrae los logs de PostgreSQL, corre `prep.py`, entrena los tres modelos con una semilla y corre `analyze.py` sobre mis 10 semillas. Después compara todo con lo que está en `resultados/` y marca cada punto como `[OK]` o `[FALLA]`. En una laptop con 2 núcleos tarda unos 4 minutos y no modifica la carpeta `resultados/`. Debería verse así:

```
  [OK]    logdb: 5719 ventanas y 93 maliciosas, iguales a resultados/logdb_windows.csv
  [OK]    logdb semilla 0 IsolationForest F1= 26.51 FPR= 5.12 AUC= 78.77   (guardado  26.51  5.12  78.77)
  [OK]    logdb semilla 0 LSTM-AE         F1= 32.50 FPR= 4.17 AUC= 82.86   (guardado  32.50  4.17  82.86)
  [OK]    logdb semilla 0 LSTM-GAN        F1= 33.75 FPR= 4.08 AUC= 80.12   (guardado  33.75  4.08  80.12)
  [OK]    tabla_descriptiva.csv igual a resultados/
  ...
  Todo coincide con los resultados del artículo.
```

Con `python verificar.py completo` repite las 10 semillas de logdb y loggit (unos 35 a 60 minutos).

## Organización del repositorio

```
├── verificar.py    comprueba los resultados con un solo comando (≈5 min)
├── articulo/       artículo en LaTeX (formato APA 7), PDF y figuras
├── codigo/         prep.py, experiment.py, analyze.py y requirements.txt
├── datos/crudos/   logdb.tar.gz y loggit.tar.gz (logsapp está en la Release)
├── resultados/     datos procesados, resultados de cada corrida, tablas y figuras
└── fuentes/        PDF de los 14 artículos que cito
```

## Datos

Los logs vienen de https://uni-logs.us-mia-1.linodeobjects.com/list.html

| Subconjunto | Servicio | Eventos | Plantillas (Drain3) | Ventanas de 5 min |
|---|---|---:|---:|---:|
| `logdb` | PostgreSQL 12 | 267 896 | 186 | 5 719 |
| `loggit` | GitLab (nginx, sshd, gitlab-shell) | 56 135 | 121 | 3 398 |
| `logsapp` | CubicERP / Odoo | 2 893 513 | 361 | 44 299 |

- `logsaap.zip` pesa 118 MB y GitHub no acepta archivos de más de 100 MB, así que lo subí como adjunto en la **Release v1.0** (columna derecha, *Releases*).
- No usé el subconjunto `logserver` porque pesa cerca de 2.4 GB y no me alcanzaba la máquina para entrenar con él.

## Cómo lo hice

1. **Parseo de los logs** con Drain3, en orden de tiempo. A las líneas de nginx y werkzeug las codifiqué como `método · ruta · extensión · código HTTP · agente`, porque si no Drain3 juntaba todo y se perdía la señal del ataque.
2. **Ventanas de 5 minutos.** Dividí cada subconjunto por fecha: 60 % entrenamiento, 20 % validación y 20 % prueba. No limpié el entrenamiento, así que tiene ataques mezclados, como pasaría en la realidad.
3. **Etiquetas débiles.** Como los logs no traen etiquetas, armé reglas de seguridad (firmas de inyección SQL, sondeos de bases de datos, URLs de ataque, ráfagas de errores 4xx) para saber qué ventanas eran maliciosas. Estas etiquetas **solo las usé para evaluar**, nunca para entrenar.
4. **Representación.** Word2Vec (300 dimensiones) sobre las plantillas; cada ventana es la suma ponderada de sus plantillas, y el modelo ve series de 20 ventanas (100 minutos).
5. **Modelos.** Isolation Forest (200 árboles), LSTM-AE (128 unidades) y LSTM–GAN, que es el LSTM-AE más un discriminador LSTM de 64 unidades.
6. **Evaluación.** El umbral es el percentil 95 de las puntuaciones de validación. Comparé los modelos con la prueba de Wilcoxon (30 pares) y con intervalos *bootstrap* del AUC.

Un problema que tuve: al principio modelé secuencias de **eventos** y los ataques salían con puntuación baja, porque son cientos de peticiones iguales y el autocodificador las reconstruía sin problema. Lo solucioné pasando a series de **ventanas de tiempo**.

## Cómo ejecutarlo todo desde cero

Necesitas Python 3.10 o superior. Todo corre en CPU; en mi caso tardó unos 105 minutos con 2 núcleos.

```bash
pip install -r codigo/requirements.txt
mkdir -p data/logsapp
tar -xzf datos/crudos/logdb.tar.gz  -C data
tar -xzf datos/crudos/loggit.tar.gz -C data
unzip logsaap.zip -d data/logsapp          # bajado de la Release v1.0

python -I codigo/prep.py data out                       # Drain3, ventanas y etiquetas débiles
python -I codigo/experiment.py out 10 30 logdb,loggit   # unos 35 min
python -I codigo/experiment.py out 10 30 logsapp        # unos 70 min
python -I codigo/analyze.py out                         # tablas, pruebas y figuras
```

Si solo quieres volver a generar las tablas y las figuras con mis resultados:

```bash
python -I codigo/analyze.py resultados
```

## Compilar el artículo

Subí la carpeta `articulo/` a Overleaf con el compilador **pdfLaTeX** y **Biber** para la bibliografía. También se puede compilar en local:

```bash
pdflatex articulo && biber articulo && pdflatex articulo && pdflatex articulo
```

## Limitaciones

- Las etiquetas salen de reglas, no de una revisión manual, así que puede haber ataques que no cubrí.
- En `logsapp` el tramo de prueba tiene solo 8 ventanas maliciosas, por eso el F1 sale muy bajo para todos los modelos.
- No hice búsqueda de hiperparámetros para no ajustar los modelos al conjunto de prueba.
