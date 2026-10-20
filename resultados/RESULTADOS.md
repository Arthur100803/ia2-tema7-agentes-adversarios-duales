## Tabla descriptiva (media y DE sobre 10 semillas)
|                                |   ('F1', 'mean') |   ('F1', 'std') |   ('FPR', 'mean') |   ('FPR', 'std') |   ('AUC', 'mean') |   ('AUC', 'std') |   ('P', 'mean') |   ('P', 'std') |   ('R', 'mean') |   ('R', 'std') |   ('MCC', 'mean') |   ('MCC', 'std') |   ('epochs', 'mean') |   ('epochs', 'std') |
|:-------------------------------|-----------------:|----------------:|------------------:|-----------------:|------------------:|-----------------:|----------------:|---------------:|----------------:|---------------:|------------------:|-----------------:|---------------------:|--------------------:|
| ('logdb', 'IsolationForest')   |           25.33  |           1.962 |             5.569 |            0.304 |            80.147 |            1.404 |          26.864 |          1.555 |          24     |          2.468 |             0.194 |            0.02  |                  0   |               0     |
| ('logdb', 'LSTM-AE')           |           32.887 |           1.437 |             4.156 |            0.485 |            83.883 |            1.625 |          37.696 |          2.968 |          29.222 |          0.75  |             0.282 |            0.019 |                 30   |               0     |
| ('logdb', 'LSTM-GAN')          |           32.276 |           1.342 |             4.184 |            0.416 |            81.667 |            2.17  |          37.021 |          2.353 |          28.667 |          1.366 |             0.276 |            0.016 |                 30   |               0     |
| ('loggit', 'IsolationForest')  |           49.031 |           6.883 |             0.744 |            0.346 |            94.438 |            0.429 |          64.305 |         10.882 |          40     |          6.023 |             0.494 |            0.073 |                  0   |               0     |
| ('loggit', 'LSTM-AE')          |           64.999 |           3.959 |             0.303 |            0     |            86.698 |            0.804 |          84.652 |          1.183 |          52.857 |          4.735 |             0.661 |            0.035 |                 30   |               0     |
| ('loggit', 'LSTM-GAN')         |           62.636 |           7.61  |             1.502 |            1.15  |            85.498 |            1.177 |          63.707 |         18.554 |          65.238 |          5.521 |             0.623 |            0.08  |                 30   |               0     |
| ('logsapp', 'IsolationForest') |            0.823 |           0.123 |            12.488 |            1.399 |            84.966 |            1.668 |           0.414 |          0.062 |          57.5   |         10.541 |             0.041 |            0.008 |                  0   |               0     |
| ('logsapp', 'LSTM-AE')         |            2.808 |           0.153 |             5.394 |            0.493 |            96.544 |            0.357 |           1.427 |          0.079 |          86.25  |          7.096 |             0.107 |            0.006 |                 23.2 |               8.443 |
| ('logsapp', 'LSTM-GAN')        |            1.854 |           1.093 |             2.954 |            1.432 |            89.881 |            7.507 |           0.958 |          0.562 |          33.75  |         27.67  |             0.051 |            0.037 |                 22.7 |               8.706 |

## Pre-prueba (IF) vs post-prueba (LSTM-AE, LSTM-GAN)
| modelo_post   | subconjunto   | indicador   |   pre |   post |   delta |   delta_pct |
|:--------------|:--------------|:------------|------:|-------:|--------:|------------:|
| LSTM-AE       | logdb         | F1          | 25.33 |  32.89 |    7.56 |       29.84 |
| LSTM-AE       | logdb         | FPR         |  5.57 |   4.16 |   -1.41 |      -25.38 |
| LSTM-AE       | logdb         | AUC         | 80.15 |  83.88 |    3.74 |        4.66 |
| LSTM-AE       | loggit        | F1          | 49.03 |  65    |   15.97 |       32.57 |
| LSTM-AE       | loggit        | FPR         |  0.74 |   0.3  |   -0.44 |      -59.18 |
| LSTM-AE       | loggit        | AUC         | 94.44 |  86.7  |   -7.74 |       -8.2  |
| LSTM-AE       | logsapp       | F1          |  0.82 |   2.81 |    1.98 |      241.16 |
| LSTM-AE       | logsapp       | FPR         | 12.49 |   5.39 |   -7.09 |      -56.8  |
| LSTM-AE       | logsapp       | AUC         | 84.97 |  96.54 |   11.58 |       13.63 |
| LSTM-AE       | consolidado   | F1          | 25.06 |  33.56 |    8.5  |       33.93 |
| LSTM-AE       | consolidado   | FPR         |  6.27 |   3.28 |   -2.98 |      -47.59 |
| LSTM-AE       | consolidado   | AUC         | 86.52 |  89.04 |    2.52 |        2.92 |
| LSTM-GAN      | logdb         | F1          | 25.33 |  32.28 |    6.95 |       27.42 |
| LSTM-GAN      | logdb         | FPR         |  5.57 |   4.18 |   -1.39 |      -24.87 |
| LSTM-GAN      | logdb         | AUC         | 80.15 |  81.67 |    1.52 |        1.9  |
| LSTM-GAN      | loggit        | F1          | 49.03 |  62.64 |   13.6  |       27.75 |
| LSTM-GAN      | loggit        | FPR         |  0.74 |   1.5  |    0.76 |      102.04 |
| LSTM-GAN      | loggit        | AUC         | 94.44 |  85.5  |   -8.94 |       -9.47 |
| LSTM-GAN      | logsapp       | F1          |  0.82 |   1.85 |    1.03 |      125.32 |
| LSTM-GAN      | logsapp       | FPR         | 12.49 |   2.95 |   -9.53 |      -76.34 |
| LSTM-GAN      | logsapp       | AUC         | 84.97 |  89.88 |    4.91 |        5.78 |
| LSTM-GAN      | consolidado   | F1          | 25.06 |  32.26 |    7.19 |       28.71 |
| LSTM-GAN      | consolidado   | FPR         |  6.27 |   2.88 |   -3.39 |      -54.04 |
| LSTM-GAN      | consolidado   | AUC         | 86.52 |  85.68 |   -0.84 |       -0.97 |

## Pruebas pareadas
| comparacion              | indicador   |   n_pares |   media_dif |   de_dif |   shapiro_p |   wilcoxon_W |   wilcoxon_p |   r_efecto |      t |         t_p |
|:-------------------------|:------------|----------:|------------:|---------:|------------:|-------------:|-------------:|-----------:|-------:|------------:|
| LSTM-AE vs IF (H1, H2)   | F1          |        30 |       8.503 |    7.436 |      0.0078 |         10   |  8.00937e-08 |      0.836 |  6.264 | 7.74788e-07 |
| LSTM-AE vs IF (H1, H2)   | FPR         |        30 |      -2.982 |    3.112 |      0.0001 |          0   |  1.71809e-06 |      0.873 | -5.25  | 1.26681e-05 |
| LSTM-AE vs IF (H1, H2)   | AUC         |        30 |       2.525 |    8.176 |      0.0039 |        155   |  0.114177    |      0.291 |  1.691 | 0.101495    |
| LSTM-AE vs IF (H1, H2)   | MCC         |        30 |       0.107 |    0.065 |      0.0009 |          1   |  3.72529e-09 |      0.869 |  9.041 | 6.16525e-10 |
| LSTM-GAN vs LSTM-AE (H3) | F1          |        30 |      -1.309 |    5.71  |      0.0005 |        158   |  0.129353    |      0.28  | -1.256 | 0.219225    |
| LSTM-GAN vs LSTM-AE (H3) | FPR         |        30 |      -0.404 |    1.839 |      0.4482 |        158.5 |  0.310856    |      0.185 | -1.204 | 0.238216    |
| LSTM-GAN vs LSTM-AE (H3) | AUC         |        30 |      -3.36  |    5.1   |      0      |         18   |  4.71249e-07 |      0.806 | -3.608 | 0.00114626  |
| LSTM-GAN vs LSTM-AE (H3) | MCC         |        30 |      -0.033 |    0.064 |      0.3447 |         89   |  0.00236749  |      0.539 | -2.84  | 0.00815493  |
| LSTM-GAN vs IF           | F1          |        30 |       7.194 |    6.734 |      0.0047 |          6   |  3.17978e-06 |      0.851 |  5.851 | 2.39614e-06 |
| LSTM-GAN vs IF           | FPR         |        30 |      -3.387 |    4.657 |      0.0006 |         79   |  0.00159225  |      0.576 | -3.983 | 0.000418429 |
| LSTM-GAN vs IF           | AUC         |        30 |      -0.835 |    7.568 |      0.0162 |        201   |  0.529105    |      0.118 | -0.604 | 0.550336    |
| LSTM-GAN vs IF           | MCC         |        30 |       0.074 |    0.07  |      0.0627 |         24   |  1.79793e-05 |      0.783 |  5.829 | 2.54978e-06 |

## Wilcoxon por subconjunto
| comparacion                 | subconjunto   | indicador   |   media_a |   media_b |   wilcoxon_W |      p |
|:----------------------------|:--------------|:------------|----------:|----------:|-------------:|-------:|
| LSTM-AE vs IsolationForest  | logdb         | F1          |     25.33 |     32.89 |          0   | 0.002  |
| LSTM-AE vs IsolationForest  | logdb         | FPR         |      5.57 |      4.16 |          0   | 0.002  |
| LSTM-AE vs IsolationForest  | logdb         | AUC         |     80.15 |     83.88 |          0   | 0.002  |
| LSTM-AE vs IsolationForest  | loggit        | F1          |     49.03 |     65    |          1   | 0.0039 |
| LSTM-AE vs IsolationForest  | loggit        | FPR         |      0.74 |      0.3  |          0   | 0.002  |
| LSTM-AE vs IsolationForest  | loggit        | AUC         |     94.44 |     86.7  |          0   | 0.002  |
| LSTM-AE vs IsolationForest  | logsapp       | F1          |      0.82 |      2.81 |          0   | 0.002  |
| LSTM-AE vs IsolationForest  | logsapp       | FPR         |     12.49 |      5.39 |          0   | 0.002  |
| LSTM-AE vs IsolationForest  | logsapp       | AUC         |     84.97 |     96.54 |          0   | 0.002  |
| LSTM-GAN vs LSTM-AE         | logdb         | F1          |     32.89 |     32.28 |         20   | 0.4922 |
| LSTM-GAN vs LSTM-AE         | logdb         | FPR         |      4.16 |      4.18 |         25.5 | 0.8652 |
| LSTM-GAN vs LSTM-AE         | logdb         | AUC         |     83.88 |     81.67 |          3   | 0.0098 |
| LSTM-GAN vs LSTM-AE         | loggit        | F1          |     65    |     62.64 |         22   | 0.625  |
| LSTM-GAN vs LSTM-AE         | loggit        | FPR         |      0.3  |      1.5  |          1   | 0.0156 |
| LSTM-GAN vs LSTM-AE         | loggit        | AUC         |     86.7  |     85.5  |          7   | 0.0371 |
| LSTM-GAN vs LSTM-AE         | logsapp       | F1          |      2.81 |      1.85 |          6   | 0.0273 |
| LSTM-GAN vs LSTM-AE         | logsapp       | FPR         |      5.39 |      2.95 |          1   | 0.0039 |
| LSTM-GAN vs LSTM-AE         | logsapp       | AUC         |     96.54 |     89.88 |          0   | 0.002  |
| LSTM-GAN vs IsolationForest | logdb         | F1          |     25.33 |     32.28 |          0   | 0.002  |
| LSTM-GAN vs IsolationForest | logdb         | FPR         |      5.57 |      4.18 |          0   | 0.002  |
| LSTM-GAN vs IsolationForest | logdb         | AUC         |     80.15 |     81.67 |         11   | 0.1055 |
| LSTM-GAN vs IsolationForest | loggit        | F1          |     49.03 |     62.64 |          0   | 0.002  |
| LSTM-GAN vs IsolationForest | loggit        | FPR         |      0.74 |      1.5  |          7   | 0.0352 |
| LSTM-GAN vs IsolationForest | loggit        | AUC         |     94.44 |     85.5  |          0   | 0.002  |
| LSTM-GAN vs IsolationForest | logsapp       | F1          |      0.82 |      1.85 |          6   | 0.0273 |
| LSTM-GAN vs IsolationForest | logsapp       | FPR         |     12.49 |      2.95 |          0   | 0.002  |
| LSTM-GAN vs IsolationForest | logsapp       | AUC         |     84.97 |     89.88 |         13   | 0.1602 |

## Sensibilidad al umbral (media de semillas)
| subconjunto   | modelo          |   percentil |    F1 |   FPR |   Recall |
|:--------------|:----------------|------------:|------:|------:|---------:|
| logdb         | IsolationForest |        95   | 25.33 |  5.57 |    24    |
| logdb         | IsolationForest |        99   | 19.96 |  1.39 |    13    |
| logdb         | IsolationForest |        99.9 |  8.3  |  0.22 |     4.44 |
| logdb         | LSTM-AE         |        95   | 32.89 |  4.16 |    29.22 |
| logdb         | LSTM-AE         |        99   | 34.53 |  1.69 |    25    |
| logdb         | LSTM-AE         |        99.9 | 32.33 |  0.53 |    20.56 |
| logdb         | LSTM-GAN        |        95   | 32.28 |  4.18 |    28.67 |
| logdb         | LSTM-GAN        |        99   | 33.68 |  1.69 |    24.33 |
| logdb         | LSTM-GAN        |        99.9 | 28.32 |  0.65 |    18.11 |
| loggit        | IsolationForest |        95   | 49.03 |  0.74 |    40    |
| loggit        | IsolationForest |        99   | 30.6  |  0    |    18.1  |
| loggit        | IsolationForest |        99.9 |  0    |  0    |     0    |
| loggit        | LSTM-AE         |        95   | 65    |  0.3  |    52.86 |
| loggit        | LSTM-AE         |        99   | 35.71 |  0.3  |    23.81 |
| loggit        | LSTM-AE         |        99.9 |  0    |  0    |     0    |
| loggit        | LSTM-GAN        |        95   | 62.64 |  1.5  |    65.24 |
| loggit        | LSTM-GAN        |        99   | 44.39 |  0.09 |    29.52 |
| loggit        | LSTM-GAN        |        99.9 | 23.26 |  0.02 |    13.33 |
| logsapp       | IsolationForest |        95   |  0.82 | 12.49 |    57.5  |
| logsapp       | IsolationForest |        99   |  0    |  1.15 |     0    |
| logsapp       | IsolationForest |        99.9 |  0    |  0.18 |     0    |
| logsapp       | LSTM-AE         |        95   |  2.81 |  5.39 |    86.25 |
| logsapp       | LSTM-AE         |        99   |  0.88 |  1.46 |    10    |
| logsapp       | LSTM-AE         |        99.9 |  0    |  0.15 |     0    |
| logsapp       | LSTM-GAN        |        95   |  1.85 |  2.95 |    33.75 |
| logsapp       | LSTM-GAN        |        99   |  1.11 |  0.61 |     6.25 |
| logsapp       | LSTM-GAN        |        99.9 |  0    |  0.04 |     0    |

## Bootstrap dif. AUC
| comparacion               | subconjunto   |   auc_a |   auc_b |   dif |   IC95_inf |   IC95_sup |
|:--------------------------|:--------------|--------:|--------:|------:|-----------:|-----------:|
| LSTM-AE - IsolationForest | logdb         |   81.05 |   84.7  |  3.63 |       1.48 |       5.93 |
| LSTM-GAN - LSTM-AE        | logdb         |   84.7  |   82.78 | -1.91 |      -2.79 |      -1.12 |
| LSTM-AE - IsolationForest | loggit        |   94.51 |   86.48 | -8.22 |     -18.84 |      -0.3  |
| LSTM-GAN - LSTM-AE        | loggit        |   86.48 |   85.76 | -0.7  |      -2.95 |       0.76 |
| LSTM-AE - IsolationForest | logsapp       |   85.31 |   96.9  | 11.81 |       5.71 |      19.8  |
| LSTM-GAN - LSTM-AE        | logsapp       |   96.9  |   93.57 | -3.35 |      -4.92 |      -1.44 |

## Dataset
|                  | 0                          | 1                                            | 2                        |
|:-----------------|:---------------------------|:---------------------------------------------|:-------------------------|
| subconjunto      | logdb                      | loggit                                       | logsapp                  |
| eventos          | 267896                     | 56135                                        | 2893513                  |
| plantillas       | 186                        | 121                                          | 361                      |
| ventanas         | 5719                       | 3398                                         | 44299                    |
| desde            | 2026-03-01 05:13           | 2025-10-30 00:44                             | 2025-11-20 05:49         |
| hasta            | 2026-05-12 21:40           | 2026-05-12 10:07                             | 2026-05-12 21:41         |
| ventanas_train   | 3431                       | 2038                                         | 26579                    |
| ventanas_val     | 1144                       | 680                                          | 8860                     |
| ventanas_test    | 1144                       | 680                                          | 8860                     |
| maliciosas_train | 2                          | 38                                           | 24                       |
| maliciosas_val   | 1                          | 28                                           | 2                        |
| maliciosas_test  | 90                         | 21                                           | 8                        |
| reglas_activadas | {"sqli": 27, "probe": 148} | {"web": 86, "scan": 0, "ssh": 0, "c4xx": 53} | {"web": 33, "c4xx": 819} |
