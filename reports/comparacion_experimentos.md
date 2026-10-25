# Comparación de experimentos — Equipo 31

Generado con `python scripts/comparar_experimentos.py`. Las métricas `val_` se usan para elegir; `test_` se reporta una sola vez.

## P1 - Regresión de demanda (viajes por estación-hora)

Criterio de selección: menor `val_mae`.

| corrida | algoritmo | caracteristicas | loss | learning_rate | max_leaf_nodes | n_estimators | min_samples_leaf | max_samples | val_mae | test_mae | test_rmse | test_mae_horas_pico | mejora_vs_baseline_% | tiempo_entrenamiento_s | tiempo_prediccion_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE P1-baseline-media-historica | — | — | — | — | — | — | — | — | 0.8355 | 0.8037 | 1.5449 | 1.3330 | 0.0000 | — | — |
| P1-hgb-caracteristicas-ampliadas | HistGradientBoostingRegressor | ampliadas | poisson | 0.05 | 127 | — | 100 | — | 0.7623 | 0.7654 | 1.3949 | 1.2526 | 4.7648 | 40.6959 | 5.5175 |
| P1-hgb-mejor-busqueda | HistGradientBoostingRegressor | base | poisson | 0.05 | 127 | — | 100 | — | 0.7640 | 0.7668 | 1.3943 | 1.2553 | 4.5948 | 45.4909 | 3.4660 |
| P1-rf-caracteristicas-ampliadas | RandomForestRegressor | ampliadas | — | — | — | 100 | 20 | 0.3 | 0.7669 | 0.7724 | 1.4370 | 1.2616 | 3.8891 | 83.0967 | 1.9801 |
| P1-rf-caracteristicas-base | RandomForestRegressor | base | — | — | — | 100 | 20 | 0.3 | 0.7686 | 0.7732 | 1.4347 | 1.2628 | 3.7893 | 51.0168 | 1.8244 |

**Elegido:** `P1-hgb-caracteristicas-ampliadas` (val_mae = 0.7623, mejora relativa en test_mae vs. baseline = 4.8 %).

## P2 - Clasificación de riesgo estimado de desbalance

Criterio de selección: mayor `val_f1`.

Nota: el umbral de `P2-hgb-umbral-validacion` se ajusta sobre validación, así que su `val_f1` es optimista; la comparación imparcial está en las columnas `test_`.

| corrida | algoritmo | learning_rate | max_leaf_nodes | l2_regularization | n_estimators | max_depth | C | umbral_prediccion | val_f1 | test_precision | test_recall | test_f1 | test_pr_auc | mejora_vs_baseline_% | tiempo_entrenamiento_s | tiempo_prediccion_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE P2-baseline-persistencia | — | — | — | — | — | — | — | — | 0.2691 | 0.2847 | 0.2847 | 0.2847 | 0.2190 | 0.0000 | — | — |
| P2-hgb-umbral-validacion | HistGradientBoostingClassifier | 0.1 | 63 | 0.0 | — | — | — | 0.7144 | 0.4235 | 0.3623 | 0.6110 | 0.4548 | 0.4278 | 59.7683 | 20.9718 | 2.4332 |
| P2-hgb-mejor-busqueda | HistGradientBoostingClassifier | 0.1 | 63 | 0.0 | — | — | — | 0.5 | 0.3765 | 0.2630 | 0.8497 | 0.4016 | 0.4278 | 41.0825 | 20.9718 | 2.1328 |
| P2-regresion-logistica | LogisticRegression | — | — | — | — | — | 1.0 | 0.5 | 0.3747 | 0.2791 | 0.7235 | 0.4028 | 0.3647 | 41.4810 | 1.2262 | 0.0685 |
| P2-rf-primera-entrega | RandomForestClassifier | — | — | — | 150 | 30 | — | 0.5 | 0.4114 | 0.3456 | 0.6166 | 0.4429 | 0.4087 | 55.5838 | 11.3264 | 4.3790 |

**Elegido:** `P2-hgb-umbral-validacion` (val_f1 = 0.4235, mejora relativa en test_f1 vs. baseline = 59.8 %).

## P3 - Clustering de estaciones

Criterio de selección: mayor `silhouette_espacio_base` con grupo mínimo >= 5 %.

| corrida | algoritmo | n_grupos | caracteristicas | covariance_type | silhouette_espacio_base | train_silhouette_score | davies_bouldin | grupo_mas_pequeno_pct | mejora_vs_baseline_% | tiempo_entrenamiento_s | tiempo_prediccion_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE P3-baseline-demanda | — | 3 | — | — | 0.0732 | 0.0732 | — | — | 0.0000 | — | — |
| P3-kmeans-caracteristicas-ampliadas | KMeans | 4 | ampliadas | — | 0.1796 | 0.2010 | 1.4753 | 14.2553 | 145.3 | 0.0082 | — |
| P3-gmm-mejor-k | GaussianMixture | 4 | base | full | 0.1214 | 0.1214 | 1.9361 | 15.3191 | 65.7881 | 0.0382 | — |
| P3-kmeans-mejor-k | KMeans | 4 | base | — | 0.2049 | 0.2049 | 1.4963 | 15.3191 | 179.8 | 0.0115 | — |

**Elegido:** `P3-kmeans-mejor-k` (silhouette_espacio_base = 0.2049, mejora relativa en silhouette_espacio_base vs. baseline = 179.8 %).
