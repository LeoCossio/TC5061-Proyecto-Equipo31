# Model Card — MiBici

**Proyecto integrador MLOps — Equipo 31**  
**Última actualización:** 26 de noviembre de 2026.

## 1. Descripción de los modelos

El proyecto utiliza tres modelos de aprendizaje automático para analizar los patrones de movilidad del sistema de bicicletas públicas MiBici y apoyar la toma de decisiones operativas.

| Modelo | Algoritmo | Versión operativa |
|---|---|---|
| P1 — Predicción de demanda | HistGradientBoostingRegressor | v2 (champion) |
| P2 — Riesgo de desbalance | HistGradientBoostingClassifier | v1 |
| P3 — Segmentación de estaciones | K-Means (k=4) | v1 |

## 2. Datos de entrenamiento

- Fuente: portal de datos abiertos de MiBici.
- Periodo disponible: octubre de 2025 a septiembre de 2026.
- Registros originales: 4,497,877.
- Registros procesados: 4,497,119.
- Estaciones identificadas: 471.
- Almacenamiento: 12 archivos Parquet.
- Trazabilidad mediante hashes SHA-256 y versionamiento del código.
- Exclusión de identificadores de usuario y atributos demográficos.

P1 v2 utiliza información disponible hasta agosto de 2026 para su entrenamiento y septiembre de 2026 para su evaluación retrospectiva.

## 3. Evaluación de desempeño

| Modelo | Métrica | Resultado |
|---|---|---:|
| P1 v1 | MAE en septiembre | 0.8330 |
| P1 v2 | MAE en septiembre | 0.8273 |
| P1 v2 | RMSE en septiembre | 1.5071 |
| P2 v1 | F1-score de prueba | 0.4548 |
| P3 v1 | Silhouette Score de selección | 0.2049 |

Las métricas corresponden a los respectivos experimentos y protocolos de evaluación.

P1 v2 redujo el MAE un 0.68 %, el RMSE un 1.43 % y el MAE en horas pico un 0.83 % respecto a P1 v1. Obtuvo mejores resultados en 25 de los 30 días evaluados y cumplió las tres condiciones establecidas para su promoción.

## 4. Uso previsto

- Estimación de demanda horaria de bicicletas.
- Apoyo a decisiones de redistribución de bicicletas.
- Identificación de estaciones con riesgo indirecto de desbalance.
- Análisis de perfiles históricos de utilización de estaciones.

Los modelos son herramientas de apoyo y no sustituyen la supervisión humana ni los sistemas de inventario en tiempo real.

## 5. Limitaciones y riesgos

- Los modelos dependen de la representatividad de los datos históricos.
- Eventos extraordinarios, cambios de infraestructura y nuevas estaciones pueden afectar su desempeño.
- P1 y P2 requieren características históricas calculadas previamente.
- P2 estima un indicador indirecto de desbalance, no la disponibilidad real de bicicletas.
- El F1 de P2 corresponde al umbral optimizado durante la experimentación; la API utiliza el comportamiento de clasificación del estimador.
- Los grupos de P3 presentan superposición y su estabilidad temporal no está garantizada.
- El A/B testing se realizó de forma retrospectiva, sin tráfico real de usuarios.
- El sistema fue validado en un entorno local contenerizado, no en producción comercial.

## 6. Consideraciones éticas y privacidad

No se utilizan identificadores personales ni atributos demográficos como variables predictoras. Sin embargo, la ausencia de estos atributos no garantiza la inexistencia de sesgos geográficos o socioeconómicos.

Se recomienda evaluar el desempeño por estación y zona geográfica, evitando que las predicciones generen decisiones que perjudiquen sistemáticamente a regiones con menor demanda histórica.

Los resultados no deben utilizarse como único criterio para reducir servicios o infraestructura.

## 7. Monitoreo y mantenimiento

El proyecto incorpora Evidently para evaluar el drift de las características de P1.

- Variables evaluadas: 17.
- Variables con drift: 1 de 17.
- Variables con drift en estaciones comunes: 0 de 17.
- Drift global: no detectado en los periodos analizados.

Se recomienda monitorear periódicamente las distribuciones de entrada y las métricas de desempeño, así como evaluar nuevas versiones antes de promoverlas.

La promoción de modelos se realiza mediante los alias champion y challenger de MLflow. El reentrenamiento y la promoción no están automatizados de extremo a extremo.

## 8. Versionamiento y trazabilidad

Modelos registrados:

- `mibici-equipo31-p1-demanda`: v2 champion y v1 challenger.
- `mibici-equipo31-p2-riesgo`: v1.
- `mibici-equipo31-p3-grupos`: v1.

P1 v2 conserva las etiquetas correspondientes al commit, hash de datos, ventana de entrenamiento, periodo de evaluación y corrida original.

Docker Compose inicializa el Registry a partir de los artefactos versionados. FastAPI consulta el alias champion de P1 y las versiones definidas de P2 y P3.

Las nuevas promociones de P1 requieren recargar el modelo en el servicio.

## 9. Validación operativa

- Despliegue exitoso mediante Docker Compose desde un volumen nuevo.
- Tres endpoints de predicción operativos.
- Health check satisfactorio.
- PyTest: 65 pruebas exitosas, 1 omitida y 0 fallidas.
- Pruebas de carga y análisis FinOps documentados para la configuración anterior a la incorporación de P1 v2.

## 10. Referencias

- Datos abiertos: https://www.mibici.net/es/datos-abiertos/
- Documentación general: `README.md`.
- Evaluación A/B: `reports/ab_testing/resultado_p1.json`.
- Monitoreo: `reports/monitoring/`.
- Registro de modelos: `scripts/registrar_modelos.py`.
- Inicialización del Registry: `scripts/inicializar_registry.py`.
- Repositorio: https://github.com/LeoCossio/TC5061-Proyecto-Equipo31