# Actividad 03 — Algoritmos de Enjambre en Aprendizaje Automático

Ejemplos prácticos de **Swarm Intelligence** (ABC y PSO) aplicados a distintas etapas
del ciclo de vida de un modelo de Machine Learning.

## 📁 Estructura del repositorio

```
.
├── feature_selection_abc/
│   └── abc_feature_selection.py      # Feature Selection con ABC (Breast Cancer dataset)
├── hyperparameter_tuning_pso/
│   └── pso_hyperparameter_tuning.py  # Hyperparameter Tuning con PSO (SVM sobre Wine dataset)
├── nn_training_pso/
│   └── nn_training_pso.py            # Entrenamiento de red neuronal 4-8-3 con PSO (Iris), sin backprop
├── clustering_pso/
│   └── pso_clustering.py             # (Extra) Clustering con PSO (Iris) vs K-Means
├── figures/                          # Gráficos de convergencia generados
├── results/                          # CSVs con resultados numéricos
├── report/
│   └── build_report.py               # Script que genera el informe PDF
└── Informe_Actividad03_Swarm_ML.pdf  # Informe final
```

## 🚀 Cómo ejecutar

Requisitos: Python 3.10+, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

```bash
pip install numpy pandas scikit-learn matplotlib

# 1. Feature Selection con ABC
python feature_selection_abc/abc_feature_selection.py

# 2. Hyperparameter Tuning con PSO
python hyperparameter_tuning_pso/pso_hyperparameter_tuning.py

# 3. Entrenamiento de red neuronal con PSO (sin backpropagation)
python nn_training_pso/nn_training_pso.py

# 4. (Extra) Clustering con PSO
python clustering_pso/pso_clustering.py
```

También pueden ejecutarse directamente en **Google Colab** o **Jupyter Notebook**
copiando el contenido de cada script en una celda.

## 🧠 Resumen de cada ejemplo

| # | Problema | Algoritmo | Dataset | Resultado |
|---|----------|-----------|---------|-----------|
| 1 | Feature Selection | ABC (Artificial Bee Colony) | Breast Cancer Wisconsin (30 features) | 9 features seleccionadas, mismo accuracy (96.5%) que con las 30 |
| 2 | Hyperparameter Tuning | PSO (Particle Swarm Optimization) | Wine (SVM-RBF: C, gamma) | Accuracy 98.9% (vs 98.3% con hiperparámetros default) |
| 3 | Entrenamiento de NN sin backprop | PSO | Iris (red 4-8-3, 67 pesos/bias) | Train acc 100%, Val acc 89.5% |
| 4 (extra) | Clustering | PSO | Iris (K=3) | SSE 112.8 vs 78.9 de K-Means; ARI 0.556 vs 0.730 |

## 🔄 Ciclo general de un algoritmo de enjambre (aplicado en todos los ejemplos)

1. **Representación de la partícula**: cómo se codifica una solución candidata (máscara
   de features, par de hiperparámetros, vector de pesos de una NN, o conjunto de centroides).
2. **Inicialización del enjambre**: generación aleatoria de la población inicial de partículas/fuentes.
3. **Función de aptitud (fitness)**: métrica que se quiere maximizar o minimizar (accuracy,
   SSE, combinación accuracy-penalización, etc.).
4. **Comportamiento de la partícula**: reglas de movimiento/actualización
   (ecuaciones de velocidad-posición en PSO; fases de abejas empleadas/observadoras/exploradoras en ABC).
5. **Evolución**: repetición del ciclo durante N iteraciones, actualizando siempre la mejor
   solución global encontrada.
6. **Finalización**: criterio de parada (número máximo de iteraciones o convergencia) y
   extracción de la mejor solución encontrada.

## 📄 Informe

El informe completo con la explicación detallada de cada algoritmo, resultados, gráficos
de convergencia y conclusiones se encuentra en
[`Informe_Actividad03_Swarm_ML.pdf`](./Informe_Actividad03_Swarm_ML.pdf).

## 🎥 Video explicativo

> Enlace al video (máx. 15 min): _agregar aquí el enlace_

## 👥 Integrantes

-Flores Mamani Adams Freddy

## 📚 Referencias

- Kennedy, J., & Eberhart, R. (1995). *Particle swarm optimization*. Proceedings of ICNN'95.
- Karaboga, D. (2005). *An idea based on honey bee swarm for numerical optimization*.
  Technical Report TR06, Erciyes University.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825-2830.
