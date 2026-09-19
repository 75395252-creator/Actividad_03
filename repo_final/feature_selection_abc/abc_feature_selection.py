"""
================================================================================
 ACTIVIDAD 03 - Algoritmos de Enjambre en Aprendizaje Automático
 Parte 1: FEATURE SELECTION usando ABC (Artificial Bee Colony)
================================================================================

Problema:
---------
Dado un dataset con N características (features), queremos encontrar el
subconjunto de características que MAXIMICE el desempeño de un clasificador
(accuracy) mientras MINIMIZA la cantidad de características usadas.

Dataset: Breast Cancer Wisconsin (sklearn) -> 30 características, 2 clases.

Ciclo del algoritmo ABC:
-------------------------
1. Representación de la partícula (fuente de alimento):
   Vector real x = [x_1, x_2, ..., x_30], x_i en [0,1].
   Se binariza con un umbral (x_i > 0.5 -> feature activa).

2. Inicialización del enjambre:
   Se generan SN (número de fuentes de alimento) vectores aleatorios en [0,1]^30.
   Cada fuente tiene un contador de "trials" (intentos sin mejora) = 0.

3. Función de aptitud (fitness):
   fitness = w1 * accuracy(subset) - w2 * (n_features_seleccionadas / n_total)
   Se entrena un KNN con validación cruzada usando solo las features activas.
   Si no se selecciona ninguna feature, fitness = 0 (solución inválida).

4. Comportamiento de la partícula (3 fases de ABC):
   a) Abejas empleadas (Employed bees): cada una explora un vecino de su
      fuente actual: v_i = x_i + phi*(x_i - x_k), k != i aleatorio.
      Si v_i es mejor, reemplaza a x_i (selección voraz) y trial=0,
      si no, trial += 1.
   b) Abejas observadoras (Onlooker bees): eligen una fuente con
      probabilidad proporcional a su fitness (selección por ruleta) y
      repiten el mismo proceso de búsqueda local.
   c) Abejas exploradoras (Scout bees): si una fuente supera 'limit'
      intentos sin mejorar, se abandona y se reemplaza por una fuente
      aleatoria nueva (evita óptimos locales / estancamiento).

5. Evolución:
   Se repiten las 3 fases durante 'max_iter' ciclos, guardando siempre la
   mejor fuente global encontrada (elitismo implícito al reportar el best).

6. Finalización:
   El algoritmo termina al alcanzar max_iter o si converge (mejor fitness
   no cambia durante N iteraciones). Se retorna el mejor subconjunto de
   características encontrado.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)

# -----------------------------------------------------------------------
# 1. Datos
# -----------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
N_FEATURES = X.shape[1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------------------------------------------------
# 2. Función de aptitud (fitness)
# -----------------------------------------------------------------------
W_ACC = 0.9      # peso para el accuracy
W_FEAT = 0.1     # peso (penalización) por número de features

def binarize(x, threshold=0.5):
    return (x > threshold).astype(int)

def fitness(x):
    mask = binarize(x)
    if mask.sum() == 0:
        return 0.0
    X_sub = X[:, mask == 1]
    clf = KNeighborsClassifier(n_neighbors=5)
    acc = cross_val_score(clf, X_sub, y, cv=5, scoring="accuracy").mean()
    feature_ratio = mask.sum() / N_FEATURES
    return W_ACC * acc - W_FEAT * feature_ratio

# -----------------------------------------------------------------------
# 3. Algoritmo ABC
# -----------------------------------------------------------------------
class ABCFeatureSelector:
    def __init__(self, dim, sn=20, limit=8, max_iter=40):
        self.dim = dim
        self.sn = sn          # número de fuentes de alimento (= empleadas = observadoras)
        self.limit = limit    # intentos máximos antes de abandonar una fuente
        self.max_iter = max_iter

        self.sources = RNG.uniform(0, 1, size=(sn, dim))
        self.fitness_vals = np.array([fitness(s) for s in self.sources])
        self.trials = np.zeros(sn, dtype=int)

        best_idx = np.argmax(self.fitness_vals)
        self.best_source = self.sources[best_idx].copy()
        self.best_fitness = self.fitness_vals[best_idx]
        self.history = [self.best_fitness]

    def _explore_neighbor(self, i):
        k = RNG.integers(0, self.sn)
        while k == i:
            k = RNG.integers(0, self.sn)
        j = RNG.integers(0, self.dim)
        phi = RNG.uniform(-1, 1)
        v = self.sources[i].copy()
        v[j] = self.sources[i, j] + phi * (self.sources[i, j] - self.sources[k, j])
        v[j] = np.clip(v[j], 0, 1)
        return v

    def _greedy_select(self, i, v):
        f_v = fitness(v)
        if f_v > self.fitness_vals[i]:
            self.sources[i] = v
            self.fitness_vals[i] = f_v
            self.trials[i] = 0
        else:
            self.trials[i] += 1

    def run(self, verbose=True):
        for it in range(self.max_iter):
            # a) Fase de abejas empleadas
            for i in range(self.sn):
                v = self._explore_neighbor(i)
                self._greedy_select(i, v)

            # b) Fase de abejas observadoras (selección por ruleta)
            fit_shift = self.fitness_vals - self.fitness_vals.min() + 1e-6
            probs = fit_shift / fit_shift.sum()
            for _ in range(self.sn):
                i = RNG.choice(self.sn, p=probs)
                v = self._explore_neighbor(i)
                self._greedy_select(i, v)

            # c) Fase de abejas exploradoras (scouts)
            for i in range(self.sn):
                if self.trials[i] > self.limit:
                    self.sources[i] = RNG.uniform(0, 1, size=self.dim)
                    self.fitness_vals[i] = fitness(self.sources[i])
                    self.trials[i] = 0

            # Actualizar mejor global
            best_idx = np.argmax(self.fitness_vals)
            if self.fitness_vals[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_vals[best_idx]
                self.best_source = self.sources[best_idx].copy()

            self.history.append(self.best_fitness)
            if verbose and (it + 1) % 5 == 0:
                print(f"Iter {it+1:3d}/{self.max_iter} | Mejor fitness = {self.best_fitness:.4f}")

        return self.best_source, self.best_fitness, self.history


if __name__ == "__main__":
    print("=" * 70)
    print("ABC - Feature Selection sobre Breast Cancer Wisconsin dataset")
    print("=" * 70)

    abc = ABCFeatureSelector(dim=N_FEATURES, sn=20, limit=8, max_iter=40)
    best_x, best_fit, history = abc.run()

    mask = binarize(best_x)
    selected = [f for f, m in zip(feature_names, mask) if m == 1]
    X_sel = X[:, mask == 1]
    acc_sel = cross_val_score(KNeighborsClassifier(5), X_sel, y, cv=5).mean()
    acc_all = cross_val_score(KNeighborsClassifier(5), X, y, cv=5).mean()

    print("\n--- RESULTADOS ---")
    print(f"Features totales: {N_FEATURES}  ->  Features seleccionadas: {mask.sum()}")
    print("Features elegidas:", selected)
    print(f"Accuracy con TODAS las features (KNN, 5-fold): {acc_all:.4f}")
    print(f"Accuracy con features SELECCIONADAS (KNN, 5-fold): {acc_sel:.4f}")
    print(f"Mejor fitness (ABC): {best_fit:.4f}")

    # Guardar resultados
    pd.DataFrame({"feature": feature_names, "selected": mask}).to_csv(
        "resultados_abc_feature_selection.csv", index=False
    )

    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o", markersize=3)
    plt.title("Convergencia de ABC - Feature Selection")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor fitness")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_abc.png", dpi=150)
    print("\nGráfico guardado en convergencia_abc.png")
