"""
================================================================================
 ACTIVIDAD 03 - Algoritmos de Enjambre en Aprendizaje Automático
 Parte 4 (ALTERNATIVA / BONUS): CLUSTERING usando PSO
================================================================================

Problema:
---------
Agrupar los datos del dataset Iris (usando solo las features, sin las
etiquetas) en K=3 clústeres, encontrando la mejor ubicación de los
centroides mediante PSO (en vez de usar K-Means clásico).

Ciclo del algoritmo PSO aplicado a clustering:
------------------------------------------------
1. Representación de la partícula:
   Cada partícula representa un CONJUNTO COMPLETO de K centroides,
   aplanado en un vector:
       x = [c1_1,...,c1_d, c2_1,...,c2_d, ..., cK_1,...,cK_d]
   Con K=3 clústeres y d=4 dimensiones (features de Iris) -> dim = 12.

2. Inicialización del enjambre:
   Se generan 'n_particles' conjuntos de centroides aleatorios, tomando
   valores dentro del rango de cada feature en los datos. Velocidades
   iniciales pequeñas y aleatorias.

3. Función de aptitud (fitness):
   Se decodifican los K centroides, se asigna cada punto al centroide
   más cercano (igual que en K-Means) y se calcula la SUMA DE DISTANCIAS
   INTRA-CLÚSTER (SSE - sum of squared errors). Queremos MINIMIZAR el SSE
   (equivalente a maximizar -SSE, que es lo que usamos como fitness).

4. Comportamiento de la partícula:
   Regla estándar de PSO:
     v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
     x_i(t+1) = x_i(t) + v_i(t+1)
   Cada partícula ajusta la posición de sus K centroides simultáneamente.

5. Evolución:
   max_iter iteraciones con inercia decreciente. Se guarda el mejor
   conjunto de centroides (menor SSE) encontrado hasta el momento.

6. Finalización:
   Al llegar a max_iter, se decodifica gbest en los centroides finales,
   se asignan las etiquetas de clúster y se compara contra K-Means y
   contra las etiquetas reales (solo para fines de evaluación/comparación).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

RNG = np.random.default_rng(11)

# -----------------------------------------------------------------------
# 1. Datos
# -----------------------------------------------------------------------
data = load_iris()
X, y_true = data.data, data.target
N, D = X.shape
K = 3
DIM = K * D

X_MIN, X_MAX = X.min(axis=0), X.max(axis=0)

# -----------------------------------------------------------------------
# 2. Decodificación y función de aptitud (SSE)
# -----------------------------------------------------------------------
def decode(vec):
    return vec.reshape(K, D)

def assign_clusters(centroids):
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (N,K)
    labels = np.argmin(dists, axis=1)
    sse = np.sum((dists[np.arange(N), labels]) ** 2)
    return labels, sse

def fitness(vec):
    centroids = decode(vec)
    _, sse = assign_clusters(centroids)
    return -sse  # maximizar -SSE == minimizar SSE

# -----------------------------------------------------------------------
# 3. Algoritmo PSO para clustering
# -----------------------------------------------------------------------
class PSOClustering:
    def __init__(self, dim, n_particles=25, max_iter=60,
                 w_max=0.9, w_min=0.4, c1=1.6, c2=1.6):
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2

        lo = np.tile(X_MIN, K)
        hi = np.tile(X_MAX, K)
        self.lo, self.hi = lo, hi
        self.pos = RNG.uniform(lo, hi, size=(n_particles, dim))
        self.vel = RNG.uniform(-1, 1, size=(n_particles, dim)) * (hi - lo) * 0.1

        self.pbest_pos = self.pos.copy()
        self.pbest_fit = np.array([fitness(p) for p in self.pos])

        best_idx = np.argmax(self.pbest_fit)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_fit = self.pbest_fit[best_idx]
        self.history = [-self.gbest_fit]  # guardamos SSE (positivo) para graficar

    def run(self, verbose=True):
        for it in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * it / self.max_iter
            r1 = RNG.uniform(0, 1, size=(self.n_particles, self.dim))
            r2 = RNG.uniform(0, 1, size=(self.n_particles, self.dim))

            self.vel = (w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pos)
                        + self.c2 * r2 * (self.gbest_pos - self.pos))
            self.pos = np.clip(self.pos + self.vel, self.lo, self.hi)

            fits = np.array([fitness(p) for p in self.pos])
            improved = fits > self.pbest_fit
            self.pbest_pos[improved] = self.pos[improved]
            self.pbest_fit[improved] = fits[improved]

            best_idx = np.argmax(self.pbest_fit)
            if self.pbest_fit[best_idx] > self.gbest_fit:
                self.gbest_fit = self.pbest_fit[best_idx]
                self.gbest_pos = self.pbest_pos[best_idx].copy()

            self.history.append(-self.gbest_fit)
            if verbose and (it + 1) % 10 == 0:
                print(f"Iter {it+1:3d}/{self.max_iter} | SSE = {-self.gbest_fit:.3f}")

        return self.gbest_pos, self.gbest_fit, self.history


if __name__ == "__main__":
    print("=" * 70)
    print("PSO - Clustering sobre dataset Iris (K=3)")
    print("=" * 70)

    pso = PSOClustering(DIM, n_particles=25, max_iter=60)
    best_vec, best_fit, history = pso.run()

    centroids = decode(best_vec)
    labels_pso, sse_pso = assign_clusters(centroids)

    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    sse_kmeans = km.inertia_

    ari_pso = adjusted_rand_score(y_true, labels_pso)
    ari_kmeans = adjusted_rand_score(y_true, km.labels_)
    sil_pso = silhouette_score(X, labels_pso)
    sil_kmeans = silhouette_score(X, km.labels_)

    print("\n--- COMPARACIÓN PSO vs K-MEANS ---")
    print(f"SSE           -> PSO: {sse_pso:.3f}   | K-Means: {sse_kmeans:.3f}")
    print(f"Adjusted Rand -> PSO: {ari_pso:.4f}   | K-Means: {ari_kmeans:.4f}")
    print(f"Silhouette    -> PSO: {sil_pso:.4f}   | K-Means: {sil_kmeans:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o", markersize=3, color="green")
    plt.title("Convergencia de PSO - Clustering (SSE)")
    plt.xlabel("Iteración")
    plt.ylabel("SSE (menor es mejor)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_pso_clustering.png", dpi=150)

    # Visualización 2D (usando 2 de las 4 features para graficar)
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 2], c=labels_pso, cmap="viridis", s=25)
    plt.scatter(centroids[:, 0], centroids[:, 2], c="red", marker="X", s=200, label="Centroides PSO")
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[2])
    plt.title("Clusters encontrados por PSO (Iris)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("clusters_pso.png", dpi=150)
    print("\nGráficos guardados: convergencia_pso_clustering.png, clusters_pso.png")
