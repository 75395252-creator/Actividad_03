"""
================================================================================
 ACTIVIDAD 03 - Algoritmos de Enjambre en Aprendizaje Automático
 Parte 2: HYPERPARAMETER TUNING usando PSO (Particle Swarm Optimization)
================================================================================

Problema:
---------
Ajustar los hiperparámetros de un SVM (Support Vector Machine) con kernel RBF:
   - C      (regularización)      rango: [0.01, 100]
   - gamma  (coeficiente del kernel) rango: [0.0001, 10]
para MAXIMIZAR el accuracy de validación cruzada sobre el dataset Wine (sklearn).

Ciclo del algoritmo PSO:
-------------------------
1. Representación de la partícula:
   Cada partícula es un vector 2D: x = [C, gamma] (posición en el espacio
   de búsqueda de hiperparámetros). Además tiene una velocidad v = [vC, vGamma].

2. Inicialización del enjambre:
   Se generan 'n_particles' posiciones aleatorias dentro de los límites
   [C_min,C_max] x [gamma_min,gamma_max], y velocidades iniciales pequeñas
   (o en cero). Se guarda pbest (mejor posición de cada partícula) y
   gbest (mejor posición global).

3. Función de aptitud (fitness):
   fitness(C, gamma) = accuracy promedio (5-fold CV) de un SVM(C=C,
   gamma=gamma, kernel='rbf') sobre el dataset.

4. Comportamiento de la partícula:
   En cada iteración, la velocidad y posición de la partícula i se
   actualizan con la ecuación clásica de PSO:

       v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
       x_i(t+1) = x_i(t) + v_i(t+1)

   donde:
     w  : inercia (controla exploración vs explotación)
     c1 : coeficiente cognitivo (atracción a su propio mejor histórico)
     c2 : coeficiente social (atracción al mejor global del enjambre)
     r1,r2 : números aleatorios en [0,1]

   Las posiciones se limitan (clip) a los rangos válidos de C y gamma.

5. Evolución:
   Se repite el proceso durante 'max_iter' iteraciones. La inercia w
   decrece linealmente (de w_max a w_min) para pasar de exploración
   amplia al inicio a explotación fina al final (mejora la convergencia).

6. Finalización:
   El algoritmo termina tras max_iter iteraciones o si gbest no mejora
   durante un número de iteraciones consecutivas (criterio de paro
   temprano). Se retorna gbest como los mejores hiperparámetros.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(7)

# -----------------------------------------------------------------------
# 1. Datos
# -----------------------------------------------------------------------
data = load_wine()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)

# -----------------------------------------------------------------------
# 2. Espacio de búsqueda y función de aptitud
# -----------------------------------------------------------------------
BOUNDS = np.array([[0.01, 100.0],    # C
                   [0.0001, 10.0]])  # gamma

def fitness(pos):
    C, gamma = pos
    C = max(C, 1e-4)
    gamma = max(gamma, 1e-5)
    clf = SVC(C=C, gamma=gamma, kernel="rbf")
    return cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()

# -----------------------------------------------------------------------
# 3. Algoritmo PSO
# -----------------------------------------------------------------------
class PSOHyperparamTuner:
    def __init__(self, bounds, n_particles=15, max_iter=30,
                 w_max=0.9, w_min=0.4, c1=1.6, c2=1.6):
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2

        lo, hi = bounds[:, 0], bounds[:, 1]
        self.pos = RNG.uniform(lo, hi, size=(n_particles, self.dim))
        self.vel = RNG.uniform(-1, 1, size=(n_particles, self.dim)) * (hi - lo) * 0.1

        self.pbest_pos = self.pos.copy()
        self.pbest_fit = np.array([fitness(p) for p in self.pos])

        best_idx = np.argmax(self.pbest_fit)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_fit = self.pbest_fit[best_idx]
        self.history = [self.gbest_fit]

    def run(self, verbose=True):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        for it in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * it / self.max_iter

            r1 = RNG.uniform(0, 1, size=(self.n_particles, self.dim))
            r2 = RNG.uniform(0, 1, size=(self.n_particles, self.dim))

            self.vel = (w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pos)
                        + self.c2 * r2 * (self.gbest_pos - self.pos))
            self.pos = np.clip(self.pos + self.vel, lo, hi)

            fits = np.array([fitness(p) for p in self.pos])

            improved = fits > self.pbest_fit
            self.pbest_pos[improved] = self.pos[improved]
            self.pbest_fit[improved] = fits[improved]

            best_idx = np.argmax(self.pbest_fit)
            if self.pbest_fit[best_idx] > self.gbest_fit:
                self.gbest_fit = self.pbest_fit[best_idx]
                self.gbest_pos = self.pbest_pos[best_idx].copy()

            self.history.append(self.gbest_fit)
            if verbose and (it + 1) % 5 == 0:
                print(f"Iter {it+1:3d}/{self.max_iter} | gbest fitness = {self.gbest_fit:.4f} "
                      f"| C={self.gbest_pos[0]:.3f} gamma={self.gbest_pos[1]:.4f}")

        return self.gbest_pos, self.gbest_fit, self.history


if __name__ == "__main__":
    print("=" * 70)
    print("PSO - Hyperparameter Tuning de SVM (RBF) sobre dataset Wine")
    print("=" * 70)

    pso = PSOHyperparamTuner(BOUNDS, n_particles=15, max_iter=30)
    best_pos, best_fit, history = pso.run()

    C_opt, gamma_opt = best_pos
    baseline_acc = cross_val_score(SVC(kernel="rbf"), X, y, cv=5).mean()  # defaults

    print("\n--- RESULTADOS ---")
    print(f"Mejor C     = {C_opt:.4f}")
    print(f"Mejor gamma = {gamma_opt:.5f}")
    print(f"Accuracy con hiperparámetros óptimos (PSO): {best_fit:.4f}")
    print(f"Accuracy con hiperparámetros por defecto (C=1, gamma='scale'): {baseline_acc:.4f}")

    pd.DataFrame({"C": [C_opt], "gamma": [gamma_opt], "cv_accuracy": [best_fit]}).to_csv(
        "resultados_pso_hyperparams.csv", index=False
    )

    plt.figure(figsize=(7, 4))
    plt.plot(history, marker="o", markersize=3, color="darkorange")
    plt.title("Convergencia de PSO - Hyperparameter Tuning (SVM)")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor accuracy (gbest)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_pso.png", dpi=150)
    print("\nGráfico guardado en convergencia_pso.png")
