"""
================================================================================
 ACTIVIDAD 03 - Algoritmos de Enjambre en Aprendizaje Automático
 Parte 3: ENTRENAMIENTO DE RED NEURONAL usando PSO (sin backpropagation)
================================================================================

Problema:
---------
Entrenar una red neuronal totalmente conectada de 3 capas
(entrada -> oculta -> salida) para clasificar el dataset Iris,
encontrando los PESOS y BIAS directamente mediante PSO, sin usar
gradiente ni backpropagation.

Arquitectura: 4 (entrada) -> 8 (oculta, ReLU) -> 3 (salida, softmax)

Ciclo del algoritmo PSO aplicado a NN training:
-------------------------------------------------
1. Representación de la partícula:
   Cada partícula es un VECTOR PLANO que contiene TODOS los pesos y bias
   de la red concatenados:
       x = [W1 (4x8), b1 (8), W2 (8x3), b2 (3)]  ->  dim = 32+8+24+3 = 67
   Es decir, la posición de una partícula en el espacio de búsqueda
   ES una red neuronal completa.

2. Inicialización del enjambre:
   Se generan 'n_particles' vectores aleatorios (pesos ~ U(-1,1)) que
   representan redes neuronales distintas, y velocidades pequeñas.
   pbest y gbest se calculan igual que en PSO estándar.

3. Función de aptitud (fitness):
   Se "decodifica" el vector de la partícula en las matrices W1,b1,W2,b2,
   se hace un forward-pass sobre el set de entrenamiento y se calcula:
       fitness = accuracy de clasificación (queremos MAXIMIZAR)
   (equivalente a minimizar la pérdida de clasificación, pero aquí usamos
   accuracy directamente porque no se necesita que la función sea diferenciable)

4. Comportamiento de la partícula:
   Igual que PSO clásico:
     v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
     x_i(t+1) = x_i(t) + v_i(t+1)
   Cada dimensión del vector (=cada peso/bias de la red) se mueve según
   esta regla. No hay cálculo de gradientes en ningún momento.

5. Evolución:
   Se repite durante 'max_iter' iteraciones, con inercia w decreciente
   (exploración global al inicio, ajuste fino al final). Se monitorea el
   accuracy de entrenamiento y validación en cada iteración.

6. Finalización:
   Se detiene tras max_iter iteraciones o al alcanzar un accuracy objetivo.
   La mejor partícula (gbest) se decodifica de vuelta a W1,b1,W2,b2 y esa
   es la red neuronal final entrenada.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(3)

# -----------------------------------------------------------------------
# 1. Datos
# -----------------------------------------------------------------------
data = load_iris()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3, stratify=y
)

N_IN, N_HIDDEN, N_OUT = 4, 8, 3
DIM = N_IN * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUT + N_OUT  # = 67

# -----------------------------------------------------------------------
# 2. Codificación / decodificación del vector de pesos (forward pass)
# -----------------------------------------------------------------------
def decode(vec):
    idx = 0
    W1 = vec[idx: idx + N_IN * N_HIDDEN].reshape(N_IN, N_HIDDEN); idx += N_IN * N_HIDDEN
    b1 = vec[idx: idx + N_HIDDEN]; idx += N_HIDDEN
    W2 = vec[idx: idx + N_HIDDEN * N_OUT].reshape(N_HIDDEN, N_OUT); idx += N_HIDDEN * N_OUT
    b2 = vec[idx: idx + N_OUT]; idx += N_OUT
    return W1, b1, W2, b2

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def forward(vec, X_data):
    W1, b1, W2, b2 = decode(vec)
    h = relu(X_data @ W1 + b1)
    out = softmax(h @ W2 + b2)
    return out

def accuracy_from_probs(probs, y_true):
    preds = np.argmax(probs, axis=1)
    return (preds == y_true).mean()

def fitness(vec, X_data=X_train, y_data=y_train):
    probs = forward(vec, X_data)
    # combinamos accuracy con una pequeña penalización de cross-entropy
    # para suavizar el paisaje de búsqueda (ayuda a distinguir soluciones
    # con igual accuracy pero distinta confianza de predicción)
    acc = accuracy_from_probs(probs, y_data)
    ce = -np.log(probs[np.arange(len(y_data)), y_data] + 1e-9).mean()
    return acc - 0.01 * ce

# -----------------------------------------------------------------------
# 3. Algoritmo PSO para entrenar la red
# -----------------------------------------------------------------------
class PSONeuralNetTrainer:
    def __init__(self, dim, n_particles=40, max_iter=150,
                 w_max=0.9, w_min=0.3, c1=1.5, c2=1.5, v_clip=1.0):
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2 = c1, c2
        self.v_clip = v_clip

        self.pos = RNG.uniform(-1, 1, size=(n_particles, dim))
        self.vel = RNG.uniform(-0.1, 0.1, size=(n_particles, dim))

        self.pbest_pos = self.pos.copy()
        self.pbest_fit = np.array([fitness(p) for p in self.pos])

        best_idx = np.argmax(self.pbest_fit)
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.gbest_fit = self.pbest_fit[best_idx]
        self.history_train, self.history_val = [], []

    def run(self, verbose=True):
        for it in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * it / self.max_iter
            r1 = RNG.uniform(0, 1, size=(self.n_particles, self.dim))
            r2 = RNG.uniform(0, 1, size=(self.n_particles, self.dim))

            self.vel = (w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pos)
                        + self.c2 * r2 * (self.gbest_pos - self.pos))
            self.vel = np.clip(self.vel, -self.v_clip, self.v_clip)
            self.pos = self.pos + self.vel

            fits = np.array([fitness(p) for p in self.pos])
            improved = fits > self.pbest_fit
            self.pbest_pos[improved] = self.pos[improved]
            self.pbest_fit[improved] = fits[improved]

            best_idx = np.argmax(self.pbest_fit)
            if self.pbest_fit[best_idx] > self.gbest_fit:
                self.gbest_fit = self.pbest_fit[best_idx]
                self.gbest_pos = self.pbest_pos[best_idx].copy()

            train_acc = accuracy_from_probs(forward(self.gbest_pos, X_train), y_train)
            val_acc = accuracy_from_probs(forward(self.gbest_pos, X_test), y_test)
            self.history_train.append(train_acc)
            self.history_val.append(val_acc)

            if verbose and (it + 1) % 15 == 0:
                print(f"Iter {it+1:3d}/{self.max_iter} | train_acc={train_acc:.4f} "
                      f"| val_acc={val_acc:.4f} | fitness={self.gbest_fit:.4f}")

        return self.gbest_pos, self.gbest_fit


if __name__ == "__main__":
    print("=" * 70)
    print("PSO - Entrenamiento de Red Neuronal (4-8-3) sobre Iris, sin backprop")
    print("=" * 70)
    print(f"Dimensión del espacio de búsqueda (n. de pesos+bias): {DIM}")

    trainer = PSONeuralNetTrainer(DIM, n_particles=40, max_iter=150)
    best_vec, best_fit = trainer.run()

    train_acc = accuracy_from_probs(forward(best_vec, X_train), y_train)
    val_acc = accuracy_from_probs(forward(best_vec, X_test), y_test)

    print("\n--- RESULTADOS FINALES ---")
    print(f"Accuracy entrenamiento: {train_acc:.4f}")
    print(f"Accuracy validación (test): {val_acc:.4f}")

    np.save("mejores_pesos_pso_nn.npy", best_vec)

    plt.figure(figsize=(7, 4))
    plt.plot(trainer.history_train, label="Train accuracy")
    plt.plot(trainer.history_val, label="Val accuracy")
    plt.title("Convergencia de PSO entrenando una NN (sin backpropagation)")
    plt.xlabel("Iteración")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_pso_nn.png", dpi=150)
    print("\nGráfico guardado en convergencia_pso_nn.png")
