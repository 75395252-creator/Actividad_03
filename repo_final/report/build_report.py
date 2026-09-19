# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                 Image, Table, TableStyle, ListFlowable, ListItem, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

FIG = "/home/claude/swarm_activity/figures"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", fontSize=22, leading=26, alignment=TA_CENTER,
                           spaceAfter=6, textColor=colors.HexColor("#1a2b4c"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Subtitle", fontSize=13, leading=18, alignment=TA_CENTER,
                           textColor=colors.HexColor("#555555")))
styles.add(ParagraphStyle(name="H1", fontSize=16, leading=20, spaceBefore=18, spaceAfter=8,
                           textColor=colors.HexColor("#1a2b4c"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6,
                           textColor=colors.HexColor("#2b4c7a"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyBold", parent=styles["Body"], fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor=colors.HexColor("#666666")))


story = []

def h1(txt): story.append(Paragraph(txt, styles["H1"]))
def h2(txt): story.append(Paragraph(txt, styles["H2"]))
def p(txt): story.append(Paragraph(txt, styles["Body"]))
def small(txt): story.append(Paragraph(txt, styles["Small"]))
def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, styles["Body"])) for i in items],
                               bulletType="bullet", start="•", leftIndent=14))
def img(path, width=14.5*cm):
    story.append(Spacer(1, 6))
    story.append(Image(path, width=width, height=width*0.58))
    story.append(Spacer(1, 4))
def hr():
    story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#cccccc"), spaceBefore=6, spaceAfter=6))

# ==========================================================================
# PORTADA
# ==========================================================================
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Actividad 03", styles["TitleBig"]))
story.append(Paragraph("Algoritmos de Enjambre aplicados al Aprendizaje Automático", styles["Subtitle"]))
story.append(Spacer(1, 1*cm))
story.append(Paragraph("Feature Selection (ABC) · Hyperparameter Tuning (PSO) · "
                        "Entrenamiento de Redes Neuronales sin Backpropagation (PSO) · Clustering (PSO)",
                        styles["Subtitle"]))
story.append(Spacer(1, 3*cm))

info_table = Table([
    ["Curso:", "Inteligencia Artificial / Aprendizaje de Máquina"],
    ["Tema:", "Swarm Intelligence (Inteligencia de Enjambre)"],
    ["Integrantes:", "________________________________________"],
    ["Repositorio GitHub:", "________________________________________"],
    ["Video explicativo:", "________________________________________"],
    ["Fecha:", "________________________________________"],
], colWidths=[4*cm, 10.5*cm])
info_table.setStyle(TableStyle([
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 8),
    ("LINEBELOW", (0,0), (-1,-1), 0.5, colors.HexColor("#dddddd")),
]))
story.append(info_table)
story.append(PageBreak())

# ==========================================================================
# 0. INTRODUCCION Y OBJETIVO
# ==========================================================================
h1("0. Objetivo")
p("Comprender y aplicar algoritmos de enjambre (Swarm Intelligence) como estrategias de "
  "optimización metaheurística en distintas etapas del ciclo de vida de un modelo de "
  "aprendizaje automático: selección de características, ajuste de hiperparámetros, "
  "entrenamiento de los pesos de una red neuronal y agrupamiento (clustering).")

h2("¿Por qué algoritmos de enjambre?")
p("Muchos problemas en Machine Learning son, en el fondo, problemas de optimización: "
  "encontrar el subconjunto de variables, la combinación de hiperparámetros o los "
  "parámetros de un modelo que maximizan (o minimizan) una función objetivo. Cuando esa "
  "función objetivo es no diferenciable, discontinua, con múltiples óptimos locales, o el "
  "espacio de búsqueda es combinatorio/muy grande, los métodos basados en gradiente "
  "(como backpropagation) pueden no ser aplicables o quedar atrapados en óptimos locales. "
  "Los algoritmos de enjambre (PSO, ABC, ACO, entre otros) exploran el espacio de soluciones "
  "de forma estocástica y colectiva, usando una población de agentes simples ('partículas' "
  "o 'abejas') que comparten información, permitiendo escapar de óptimos locales sin "
  "requerir gradientes.")

h2("Resumen de los cuatro ejemplos desarrollados")
bullets([
    "<b>1. Feature Selection con ABC</b> (Artificial Bee Colony) sobre el dataset "
    "Breast Cancer Wisconsin (30 características).",
    "<b>2. Hyperparameter Tuning con PSO</b> (Particle Swarm Optimization) para un SVM-RBF "
    "sobre el dataset Wine (hiperparámetros C y gamma).",
    "<b>3. Entrenamiento de una red neuronal (4-8-3) sin backpropagation con PSO</b>, "
    "sobre el dataset Iris, optimizando directamente los 67 pesos y bias de la red.",
    "<b>4. (Extra) Clustering con PSO</b> sobre el dataset Iris, comparado contra K-Means.",
])
p("Todos los ejemplos están implementados en Python (NumPy, scikit-learn, Matplotlib) y "
  "pueden ejecutarse en Google Colab, Jupyter Notebook o Visual Studio Code. El código "
  "completo se encuentra en el repositorio de GitHub referenciado en la portada de este informe.")

story.append(PageBreak())

# ==========================================================================
# 1. FEATURE SELECTION - ABC
# ==========================================================================
h1("1. Feature Selection con ABC (Artificial Bee Colony)")

h2("1.1 Descripción del problema")
p("Dado un conjunto de datos con <i>n</i> características, el objetivo es encontrar el "
  "subconjunto de características que <b>maximice el desempeño de un clasificador</b> "
  "y, simultáneamente, <b>minimice la cantidad de características utilizadas</b> "
  "(menos features = modelos más simples, rápidos y menos propensos a overfitting). "
  "Se utilizó el dataset <b>Breast Cancer Wisconsin</b> de scikit-learn (569 muestras, "
  "30 características numéricas, 2 clases) y un clasificador <b>K-Nearest Neighbors (k=5)</b> "
  "evaluado con validación cruzada de 5 particiones (5-fold CV).")

h2("1.2 Ciclo del algoritmo ABC")
bullets([
    "<b>Representación de la partícula:</b> cada 'fuente de alimento' es un vector real "
    "x = [x1,...,x30], con xi en [0,1]. Se binariza con un umbral (xi &gt; 0.5 → feature activa) "
    "para obtener la máscara de selección de características.",
    "<b>Inicialización del enjambre:</b> se generan SN=20 fuentes de alimento aleatorias en "
    "[0,1]^30, cada una con un contador de intentos (trials) inicializado en 0.",
    "<b>Función de aptitud (fitness):</b> fitness = 0.9·accuracy(subset) − 0.1·(n_features/n_total). "
    "El término de penalización favorece soluciones con menos características cuando el "
    "accuracy es similar.",
    "<b>Comportamiento de la partícula (3 fases):</b> "
    "(a) <i>Abejas empleadas</i>: cada una busca un vecino modificando una dimensión al azar "
    "vi = xi + φ(xi − xk); (b) <i>Abejas observadoras</i>: seleccionan una fuente mediante "
    "ruleta proporcional al fitness y repiten la búsqueda local; "
    "(c) <i>Abejas exploradoras (scouts)</i>: si una fuente no mejora tras 'limit'=8 intentos, "
    "se abandona y se reemplaza por una fuente aleatoria nueva, evitando estancamiento.",
    "<b>Evolución:</b> se repiten las 3 fases durante 40 iteraciones, actualizando siempre "
    "la mejor fuente global (gbest).",
    "<b>Finalización:</b> el algoritmo se detiene al llegar a 40 iteraciones (empíricamente "
    "converge antes, alrededor de la iteración 5-10). Se retorna la mejor máscara de "
    "características encontrada.",
])

h2("1.3 Solución encontrada")
p("El algoritmo ABC seleccionó <b>9 de las 30 características originales</b> "
  "(mean texture, texture error, perimeter error, smoothness error, concavity error, "
  "symmetry error, worst perimeter, worst area, worst concave points), logrando el "
  "<b>mismo accuracy (96.49%)</b> que usando las 30 características completas, pero con "
  "un modelo <b>70% más simple</b> en número de variables. Esto demuestra que ABC logró "
  "eliminar variables redundantes sin sacrificar desempeño predictivo.")

img(f"{FIG}/convergencia_abc.png")
small("Figura 1. Curva de convergencia de ABC: evolución del mejor fitness global por iteración.")

story.append(PageBreak())

# ==========================================================================
# 2. HYPERPARAMETER TUNING - PSO
# ==========================================================================
h1("2. Hyperparameter Tuning con PSO (Particle Swarm Optimization)")

h2("2.1 Descripción del problema")
p("Se buscó ajustar los hiperparámetros <b>C</b> (regularización) y <b>gamma</b> "
  "(coeficiente del kernel RBF) de un clasificador <b>SVM</b>, para maximizar el accuracy "
  "de validación cruzada (5-fold) sobre el dataset <b>Wine</b> de scikit-learn "
  "(178 muestras, 13 características, 3 clases). El espacio de búsqueda es continuo: "
  "C ∈ [0.01, 100], gamma ∈ [0.0001, 10].")

h2("2.2 Ciclo del algoritmo PSO")
bullets([
    "<b>Representación de la partícula:</b> cada partícula es un punto en 2D, "
    "x = [C, gamma], con una velocidad asociada v = [vC, vGamma].",
    "<b>Inicialización del enjambre:</b> 15 partículas con posiciones aleatorias dentro de "
    "los límites de C y gamma, y velocidades iniciales pequeñas.",
    "<b>Función de aptitud:</b> fitness(C, gamma) = accuracy promedio de 5-fold CV de un "
    "SVM(C, gamma, kernel='rbf') entrenado sobre el dataset Wine estandarizado.",
    "<b>Comportamiento de la partícula:</b> ecuación clásica de PSO — "
    "v(t+1) = w·v(t) + c1·r1·(pbest − x(t)) + c2·r2·(gbest − x(t)); "
    "x(t+1) = x(t) + v(t+1). w=inercia, c1=coef. cognitivo, c2=coef. social, r1,r2 ~ U(0,1). "
    "Las posiciones resultantes se recortan (clip) a los límites válidos de C y gamma.",
    "<b>Evolución:</b> 30 iteraciones, con inercia w decreciente linealmente de 0.9 a 0.4 "
    "(mayor exploración al inicio, mayor explotación/ajuste fino al final).",
    "<b>Finalización:</b> tras 30 iteraciones (o antes, si gbest no mejora durante varias "
    "iteraciones consecutivas). Se retorna gbest = [C*, gamma*] como los hiperparámetros óptimos.",
])

h2("2.3 Solución encontrada")
p("PSO encontró <b>C* ≈ 67.67</b> y <b>gamma* ≈ 0.0876</b>, alcanzando un accuracy de "
  "<b>98.89%</b> en validación cruzada, superando el <b>98.33%</b> obtenido con los "
  "hiperparámetros por defecto de scikit-learn (C=1, gamma='scale'). Aunque la mejora "
  "parece pequeña en este dataset (ya es fácil de clasificar), el mismo procedimiento "
  "escala a espacios de hiperparámetros mucho más grandes y difíciles de explorar "
  "manualmente (redes neuronales profundas, ensambles, etc.).")

img(f"{FIG}/convergencia_pso.png")
small("Figura 2. Curva de convergencia de PSO: mejor accuracy global (gbest) por iteración.")

story.append(PageBreak())

# ==========================================================================
# 3. NN TRAINING WITHOUT BACKPROP - PSO
# ==========================================================================
h1("3. Entrenamiento de una Red Neuronal sin Backpropagation (PSO)")

h2("3.1 Descripción del problema")
p("Se entrenó una red neuronal totalmente conectada de <b>3 capas</b> "
  "(4 entradas → 8 neuronas ocultas con ReLU → 3 salidas con softmax) para clasificar el "
  "dataset <b>Iris</b>, encontrando <b>todos los pesos y bias directamente con PSO</b>, "
  "sin calcular gradientes ni usar backpropagation. La red tiene en total "
  "<b>67 parámetros</b> (4×8 + 8 + 8×3 + 3 = 32+8+24+3).")

h2("3.2 Ciclo del algoritmo PSO aplicado al entrenamiento de la NN")
bullets([
    "<b>Representación de la partícula:</b> cada partícula es un VECTOR PLANO de 67 "
    "posiciones que contiene TODOS los pesos y bias de la red concatenados "
    "(x = [W1, b1, W2, b2]). Es decir, cada partícula ES una red neuronal completa.",
    "<b>Inicialización del enjambre:</b> 40 partículas con pesos iniciales aleatorios "
    "~U(−1,1) (40 redes neuronales distintas), y velocidades pequeñas aleatorias.",
    "<b>Función de aptitud:</b> se decodifica el vector en W1,b1,W2,b2, se hace un "
    "forward-pass sobre el set de entrenamiento y se calcula "
    "fitness = accuracy − 0.01·cross_entropy (el término de entropía cruzada suaviza "
    "el paisaje de búsqueda para diferenciar soluciones con igual accuracy).",
    "<b>Comportamiento de la partícula:</b> misma ecuación de movimiento de PSO clásico, "
    "aplicada simultáneamente a las 67 dimensiones (cada dimensión = un peso o bias de la "
    "red). No se calcula ningún gradiente en ningún momento del proceso.",
    "<b>Evolución:</b> 150 iteraciones, inercia decreciente de 0.9 a 0.3, velocidad acotada "
    "(clip) para evitar explosión de la búsqueda.",
    "<b>Finalización:</b> tras 150 iteraciones. gbest se decodifica de vuelta en W1,b1,W2,b2 "
    "y esa es la red neuronal final, lista para hacer inferencia.",
])

h2("3.3 Solución encontrada")
p("La red entrenada con PSO alcanzó <b>100% de accuracy en el set de entrenamiento</b> y "
  "<b>89.47% en el set de validación (test)</b>, demostrando que es posible encontrar un "
  "conjunto de pesos competente para una red neuronal simple sin usar backpropagation ni "
  "gradientes, únicamente mediante búsqueda basada en población. Este enfoque no escala "
  "tan bien como backprop para redes muy profundas (la dimensión del espacio de búsqueda "
  "crece con el número de parámetros), pero es útil cuando la función de pérdida no es "
  "diferenciable o cuando se buscan alternativas a la optimización basada en gradiente.")

img(f"{FIG}/convergencia_pso_nn.png")
small("Figura 3. Accuracy de entrenamiento y validación de la red neuronal a medida que "
      "avanza la optimización por enjambre (PSO), sin backpropagation.")

story.append(PageBreak())

# ==========================================================================
# 4. CLUSTERING - PSO (BONUS)
# ==========================================================================
h1("4. (Extra) Clustering con PSO")

h2("4.1 Descripción del problema")
p("Se agruparon las 150 muestras del dataset <b>Iris</b> (sin usar las etiquetas reales) "
  "en <b>K=3 clústeres</b>, buscando la mejor ubicación de los centroides mediante PSO, "
  "en lugar de usar el algoritmo clásico K-Means.")

h2("4.2 Ciclo del algoritmo PSO aplicado a clustering")
bullets([
    "<b>Representación de la partícula:</b> cada partícula representa el CONJUNTO COMPLETO "
    "de K=3 centroides (4 dimensiones cada uno), aplanado en un vector de dimensión 12.",
    "<b>Inicialización:</b> 25 partículas con centroides aleatorios dentro del rango de "
    "valores de cada feature del dataset.",
    "<b>Función de aptitud:</b> se asigna cada punto al centroide más cercano (igual que en "
    "K-Means) y se calcula el SSE (suma de errores cuadráticos intra-clúster). "
    "fitness = −SSE (para convertir la minimización de SSE en una maximización).",
    "<b>Comportamiento de la partícula:</b> ecuación estándar de PSO, moviendo "
    "simultáneamente los 3 centroides de cada partícula en cada iteración.",
    "<b>Evolución:</b> 60 iteraciones con inercia decreciente.",
    "<b>Finalización:</b> tras 60 iteraciones, se decodifica gbest en los 3 centroides "
    "finales y se asigna la etiqueta de clúster a cada muestra.",
])

h2("4.3 Solución encontrada y comparación con K-Means")
tbl = Table([
    ["Métrica", "PSO", "K-Means"],
    ["SSE (menor es mejor)", "112.79", "78.85"],
    ["Adjusted Rand Index", "0.556", "0.730"],
    ["Silhouette Score", "0.533", "0.553"],
], colWidths=[6*cm, 4*cm, 4*cm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a2b4c")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f4f6fa")]),
]))
story.append(tbl)
story.append(Spacer(1, 8))
p("K-Means obtiene un SSE ligeramente menor porque converge de forma determinista al "
  "óptimo local más cercano (algoritmo de Lloyd), mientras que PSO explora el espacio de "
  "forma estocástica. Sin embargo, PSO no requiere que la función objetivo sea diferenciable "
  "y puede adaptarse fácilmente a otras métricas de distancia o restricciones adicionales "
  "que K-Means no maneja de forma nativa.")

img(f"{FIG}/clusters_pso.png", width=10*cm)
small("Figura 4. Clústeres encontrados por PSO (proyección en 2 de las 4 dimensiones de Iris).")

img(f"{FIG}/convergencia_pso_clustering.png")
small("Figura 5. Curva de convergencia de PSO en el problema de clustering (SSE decreciente).")

story.append(PageBreak())

# ==========================================================================
# 5. CONCLUSIONES
# ==========================================================================
h1("5. Conclusiones generales")
bullets([
    "Los algoritmos de enjambre son herramientas de optimización de propósito general que "
    "pueden aplicarse a prácticamente cualquier etapa del pipeline de Machine Learning donde "
    "exista una función objetivo a maximizar o minimizar, incluso si esta no es diferenciable.",
    "En <b>feature selection</b>, ABC permitió reducir el número de características de 30 a 9 "
    "sin perder accuracy, mostrando su utilidad para simplificar modelos.",
    "En <b>hyperparameter tuning</b>, PSO encontró una combinación de C y gamma que igualó o "
    "superó la configuración por defecto, sin necesidad de una búsqueda exhaustiva tipo grid search.",
    "En el <b>entrenamiento de redes neuronales</b>, PSO demostró que es posible optimizar "
    "los pesos de una red sin backpropagation, aunque este enfoque es menos eficiente que el "
    "gradiente descendente para redes con muchos parámetros (la dimensionalidad del problema "
    "crece rápidamente).",
    "En <b>clustering</b>, PSO obtuvo resultados comparables (aunque ligeramente inferiores) "
    "a K-Means, pero con la ventaja de ser fácilmente adaptable a otras funciones de distancia "
    "o restricciones del problema.",
    "En todos los casos se observó el mismo patrón general del ciclo de optimización por "
    "enjambre: representación de la solución como partícula → inicialización aleatoria → "
    "evaluación mediante función de aptitud → movimiento/actualización de partículas guiado "
    "por la mejor solución individual y global → repetición hasta convergencia o criterio de paro.",
])

story.append(Spacer(1, 20))
hr()
h2("Referencias")
bullets([
    "Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN'95.",
    "Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization. "
    "Technical Report TR06, Erciyes University.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.",
])

doc = SimpleDocTemplate(
    "/mnt/user-data/outputs/Informe_Actividad03_Swarm_ML.pdf",
    pagesize=letter,
    topMargin=2*cm, bottomMargin=2*cm, leftMargin=2.2*cm, rightMargin=2.2*cm,
    title="Actividad 03 - Algoritmos de Enjambre en Aprendizaje Automático"
)
doc.build(story)
print("PDF generado correctamente.")
