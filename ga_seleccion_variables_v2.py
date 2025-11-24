import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random

# ============================================================
# 1. Cargar datos reales de deserción
# ============================================================

# Nombre del archivo CSV
DATA_FILE = "datos_desercion.csv"

df = pd.read_csv(DATA_FILE)

# La variable objetivo: 1 = deserta, 0 = no deserta
target_col = "DESERTA"

# Comprobamos que existe la columna objetivo
if target_col not in df.columns:
    raise ValueError(f"No se encontró la columna objetivo '{target_col}' en el CSV.")

# ============================================================
# 2. Definir variables candidatas (las que el GA va a seleccionar)
# ============================================================

candidate_features = [
    "EDAD",
    "PROM_INGRESO",
    "PUNTAJE_PAA",
    "GENERO",
    "CASADO",
    "DIVORCIADO",
    "UNION_LIBRE",
    "TRABAJA",
    "DISCAPACIDAD"
]

# Verificamos que todas las variables existan en el DataFrame
for col in candidate_features:
    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no se encontró en el CSV.")

X = df[candidate_features]
y = df[target_col]

print("Variables candidatas que usará el algoritmo genético:")
print(candidate_features)

# ============================================================
# 3. Modelo base: Regresión logística + estandarización
# ============================================================

# Todas las variables son numéricas (0/1 o continuas),
# Por lo que sólo usamos StandardScaler y LogisticRegression.

base_model = LogisticRegression(max_iter=1000)

def evaluate_individual(individual):
    """
    individual: lista de 0/1 indicando qué variables se usan.
    Regresa el fitness como el AUC promedio (o accuracy si AUC falla).
    """
    # Si todos los genes son 0, penalizamos
    if sum(individual) == 0:
        return 0.0

    # Seleccionamos sólo las columnas activas (con 1)
    selected_cols = [f for bit, f in zip(individual, candidate_features) if bit == 1]
    X_sel = X[selected_cols]

    # Pipeline: escalado + regresión logística
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", base_model)
    ])

    # Intentamos usar AUC; si hay error, usamos accuracy
    if len(np.unique(y)) == 2:
        try:
            scores = cross_val_score(pipe, X_sel, y, cv=3, scoring="roc_auc")
        except Exception as e:
            print("Error al calcular AUC, usando accuracy en su lugar:", e)
            scores = cross_val_score(pipe, X_sel, y, cv=3, scoring="accuracy")
    else:
        scores = cross_val_score(pipe, X_sel, y, cv=3, scoring="accuracy")

    return scores.mean()

# ============================================================
# 4. Definir Algoritmo Genético
# ============================================================

POP_SIZE = 20       # Tamaño de la población
N_GENERATIONS = 15  # Número de generaciones
P_CROSS = 0.8       # Probabilidad de cruce
P_MUT = 0.1         # Probabilidad de mutación por bit

n_features = len(candidate_features)

def create_individual():
    """
    Crea un cromosoma binario aleatorio (lista de 0/1).
    Garantiza que al menos una variable esté activa.
    """
    individual = [random.randint(0, 1) for _ in range(n_features)]
    if sum(individual) == 0:
        individual[random.randint(0, n_features - 1)] = 1
    return individual

def mutate(individual):
    """
    Mutación bit a bit: con probabilidad P_MUT, cambia 0->1 o 1->0.
    """
    for i in range(n_features):
        if random.random() < P_MUT:
            individual[i] = 1 - individual[i]
    # evitar cromosoma con todas las variables apagadas
    if sum(individual) == 0:
        individual[random.randint(0, n_features - 1)] = 1
    return individual

def crossover(parent1, parent2):
    """
    Cruce de un punto: se intercambia la cola de los padres.
    """
    if random.random() < P_CROSS:
        point = random.randint(1, n_features - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        # Sin cruce: copia exacta
        return parent1[:], parent2[:]

def select_parent(population, fitnesses):
    """
    Selección por torneo: se eligen dos individuos al azar
    y gana el que tiene mayor fitness.
    """
    i, j = random.sample(range(len(population)), 2)
    if fitnesses[i] > fitnesses[j]:
        return population[i]
    else:
        return population[j]

# ============================================================
# 5. Bucle principal del SGA
# ============================================================

population = [create_individual() for _ in range(POP_SIZE)]
best_fitness_per_gen = []
best_individual = None
best_fitness = -1

for gen in range(N_GENERATIONS):
    print(f"\nGeneración {gen + 1}/{N_GENERATIONS}")

    # Evaluar cada individuo
    fitnesses = []
    for ind in population:
        fit = evaluate_individual(ind)
        fitnesses.append(fit)

    # Mejor de la generación
    gen_best = max(fitnesses)
    gen_best_ind = population[np.argmax(fitnesses)]

    # Actualizar mejor global
    if gen_best > best_fitness:
        best_fitness = gen_best
        best_individual = gen_best_ind[:]

    best_fitness_per_gen.append(gen_best)

    print("----------------------------------------------------------")
    print(f"Mejor fitness de la generación: {gen_best:.4f}")
    print(f"Mejor fitness global hasta ahora: {best_fitness:.4f}")

    # ============================================================
    # ELITISMO: el mejor individuo pasa directo a la siguiente generación
    # ============================================================
    new_population = [gen_best_ind[:]]   # <-- Aquí agregamos un único individuo élite

    # Crear el resto de la población mediante selección + cruce + mutación
    while len(new_population) < POP_SIZE:
        p1 = select_parent(population, fitnesses)
        p2 = select_parent(population, fitnesses)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        new_population.extend([c1, c2])

    # Ajustar tamaño si se pasó por tener pares
    population = new_population[:POP_SIZE]


# ============================================================
# 6. Resultados finales
# ============================================================

print("\n===== RESULTADOS FINALES =====")
print(f"\nMejor fitness encontrado: {best_fitness:.4f}")
print("Mejor individuo (máscara de variables 0/1):")
print(best_individual)

selected_features = [f for bit, f in zip(best_individual, candidate_features) if bit == 1]
print("\nVariables seleccionadas por el mejor individuo:")
print(selected_features)
print("----------------------------------------------------------")

# ============================================================
# 7. Gráficas para el informe
# ============================================================

# 7.1 Evolución del fitness por generación
plt.figure(figsize=(8, 4))
plt.plot(range(1, N_GENERATIONS + 1), best_fitness_per_gen, marker="o")
plt.xlabel("Generación")
plt.ylabel("Mejor fitness (AUC/Accuracy)")
plt.title("Evolución del mejor fitness por generación (SGA selección de variables)")
plt.grid(True)
plt.tight_layout()
plt.savefig("ga_fitness_por_generacion.png", dpi=300)
plt.close()

# 7.2 Variables seleccionadas (en el mejor individuo)
frecuencias = np.array(best_individual)  # 1 = seleccionada, 0 = no seleccionada
plt.figure(figsize=(10, 4))
plt.bar(candidate_features, frecuencias)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Seleccionada en el mejor individuo (1 = sí, 0 = no)")
plt.title("\nVariables seleccionadas por el mejor individuo del SGA")
plt.tight_layout()
plt.savefig("ga_variables_seleccionadas.png", dpi=300)
plt.close()
