#Lab 1 
"""Andrés Mazariegos
June Herrera
Dilary Cruz"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Ejercicio1 
def kmeans_basico(X, k, max_iter=200):
    # X: matriz n x d
    X = np.array(X)
    n, d = X.shape

    # 1. Elegimos k puntos aleatorios como centroides iniciales
    indices = np.random.choice(n, k, replace=False)
    centroides = X[indices]

    for _ in range(max_iter):

        # 2. Asignar cada punto al centroide más cercano
        labels = []
        for punto in X:
            distancias = []
            for c in centroides:
                distancia = np.linalg.norm(punto - c)
                distancias.append(distancia)
            labels.append(np.argmin(distancias))

        labels = np.array(labels)

        # 3. Recalcular centroides
        nuevos_centroides = []
        for i in range(k):
            puntos_del_grupo = X[labels == i]
            nuevo_centroide = puntos_del_grupo.mean(axis=0)
            nuevos_centroides.append(nuevo_centroide)

        nuevos_centroides = np.array(nuevos_centroides)

        # 4. Si no cambian, terminamos
        if np.allclose(centroides, nuevos_centroides):
            break

        centroides = nuevos_centroides

    return labels, centroides

#Ejercicio 2

# ======================================================
# 2a) DATASET IRIS
# ======================================================

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Cargar dataset Iris
iris = load_iris()

# Convertir a DataFrame
df_iris = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Agregar la clase real (target)
df_iris["target"] = iris.target

# Separar X (variables) y target
X_iris = df_iris.drop(columns=["target"]).values

# Ejecutar k-means desde cero (k=3 porque hay 3 especies)
labels_iris, centroids_iris = kmeans_basico(X_iris, k=3, max_iter=200)

# Guardar clusters en el DataFrame
df_iris["cluster"] = labels_iris

# Mostrar tabla de comparación
print("\n=== IRIS: Cluster vs Target ===")
print(pd.crosstab(df_iris["cluster"], df_iris["target"]))

# Visualización (2D)
plt.figure()
plt.scatter(X_iris[:, 2], X_iris[:, 3], c=labels_iris)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("Iris - KMeans desde cero")
plt.savefig("iris_kmeans.png", dpi=200, bbox_inches="tight")
plt.close()


# ======================================================
# 2b) DATASET PENGUINS (seaborn)
# ======================================================

import seaborn as sns

penguins = sns.load_dataset("penguins")

# Solo columnas numéricas y quitamos filas con NaN
df_peng = penguins.select_dtypes(include="number").dropna()

X_peng = df_peng.values

# K = 3 porque hay 3 especies 
labels_peng, centroids_peng = kmeans_basico(X_peng, k=3, max_iter=200)

print("\n==============================")
print("PENGUINS: info")
print("==============================")
print("X_peng shape:", X_peng.shape)
print("labels shape:", labels_peng.shape)
print("centroids shape:", centroids_peng.shape)
print("Columnas numéricas usadas:", list(df_peng.columns))

# Visualización en 2D (primeras dos columnas numéricas)
plt.figure()
plt.scatter(X_peng[:, 0], X_peng[:, 1], c=labels_peng)
plt.xlabel(df_peng.columns[0])
plt.ylabel(df_peng.columns[1])
plt.title("Penguins - KMeans desde cero (numéricas)")
plt.savefig( "penguins_kmeans.png", dpi=200, bbox_inches="tight")
plt.close()

# ======================================================
# 2c) WINE QUALITY (UCI) con ucimlrepo (id=186)
# ======================================================

from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)

X_wq = wine_quality.data.features
y_wq = wine_quality.data.targets

df_wine = X_wq.copy()

# puede venir como DataFrame
if isinstance(y_wq, pd.DataFrame):
    df_wine["quality"] = y_wq.iloc[:, 0].values
else:
    df_wine["quality"] = y_wq

# Filtrar solo red
# ======================================================
# 2c) WINE QUALITY (UCI) con ucimlrepo (sin columna 'type')
# ======================================================

from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)

X_wq = wine_quality.data.features      # DataFrame con 11 columnas
y_wq = wine_quality.data.targets       # DataFrame/Serie con quality

# Aseguramos que y sea una Serie 1D
if isinstance(y_wq, pd.DataFrame):
    y_series = y_wq.iloc[:, 0]         # primera columna (quality)
else:
    y_series = y_wq

# Dataset completo (features + quality)
df_wine = X_wq.copy()
df_wine["quality"] = y_series.values

# Matriz X para k-means: solo features (sin quality)
X_wine = df_wine.drop(columns=["quality"]).values

# Ejecutar k-means desde cero (k=3)
labels_wine, centroids_wine = kmeans_basico(X_wine, k=3, max_iter=200)

df_wine["cluster"] = labels_wine

print("\n=== WINE QUALITY: Cluster vs Quality ===")
print(pd.crosstab(df_wine["cluster"], df_wine["quality"]))

# Guardar gráfica (primeras 2 features)
plt.figure()
plt.scatter(X_wine[:, 0], X_wine[:, 1], c=labels_wine)
plt.xlabel(df_wine.drop(columns=["quality", "cluster"]).columns[0])
plt.ylabel(df_wine.drop(columns=["quality", "cluster"]).columns[1])
plt.title("Wine Quality - KMeans desde cero")
plt.savefig("wine_kmeans.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nListo ✅ Se guardaron las imágenes: iris_kmeans.png, penguins_kmeans.png, wine_red_kmeans.png")