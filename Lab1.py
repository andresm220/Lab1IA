#Lab 1 
"""Andrés Mazariegos
June Herrera
Dilary Cruz"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

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

# Ejecutar k-means de Scikit-Learn (Comparación)
kmeans_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_iris_sklearn = kmeans_sklearn.fit_predict(X_iris)

# Calcular similitud (ARI: 1.0 es idéntico, 0.0 es aleatorio)
ari_iris = adjusted_rand_score(labels_iris, labels_iris_sklearn)

# Guardar clusters en el DataFrame
df_iris["cluster"] = labels_iris
df_iris["cluster_sklearn"] = labels_iris_sklearn

# Mostrar tabla de comparación
print("\n=== IRIS: Comparación ===")
print(f"Similitud (ARI) entre Mi K-Means y Sklearn: {ari_iris:.4f}")
print("Crosstab (Mi Cluster vs Target):")
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

# Sklearn Comparison
kmeans_sklearn_peng = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_peng_sklearn = kmeans_sklearn_peng.fit_predict(X_peng)
ari_peng = adjusted_rand_score(labels_peng, labels_peng_sklearn)

print("\n==============================")
print("PENGUINS: info")
print("==============================")
print(f"Similitud (ARI) entre Mi K-Means y Sklearn: {ari_peng:.4f}")
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

# Ejecutar k-means desde cero (k=6, calidad va de 3 a 8 usualmente)
labels_wine, centroids_wine = kmeans_basico(X_wine, k=6, max_iter=200)

# Sklearn Comparison
kmeans_sklearn_wine = KMeans(n_clusters=6, random_state=42, n_init=10)
labels_wine_sklearn = kmeans_sklearn_wine.fit_predict(X_wine)
ari_wine = adjusted_rand_score(labels_wine, labels_wine_sklearn)

df_wine["cluster"] = labels_wine

print("\n=== WINE QUALITY: Comparación ===")
print(f"Similitud (ARI) entre Mi K-Means y Sklearn: {ari_wine:.4f}")
print("Crosstab (Cluster vs Quality):")
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

# 1. Cargar los datos
# Asegúrate de que el archivo esté en la misma carpeta o usa la ruta completa
df = pd.read_csv('countries_binary.csv')
X = df.drop('Country', axis=1)
countries = df['Country'].values

# --- Ejercicio 3: Agrupamiento Jerárquico (Generación de Imagen) ---
methods = ['single', 'complete', 'average', 'ward']
metrics = ['euclidean', 'hamming']

# Configuración clave para evitar superposición:
fig, axes = plt.subplots(4, 2, figsize=(15, 24), constrained_layout=True)

plot_idx = 0
for method in methods:
    for metric in metrics:
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        # Caso especial: Ward solo funciona con distancia Euclideana
        if method == 'ward' and metric != 'euclidean':
            ax.text(0.5, 0.5, 'Ward requiere Euclideana', ha='center', va='center', fontsize=12)
            ax.set_title(f"Método: {method}, Métrica: {metric}", fontsize=14, fontweight='bold')
            ax.axis('off') # Ocultar el recuadro vacío
        else:
            Z = linkage(X, method=method, metric=metric)
            dendrogram(Z, labels=countries, ax=ax, leaf_rotation=90)
            
            # Ajustes visuales
            ax.set_title(f"Método: {method}, Métrica: {metric}", fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
        
        plot_idx += 1

# Guardar la imagen en lugar de lanzarla con plt.show()
plt.savefig('dendrogramas_finales.png')
print("Imagen 'dendrogramas_finales.png' guardada exitosamente.")

# --- Ejercicio 4: Comparación con K-means ---
k = 3
kmeans_labels, _ = kmeans_basico(X, k=k, max_iter=200)

# Comparar con el método jerárquico Ward/Euclidean
Z_ward = linkage(X, method='ward', metric='euclidean')
hierarchical_labels = fcluster(Z_ward, k, criterion='maxclust')

# ARI Countries
ari_countries = adjusted_rand_score(kmeans_labels, hierarchical_labels)
print(f"\nSimilitud (ARI) Países: KMeans vs Jerárquico (Ward): {ari_countries:.4f}")
print("¿Son iguales? Si el ARI es < 1, no son idénticos. Si es bajo, son muy distintos.")

comparacion = pd.DataFrame({
    'Pais': countries,
    'Cluster_KMeans': kmeans_labels,
    'Cluster_Jerarquico_Ward': hierarchical_labels
})

# Guardar resultados en CSV
comparacion.to_csv('clustering_comparison_v4.csv', index=False)
print("Comparación guardada en 'clustering_comparison_v4.csv'")
print(comparacion)

# ======================================================
# Ejercicio 5: Datos Sintéticos (Moons/Círculos)
# ======================================================
print("\n=== Ejercicio 5: Datos Sintéticos ===")
from sklearn.datasets import make_moons, make_circles

# Generar make_moons
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

# Comparación K-Means vs Jerárquico
k = 2

# K-Means
labels_kmeans_moons, _ = kmeans_basico(X_moons, k=k)

# Jerárquico (Single Linkage)
# 'single' linkage es ideal para formas no convexas como lunas
Z_moons = linkage(X_moons, method='single', metric='euclidean')
labels_hier_moons = fcluster(Z_moons, k, criterion='maxclust')

# Visualización
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot K-means
axs[0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_kmeans_moons, cmap='viridis')
axs[0].set_title("K-Means (Moons)")
axs[0].set_xlabel("Falla debido a la no convexidad")

# Plot Jerárquico
axs[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_hier_moons, cmap='viridis')
axs[1].set_title("Jerárquico - Single Linkage (Moons)")
axs[1].set_xlabel("Funciona bien (encadenamiento)")

plt.savefig('moons_comparison.png')
print("Imagen 'moons_comparison.png' guardada.")
plt.close()

# ======================================================
# Ejercicio 6: Cuantización de Colores
# ======================================================
print("\n=== Ejercicio 6: Cuantización de Colores ===")
from skimage import data
from skimage.transform import resize

def cuantizar_imagen_kmeans(imagen, k_colores):
    # imagen: (H, W, 3)
    h, w, d = imagen.shape
    
    # Aplanar la imagen a (N, 3)
    pixels = imagen.reshape((-1, 3))
    
    n_sample = 1000  # Muestra para encontrar centroides
    if len(pixels) > n_sample:
        indices = np.random.choice(len(pixels), n_sample, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
        
    # Entrenar K-means con la muestra
    # Usamos nuestra función kmeans_basico
    _, centroides = kmeans_basico(pixels_sample, k=k_colores, max_iter=50)
    

    labels_all = []
 
    
    
    # Implementación vectorizada por bloques para no saturar memoria
    chunk_size = 10000
    for i in range(0, len(pixels), chunk_size):
        chunk = pixels[i:i+chunk_size]
        # Distancias: ||chunk[:, None] - centroides[None, :]||
        dists = np.linalg.norm(chunk[:, np.newaxis] - centroides, axis=2)
        chunk_labels = np.argmin(dists, axis=1)
        labels_all.extend(chunk_labels)
        
    labels_all = np.array(labels_all)
    
    # Reconstruir imagen cuantizada
    # Cada pixel toma el color de su centroide
    quantized_pixels = centroides[labels_all]
    quantized_image = quantized_pixels.reshape((h, w, d))
    
    # Mapa de clases (labels) reformado
    labels_image = labels_all.reshape((h, w))
    
    return labels_image, quantized_image

# Cargar 3 imágenes de ejemplo de skimage
# Nos aseguramos de que sean RGB y valores 0-1 o 0-255 consistente
images = [
    ("Chelsea (Gato)", data.chelsea()),       # RGB
    ("Astronauta", data.astronaut()),         # RGB
    ("Café", data.coffee())                   # RGB
]

k_colors = 4  # Número de colores para la cuantización

# Crear figura para mostrar resultados
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for i, (name, img) in enumerate(images):
    # Normalizar a 0-1 si es entero
    if img.dtype == 'uint8':
        img = img / 255.0

    print(f"Procesando imagen: {name}...")
    labels_map, quant_img = cuantizar_imagen_kmeans(img, k_colors)
    
    # Original
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Original: {name}")
    axes[i, 0].axis('off')
    
    # Mapa de clases
    axes[i, 1].imshow(labels_map, cmap='viridis')
    axes[i, 1].set_title("Mapa de Clases (Clusters)")
    axes[i, 1].axis('off')
    
    # Cuantizada
    axes[i, 2].imshow(quant_img)
    axes[i, 2].set_title(f"Cuantizada (k={k_colors})")
    axes[i, 2].axis('off')

plt.savefig('color_quantization.png')
print("Imagen 'color_quantization.png' guardada con 3 ejemplos.")