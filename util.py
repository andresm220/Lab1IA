import numpy as np
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