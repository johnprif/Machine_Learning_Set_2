import numpy as np
from tensorflow import keras
from sklearn import metrics
from sklearn.metrics import f1_score

#Downloading mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(trImages, trLabels), (tImages, tLabels) = fashion_mnist.load_data()

# Conversion to float and normalize R1
trImages = trImages.astype('float32')
tImages = tImages.astype('float32')
trImages=trImages/255
tImages=tImages/255
# Reshaping input data
trImages = trImages.reshape(trImages.shape[0], trImages.shape[1] * trImages.shape[2]) #one dimensional
tImages = tImages.reshape(tImages.shape[0], tImages.shape[1] * tImages.shape[2])


class KMeans:
    def __init__(self, k):
        self.k = k

    def train(self, X, MAXITER=100, TOL=1e-3):
        centroids = np.random.rand(self.k, X.shape[1])
        centroidsold = centroids.copy()
        for iter_ in range(MAXITER):
            dist = np.linalg.norm(X - centroids[0, :], axis=1).reshape(-1, 1) #linalg is Euclidean distance
            for class_ in range(1, self.k):
                dist = np.append(dist, np.linalg.norm(X - centroids[class_, :], axis=1).reshape(-1, 1), axis=1)
            classes = np.argmin(dist, axis=1)
            # update position
            for class_ in set(classes):
                centroids[class_, :] = np.mean(X[classes == class_, :], axis=0)
            if np.linalg.norm(centroids - centroidsold) < TOL:
                break
                print('Centroid converged')
        self.centroids = centroids

    def predict(self, X):
        dist = np.linalg.norm(X - self.centroids[0, :], axis=1).reshape(-1, 1)
        for class_ in range(1, self.k):
            dist = np.append(dist, np.linalg.norm(X - self.centroids[class_, :], axis=1).reshape(-1, 1), axis=1)
        classes = np.argmin(dist, axis=1)
        return classes

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

kmeans = KMeans(10)
kmeans.train(trImages)
classes = kmeans.predict(trImages)
classes
print("PURITY: ")
print(purity_score(classes,trLabels)) #trLabels
print("F-MEASURE: ")
print(f1_score(trLabels, classes,  average='weighted' )) #trLabels


