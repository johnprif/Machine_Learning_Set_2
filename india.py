import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics

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


class K_Means:
    def __init__(self, k=10, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

def evaluate(y_true, y_clusters):
    cm = np.zeros((np.max(y_true)+1, np.max(y_clusters)+1))
    for i in range(y_true.size):
        cm[y_true[i], y_clusters[i]]+=1
    purity = 0.
    for cluster in range(cm.shape[1]):  # clusters are along columns
        purity += np.max(cm[:, cluster])
    return purity/y_true.size

model = K_Means()
labels = model.fit(trImages)
predictions = model.predict(trImages)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def retrieve_info(cluster_labels,y_train):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(labels.labels))):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num
    return reference_labels

reference_labels = retrieve_info(labels.labels,trLabels)
number_labels = np.random.rand(len(labels.labels))
for i in range(len(labels.labels)):
  number_labels[i] = reference_labels[labels.labels[i]]

print("ACCURACY: ")
print(accuracy_score(number_labels,trLabels))
print("F-MEASURE: ")
print(f1_score(number_labels,trLabels,average='weighted' ))
print("PURITY: ")
print(purity_score(number_labels,trLabels))
print("what is this")
print(number_labels)
print(trLabels)
