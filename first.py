from tensorflow import keras
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
import numpy as np
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


total_clusters = len(np.unique(tLabels))
print(total_clusters)
# Initialize the K-Means model
kmeans = MiniBatchKMeans(n_clusters = total_clusters)
# Fitting the model to training set
kmeans.fit(trImages)

#kmeans.labels_

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def retrieve_info(cluster_labels,y_train):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num
    return reference_labels

reference_labels = retrieve_info(kmeans.labels_,trLabels)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
  number_labels[i] = reference_labels[kmeans.labels_[i]]

print("ACCURACY: ")
print(accuracy_score(number_labels,trLabels))
print("F-MEASURE: ")
print(f1_score(number_labels,trLabels,average='weighted' ))
print("PURITY: ")
print(purity_score(number_labels,trLabels))
