import random
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

# Manhattan Distance
def L1(v1, v2):
    if (len(v1) != len(v2)):
        return -1
    return sum([abs(v1[i] - v2[i]) for i in range(len(v1))])


# kmeans with L1 distance.
# rows refers to the NxM feature vectors
def kcluster(rows, distance=L1, k=10):  # Cited from Programming Collective Intelligence
    # Determine the minimum and maximum values for each point
    ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows])) for i in range(len(rows[0]))]

    # Create k randomly placed centroids
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0] for i in range(len(rows[0]))] for j in
                range(k)]

    lastmatches = None
    for t in range(100):
        print ('Iteration %d' % t)
        bestmatches = [[] for i in range(k)]
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row = rows[j]
            bestmatch = 0
            for i in range(k):
                print("EDW MESA")
                d = distance(clusters[i], row)
                if d < distance(clusters[bestmatch], row):
                    bestmatch = i
            bestmatches[bestmatch].append(j)
        ## If the results are the same as last time, this is complete
        if bestmatches == lastmatches:
            break
        lastmatches = bestmatches

        # Move the centroids to the average of their members
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(bestmatches[i]) > 0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                clusters[i] = avgs
    return bestmatches

kcluster(trImages)