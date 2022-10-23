from math import*
import random
import time
import tensorflow as tf
from tensorflow import keras

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


def distance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquares = 0
    for i in range(1, (instance1).size):
        sumOfSquares += (instance1[i] - instance2[i])**2
    return sqrt(sumOfSquares)

from scipy import spatial
def cdistance(x, y):
    from math import sqrt
    def dot_product(x, y):
        return sum(a * b for a, b in zip(x, y))
    def magnitude(vector):
        return sqrt(dot_product(vector, vector))
    def similarity(x, y):
        return (1-(dot_product(x, y) / (magnitude(x) * magnitude(y) + .00000000001)))

import numpy as np
def meanInstance(name, instanceList):
    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = (instanceList[0]).size
    means = [name] + [0] * (numAttributes-1)
    for instance in instanceList:
        for i in range(1, numAttributes):
            means[i] += instance[i]
    for i in range(1, numAttributes):
        means[i] /= float(numInstances)
    return tuple(means)

def assign(instance, centroids):
    minDistance = distance(instance, centroids[0])
    minDistanceIndex = 0
    for i in range(1, len(centroids)):
        d = distance(instance, centroids[i])
        if (d < minDistance):
            minDistance = d
            minDistanceIndex = i
    return minDistanceIndex

def createEmptyListOfLists(numSubLists):
    myList = []
    for i in range(numSubLists):
        myList.append([])
    return myList

def assignAll(instances, centroids):
    clusters = createEmptyListOfLists(len(centroids))
    for instance in instances:
        clusterIndex = assign(instance, centroids)
        clusters[clusterIndex].append(instance)
    return clusters

def computeCentroids(clusters):
    centroids = []
    for i in range(len(clusters)):
        name = "\nCentroid " + str(i+1)
        centroid = meanInstance(name, clusters[i])
        centroids.append(centroid)
    return centroids

def kmeans(instances, k, initCentroids=None):
    result = {}
    if (initCentroids == None or len(initCentroids) < k):
        # randomly select k initial centroids
        random.seed(time.time())
        centroids = random.sample(instances, k)
    else:
        centroids = initCentroids
    prevCentroids = []
    iteration = 0
    while (centroids != prevCentroids):
        iteration += 1
        clusters = assignAll(instances, centroids)
        prevCentroids = centroids
        centroids = computeCentroids(clusters)
        withinss = computeWithinss(clusters, centroids)
    result["clusters"] = clusters
    result["centroids"] = centroids
    result["withinss"] = withinss
    result["sse"]=withinss
    return result

def computeWithinss(clusters, centroids):
    result = 0
    try:
        for i in range(len(centroids)):
            centroid = centroids[i]
            cluster = clusters[i]
            for instance in cluster:
                result += cdistance(centroid, instance)
    except:
        pass
    return result

# Repeats k-means clustering n times, and returns the clustering
# with the smallest withinss
def repeatedKMeans(instances, k, n):
    bestClustering = {}
    bestClustering["withinss"] = float("inf")
    for i in range(1, n+1):
        print ("k-means trial %d," % i)
        trialClustering = kmeans(instances, k)
        print ("withinss: %.1f" % trialClustering["withinss"])
        if trialClustering["withinss"] < bestClustering["withinss"]:
            bestClustering = trialClustering
            minWithinssTrial = i
    print ("Trial with minimum withinss:", minWithinssTrial)
    return bestClustering

for i in range(0,len(trImages)):
    clustering = kmeans(set(trImages[i]), 10)
