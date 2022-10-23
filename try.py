#https://towardsdatascience.com/building-your-first-neural-network-in-tensorflow-2-tensorflow-for-hackers-part-i-e1e2f1dfe7a0
#https://becominghuman.ai/simple-neural-network-on-mnist-handwritten-digit-dataset-61e47702ed25
#https://keras.io/guides/sequential_model/
#https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
#https://stackoverflow.com/questions/48987959/classification-metrics-cant-handle-a-mix-of-continuous-multioutput-and-multi-la
#https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow import keras
import numpy as np
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

#neurons
K1=500
K2=200

(trImages, trLabels), (tImages, tLabels) = keras.datasets.fashion_mnist.load_data()

#normalize
trImages = trImages.astype('float32')
tImages = tImages.astype('float32')
trImages=trImages/255
tImages=tImages/255

# Convert tLabels into one-hot format
temp = []
for i in range(len(tLabels)):
    temp.append(to_categorical(tLabels[i], num_classes=10))
tLabels = np.array(temp)

# Convert trLabels into one-hot format
temp = []
for i in range(len(trLabels)):
    temp.append(to_categorical(trLabels[i], num_classes=10))
trLabels = np.array(temp)

model = keras.Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=K1, activation='relu'))#or activation='sigmoid'
model.add(Dense(units=K2, activation='relu'))#or activation='sigmoid'
model.add(Dense(units=10, activation='softmax'))

model.summary()

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.fit(trImages, trLabels, epochs=10,
          validation_data=(tImages,tLabels))

predictions = model.predict(tImages)

tLabels=np.argmax(tLabels, axis=1)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(tLabels, predictions)
print('Accuracy: %f' % (accuracy*100))

# precision tp / (tp + fp)
precision = precision_score(tLabels, predictions, average='weighted')
print('Precision: %f' % (precision*100))

# recall: tp / (tp + fn)
recall = recall_score(tLabels, predictions, average='weighted')
print('Recall: %f' % (recall*100))

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(tLabels, predictions, average='weighted')
print('F1 score: %f' % (f1*100))