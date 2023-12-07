import os
import datetime
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds 
import seaborn as sns
import scipy.spatial
import sklearn.manifold
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from PIL import Image

k_values = []
Accuracy = []

network = tf.keras.applications.ResNet50(weights = "imagenet")                                        #load the resnet-50 pretrained weights

feature_extractor = tf.keras.Model(                                                                   #sperate the model and get the feature extractor part(without FC)
    inputs = network.input,
    outputs = network.layers[-2].output
)

def preprocess_and_extract_embedding(example):                                                        #create a function to get the feature embeddings
    image = example['image']
    resized_image = tf.image.resize(image, (224, 224))
    preprocessed_image = tf.keras.applications.resnet50.preprocess_input(resized_image)
    embedding = feature_extractor(preprocessed_image[np.newaxis, ...])
    return tf.squeeze(embedding)

dataset_name = 'oxford_iiit_pet'                                                                       
dataset, info = tfds.load(name=dataset_name, split='train', with_info=True)                           #load the oxford_iiit_pet dateset and split it into given train,test ratio 
test_dataset, test_info = tfds.load(name=dataset_name, split='test', with_info=True)

train_embeddings = [preprocess_and_extract_embedding(example) for example in dataset]                 #get the train embeddings using the model
train_labels = [example['label'] for example in dataset]                                              #get corresponding train labels

test_embeddings = [preprocess_and_extract_embedding(example) for example in test_dataset]             #get the test embeddings using the model
test_labels = [example['label'] for example in test_dataset]                                          #get corresponding train labels

f_train = tf.convert_to_tensor(train_embeddings)
y_train = tf.convert_to_tensor(train_labels)
vals = sklearn.manifold.TSNE(2).fit_transform(f_train)                                                #reduce the dimension of train embeddings using TSNE

plt.figure(figsize=(8, 8))                                                                            #plot the train embeddings and the labels 
sns.scatterplot(x=vals[:, 0], y=vals[:, 1], hue=y_train, palette='pastel')                          
plt.show()



for i in range(1,int(math.sqrt(len(train_embeddings)))):                                              
    knn_classifier = KNeighborsClassifier(n_neighbors=i)                                              
    knn_classifier.fit(train_embeddings, train_labels)                                               #train the knn classifier using the train data

    predicted_labels = knn_classifier.predict(test_embeddings)                                       #take the predict labels of the test data using the knn classifier 

    accuracy = accuracy_score(test_labels, predicted_labels)                                         #calculate the accuracy 
    
    k_values.append(i)
    Accuracy.append(accuracy)


Max_acc = max(Accuracy)                                                                              #get the maximum of the accuracy
K = k_values[Accuracy.index(Max_acc)]                                                                #get the k value relavant to the maximum accuracy
print(f'Accuracy on the test set: {Max_acc * 100:.2f}%')                                             #display the maximum accuracy

plt.plot(k_values, Accuracy, '.', color = "blue")                                                    #plot k values vs accuracy graph
plt.plot(K, Max_acc,'.', color = "red")
plt.xlabel("k values")
plt.ylabel("test acuuracy")
plt.show()