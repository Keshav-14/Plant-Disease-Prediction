import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib

train_path = 'Dataset/Healthy_disease_Classification/train'

training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0

def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# print(len(imglist(os.path.join(train_path, 'Diseased'))))

descriptors_list = []

brisk = cv2.BRISK_create(30)

for image_path in image_paths:
    img = cv2.imread(image_path)
    keypoints, descriptors = brisk.detectAndCompute(img, None)
    descriptors_list.append((image_path, descriptors))

descriptors = descriptors_list[0][1]

for image_path, descriptor in descriptors_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

descriptors_float = descriptors.astype(float)

k = 200
voc, variance= kmeans(descriptors_float, k, 1)

img_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(descriptors_list[i][1], voc)
    for w in words:
        img_features[i][w] += 1
        
number_of_occurances = np.sum((img_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * number_of_occurances * 1)), 'float32')

stdSlr = StandardScaler().fit(img_features)
img_features = stdSlr.transform(img_features)

model = LinearSVC(max_iter = 10000)
model.fit(img_features, np.array(image_classes))


joblib.dump((model, training_names, stdSlr, k, voc), 'bovw.pkl', compress=3)

