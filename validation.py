import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
from scipy.cluster.vq import vq
import pylab as pl

model, classes_names, stdSlr, k, voc = joblib.load('bovw.pkl')

test_path = 'Dataset/Healthy_disease_Classification/test'

testing_names = os.listdir(test_path)

image_paths = []
image_classes = []
class_id = 0


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def showConfusionMatrix(con_mat):
    pl.matshow(con_mat)
    pl.title("Confusion Matrix")
    pl.colorbar()
    pl.show()

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1


descriptors_list = []

brisk = cv2.BRISK_create(30)

for image_path in image_paths:
    img = cv2.imread(image_path)
    keypoints, descriptors = brisk.detectAndCompute(img, None)
    descriptors_list.append((image_path, descriptors))

descriptors = descriptors_list[0][1]

for image_path, descriptor in descriptors_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(descriptors_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1

number_of_occurances = np.sum((test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * number_of_occurances * 1)), 'float32')

test_features = stdSlr.transform(test_features)

true_classes = [classes_names[i] for i in image_classes]

predicted_values = model.predict(test_features)
predictions = [classes_names[i] for i in predicted_values]

print("True Class : ", str(true_classes))
print("Predicted Class : ", str(predictions))

accuracy = accuracy_score(true_classes, predictions)
print("Accuracy : ", accuracy)

con_mat = confusion_matrix(true_classes, predictions)
print(con_mat)

showConfusionMatrix(con_mat)
