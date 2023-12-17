from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from KNN import KNN

mnist = fetch_openml(name='mnist_784', parser='auto')

def display_some_images():
  image= mnist.data.to_numpy()
  plt.subplot(431)
  plt.imshow((image[0].reshape(28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
  plt.subplot(432)
  plt.imshow(image[1].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
  plt.subplot(433)
  plt.imshow(image[3].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
  plt.subplot(434)
  plt.imshow(image[4].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
  plt.subplot(435)
  plt.imshow(image[5].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
  plt.subplot(436)
  plt.imshow(image[6].reshape(28,28), cmap=plt.cm.gray_r, interpolation='nearest')
  plt.show()

index_number= np.random.permutation(70000)
x1,y1=mnist.data.loc[index_number],mnist.target.loc[index_number]
x1.reset_index(drop=True,inplace=True)
y1.reset_index(drop=True,inplace=True)
x_train , x_test = x1[:55000], x1[55000:]
y_train , y_test = y1[:55000], y1[55000:]

knn = KNN(5)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)