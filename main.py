import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def classifier_1(threshold, characteristic_list, classification_list, Non_list):
    classified_positive = []
    classified_negative = []
    for i in range(len(characteristic_list)):
        if characteristic_list[i] <= threshold:
            classified_positive.append(i)
        else:
            classified_negative.append(i)
    TP = len(np.intersect1d(classified_positive, classification_list))
    P = len(classification_list)
    TPR = TP / P
    FP = len(np.intersect1d(classified_positive, Non_list))
    N = len(Non_list)
    TN = len(np.intersect1d(classified_negative, Non_list))
    FPR = FP / N
    TNR = TN / N
    return FPR, TPR

def classifier_2(threshold, characteristic_list, classification_list, Non_list):
    classified_positive = []
    classified_negative = []
    for i in range(len(characteristic_list)):
        if characteristic_list[i] >= threshold:
            classified_positive.append(i)
        else:
            classified_negative.append(i)
    TP = len(np.intersect1d(classified_positive, classification_list))
    P = len(classification_list)
    TPR = TP / P
    FP = len(np.intersect1d(classified_positive, Non_list))
    N = len(Non_list)
    TN = len(np.intersect1d(classified_negative, Non_list))
    FPR = FP / N
    TNR = TN / N
    return FPR, TPR

data = datasets.load_iris()
X = data.data

Sepal_length = []
Sepal_width = []
Petal_length = []
Petal_width = []

for i in range(len(X)):
    Sepal_length.append(X[i][0])
    Sepal_width.append(X[i][1])
    Petal_length.append(X[i][2])
    Petal_width.append(X[i][3])

Setosa = [i for i in range(0, 50)]
Versicolor = [i for i in range(50, 100)]
Virginica = [i for i in range(100, 150)]

Non_setosa = np.hstack([Versicolor, Virginica])
Non_versicolor = np.hstack([Setosa, Virginica])
Non_virginica = np.hstack([Setosa, Versicolor])

'''Setosa petal width'''
x_1 = []
y_1 = []
tresholds_1 = np.linspace(min(Petal_width), max(Petal_width), 1000)
for i in tresholds_1:
    x, y = classifier_1(i, Petal_width, Setosa, Non_setosa)
    x_1.append(x)
    y_1.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_1, y_1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Setosa), petal width')
plt.show()

'''Versicolor petal width'''
x_2 = []
y_2 = []
for i in tresholds_1:
    x, y = classifier_2(i, Petal_width, Versicolor, Non_versicolor)
    x_2.append(x)
    y_2.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_2, y_2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Versicolor), petal width')
plt.show()

'''Virginica petal width'''
x_3 = []
y_3 = []
for i in tresholds_1:
    x, y = classifier_2(i, Petal_width, Virginica, Non_virginica)
    x_3.append(x)
    y_3.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_3, y_3)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Virginica), petal width')
plt.show()

'''Setosa petal length'''
x_4 = []
y_4 = []
tresholds_2 = np.linspace(min(Petal_length), max(Petal_length), 1000)
for i in tresholds_2:
    x, y = classifier_1(i, Petal_length, Setosa, Non_setosa)
    x_4.append(x)
    y_4.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_4, y_4)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Setosa), petal length')
plt.show()

'''Versicolor petal length'''
x_5 = []
y_5 = []
for i in tresholds_2:
    x, y = classifier_2(i, Petal_length, Versicolor, Non_versicolor)
    x_5.append(x)
    y_5.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_5, y_5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Versicolor), petal length')
plt.show()

'''Virginica petal length'''
x_6 = []
y_6 = []
for i in tresholds_2:
    x, y = classifier_2(i, Petal_length, Virginica, Non_virginica)
    x_6.append(x)
    y_6.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_6, y_6)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Virginica), petal length')
plt.show()

'''Setosa sepal width'''
x_7 = []
y_7 = []
tresholds_3 = np.linspace(min(Sepal_width), max(Sepal_width), 1000)
for i in tresholds_3:
    x, y = classifier_2(i, Sepal_width, Setosa, Non_setosa)
    x_7.append(x)
    y_7.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_7, y_7)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Setosa), sepal width')
plt.show()

'''Versicolor sepal width'''
x_8 = []
y_8 = []
for i in tresholds_3:
    x, y = classifier_1(i, Sepal_width, Versicolor, Non_versicolor)
    x_8.append(x)
    y_8.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_8, y_8)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Versicolor), sepal width')
plt.show()

'''Virginica sepal width'''
x_9 = []
y_9 = []
for i in tresholds_3:
    x, y = classifier_1(i, Sepal_width, Virginica, Non_virginica)
    x_9.append(x)
    y_9.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_9, y_9)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Virginica), sepal width')
plt.show()

'''Setosa sepal length'''
x_10 = []
y_10 = []
tresholds_4 = np.linspace(min(Sepal_length), max(Sepal_length), 1000)
for i in tresholds_4:
    x, y = classifier_1(i, Sepal_length, Setosa, Non_setosa)
    x_10.append(x)
    y_10.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_10, y_10)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Setosa), sepal length')
plt.show()

'''Versicolor sepal length'''
x_11 = []
y_11 = []
for i in tresholds_4:
    x, y = classifier_2(i, Sepal_length, Versicolor, Non_versicolor)
    x_11.append(x)
    y_11.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_11, y_11)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Versicolor), sepal length')
plt.show()

'''Virginica sepal length'''
x_12 = []
y_12 = []
for i in tresholds_4:
    x, y = classifier_2(i, Sepal_length, Virginica, Non_virginica)
    x_12.append(x)
    y_12.append(y)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(x_12, y_12)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve(Virginica), sepal length')
plt.show()