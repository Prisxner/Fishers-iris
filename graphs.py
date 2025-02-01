import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

data = iris.data
target = iris.target
target_names = iris.target_names

setosa = data[target == 0]
versicolor = data[target == 1]
virginica = data[target == 2]

setosa_sepal_length = setosa[:, 0]
setosa_sepal_width = setosa[:, 1]
setosa_petal_length = setosa[:, 2]
setosa_petal_width = setosa[:, 3]

versicolor_sepal_length = versicolor[:, 0]
versicolor_sepal_width = versicolor[:, 1]
versicolor_petal_length = versicolor[:, 2]
versicolor_petal_width = versicolor[:, 3]

virginica_sepal_length = virginica[:, 0]
virginica_sepal_width = virginica[:, 1]
virginica_petal_length = virginica[:, 2]
virginica_petal_width = virginica[:, 3]

'''lenght'''

x_1 = setosa_sepal_length
y_1 = [1 for i in range(50)]

x_2 = versicolor_sepal_length
y_2 = [1.01 for i in range(50)]

x_3 = virginica_sepal_length
y_3 = [1.02 for i in range(50)]

plt.scatter(x_1, y_1, label='setosa', color='red')
plt.scatter(x_2, y_2, label='versicolor', color='green')
plt.scatter(x_3, y_3, label='virginica', color='blue')
plt.yticks([])
plt.ylim(0.98, 1.05)
plt.legend()
plt.title('Sepal length')
plt.show()

'''width'''

x_1 = setosa_sepal_width
y_1 = [1 for i in range(50)]

x_2 = versicolor_sepal_width
y_2 = [1.01 for i in range(50)]

x_3 = virginica_sepal_width
y_3 = [1.02 for i in range(50)]

plt.scatter(x_1, y_1, label='setosa', color='red')
plt.scatter(x_2, y_2, label='versicolor', color='green')
plt.scatter(x_3, y_3, label='virginica', color='blue')
plt.yticks([])
plt.ylim(0.98, 1.05)
plt.legend()
plt.title('Sepal width')
plt.show()

'''lenght'''

x_1 = setosa_petal_width
y_1 = [1 for i in range(50)]

x_2 = versicolor_petal_width
y_2 = [1.01 for i in range(50)]

x_3 = virginica_petal_width
y_3 = [1.02 for i in range(50)]

plt.scatter(x_1, y_1, label='setosa', color='red')
plt.scatter(x_2, y_2, label='versicolor', color='green')
plt.scatter(x_3, y_3, label='virginica',color='blue')
plt.yticks([])
plt.ylim(0.98, 1.05)
plt.legend()
plt.title('Petal width')
plt.show()

'''width'''

x_1 = setosa_petal_length
y_1 = [1 for i in range(50)]

x_2 = versicolor_petal_length
y_2 = [1.01 for i in range(50)]

x_3 = virginica_petal_length
y_3 = [1.02 for i in range(50)]

plt.scatter(x_1, y_1, label='setosa', color='red')
plt.scatter(x_2, y_2, label='versicolor', color='green')
plt.scatter(x_3, y_3, label='virginica', color='blue')
plt.yticks([])
plt.ylim(0.98, 1.05)
plt.legend()
plt.title('Petal length')
plt.show()


