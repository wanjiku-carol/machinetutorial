import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection


filepath = './Iris_Data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(filepath, names=names)

# DIMENSIONS OF THE DATA
# get instances(rows) and attributes(columns)
print(dataset.shape)
#  Result: (151, 5)

# PEEK AT THE DATA
# eyeball the data. print the first 20 rows
print(dataset.head(20))

# STATISTICAL SUMMARY
# Summary of each attribute:  count, mean, the min and max values
# print(dataset.describe())
# Result:
#       sepal-length sepal-width petal-length petal-width        class
# count           151         151          151         151          151
# unique           36          24           44          23            4
# top             5.0         3.0          1.5         0.2  Iris-setosa
# freq             10          26           14          28           50

# CLASS DISTRIBUTION
# look at the number of instances (rows) that belong to each class as an absolute count
print(dataset.groupby('class').size())
# Result:
# class
# Iris-setosa        50
# Iris-versicolor    50
# Iris-virginica     50
# dtype: int64

# DATA VISUALIZATION
# Univariate Plots: to better understand each attribute
# Given that the input variables are numeric, we can create box and whisker plots of each.
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
dataset.hist()
plt.show()

# Multivariate Plots: to look at the interactions between the variables

# This can be helpful to spot structured relationships between input variables.
scatter_matrix(dataset)
plt.show()
