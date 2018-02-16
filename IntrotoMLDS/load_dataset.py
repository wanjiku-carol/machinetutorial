import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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

# EVALUATE SOME ALGORITHMS
# Here is what we are going to cover in this step:

# Separate out a validation dataset.
# Set-up the test harness to use 10-fold cross validation.
# Build 5 different models to predict species from flower measurements
# Select the best model.
#  1.  Create a Validation Dataset
# We will split the loaded dataset into two, 80% of which we will use 
# to train our models and 20% that we will hold back as a validation dataset.
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation, = model_selection.train_test_split(
  X, Y, test_size=validation_size, random_state=seed
)
# You now have training data in the X_train and Y_train for preparing models 
# and a X_validation and Y_validation sets that we can use later.
# 2. Test harness


