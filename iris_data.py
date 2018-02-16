import pandas
from pandas.tools.plotting import scatter_matrix
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

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
# check how many instances(rows) and attributes(columns) the data set has
# print(dataset.shape)

# print the first 20 rows
# print(dataset.head(20))

# each attribute's count, mean, the min and max and percentile
# print(dataset.describe())

# class distribution(number of instances that belong to each class)
# print(dataset.groupby('class').size())

# Visualisation of data using univariate plots
# i.e. plots of each individual variable
# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show()

# create a histogram of each input variable to get an idea of the distribution
# dataset.hist()
# plt.show()

scatter_matrix(dataset)
plt.show()
