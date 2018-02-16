import pandas as pd
import numpy as np

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
import matplotlib.pyplot as plt

filepath = './Iris_Data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(filepath, names=names)
#  EVALUATE SOME ALGORITHMS
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
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation, = model_selection.train_test_split(
  X, Y, test_size=validation_size, random_state=seed
)
# You now have training data in the X_train and Y_train for preparing models 
# and a X_validation and Y_validation sets that we can use later.
# 2. Test harness
# 3. Build Models
# spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#results
# LR: 0.966667 (0.040825)
# LDA: 0.975000 (0.038188)
# KNN: 0.983333 (0.033333)
# CART: 0.983333 (0.033333)
# NB: 0.975000 (0.053359)
# SVM: 0.991667 (0.025000)

# 4 Select Best Model
# We can also create a plot of the model evaluation results and 
# compare the spread and the mean accuracy of each model. 
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# 6. MAKE PREDICTIONS
# to get an idea of the accuracy of the model on our validation set.

# This will give us an independent final check on the accuracy of the best model. 
# It is valuable to keep a validation set just in case you made a slip during training, 
# such as overfitting to the training set or a data leak. Both will result in an 
# overly optimistic result.

# We can run the KNN model directly on the validation set and summarize the results 
# as a final accuracy score, a confusion matrix and a classification report.

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Result
# [[ 7  0  0]
#  [ 0 11  1]
#  [ 0  2  9]]
#                  precision    recall  f1-score   support

#     Iris-setosa       1.00      1.00      1.00         7
# Iris-versicolor       0.85      0.92      0.88        12
#  Iris-virginica       0.90      0.82      0.86        11

#     avg / total       0.90      0.90      0.90        30
