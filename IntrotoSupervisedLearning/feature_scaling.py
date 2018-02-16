# Feature Scaling: The Syntax
# Import the class containing the scaling method
from sklearn.preprocessing import StandardScaler

# Create an instance of the class
StdSc = StandardScaler()

# Fit the scaling parameters and then transform the data
StdSc = StdSc.fit(X_data)
X_scaled = KNN.transform(X_data)

# Other scaling methods exist: MinMaxScaler, MaxAbsScaler.

# K Nearest Neighbors: The Syntax
# Import the class containing the classification method
from sklearn.neighbors import KNeighborsClassifier
# Create an instance of the class
KNN = KNeighborsClassifier(n_neighbors=3)
# Fit the instance on the data and then predict the expected value
KNN = KNN.fit(X_data, y_data)
y_predict = KNN.predict(X_data)
