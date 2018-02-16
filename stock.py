import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
# REGRESSIONS

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HGH_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HGH_PCT', 'PCT_change', 'Adj. Volume']]

# PREDICTIONS
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# we'll try to predict ten percent of the data frame
# math.ceil rounds everything up to the nearest whole
forecast_out = int(math.ceil(0.01 * len(df)))
# we have features line 8 and line 11, now we create labels

df['labels'] = df[forecast_col].shift(-forecast_out)
# shift the columns negatively. The label column will be the adjusted column
# ten percent into the future
df.dropna(inplace=True)
# print(df.head())
print(df.tail())

# Features and labels are defined features = X, labels = y

X = np.array(df.drop(['labels'], 1))  # everything except labels
y = np.array(df['labels'])
X = preprocessing.scale(X)

df.dropna(inplace=True)
y = np.array(df['labels'])

print(len(X), len(y))
