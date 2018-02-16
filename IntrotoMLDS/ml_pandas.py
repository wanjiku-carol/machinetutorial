import pandas as pd
import numpy as np
# Use data from step tracking application to create a Pandas Series

step_data = [3620, 7891, 9761, 3907, 4338, 5373]
step_counts = pd.Series(step_data, name='steps')
print(step_counts)

# Output
# 0    3620
# 1    7891
# 2    9761
# 3    3907
# 4    4338
# 5    5373

step_counts.index = pd.date_range('20150329', periods=6)
print(step_counts)
# Output
# 2015-03-29    3620
# 2015-03-30    7891
# 2015-03-31    9761
# 2015-04-01    3907
# 2015-04-02    4338
# 2015-04-03    5373

print(step_counts[3])
# Output
# 3907

print(step_counts['2015-03-29'])
# Output
# 3620

print(step_counts['2015-04'])
# Output
# 2015-04-01 3907
# 2015-04-02 4338
# 2015-04-03 5373
# Freq: D, Name: steps, dtype: int64

print(step_counts.dtypes)
# Output
# int64
# step_counts = step_counts.astype(np.float)
print(step_counts.dtypes)
# Output
# float64

cycling_data = [10.7, 0, None, 2.4, 15.3, 10.9, 0, None]
joined_data = list(zip(step_data, cycling_data))
# activity_df = pd.DataFrame(joined_data)
print(activity_df)
# Output
#  0     1
# 0  3620  10.7
# 1  7891   0.0
# 2  9761   NaN
# 3  3907   2.4
# 4  4338  15.3
# 5  5373  10.9

activity_df = pd.DataFrame(joined_data, 
            index=pd.date_range('20150329',periods=6),
            columns=['Walking', 'Cycling'])

print(activity_df)
# Output
#   Walking  Cycling
# 2015-03-29     3620     10.7
# 2015-03-30     7891      0.0
# 2015-03-31     9761      NaN
# 2015-04-01     3907      2.4
# 2015-04-02     4338     15.3
# 2015-04-03     5373     10.9

# DataFrame rows can be indexed by row using the 'loc' and 'iloc' methods
print(activity_df.loc['2015-04-01'])
# Output
# Walking    3907.0
# Cycling       2.4
# Name: 2015-04-01 00:00:00, dtype: float64

print(activity_df.iloc[-3])
# Output
# Walking    3907.0
# Cycling       2.4
# Name: 2015-04-01 00:00:00, dtype: float64

print(activity_df['Walking']) # or
print(activity_df.Walking) # or
print(activity_df.iloc[:,0])
# Output
# 2015-03-29    3620
# 2015-03-30    7891
# 2015-03-31    9761
# 2015-04-01    3907
# 2015-04-02    4338
# 2015-04-03    5373
# Freq: D, Name: Walking, dtype: int64

# Reading Data with Pandas
# The location of the data file
filepath = './Iris_Data.csv'
# import the data
data = pd.read_csv(filepath)
# Print a few rows
print(data.iloc[:5])

#   sepal_length  sepal_width  petal_length  petal_width      species
# 0           5.1          3.5           1.4          0.2  Iris-setosa
# 1           4.9          3.0           1.4          0.2  Iris-setosa
# 2           4.7          3.2           1.3          0.2  Iris-setosa
# 3           4.6          3.1           1.5          0.2  Iris-setosa
# 4           5.0          3.6           1.4          0.2  Iris-setosa

# Assigning New Data to a DataFrame
# Create a new column that is a product of both measurements
# data['sepal_area'] = data.sepal_length * data.sepal_width

#   petal_width      species  sepal_area
# 0          0.2  Iris-setosa       17.85
# 1          0.2  Iris-setosa       14.70
# 2          0.2  Iris-setosa       15.04
# 3          0.2  Iris-setosa       14.26
# 4          0.2  Iris-setosa       18.00

# data['abbrev'] = (data.species.apply(lambda x: x.replace('Iris-','')))
print(data.iloc[:5, -3:])

# petal_width      species  abbrev
# 0          0.2  Iris-setosa  s  tosa
# 1          0.2  Iris-setosa  setosa
# 2          0.2  Iris-setosa  setosa
# 3          0.2  Iris-setosa  setosa
# 4          0.2  Iris-setosa  setosa


# Concatenating Two DataFrames
# Concatenate the first two and
# last two rows
# small_data = pd.concat([data.iloc[:2],
#                           data.iloc[-2:]])

print (small_data.iloc[:,-3:])

#     petal_length  petal_width         species
# 0             1.4          0.2     Iris-setosa
# 1             1.4          0.2     Iris-setosa
# 148           5.4          2.3  Iris-virginica
# 149           5.1          1.8  Iris-virginica

# See the 'join' method for
# SQL style joining of dataframes

# Aggregated Statistics with GroupBy
# Using the groupby method calculated aggregated DataFrame statistics
# Use the size method with a DataFrame to get count
# For a Series, use the .value_counts method

# group_sizes = (data.groupby('species').size())

print(group_sizes)

# species
# Iris-setosa        50
# Iris-versicolor    50
# Iris-virginica     50
# dtype: int64

# Performing Statistical Calculations
# Mean calculated on a DataFrame
print(data.mean())

# sepal_length    5.843333
# sepal_width     3.054000
# petal_length    3.758667
# petal_width     1.198667
# dtype: float64

# Median
print(data.petal_length.median())
# 4.35

# Mode
print(data.petal_length.mode())

# 0    1.5
# dtype: float64

# Standard dev, variance, and SEM
print(data.petal_length.std(),
#       data.petal_length.var(),
#       data.petal_length.sem())

# 1.7644204199522626, 3.1131794183445192, 0.14406432402100849

# As well as quantile
print(data.quantile(0))

# l_length    4.3
# sepal_width     2.0
# petal_length    1.0
# petal_width     0.1
# Name: 0, dtype: float64

# Multiple calculations can be presented in a DataFrame
print(data.describe())

# sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

# Sampling from DataFrames
# DataFrames can be randomly sampled from
# Sample 5 rows without replacement
# data_sample = (data.sample(n=5, replace=False, random_state=42))
print(data_sample.iloc[:,-3:])

#  petal_length  petal_width          species
# 73            4.7          1.2  Iris-versicolor
# 18            1.7          0.3      Iris-setosa
# 118           6.9          2.3   Iris-virginica
# 78            4.5          1.5  Iris-versicolor
# 76            4.8          1.4  Iris-versicolor
