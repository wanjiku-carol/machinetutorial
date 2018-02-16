# Basic Scatter Plots with Matplotlib
# Scatter plots can be created from Pandas Series

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

filepath = './Iris_Data.csv'
data = pd.read_csv(filepath)

plt.plot(data.sepal_length,
          data.sepal_width,
          ls='', marker='o')

# Multiple layers of data can also be added

plt.plot(data.sepal_length,
          data.sepal_width, ls='', marker='o', 
          label='sepal')
plt.plot(data.petal_length, data.petal_width, 
          ls='', marker='o', label='petal')

# Histograms with Matplotlib
plt.hist(data.sepal_length, bins=25)

# Customizing Matplotlib Plots
# Every feature of Matplotlib plots can be customized
fig, ax = plt.subplots()

ax.barh(np.arange(10),
          data.sepal_width.iloc[:10])

# Set position of ticks and tick labels
ax.set_yticks(np.arange(0.4,10.4,1.0))
ax.set_yticklabels(np.arange(1,11))
ax.set(xlabel='xlabel', ylabel='ylabel',
        title='Title')

# Incorporating Statistical Calculations
# Statistical calculations can be included with Pandas methods
(data.groupby('species').mean().plot(color=['red','blue','black','green'],
fontsize=10.0, figsize=(4,4)))

# Statistical Plotting with Seaborn
# Joint distribution and scatter plots can be created
sns.joinplot(x='sepal_length', y='sepal_width',data=data, size=4)

# Correlation plots of all variable pairs can also be made with Seaborn
sns.pairplot(data, hue='species', size=3)
