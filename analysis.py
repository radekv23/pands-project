# Importing Package 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



# Reading the CSV files and passing the filename  to a variable named iris.
iris = pd.read_csv("iris.csv")
print(iris.head())


# Printing information about dataset
print(iris.info())

# Checking number of columns and rows
print(iris.shape)

# Columns indexes
print(iris.columns)

# Identifiy missing values
pd.isnull(iris)

# counting spiecies of iris
print(iris["species"].value_counts())

# Getting statistics about dataset
print(iris.describe())

# 1D Scatter Plots
#dividing our data set into three species
iris_setsoa = iris.loc[iris["species"] == "setosa"]
iris_virginica = iris.loc[iris["species"] == "virginica"]
iris_versicolor = iris.loc[iris["species"] == "versicolor"]

# creating plot
plt.plot(iris_setsoa["petal_length"],np.zeros_like(iris_setsoa["petal_length"]), 'o', label='setosa')
plt.plot(iris_versicolor["petal_length"],np.zeros_like(iris_versicolor["petal_length"]), 'o', label='versicolor')
plt.plot(iris_virginica["petal_length"],np.zeros_like(iris_virginica["petal_length"]), 'o', label='virginica')
plt.legend()
plt.grid()
plt.show()  

# 2D scatter plots in seaborn
sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species",height=6).map(plt.scatter,"sepal_length","sepal_width").add_legend()
sns.FacetGrid(iris,hue="species",height=6).map(plt.scatter,"petal_length","petal_width").add_legend()
plt.show()  

# Pair plots in seaborn
sns.set_style("whitegrid")
sns.pairplot(iris,hue="species",height=3)
plt.show()

# plotting the histogram’s of each flowers.
sns.FacetGrid(iris,hue="species",height=6).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(iris,hue="species",height=6).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(iris,hue="species",height=6).map(sns.distplot,"sepal_length").add_legend()
sns.FacetGrid(iris,hue="species",height=6).map(sns.distplot,"sepal_width").add_legend()
plt.show()

# Cumulative Distribution Function (CDF) plots 
plt.figure(figsize=(10,7))
counts, bin_edges = np.histogram(iris[iris['species'] == 'setosa']['petal_length'],
                                 bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'Setosa PDF')
plt.plot(bin_edges[1:], cdf, label = 'Setosa CDF')
counts, bin_edges = np.histogram(iris[iris['species'] == 'versicolor']['petal_length'],
                                 bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'Versicolor PDF')
plt.plot(bin_edges[1:], cdf, label = 'Versicolor CDF')
counts, bin_edges = np.histogram(iris[iris['species'] == 'virginica']['petal_length'],
                                 bins = 10, density = True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label = 'Virginica PDF')
plt.plot(bin_edges[1:], cdf, label = 'Virginica CDF')
plt.legend()
plt.show()

# Plotting the boxplots using Seaborn
sns.set(style="ticks") 
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.boxplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,2)
sns.boxplot(x='species',y='sepal_width',data=iris)
plt.subplot(2,2,3)
sns.boxplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,4)
sns.boxplot(x='species',y='petal_width',data=iris)
plt.show()


# Plotting Violin plots
sns.set_style("whitegrid")
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.show()
