#Importing Package 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the CSV files and passing the filename  to a variable named iris.
iris = pd.read_csv("iris.csv")
print(iris.head())


#Printing information about dataset
print(iris.info())

#Checking number of columns and rows
print(iris.shape)

#Columns indexes
print(iris.columns)

#counting spiecies of iris
print(iris["species"].value_counts())

#Getting statistics about dataset
print(iris.describe())

#1D Scatter Plots
iris_setsoa = iris.loc[iris["species"] == "setosa"];
iris_virginica = iris.loc[iris["species"] == "virginica"];
iris_versicolor = iris.loc[iris["species"] == "versicolor"];
plt.plot(iris_setsoa["petal_length"],np.zeros_like(iris_setsoa["petal_length"]), 'o', label='setosa')
plt.plot(iris_versicolor["petal_length"],np.zeros_like(iris_versicolor["petal_length"]), 'o', label='versicolor')
plt.plot(iris_virginica["petal_length"],np.zeros_like(iris_virginica["petal_length"]), 'o', label='virginica')
plt.legend()
plt.grid()
plt.show()  

#2D Scatter Plots
iris.plot(kind="scatter",x="sepal_length",y="sepal_width")
plt.show() 

#2D scatter plot in seaborn
sns.set_style("whitegrid");
sns.FacetGrid(iris,hue="species",height=4) \
.map(plt.scatter,"sepal_length","sepal_width") \
.add_legend()
plt.show()  

#Pair plots in seaborn
sns.set_style("whitegrid");
sns.pairplot(iris,hue="species",height=3);
plt.show()