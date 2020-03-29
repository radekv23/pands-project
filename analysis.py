#Importing Package 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



#Reading the CSV files and passing the filename  to a variable named iris.
iris = pd.read_csv("iris.csv")
print(iris.head())


#Printing information about dataset
print(iris.info())

#Checking number of columns and rows
print(iris.shape)

#Columns indexes
print(iris.columns)

#Identifiy missing values
pd.isnull(iris)

#counting spiecies of iris
print(iris["species"].value_counts())

#Getting statistics about dataset
print(iris.describe())

#1D Scatter Plots
iris_setsoa = iris.loc[iris["species"] == "setosa"]
iris_virginica = iris.loc[iris["species"] == "virginica"]
iris_versicolor = iris.loc[iris["species"] == "versicolor"]
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
sns.set_style("whitegrid")
sns.FacetGrid(iris,hue="species",height=4) \
.map(plt.scatter,"sepal_length","sepal_width") \
.add_legend()
plt.show()  

#Pair plots in seaborn
sns.set_style("whitegrid")
sns.pairplot(iris,hue="species",height=3)
plt.show()


#RandomForest algorithm
# Isolate Data, class labels and column values
X = iris.iloc[:,0:4]
Y = iris.iloc[:,-1]
names = iris.columns.values

# Build the model
model = RandomForestClassifier(n_estimators=100)

# Fit the model
model.fit(X, Y)

# Print the results
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), reverse=True))

# Isolate feature importances 
importance = model.feature_importances_

# Sort the feature importances 
sorted_importances = np.argsort(importance)

# Insert padding
padding = np.arange(len(names)-1) + 0.5

# Plot the data
plt.barh(padding, importance[sorted_importances], align='center')

# Customize the plot
plt.yticks(padding, names[sorted_importances])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")

# Show the plot
plt.show()
