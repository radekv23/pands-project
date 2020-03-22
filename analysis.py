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

