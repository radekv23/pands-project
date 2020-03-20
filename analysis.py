#Importing Package 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#To read the CSV files, we use the read_csv method, as follows and pass the filename  to a variable (iris).
iris = pd.read_csv("iris.csv")
print(iris.head())