#Importing Package 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Reading the CSV files and passing the filename  to a variable named iris.
iris = pd.read_csv("iris.csv")
print(iris.head())

# RandomForest algorithm

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

# Logistic regression

# Training set preparation
# feature, columns except the last column
X = iris.iloc[:, :-1]
# target values, last column of the data frame
y = iris.iloc[:, -1]

