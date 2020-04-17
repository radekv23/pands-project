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
from sklearn import metrics


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
plt.show()

# Linear regression

# plotting the petal_width against the petal_length
sns.regplot(x='petal_width', y='petal_length', data=iris)
plt.show()

# Logistic regression

# Training set preparation
# feature, columns except the last column
X = iris.iloc[:, :-1]
# target values, last column of the data frame
y = iris.iloc[:, -1]

# Plots of relation between features and species
plt.xlabel('Features')
plt.ylabel('Species')

pltX = iris.loc[:, 'sepal_length']
pltY = iris.loc[:,'species']
plt.scatter(pltX, pltY, color='blue', label='sepal_length')

pltX = iris.loc[:, 'sepal_width']
pltY = iris.loc[:,'species']
plt.scatter(pltX, pltY, color='green', label='sepal_width')

pltX = iris.loc[:, 'petal_length']
pltY = iris.loc[:,'species']
plt.scatter(pltX, pltY, color='red', label='petal_length')

pltX = iris.loc[:, 'petal_width']
pltY = iris.loc[:,'species']
plt.scatter(pltX, pltY, color='black', label='petal_width')

plt.legend(loc=4, prop={'size':8})
plt.show()

# Split the data: training (80%), testing (20%)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model
model = LogisticRegression(solver='lbfgs', multi_class='auto')
model.fit(x_train, y_train)
#Test the model
predictions = model.predict(x_test)
print(predictions)
print()
# precision, recall, f1-score
print( classification_report(y_test, predictions) )
print("Accuracy:",(accuracy_score(y_test, predictions)*100), "%")
