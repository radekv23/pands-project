# PandS-project: Data Analysis of IRIS DATASET

## Iris flower data set
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

![Iris species](https://thegoodpython.com/assets/images/iris-species.png)

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

## Use of the data set
Based on Fisher's linear discriminant model, this data set became a typical test case for many statistical classification techniques in machine learning such as support vector machines.

The use of this data set in cluster analysis however is not common, since the data set only contains two clusters with rather obvious separation. One of the clusters contains Iris setosa, while the other cluster contains both Iris virginica and Iris versicolor and is not separable without the species information Fisher used. This makes the data set a good example to explain the difference between supervised and unsupervised techniques in data mining: Fisher's linear discriminant model can only be obtained when the object species are known: class labels and clusters are not necessarily the same.

Nevertheless, all three species of Iris are separable in the projection on the nonlinear and branching principal component. The data set is approximated by the closest tree with some penalty for the excessive number of nodes, bending and stretching. Then the so-called "metro map" is constructed. The data points are projected into the closest node. For each node the pie diagram of the projected points is prepared. The area of the pie is proportional to the number of the projected points. It is clear from the diagram (left) that the absolute majority of the samples of the different Iris species belong to the different nodes. Only a small fraction of Iris-virginica is mixed with Iris-versicolor (the mixed blue-green nodes in the diagram). Therefore, the three species of Iris (Iris setosa, Iris virginica and Iris versicolor) are separable by the unsupervising procedures of nonlinear principal component analysis. To discriminate them, it is sufficient just to select the corresponding nodes on the principal tree.

# Steps of Analysis:
## 1. Importing libraries and loading Iris dataset.
We are importing pandas, numpy, matplotlib and seaborns libaries.

And loading our dataset:

<img src="https://github.com/radekv23/pands-project/blob/master/img/dataset.JPG">

## 2. Data preparation and statistics.
We are printing info about our dataset. We could notice is no missing values and all our data is of the float type.

Shape of data showing us that it is 150 instances of data and 5 attributes present.

We can see the labels of columns present, object type for all columns present in data and data points for each class present.

<img src="https://github.com/radekv23/pands-project/blob/master/img/dataInfo.JPG">

By using pandas describe() we can print basic statistical details like percentile, mean, std, etc. of a data frame or a series of numeric values.

<img src="https://github.com/radekv23/pands-project/blob/master/img/describe.JPG">

## 3. Visualization
### Scatter plots
1D scatter plot of the iris data:

<img src="https://github.com/radekv23/pands-project/blob/master/img/1dScatter.png">


2D scatter plot in seaborn:

<img src="https://github.com/radekv23/pands-project/blob/master/img/2dScaSeaborn.png">

#### Conclusion

Setosa (blue) points can be easily separated from versicolor (orange) and virginica (green) by drawing a line.
But versicolor and virginica data points cannot be easily separated.
Using sepal_length and sepal_width features, we can distinguish Setosa flowers from others.
Separating Versicolor from Viginica is much harder as they have considerable overlap.

### Pair plots in Seaborn

<img src="https://github.com/radekv23/pands-project/blob/master/img/pairPlots.png">

#### Conclusion:

Petal length and petal width are the most useful features to identify various flower types.
While Setosa can be easily identified (linearly separable), virginica and Versicolor have some overlap (almost linearly separable).

### Random Forest 

Random Forest algorithm randomly generates thousands of decision trees and takes turns leaving out each variable in fitting the model. It calculate how much better or worse a model does when you leave one variable out of the equation.

<img src="https://github.com/radekv23/pands-project/blob/master/img/RandomForest.png">

#### Conclusion:

We can clearly see tha most important variable to determine species is petal lenght  and least important is sepal width.

## Built With
Python - https://www.python.org/downloads/

Visual studio code - https://code.visualstudio.com/download

cmder - https://cmder.net/

Anaconda - https://www.anaconda.com/distribution/

### Author

Student: Radoslaw Wojtczak

GMIT student id: G00352936

### Acknowledgments

StackOverFlow ultimate help source when you stuck
https://stackoverflow.com

Github help
https://guides.github.com

W3Schools
https://www.w3schools.com/python/

Real python
https://realpython.com/

Other resources:

https://en.wikipedia.org/wiki/Iris_flower_data_set

https://www.datacamp.com

https://www.udemy.com

https://www.coursera.org

https://matplotlib.org

https://developers.google.com/edu/python/

### Inspiration

GMIT PandS Team:
Ian McLoughlin
      &
Andrew Beatty

Thank you for all the help, and great simple to understand lectures, labs, tutorials and workshops.
