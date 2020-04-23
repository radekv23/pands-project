# PandS-project: Data Analysis of IRIS DATASET

## Iris flower data set
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

![Iris species](https://thegoodpython.com/assets/images/iris-species.png)

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

## Use of the data set
Based on Fisher's linear discriminant model, this data set became a typical test case for many statistical classification techniques in machine learning such as support vector machines.

The use of this data set in cluster analysis however is not common, since the data set only contains two clusters with rather obvious separation. One of the clusters contains Iris setosa, while the other cluster contains both Iris virginica and Iris versicolor and is not separable without the species information Fisher used. This makes the data set a good example to explain the difference between supervised and unsupervised techniques in data mining: Fisher's linear discriminant model can only be obtained when the object species are known: class labels and clusters are not necessarily the same.

Nevertheless, all three species of Iris are separable in the projection on the nonlinear and branching principal component. The data set is approximated by the closest tree with some penalty for the excessive number of nodes, bending and stretching. Then the so-called "metro map" is constructed. The data points are projected into the closest node. For each node the pie diagram of the projected points is prepared. The area of the pie is proportional to the number of the projected points. It is clear from the diagram (left) that the absolute majority of the samples of the different Iris species belong to the different nodes. Only a small fraction of Iris-virginica is mixed with Iris-versicolor (the mixed blue-green nodes in the diagram). Therefore, the three species of Iris (Iris setosa, Iris virginica and Iris versicolor) are separable by the unsupervising procedures of nonlinear principal component analysis. To discriminate them, it is sufficient just to select the corresponding nodes on the principal tree.

# Steps of Analysis - [analysis.py](https://github.com/radekv23/pands-project/blob/master/analysis.py):
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


2D scatter plots in seaborn:

<img src="https://github.com/radekv23/pands-project/blob/master/img/2dScaSeaborn.png">

<img src="https://github.com/radekv23/pands-project/blob/master/img/sca2.png">

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

###  Histogram’s of each flowers

<img src="https://github.com/radekv23/pands-project/blob/master/img/hist1.png">
<img src="https://github.com/radekv23/pands-project/blob/master/img/hist2.png">
<img src="https://github.com/radekv23/pands-project/blob/master/img/hist3.png">
<img src="https://github.com/radekv23/pands-project/blob/master/img/hist4.png">

#### Conclusion:

When we are using petal length we can separate setosa, because when we are using sepal length,sepal width we can’t do anything because we can’t separate the flowers as they are too close to each other.

### Cumulative Distribution Function (CDF)

The cumulative distribution function (CDF) of a random variable is another method to describe the distribution of random variables. The cumulative distribution function (CDF) of random variable X is defined as FX(x)=P(X≤x), for all x∈R. Note that the subscript X indicates that this is the CDF of the random variable X.

<img src="https://github.com/radekv23/pands-project/blob/master/img/cdf1.png">

#### Conclusion:

From the above CDF plots, we can see that:
- 100 % of the Setosa, petal length < 1.9,
- 95 % of the Versicolor, petal length < 5,
- 90% of the Virginica, petal length > 5.

###  Box plots

<img src="https://github.com/radekv23/pands-project/blob/master/img/boxplot.png">

#### Conclusion:

A boxplot can tell us about our outliers: isolated points that can be seen on our box plots. There are very few of them and it wouldn't have any major impact on our analysis.

###  Violin plots

<img src="https://github.com/radekv23/pands-project/blob/master/img/violin.png">

#### Conclusion:

The Violin Boxplot shows that Virginica has highest median value in petal length, petal width and sepal length. However, Setosa has the highest sepal width median value. We can also see significant difference between Setosa’s sepal lenght and width against its petal length and width. That differene is smaller in Versicolor and Virginica. Furthermore the weight of the Virginica sepal width and petal width are highly concentrated around the median.

# Machine learning - [ml.py](https://github.com/radekv23/pands-project/blob/master/ml.py)

Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop a conventional algorithm for effectively performing the task.

Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory data analysis through unsupervised learning. In its application across business problems, machine learning is also referred to as predictive analytics.

### Random Forest 

Random Forest algorithm randomly generates thousands of decision trees and takes turns leaving out each variable in fitting the model. It calculate how much better or worse a model does when you leave one variable out of the equation.

<img src="https://github.com/radekv23/pands-project/blob/master/img/RandomForest.png">

#### Conclusion:

We can clearly see tha most important variable to determine species is petal lenght  and least important is sepal width.

### Linear regression 

Linear Regression is one of the most simple Machine learning algorithm that comes under Supervised Learning technique and used for solving regression problems.
It is used for predicting the continuous dependent variable with the help of independent variables. The goal of the Linear regression is to find the best fit line that can accurately predict the output for the continuous dependent variable.
By finding the best fit line, algorithm establish the relationship between dependent variable and independent variable. And the relationship should be of linear nature.

<img src="https://github.com/radekv23/pands-project/blob/master/img/linreg.png">

### LM Plots

LM plots are a convenient interface to fit regression models across conditional data subsets.
We have plotted sepal_width vs. sepal_length. The data set is divided into three subsets (species of iris flower).

<img src="https://github.com/radekv23/pands-project/blob/master/img/lmplots.png">

### Logistic regression 

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).

<img src="https://github.com/radekv23/pands-project/blob/master/img/logreg.png">

#### Model:

<img src="https://github.com/radekv23/pands-project/blob/master/img/lrModel.JPG">

# Neural networks [nn.py](https://github.com/radekv23/pands-project/blob/master/nn.py)

Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input.

#### Model:

<img src="https://github.com/radekv23/pands-project/blob/master/img/nnModel.JPG">


## Built With
Python - https://www.python.org/downloads/

Visual studio code - https://code.visualstudio.com/download

cmder - https://cmder.net/

Anaconda - https://www.anaconda.com/distribution/

Pytorch - https://pytorch.org install (conda install pytorch torchvision -c pytorch)

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

https://en.wikipedia.org/wiki/Machine_learning

https://www.datacamp.com/community/tutorials

https://www.udemy.com

https://www.coursera.org

https://matplotlib.org

https://developers.google.com/edu/python/

https://towardsdatascience.com

https://machinelearningmastery.com/blog/

https://medium.com/topic/data-science

https://ksopyla.com/python/

https://jakevdp.github.io/PythonDataScienceHandbook/

https://www.dataquest.io/blog/

https://www.kaggle.com/notebooks

https://seaborn.pydata.org/tutorial.html

https://scikit-learn.org/stable/user_guide.html

https://pandas.pydata.org/docs/development/index.html#development

https://pythonspot.com/matplotlib-scatterplot/

https://honingds.com/blog/seaborn-scatterplot/

https://www.tutorialspoint.com/seaborn/seaborn_visualizing_pairwise_relationship.htm

https://cmdlinetips.com/2019/02/how-to-make-histogram-in-python-with-pandas-and-seaborn/

https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python

https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html

https://pythonbasics.org/seaborn-boxplot/

https://cmdlinetips.com/2018/03/how-to-make-boxplots-in-python-with-pandas-and-seaborn/

https://mode.com/blog/violin-plot-examples

https://fcpython.com/visualisation/violin-plots-in-seaborn

https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/

https://blog.goodaudience.com/machine-learning-using-logistic-regression-in-python-with-code-ab3c7f5f3bed

https://realpython.com/logistic-regression-python/

https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

https://gilberttanner.com/blog/introduction-to-data-visualization-inpython

https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly/

https://pytorch.org

### Inspiration

GMIT PandS Team:

Ian McLoughlin

      &
      
Andrew Beatty

Thank you for all the help, and great simple to understand lectures, labs, tutorials and workshops.
