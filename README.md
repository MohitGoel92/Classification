# Classification

There are different types of machine learning, they include:

- Supervised: When the data points have a known outcome.
  - Regression
  - Classification
- Unsupervised: When the data points have an unknown outcome.
- Semi-Supervised: When we have data with known outcomes and data without outcomes.

Regression is used when we predict a continuous outcome. Typical examples include:

- House prices
- Box office revenues
- Event attendance
- Network load
- Portfolio losses

Classification is used when we predict a categorical outcome. Typical examples include:

- Detecting fraudulent transactions
- Customer churn
- Event attendance
- Network load
- Loan default

The most common models used for classification are:

- Logistic Regression
- K-Nearest Neighbours
- Support Vector Machines
- Decision Tree
- Neural Networks
- Ensemble Models
  - Random Forest
  - Bagging (Bootstrap Aggregating)
  - Boosting

## Logistic Regression

Logistic Regression is a type of regression that models the probability of a certain class occuring given other independent variables. It uses a logistic or logit function to model a dependent variable. It is a very common predictive model because of its high interpretability.

The diagram below is of a Sigmoid function, which is used for Logistic regression.

**Note:** Logistic regression is closely related to Linear regression.

<p align="center"> <img width="600" src= "/Pics/W11.png"> </p>

The derivation from the *Logistic Function* to the *Odds Ratio* given below highlights the relationship between the Logistic and Linear regression. Notice that the coefficient of the exponential is a linear line (in the form of Y = mX + c, where m and c are learnt coefficients).

<p align="center"> <img width="600" src= "/Pics/W12.gif"> </p>

**Note:** 
- P(x) is the probability of a particular observation x belonging to that class.
- The log odds demonstrates how a unit increase or decrease in our x-values will change our log odds in a linear fashion, according to the coefficient β1 which we have learnt.

The syntax used with logistic regression is as follows:

```
# Importing the class containing the classification method

from sklearn.linear_model import LogisticRegression

# Create an instance of the class
# The l2 penalty refers to Ridge regularisation, and c is the regularisation parameter (also known as inverse lambda). 
# Therefore, the higher the value for c, the lower the penalty.

lr = LogisticRegression(penalty = 'l2', c = 10)

# Fitting the instance on the data and then predicting

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Viewing the output fitted coefficients

lr.coef_

# Tuning the regularisation parameters with cross-validation

LogisticRegressionCV
```

## Confusion Matrix

A **confusion matrix** tabulates true positives, false negatives, false positives and true negatives. The below is a diagrammatic representation of a confusion matrix.

<p align="center"> <img width="500" src= "/Pics/W13.png"> </p>

**Accuracy** is defined as the ratio of true positives and true negatives divided by the total number of observations. It is a measure related to correctly predicting positive and negative instances. The formula is given below:

<p align="center"> <img width="400" src= "/Pics/W14.png"> </p>

**Recall** or **Sensitivity** identifies the ratio of true positives divided by the total number of actual positives. It quantifies the percentage of positive instances correctly identified. In other words, for all the positive cases how many did the model predict correctly. The formula is given below:

<p align="center"> <img width="400" src= "/Pics/W15.png"> </p>

**Note:** Although this portrays the overall model accuracy, this can be thrown off or misleading by heavily skewed (imbalanced) data.

**Precision** is the ratio of true positives divided by the total of predicted positives. In other words, for all the positive predictions how many did the model get correct. The closer this value is to 1.0, the better job the model does at identifying only positive instances. The formula is given below:

<p align="center"> <img width="280" src= "/Pics/W16.png"> </p>

**Specificity** is the ratio of true negatives divided by the total number of actual negatives. In other words, for all the negative cases how many did the model predict correctly. The closer this value is to 1.0, the better job this model does at avoiding false alarms. The formula is given below:

<p align="center"> <img width="280" src= "/Pics/W17.png"> </p>

**Type I error:** Type I errors refer to false positives. For instance, the model predicted that an event would occur but it did not.

**Type II error:** Type II errors refer to false negatives. For instance, the model predicted that an event would not occur but it did.

**Note:** Type I errors are seen as yellow flags and Type II errors are seen red flags. This is due to type II errors being interpreted as worse. For example, we have built a model that predicts whether an earthquake will occur. If the model predicts that an earthquake will occur but it does not (type I error), this will only result in a potential waste of time and resources. For instance, evacuations and shutting down of businesses. However, if the model predicts that it is safe and an earthquake will not happen, but it does (Type II error), this may result in needless loss of life. In this example, it is better to waste resources than there be a huge loss in life to the population.

The **F1 Score** or *Harmonic Mean* captures the tradeoff between Recall and Precision. Optimising F1 will not allow for the corner cases (true positives and true negatives) to have a misleading implication on accuracy if, for instance, everything was to be predicted as positive or negative. The formula is given below:

<p align="center"> <img width="350" src= "/Pics/W18.png"> </p>

## Receiver Operating Characteristic (ROC)

The **Receiver Operating Characteristic (ROC)** plots the true positive rate (sensitivity) of a model against its false positive rate (1-sensitivity).

The diagram below illustrates the ROC being evaluated at all possible thresholds.

<p align="center"> <img width="500" src= "/Pics/W19.png"> </p>

The area under the curve of the ROC plot is a very common method of selecting a classification method. The ROC under the curve is also referred to as **ROC AUC**, which gives a measure of how well we are seperating the two classes. This is illustrated by the diagram below.

<p align="center"> <img width="500" src= "/Pics/W20.png"> </p>

From the diagram above, we state that AUC of 0.9 means high true positives and low false positives.

**Note:** It is better to use the ROC for data with balanced classes.

## Precision-Recall Curve

This curve measures the trade-off between precision and recall. The diagram below illustrates this curve.

**Note:** The curve will usually be decreasing.

<p align="center"> <img width="450" src= "/Pics/W21.png"> </p>

**Note:** It is better to use the precision-recall curve for data with imbalanced classes.

The syntax used for error metrics is as follows:

```
# Importing the desired error function

from sklearn.metrics import accuracy_score

# Calculating the error on the test and predicted data sets

accuracy_value = accuracy_score(y_test, y_pred)

# List of error metrics and diagnostic tools:

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, 
                            confusion_matrix, roc_curve, precision_recall_curve
```

# K-Nearest Neighbours (KNN)

The diagram below illustrates how the KNN algorithm works. In the below example, we have two classes (A and B) and a new point depicted by the red star. If we choose the number of K-Neighbours to be 3, we observe 2 points from class B and only one from class A. Therefore, the algorithm will classify the new point as class B. If we choose the number of K-Neighbours to be 6 however, we observe 4 points belonging to class A and only 2 points belonging to class B, therefore classifying the new point as class A.

<p align="center"> <img width="500" src= "/Pics/KNN1.png"> </p>

### Requirements for the K-Nearest Neighbours Model

- The correct value for 'K' (number of neighbours).
- How to measure the distance or closeness of the neighbours.

The diagrams below show the two extreme cases when choosing the number of neighbours 'K'.

<p align="center"> <img width="1000" src= "/Pics/KNN2.png"> </p>

For K=1, we observe a decision boundary that seperates the dataset into two regions (blue and magenta). These can be thought of as the predictive regions when classifying new data. However, we also notice that the model has overfit; indicating we have traded *Low Bias* for *High Variance*. 

For the other extreme where K=All (K = No. of total points), we observe no decision boundary. This is due to the whole region being classified as the majority class, and as there are more blue points than pink the model will classify any new data to the class represented by the colour blue. This indicates that we have traded *Low Variance* for *High Bias*.

### Choosing the right value for K

- 'K' is a hyperparameter, which means it is not a learnt parameter. We must therefore obtain the correct/optimal number of K-Neigbours from parameter tuning.
- The right value depends on which error metric is most important to us in regards to the business objective. For instance, if we need to capture all the true positives we focus more on recall. If we must ensure all the predicted positives are correct, we may focus more on precision; finding the balance using the F1 score.
- A common approach is to use the *Elbow Method*. This emphasises the kinks in the curve of the error rate as a function of K. Beyond this point, the rate of improvement slows or stops. The main aim therefore being to identify a model with low 'K' and low error rate. The diagram below illustrates this.

<p align="center"> <img width="500" src= "/Pics/KNN3.png"> </p>

### Measuring the distances for K-Nearest Neighbours

When measuring the distances for the KNN model, we have two options. We have the *Euclidean Distance (L2)* and the *Manhattan Distance (L1)*.

The diagram below depicts the Euclidean Distance (L2).

<p align="center"> <img width="500" src= "/Pics/KNN4.png"> </p>

The diagram below depicts the Manhattan Distance (L1).

<p align="center"> <img width="500" src= "/Pics/KNN5.png"> </p>

**Note:** Feature Scaling is now crucial.

### Regression with K-Nearest Neighbours

KNN can also be used for regression. Instead of predicting the class a point belongs to as we learnt in classification, for regression each value predicted is the mean value of its neighbours. So in essence, the KNN works as a smoothing function. Let's examine the diagram below.

<p align="center"> <img width="1000" height="225" src= "/Pics/KNN6.png"> </p>

- In the first case where K=20, we observe a straight line. This is because a single value will be predicted, which is the mean of all the values. This is another case where we have low variance but high bias, indicating underfitting.
- For the second case where K=3, we visually see the KNN behaving like a smoothing function. It can be thought of as a rolling average, where the predicted point will be the average of the closest 3 neighbours. This behaviour can be expected for K where K is greater than 1 and less than the total number of samples. 
- For our third case where K=1, the nearest neighbour will act like the predicton for that value. This is another case where we have low bias but high variance, indicating overfitting.

### Pros of K-Nearest Neighbours

- Simple to implement as it does not require the estimation of any parameters, just which neighbours are closest.
- Adapts well as new training data is introduced for trying to predict one class against the other. As the data changes so will the nearest neighbours.
- Easy to interpret which makes intuitive sense to decision makers, resulting in that model becoming much more powerful in a business setting.

### Cons of K-Nearest Neighbours

- Slow to predict because of the many distance calculations. Therefore, the larger the dataset the more computationally intensive the calculations will be as the number of distance calculations will rise.
- Does not generate insights into the data generation process as it does not give us a model like, for instance, Logistic Regression. For a model like Logistic Regression, it can give us insights such as the contribution of features impacting the likelihood of each point falling into a certain predicted category.
- Can require a lot of memory if the dataset is large (or it grows), as the model needs to store all the values in the training set every time it fits the model.
- When there are many predictors, the KNN accuracy can break down due to the curse of dimensionality. If there are a lot of features, the distances are generally further and further away as we increase the dimensions. Remember, the number of dimensions will increase as we increase the number of features.

### Characteristics of a K-Nearest Neighbour Model Vs Linear/Logistic Regression

**Fitting**

- For Linear Regression, the fitting involves minimising the cost function which is slow.
- For KNN, the fitting involves storing the training data which is fast.

**Memory**

- For Linear Regression, the model has only a few parameters as once it is fit, it only has to remember the coefficients that determined the line or hyperplane in the higher dimensional space, resulting in being memory efficient.
- For KNN, the model has many parameters as it needs to remember the entire training dataset, resulting in being memory intensive.

**Prediction**

- For Linear Regression, predictions are just a simple combination of out vectors, resulting in fast computations.
- For KNN, we must compute several different distances from each of the points, resulting in slow computations.

The syntax used with K-Nearest Neighbours is as follows:

```
# Importing the class containing the classification method

from sklearn.neighbors import KNeighborsClassifier

# Creating an instance of the class

KNN = KNeighborsClassifier(n_neighbors = 5)

# Fitting the instance on the training dataset and predicting the test set value

KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)

Regression can be done with KNeighborsRegressor
```
# Support Vector Machines

Please see the GitHub page given by the below link for an in depth explanation of classification models, especially for SVM Linear and SVM Gaussian Radial Basis Function (RBF):

https://github.com/MohitGoel92/Predicting-Customer-Purchases

In essence, the main idea behind Support Vector Machines is to find a hyperplane that seperates classes by determining decision boundaries that maximise the distance between classes.

### Regularisation in SVMs

The function below is the overall SVM *Cost* function:

<p align="center"> <img width="500" src= "/Pics/SVM1.gif"> </p>

- The first term will be referred to as the *Hinge Loss* function, which is the cost for misclassifications. The greater the number of misclassifications, the higher the Hinge Loss value.
- The second term is the *Regularisation* term, which adds an additional cost for having a more complex decision boundary with higher coefficients, thus preventing overfitting. Therefore, a complex boundary will result in a lower Hinge Loss function, but the impact will be heavily outweighed by the large Regularisation term.
- If we have a more generalised decision boundary, although our Hinge Loss function will be slightly higher due to potential misclassifications, our regularisation term will be much smaller, resulting in a lower overall Cost function.

**Note:** For the Regulariastion term, its impact can be tweeked by the use of the value of 'C', where the smaller the value of 'C' will mean more regularisation (like inverse lambda).

**Note:** When comparing Logistic Regression and SVMs, one of the main differences is that the cost function for Logistic Regression decreases to zero, but rarely reaches zero. On the other hand, SVMs use the Hinge Loss function as the cost function to penalise misclassification. This tends to lead to better accuracy at the cost of having less sensitivity on the predicted probabilities. 

The syntax used with SVM (Linear) is as follows:

```
# Importing the class containing the classification method

from sklearn.svm import LinearSVC

# Creating an instance of the class

LinSVC = LinearSVC(penalty = 'l2', C = 10)

# Fitting the instance on the data and then predicting the expected value

LinSVC = LinSVC.fit((X_train, y_train))
y_pred = LinSVC.predict(X_test)

# Tune regularisation parameters with cross-validation.

# Use LinearSVM for regression
```

**Note:** SVM does not calculate the predicted probabilities in the range between 0 and 1. It outputs labels determined by the decision boundary assigned to the region where an observation belongs.

### Support Vector Machines Kernels

Please see the GitHub page given by the below link for an in depth explanation of classification models, especially for SVM Linear and SVM Gaussian Radial Basis Function (RBF):

https://github.com/MohitGoel92/Predicting-Customer-Purchases

**Note:** By using Gaussian Kernels, we can transform our data space vectors into a different coordinate system, and therefore having a better chance for finding a hyperplane that classifies our data well. SVMs with RBF kernels are slow to train with data sets that are large or have many features.

The syntax used with SVM (Kernel) is as follows:

```
# Imorting the class containing the classification method

from sklearn.svm import SVC

# Creating an instance of the class

rbfSVC = SVC(kernel = 'rbf', gamma=1.0, C=10.0)

# Fitting the instance on the training dataset and predicting the test set values

rbfSVC.fit(X_train, y_train)
y_pred = rbfSVC.predict(X_test)

# We tune the kernel and associated parameters with cross-validation
```

### Machine Learning Workflow

**Problem:** SVMs and RBF Kernels are very slow to train with lots of features or data.

**Data Collection:** The construction of approximate kernal gaps with Stochastic Gradient Descent (SGD) using Nystroem or RBF Sampler. We then fit a much simpler, linear classifier.

The syntax used for faster kernel transformations (Nyostroem) is as follows:

```
# Importing the class containing the classification method

from sklearn.kernel_approximation import Nystroem

# Creating an instance of the class
# Note: Multiple non-linear kernels can be used.
# Note: kernel and gamma are identical to SVC.
# Note: n_components are the number of samples used to come up with our kernel approximation.

NystroemSVC = Nystroem(kernel = 'rbf', gamma = 1.0, n_components = 100)

# Fitting and transforming the instance on the dataset

X_train = NystroemSVC.fit_transform(X_train)
X_test = NystroemSVC.transform(X_test)

# Tune the kernel and associated parameters with cross-validation
```

The syntax used for faster kernel transformations (RBFsampler) is as follows:

```
# Importing the class containing the classification method

from sklearn.kernel_approximation import RBFsampler

# Creating an instance of the class
# Note: RBF is the only kernel that can be used.
# Note: kernel and gamma are identical to SVC.
# Note: n_components are the number of samples used to come up with our kernel approximation.

rbfSampler = RBFsampler(gamma = 1.0, n_components = 100)

# Fitting and transforming the instance on the dataset

X_train = rbfSampler.fit_transform(X_train)
X_test = rbfSampler.transform(X_test)

# Tune the kernel parameters and compnents with cross-validation
```
The below is a short summary of the conditions we should look out for when making model choices.

<p align="center"> <img width="700" src= "/Pics/mlw2.png"> </p>

# Decision Trees

Please see the GitHub page given by the link below for an in depth explanation of classification models, especially for Classification and Regression Trees (CART):

https://github.com/MohitGoel92/Predicting-Customer-Purchases

Please use the GitHub page given by the link below for a further explanation into Entropy (information gain) Vs Gini Index:

https://github.com/MohitGoel92/E-Signing-Loan-Prediction

Let's observe the dataset below:

<p align="center"> <img width="700" src= "/Pics/dt1.png"> </p>

We wish to predict whether customers will play tennis based on the predictive features: Temperature, Humidity, Wind and Outlook. We then segment the data based on features to predict the result. The diagram below illustrates this step.

<p align="center"> <img width="450" src= "/Pics/dt2.png"> </p>

**Note:** Trees that predict categorical results are decision trees. To predict quantities we use regression trees, where the values at the leaves of the regression trees are the averages of all the members.

For example, let's say we wish to use the slope and elevation in the Himalayas to predict the average precipitation, which is a continuous variable. The tree given below illustrates this example.

<p align="center"> <img width="450" src= "/Pics/dt3.png"> </p>

The values at the leaves are the averages of the members that fall within each terminal leaf. 

Graphically, a tree that predicts continuous values will look like the below.

<p align="center"> <img width="550" src= "/Pics/dt4.png"> </p>

In the diagram above, max_depth of 2 is given by the blue line and max_depth of 5 is given by the pink. For max_depth of 2, we will get four different projected values for our predictions. For max_depth of 5 we observe overfitting, therefore finding the optimal balance of the right depth being crucial.

We keep splitting until the leaf node(s) are pure (only one class remains). We may also stop if a maximum depth is reached, or a performance metric is achieved. However, methodically we can use *Greedy Search* to find the best split at each step. The best split is the one that maximises the *Information Gained* from the split.

**Note:** Decision Trees split data using impurity measures. This is the Greedy algorithm and is not based on statistical assumptions.

### Splitting based on Classification Error

The *Classification Error* function is given below:

<p align="center"> <img width="400" src= "/Pics/dt5.gif"> </p>

Let's examine the below tree diagram.

<p align="center"> <img width="350" src= "/Pics/dt5.png"> </p>

- The first node called the *Parent* node has a classification error of: 1 - 8/12 = 1/3.
- The second node called the *Child* node has a classification error of: 1 - 2/4 = 1/2. As 1/2 > 1/3, we conclude that information has been lost instead of gained, but on the smaller proportion of the data point.
- The third node which is also called the Child node has a classification error of: 1 - 6/8 = 1/4. As 1/4 < 1/3, we conclude that information has been gained.
- The overall change according to the weighted average of our two new classification errors (Classification Error Change) is: 1/3 - 4/12(1/2) - 8/12(1/4) = 0. This means we have not had any information gained given our new split.
  - If this was the best split possible, we would claim that we should not split the nodes any further. Therefore, using classification error no further splits would occur.

**Note:** We're forced to stop well short of a point where all of our leaves are homogeneous. In other words, the leaves do not contain all yes or all no in each of the child leads/nodes).

### Entropy-based Splitting

The *Entropy* function is given below:

<p align="center"> <img width="400" src= "/Pics/dt6.png"> </p>

Let's examine the below tree diagram.

<p align="center"> <img width="350" src= "/Pics/dt5.png"> </p>

- For the first node, the Entropy before is: -8/12 log2 (8/12) - 4/12 log2 (4/12) = 0.9183
- For the second node, the Entropy is: -2/4 log2 (2/4) - 2/4 log2 (2/4) = 1.00
- For the third node, the Entropy is: -6/8 log2 (6/8) - 2/8 log2 (2/8) = 0.8113
- The overall Entropy change by weighted average is: 0.9183 - 4/12(1) - 8/12(0.8113) = 0.0441
- We observe that Entropy has decreased, therefore concluding we are able to have information gain when we use entropy. 

**Note:** We can now allow further splits to occur, eventually reaching the goal of homogeneous nodes.

### Classification Error Vs Entropy

The Classifiation Error is a flat function with the maximum at the center. The center represents ambiguity (largest error) which will occur at the 50/50 split. Splitting metrics favour results that are furthest away from the centre.

Entropy has the same maximum but is curved. The curvature allows splitting to continue until the nodes are pure.

The diagram below illustrates this for clarification.

<p align="center"> <img width="400" src= "/Pics/dt7.png"> </p>

### Information Gained by Splitting

With Classification Error, the function is flat. The final average Classification Error can be identical to the parent, resulting in premature stopping. This was observed in the previous example of "Splitting based on Classification Error".

The diagram below depicts this.

<p align="center"> <img width="400" src= "/Pics/dt8.png"> </p>

With Entropy gain, the function has a curve. This allows the average information of children to be less than the parent, resulting in information gain and continued splitting.

The diagram below depicts this.

<p align="center"> <img width="400" src= "/Pics/dt9.png"> </p>

### The Gini Index

In practice, the *Gini* index is often used for splitting. This function is similar to Entropy as it is also curved. However, unlike Entropy it does not contain a logarithm so it is easier to compute. 

The Gini index function is given below:

<p align="center"> <img width="375" src= "/Pics/dt10.png"> </p>

The diagram below illustrates the comparison between the Classification Error, Cross Entropy and Gini Index.

<p align="center"> <img width="450" src= "/Pics/dt112.png"> </p>

**Note:** The most common splitting impurity measures are Entropy and Gini Index.

**Note:** Getting perfect splits will eventually lead to overfitting. Therefore, if we get down to homogeneous nodes we will overfit. We must therefore ensure the right balance between the bias and variance.

### Decision Trees and High Variance

- One of the problems as mentioned above is that Decision Trees tend to add high variance as they tend to overfit. 
- Since Decision Trees don't make many strong assumptions such as linearly seperable classes, they usually find structures that attempt to explain the training set too well. 
- Therefore, small changes in the data greatly affect predictions (high variance) as the model does not generalise outside the current dataset. 
- A solution to this problem is to prune trees. This means we have a preset maximum depth which results in only a certain number of splits being performed.
- We can prune leaves based on the Classification Error threshold. For instance, if a leaf correctly classifies 90% of its samples, we may deem that good enough for us; resulting in no further splitting required.
- We can also prune leaves by deciding a certain threshold of Information Gain. For instance, if we require a certain number of information gain in order to keep splitting.
- In addition, we may have a minimum amount of rows in the subset in which we're no longer allowing further splits.

### Advantages of the Decision Tree Classifier

- As Decision Trees are a sequence of questions and answers, they're easy to interpret and implement. In other words, we are using an "if ... then ... else" logic.
- Easy to understand the visualisation and high interpretability. This is hugely advantageous in a business setting as it is easy to communicate the findings to management. For instance, we can logically explain why certain features within subsets make it more likely for customers to churn.
- The algorithm will easily turn any features into binary features on its own. Therefore being able to handle any data category (binary, ordinal, continuous).
- As opposed to the distance or linear-based algorithms, we require no scaling.

The syntax used for Decision Tree Classifier is as follows:

```
# Importing the class containing the classification method

from sklearn.tree import DecisionTreeClassifier

# Creating an instance of the class

DTC = DecisionTreeClassifier(criterion = 'Gini', max_features = 10, max_depth = 5)

# Fitting the instance on the training set and predicting the test set results

DTC.fit(X_train, y_train)
y_pred = DTC.predict(X_test)

# We tune parameters with cross-validation.
# We use DecisionTreeRegressor for regression.
```

# Ensemble Based Methods and Bagging

**Ensemble Methods:** This technique will create multiple models and then combine them to produce improved results.

Previously, we have discussed how Decision Trees tend to overfit, and how pruning may help to reduce variance to a certain degree. However, this is often not significant enough for a model to generalise well. An improvement would be to use many trees, where we would now combine the predictions of all the trees to reduce variance.

**Aggregate Results:** Trees vote on or average the results for each data point. This is called 'Meta Classification'. 

**Bagging:** Bagging stands for Bootstrap Aggregating. Bagging is an ensemble algorithm that combines the predictions of several models that we've trained on Bootstrap samples of data.

We bootstrap by getting each one of our smaller samples, build out our decision trees and bring together all of those different decision trees of those samples; that's the aggregation step. A model that averages the predictions of multiple models reduces the variance of a single model and has high chances to generalise well when scoring new data. The diagram below clarifies this.

<p align="center"> <img width="450" src= "/Pics/bs1.png"> </p>

Now a question will arise and it is this, "How many trees do we fit?". This will end up being another hyperparameter that we can tune (the number of trees). The greater the number of trees, the more overfit our Decision Trees will be. In practice however, there's a point of diminishing returns which is usually around 50 trees. Let's observe the graph below which shows the RMSE (Cross-Validated) against the Number of Bagged Trees.

<p align="center"> <img width="450" src= "/Pics/dt12.png"> </p>

From the above we observe that bagging performance increases as the number of trees increases. Additionally, we observe at around 50 trees the RMSE plateaus, resulting in diminished returns. We should therefore select the optimal number of trees as the greater the number of trees we have, the greater the computational power that is required.

Similar to Decision Trees, Bagging Trees:
  - Are easy to interpret and implement.
  - Allow heterogeneous input data (different data types), with no preprocessing required.

Unlike Decision Trees, Bagging Trees:
  - Less variability than Decision Trees. We reduce the variance, thus increasing the chances of overfitting once we introduce Bagging.
  - Can grow trees in parallel (each tree is not dependent on any other tree because it's specific to its own dataset).

The syntax used for Bagging is as follows:

```
# Importing the class containing the classification method

from sklearn.ensemble import BaggingClassifier

# Creating an instance of the class

BC = BaggingClassifier(n_estimators = 50)

# Fitting the instance on the training set and predicting the test set results

BC.fit(X_train, y_train)
y_pred = BC.predict(X_test)

# Tune parameters with cross-validation
# Use BaggingRegressor for regression
```

### Reducing Variance Due to Bagging

If our Bagging produced *n* independent trees, each with variance σ-squared (sigma squared), the bagged variance is:

<p align="center"> <img width="30" src= "/Pics/dt13.png"> </p>

Therefore, the larger *n* is (the larger the number of trees we use), assuming they're independent trees, the more we can reduce the overall variance. In reality however, these trees are not independent. As we are sampling with replacement, the trees are very likely correlated.

The equation below explains how the Bootstrap samples are correlated (ρ):

<p align="center"> <img width="225" src= "/Pics/dt14.png"> </p>

From the above, we observe that if the correlation is close to one, we end up with no reduction in variance (as we're using similar trees). As a result, we are not gaining any new information. We therefore must ensure that each one of these Decision Trees are somewhat different than one another.

The solution is to further de-correlate trees by introducing randomness. We must try to make sure that the trees are significantly different from one another and thus decorrelated. This is achieved by restricting the number of features the trees are allowed to be built from. The number of randomly selected features for each tree is:

<p align="center"> <img width="250" src= "/Pics/dt15.png"> </p>

where *m* is the number of randomly selected features.

This resulting algorithm is called ***Random Forest***.

**Random Forest:** A random selection of the subsets of rows and columns of the data, of which are used to produce regression or classification trees.

In general, having extra trees will eventually result in better out-of-sample (new data) accuracy compared to just simple bagging. This is illustrated by the graph below.

<p align="center"> <img width="450" src= "/Pics/dt16.png"> </p>

From the above, we observe that errors are further reduced for Random Forest relative to Bagging. In addition, we grow enough trees until the error settles down (flattens out). When this happens, additional trees will not improve results.

**Note:** The main difference between Random Forest and Bagging is that Random Forest introduces more randomness by using only a subset of features, not only subsets of observation. In general, they tend to have better out of sample accuracy.

The syntax used for Random Forest is as follows:

```
# Importing the class containing the classification method

from sklearn.ensemble import RandomForestClassifier

# Creating an instance of the class

RFC = RandomForestClassifier(n_estimators = 50)

# Fitting the instance on the training set and predicting the test set results

RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_test)

# Tune parameters with cross-validation
# Use RandomForestRegressor for regression
```

But what if Random Forest does not reduce the variance enough? In the cases where Random Forest has overfit, we can introduce even more randomness. Here, we select features randomly and create splits randomly (no fixed number *m* of features to be selected); therefore not choosing greedily. Fittingly, these extra random trees are called ***Extra Random Trees***.

```
# Importing the class containing the classification method

from sklearn.ensemble import ExtraTreeClassifier

# Creating an instance of the class

ETC = ExtraTreeClassifier(n_estimators = 50)

# Fitting the instance on the training set and predicting the test set results

ETC.fit(X_train, y_train)
y_pred = ETC.predict(X_test)

# Tune parameters with cross-validation
# Use ExtraTreeRegressor for regression
```

**Note:** In general, Ensemble models tend to have less overfitting than single models. This is also the case for Bagging when compared to Decision Trees. Therefore concluding that Bagging has less overfitting than Decision Trees.

# Boosting and Stacking

As opposed to Bagging, whose main function is to find a way of reducing variance, Boosting also helps with reducing variance but is meant for business problems where we may want to continue to better fit or correct our model. This enables us to correctly capture rare events. Consequently, unlike Bagging it is easier to overfit the data.

**Note:** For Bagging the trees are independent of each other, but for Boosting the trees depend on the prior, therefore being dependent on each other.

In Bagging, we grow Decision Trees from multiple bootstrapped samples. We vote on the trees or average the results for each data point (aggregate results). For Boosting, we have a weighting of our different trees with different weights.

**Note:** A Decision Tree with just one split is called a *Decision Stump*.

### Boosting Overview

Boosting methods are additive in the sense that they sequentially retrain Decision Trees using the observations with the highest residuals on the previous tree. To do so, observations with a high residual are assigned a higher weight.

The diagram below illustrates how Boosting will work on a small dataset.

<p align="center"> <img width="550" src= "/Pics/bs2.png"> </p>

- Firstly, we create an initial decision stump with one node and two leaves, which results in the splitting of our dataset into two. 
- After the split, we calculate the residuals by attributing a larger weight to the misclassified points, so if the subsequent weak learner misclassifies them again, a greater weight is attributed. 
- On the contrary, we lower the weights of the points that the first weak learner got correct. This is a fundamental aspect in Boosting.

**Note:** The new decision stumps that are produced to be combined as the final classifier are called weak learners. Better classifiers get more weight.

The result is a weighted sum of classifiers, where the successive classifiers are weighted by the *Learning Rate* (λ). This is depicted by the diagram below.

<p align="center"> <img width="550" src= "/Pics/bs3.png"> </p>

**Learning Rate (λ):** 
- The learning rate is the magnitude of allowable correctness of our errors at each one of our steps.
- It is a hyperparameter which decides how much weight we provide to each of our successive decision trees.
- A lower learning rate will mean more trees are desirable, as we are correcting our errors at a very slow pace.
- A higher learning rate will mean an easier overfit by allowing each successive tree to have too much influence on our final decision.
- This goes back to the bias-variance tradeoff where lower variance implies higher bias (less overfitting), and higher variance indicates lower bias (more overfitting).
- A learning rate (λ) < 1 will prevent overfitting (regularisation).

**Note:** The nature of Boosting algorithms tend to produce good results in the presence of outliers and rare events.

### Loss Functions

Boosting utilises different loss functions.

<p align="center"> <img width="450" src= "/Pics/bs4.png"> </p>

- At each stage, or for each of our weak learners, the margin is determined for each point.
- The margin is positive for the correctly classified points and negative for the misclassifications.
- The value of the loss function is calculated from the margin. It can be thought of as the distance from our decision boundary (from the margin). This allows us to heavily penalise faraway points.

**0-1 Loss Function**

- This loss function multiplies the misclassified points by 1, and multiplies the correctly classified points by 0.
- Therefore, correctly classified points are ignored.
- Although this is theoretically the ideal loss function, it is not used in reality. This is because it is difficult to optimise, as it is non-smooth and non-convex.

**AdaBoost Loss Function**

- AdaBoost = Adaptive Boosting.
- The AdaBoost loss function is exponential, it is given by: e^(-margin)
- This makes AdaBoost more sensitive to outliers or further away points, compared to other types of Boosting. If the distance from our margin is large and incorrect, we end up with a large contribution to that overall error for that given point.

**Gradient Boosting Loss Function**

- Gradient Boosting is a generalised Boosting method that can use various loss functions.
- The common implementation uses the binomial log likelihood loss function (deviance), it is given by: log(1 + e^(-margin)).
- This is more robust to outliers than AdaBoost.

The diagram below illustrates the the loss functions discussed above.

<p align="center"> <img width="450" src= "/Pics/bs5.png"> </p>

### Bagging Vs Boosting

The table below summarises the key difference between Bagging and Boosting.

<p align="center"> <img width="550" src= "/Pics/bs7.png"> </p>

**Note:** When tuning a gradient boosted model, we must remember that as Boosting is additive, overfitting is now likely. This is due to the increased trees trying to continually improve on mistakes that were made by prior trees, therefore continually trying to improve off the errors from the past. We must use cross-validation to set the numer of tree. The diagram below compares the Test Set Error for the number of iterations for Boosting and Random Forest.

<p align="center"> <img width="450" src= "/Pics/bs8.png"> </p>

### Tuning a Gradient Boosted Model

The diagram belows compares the Test Set Error for the number of iterations with different parameters.

<p align="center"> <img width="550" src= "/Pics/bs9.png"> </p>

- For the base line, the Learning Rate (λ) is set to less than 1 for regularisation. The lower the Learning Rate, the higher the number of trees we may use as the Learning Rate represents how much we're correcting the model at each step. So these two hyperparameters (Learning Rate and number of trees) are going to be related.
- For λ = 0.1, as λ < 1, this is also called *Shrinkage* as it shrinks the impact of each successive learner.
- Another parameter we can use to add randomness in order to reduce overfitting is the subsample. For subsample < 1, we use a fraction of the dataset for the base learners. This is called *Stochastic Gradient Boosting*. By using a subsample, our base learners don't train on the entire dataset. This alone allows for faster optimisation, as well as a bit of regularisation as it will not perfectly fit to our entire dataset.
- From the graph, we observe that using a combination of the Learning Rate (λ) = 0.1 and subsample = 0.5, we obtain a good result.
- Max_features is the number of features to consider in the base learners when splitting. This reduces the possible complexity of our model, resulting in improving our test set error. When using the combination of λ = 0.1 and max_features = 2, we observe that we have obtained a respectable result.

**Note:** In practice, the performance of the hyperparameter tuning will depend on the dataset. As usual, we may use cross-validation when deciding between each one of the hyperparameters.

The syntax used for Gradient Boosting Classifier is as follows:

```
# Importing the class containing the classification method

from sklearn.ensemble import GradientBoostingClassifier

# Creating an instance of the class
# Note: learning_rate = 0.1 is a usual choice, max_features = 1 is a lower than usual choice (higher bias, lower variance), subsample = 0.5 is a usual choice, and n_estimators = 200 also a usual choice but should be optimised.

GBC = GradientBoostingClassifier(learning_rate = 0.1, max_features = 1, subsample = 0.5, n_estimators = 200)

# Fitting the instance on the training set and predicting the test set results

GBC.fit(X_train, y_train)
y_pred = GBC.predict(X_test)

# Tune parameters with cross-validation
# Use GradientBoostingRegressor for regression
```
**Note:** For Boosting, we require successive trees. As a result, as we have a lot of trees, a lot of time may be taken to fit our model. Time considerations must therefore be taken into account.

The syntax used for Adaptive Boosting classifier is as follows:

```
# Importing the class containing the classification method

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Creating an instance of the class
# Note: The base learner can be set manually, in addition to setting the max depth.

ABC = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), learning_rate = 0.1, n_estimators = 200)

# Fitting the instance on the training set and predicting the test set results

ABC.fit(X_train, y_train)
y_pred = ABC.predict(X_test)

# Tune parameters with cross-validation
# Use AdaBoostRegressor for regression
```

# Stacking: Combining Classifiers

Stacking is an ensemble method that combines any type of model by combining the predicted probabilities of classes. In that sense, it is a generalised case of bagging. The two most common ways to combine the predicted probabilities in stacking are: using a majority vote or using weights for each predicted probability.

The diagram below gives an overview of how Stacking works.

<p align="center"> <img width="550" src= "/Pics/bs10.png"> </p>

- This is similar to Bagging, but we are not limited to Decision Trees. The base learners can be anything, so there is no bias towards needing to use a decision tree, which is an ensemble method in itself.
- The idea is to fit several algorithms to our training set and use their predictions (or scores) of each of the individual base learners as a new training set.
- The output of each of the base learners creates features known as *Meta Features*, that we use to combine with the data, feeding it into the final classifier.
- The final classifier is called the *Meta Classifier*, and it is used to arrive at a single prediction.
- We train several different algorithms, and we think of this as testing many different assumptions on our dataset.
- The output of the base learners can be combined via majority vote or weighted.

**Note:** An additional hold-out and test set is needed for each classifier if meta learner parameters are used.

**Note:** We must be aware of increasing model complexity, or in other words, overfitting which is what we wish to avoid. The final prediction is achieved by voting or with an alternative model.

**Note:** Taking the average of each possibility is called *Soft Voting*. If we have a certain model that is very certain that the data point should be classified as 0 or 1, we will allow that to have more weight.

The syntax used for Voting Classifier is given below. 

**Note:** A Voting Classifier is the type of ensemble that we would use to combine several classifiers.

```
# Importing the class containing the classification method

from sklearn.ensemble import VotingClassifier

# Creating an instance of the class
# Note: This is a list of the fitted models and we are to combine them.

VC = VotingClassifier(estimator_list)

# Fitting the instance on the training set and predicting the test set results

VC.fit(X_train, y_train)
y_pred = VC.predict(X_test)

# Use VotingRegressor for regression.
# The StackingClassifier (for StackingRegressor) works similarly:

SC = StackingClassifier(estimator_list, final_estimator = LogisticRegression())
```
**Note:** XGBoost is another popular Boosting algorithm, but is not in scikit-learn. This involves essential gradient Boosting with a little parallelisation which allows us to speed up how fast we are able to fit our model.

# Unbalanced Classes

Classifiers are usually built to optimise accuracy and hence often perform poorly on unbalanced classes. For unbalanced datasets, we can balance the size of the classes by either downsampling the majority class or upsampling the minority class.

With unbalanced classes, the data often isn't easily seperable. We must choose to make sacrifices to one class or the other.

For instance, for every minority class data point identified as such, we may wrongly label a few of the majority class points as the minority class. By upsampling or downsampling, we may add more weight to the features relating to the minority class as a proportion of the full dataset. This may result in wrongly labelling a few majority class points.

**Note:** As recall goes up, precision is likely to go down.

**Downsample:** We randomly select the same number of samples of the majority class that exist in the minority class. This is demonstrated by the diagram below.

<p align="center"> <img width="450" src= "/Pics/uc1.png"> </p>

Downsampling adds tremendous importance to the minor class, typically shooting up the recall and bringing down precision. Values like 0.8 recall and 0.15 precision are not uncommon. Increasing the ability of our model to correctly predict the minority class will be at the cost of losing a lot of valuable data that may help us in predicting the majority class.

**Upsample:** We create copies of the minority class until we have a balanced sample. This is demonstrated by the diagram below.

<p align="center"> <img width="450" src= "/Pics/uc2.png"> </p>

Upsampling mitigates some of the excessive weight on the minor class. Recall is still typically higher than precision, but the gap is lesser. Values like 0.7 recall and 0.4 precision are not uncommon, and are often considered good results for an unbalanced dataset. The downside here is that we are then fitting to duplications of the same data in the minority class, thus giving more weight to and overfitting to these repeated rows.

**Resample:** We set a number of which each of the minority and majorty class should contain. We then increase the minority set and decrease the majority set appropriately until we achieve balanced classes. This is demonstrated by the diagram below.

<p align="center"> <img width="450" src= "/Pics/uc3.png"> </p>

Cross-validation works for any global model-making choice, including sampling. It is not only used for finding optimal hyperparameters of the model.

Below is a digram of the ROC curve comparing different samples sizes. The sample size 's' being the number of rows for both the minority and majority class.

<p align="center"> <img width="400" src= "/Pics/uc4.png"> </p>

From the above, we observe that a higher value for our sampling technique is better.

**Notes:**

- Every classifier used produces a different model.
- Every dataset we use that is produced by various sampling will produce a different model.
- We can choose the best model using any criteria including AUC (Area Under Curve). We must remember that each model produces a different ROC curve.
- Once a model is chosen, you can walk along the ROC curve and pick any point on it. Each point has different precision/recall values.

### Steps for Unbalanced Datasets

- At the step where we usually split the dataset into the training set and test set, use the *StratifiedShuffleSplit*.
- Perform upsampling, downsampling or resampling to the dataset.

**Note:** Perform the split before we upsample or downsample as, if we upsample before the split, we may have duplicate observations in the training and test sets.

- Build the model.

### Modelling Approaches

- General sklearn approaches. By default, the hyperparameter called class weight is usually set to none, however it can also be set to balance. This string helps to balance out the error attributed to our minority and majority classes.
- Oversampling the minority class.
- Undersampling the majority class.
- A combination of oversampling and undersampling.
- Ensemble methods to leverage the oversampling or undersampling techniques to ensure balance between each one of the classes.

### Weighting

- Many models allow weighted observations.
- We adjust these total weights so they are equal across classes.
- It is easy to do when available.
- There is no need to sacrifice data such as with undersampling or oversampling.

### Random and Synthetic Oversampling

**Random Oversampling:**

- Simplest oversampling approach.
- Resample with replacement from minority class.
- No concerns about geometry of feature space.
- Good for categorical data.

**Synthetic Oversampling:**

- We create new samples of the minority class that do not exist yet.
- Start with a point in the minority class.
- Choose one of k-nearest neighbours.
- Add a new point between them. Repeat this for each one of the number of neighbours that we have set out. If we have k-neighbours when the number of neighbours is 3, we do this for all three of the points.
- There are 2 main approaches:
  - SMOTE
  - ADASYN

### SMOTE: Synthetic Minority Oversampling Technique

**Regular:** Connect minority class points to any neighbour (even other classes). We generate the new points between the connected points on the connected lines.

**Borderline:** Classify points as outliers, safe, or in-danger.
  - Outliers: Neighbours that are from a different class.
  - Safe: All neighbours are from the same class.
  - In-danger: At least half of the nearest neighbours are from the same class but they're all not from the same class (2 out of 3 are from the same class).

**Borderline 1 SMOTE:** Connect the minority of in-danger points only to minority points.

**Borderline 2 SMOTE:** Connect the minority in-danger points to whatever is nearby.

**SVM SMOTE:** Use minority support vectors to generate new points.

For both the Borderline and SVM SMOTE, a neighbourhood is defined using the parameters and neighbours to decide the number of neighbours to use and to decide whether a sample is in-danger, whether it is safe or whether it in an outlier.

### ADASYN: ADAptive SYNthetic sampling

For each minority point:

- Look at the classes in the neighbourhood of each minority point.
- Generate new samples proportional to the competing classes.

Therefore with ADASYN, more samples will be generated in the area that the nearest neighbour rule is not respected, thus putting more weight on values that would have been originally misclassified.

**Note:** All of these are motivated by K-Nearest Neighbours, but these oversampling techniques will help with any classification for which balance is an issue.

### Nearest Neighbour Methods

**Undersampling:** Decreasing the size of the majority class so that it is similar in size to the minority class.

**NearMiss-1:**

- We keep the points that are closest to the nearby minority points.
- We keep points that are near the decision boundaries.

The diagrams below illustrate this.

<p align="center"> <img width="850" src= "/Pics/uc6.png"> </p>

**Note:** This type of downsampling can easily be skewed by the presence of some outliers or some noise. This may result in decision boundaries that we do not want optimally as it may be easily thrown off by outliers.

**NearMiss-2:**

<p align="center"> <img width="850" src= "/Pics/uc6.png"> </p>
