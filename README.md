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

**Bagging:** Bagging stands for Bootstrap Aggregating. Bagging is a tree ensemble that combines the predictions of several trees that we've trained on Bootstrap samples of data.

We bootstrap by getting each one of our smaller samples, build out our decision trees and bring together all of those different decision trees of those samples; that's the aggregation step. A model that averages the predictions of multiple models reduces the variance of a single model and has high chances to generalise well when scoring new data.

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

**Note:** In general, Ensemble models tend to have less overfitting than single models. This is also the case for Bagging, compared to Decision Trees. For instance, Bagging has less overfitting than Decision Trees.
