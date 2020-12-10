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
- Random Forests
- Boosting
- Ensemble Models

## Logistic Regression

Logistic Regression is a type of regression that models the probability of a certain class occuring given other independent variables. It uses a logistic or logit function to model a dependent variable. It is a very common predictive model because of its high interpretability.

The diagram below is of a Sigmoid function, which is used for Logistic regression.

**Note:** Logistic regression is closely related to Linear regression.

<p align="center"> <img width="600" src= "/Pics/W11.png"> </p>

The derivation from the *Logistic Function* to the *Odds Ratio* given below highlights the relationship between the Logistic and Linear regression. Notice that the coefficient of the exponential is a linear line (in the form of Y = mX + c, where m and c are learnt coefficients).

<p align="center"> <img width="600" src= "/Pics/W12.gif"> </p>

**Note:** 
- P(x) is the probability of a particular observation x belonging to that class.
- The log odds demonstrates how a unit increase or decrease in our x-values will change our log odds in a linear fashion, according to the coefficient beta1 we have learnt.

The syntax used to logistic regression is as follows:

```
# Importing the class containing the classification method
from sklearn.linear_model import LogisticRegression

# Create an instance of the class
# The l2 penalty refers to the regularisation and c is the regularisation, also known as inverse lambda. 
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

A confusion matrix tabulates true positives, false negatives, false positives and true negatives. The below is a diagrammatic representation of a confusion matrix.

<p align="center"> <img width="500" src= "/Pics/W13.png"> </p>

Accuracy is defined as the ratio of true positives and true negatives divided by the total number of observations. It is a measure related to correctly predicting positive and negatives instances. The formula is given below.

<p align="center"> <img width="400" src= "/Pics/W14.png"> </p>

Recall or sensitivity identifies the ratio of true positives divided by the total number of actual positives. It quantifies the percentage of positive instances correctly identified. In other words, for all the positive cases how many did the model predict correctly. The formula is given below.

<p align="center"> <img width="400" src= "/Pics/W15.png"> </p>

**Note:** Although this portrays the overall model accuracy, this can be thrown off or misleading by heavily skewed data.

Precision is the ratio of true positives divided by the total of predicted positives. In other words, for all the positive predictions how many did the model get correct. The closer this value is to 1.0, the better job the model does at identifying only positive instances. The formula is given below.

<p align="center"> <img width="280" src= "/Pics/W16.png"> </p>

Specificity is the ratio of true negatives divided by the total number of actual negatives. In other words, for all the negative cases how many did the model predict correctly. The closer this value is to 1.0, the better job this model does at avoiding false alarms. The formula is given below.

<p align="center"> <img width="280" src= "/Pics/W17.png"> </p>

**Type I error:** Type I errors refer to false positives. For instance, the model predicted that an event would occur but it did not.

**Type II error:** Type II errors refer to false negatives. For instance, the model predicted that an event would not occur but it did.

**Note:** Type I errors are seen as yellow flags and Type II errors are seen red flags. This is due to type II errors being interpreted as worse. For example, we have built a model that predicts whether an earthquake will occur. If the model predicts that an earthquake will occur but it does not (type I error), this will only result in a potential waste of time and resources. For instance, evacuations and shutting down of businesses. However, if the model predicts that it is safe and an earthquake will not happen, but it does (Type II error), this may result in needless loss of life. In this example, it is better to waste resources than there be a huge loss in life to the population.

The F1 Score or *Harmonic Mean* captures the tradeoff between Recall and Precision. Optimising F1 will not allow for the corner cases (true positives and true negatives) to have a misleading implication on accuracy if, for instance, everything was to be predicted as positive or negative. The formula is given below.

<p align="center"> <img width="350" src= "/Pics/W18.png"> </p>

## Receiver Operating Characteristic (ROC)

The receiver operating characteristic (ROC) plots the true positive rate (sensitivity) of a model against its false positive rate (1-sensitivity).

The diagram below illustrates the ROC being evaluated at all possible thresholds.

<p align="center"> <img width="500" src= "/Pics/W19.png"> </p>
