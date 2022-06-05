# Credit Risk Report 

## Overview of the Analysis

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this analysis, I'll use various techniques to train and evaluate models with imbalanced classes. I’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

I'll use the lending data to predict the credit risk (**healthy** or **high-risk** loan). 

First, I'll split the data into targets (`loan_status`), features (other columns besides `loan_status`) and the data into `training`, `testing` set by using `train_test_split`. After, I'll use `value_counts` to check the balance of the target value, which is 75036 for **healthy loan** and 2500 for **high-risk** loan. 

Next, I'll use `LogisticRegression` module to instantiate the Logistic Regression model and fit the model using `X_train` and `y_train`. Make a prediction using `X_test`. Save the prediction on the `X_test`. Calculate the `accuracy score`, generate the `test_matrix`, and print `classification_report` by using `y_test` and `testing_prediction`. 

Lastly, I'll use `RandomOverSampler` module to instantiate the random oversampler model. Fit the original training data to the `random_oversampler` model by using `fit_resample`. The distince value of resampled data will both be 56271. Use `LogisticRegression` classifier and the resampled data to fit the model and make predictions. Evaluate and model performance as steps mentioned above. 


## Results

* Machine Learning Model 1:
  * The `balanced_accuracy_score` is 95%.
  * The `test_matrix` is:
        [18663,   102],
        [56,   563]
  * The `classification_report` is:

![<1>](<Image/1.png>)


* Machine Learning Model 2:
  * The `balanced_accuracy_score` is 99%
  * The `test_matrix` is:
        [18649,   116],
        [4,   615]
  * The `classification_report` is:

![<2>](</Image/2.png>)


## Summary

The precision and recall for 0 (healthy loan) which both are 1.0 is better than that for the 1 (high-risk loan) class which is 0.85 and 0.88. Overall, the regression model predict both labels pretty well in Module 1. 

The precision and F1 did not change with oversampled data (both remained at 1.0) for 0. The precision is getting worse by 1% but F1 is getting better by 3% for 1. Overall, the oversampled data did not make a big difference in Moduel 2. 

However, the accuracy score is higher in Module 2 with oversampled data. 


## Contributor
Feier Ou
ffeierou1003@gmail.com