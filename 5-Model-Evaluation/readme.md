# Week 5: Model Evaluation and Refinement
## Video 5.1: Model Evaluation and Refinement
### Model Evaluation
* In-sample evaluation tells us how well our model will fit the data used to train it
* Problem? -It does not tell us how well the trained model can be used to predict new data
* Solution?
    * In-sample data or training data
    * Out-of-sample evaluation or test set

### Training/Testing Sets
* Split the dataset into:
    * Training set: 70%
    * Testing Set: 30%
* Build and train the model with a training set
* Use testing set to assess the performance of a predictive model
* When we have completed testing our model, we should use all the data to train the model to get the best performance
### Function train_test_split()
```py
# Split data into random train and test subjects
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.3,random_state=0)

'''
x_data: features or independent variables

y_data: dataset target: df['price']

x_train,y_train: parts of available data as training set

x_test,y_test: parts of available data as testing set

random_state: number generator used for random sampling
'''
```

### Generalizaiton Performance
* Generalization Error is the measure of how well our data does at predicting previously unseen data
* The error we obtain using our testing data is an approximation of this error

### Cross Validation
* Most common out-of-sample evaluation metrics
* More Effective use of data (ea. observation is used for both training and testing)
### Function cross_val_score()
* Returns an array of R2 scores
```py
# Function cross_val_score()
from sklearn/model_selection import cross_val_score

# data is split into 3 folds and R2 is calulated
scores = cross_val_score(Ir, x_data, y_data, cv=3)

np.mean(scores)
```
### Function cross_val_predict()
* It returns the prediction that was obtained for ea. element when it was in the test set
* Has a similar interface to cross_val_score()
* The input parameters are exactly the same as the cross_val_score function, but the output is a prediction.
```py
from sklearn.model_selection import cross_val_predict

yhat=cross_val_predict(lr2e,x_data,y_data,cv=3)
```
## Video 5.2: Overfitting, Unverfitting, and model Selection
### Model Selection
$$y(x)+noise$$
```py
# calculate different R-squared values

Rsqu_test=[]
order=[1,2,3,4]

for n in order:
    pr=PolynomialFeatures(degree=n)
    x_train_pr=pr.fit_transform(x_train[['horsepower']])
    x_tesr_pr=pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr,y_train)
    Rsqu_test.append(lr.score(x_test_pr,y_test))
```
## Video 5.3: Ridge Regression - preventing overfitting
```py
from sklearn.linear_model import Ridge
RigeModel=Ridge(alpha=0.1)

RigeModel.fit(X,y)

Yhat=RigeModel.predict(X)
```
![](./1.png)
* Select the alpha value that maximizes R2
## Video 5.4: Grid Search
* allows to scan through multiple free parameters with few lines of code
### Hyperparameters
* In the last section, the term alpha in Ridge regression is called a hyperparameter
* Scikit-learn has a means of automatically iterating over these hyperparameters using cross-validation called Grid-Search
```py
# in python, a dictionary with the key 'alpha' is created, with the values of the dictionary being the different values of the free parameter
parameters = [{'alpha'}:[1,10,100,1000]]
```
![](./2.png)
```py
# import packages
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

parameters1=[{'alpha':[0.001,0.1,1,10,100,1000,10000,100000,100000]}]

# create ridge regression model
RR=Ridge()

# create Grid object and input parametsers into GridSearch function, R2 is the default scoring method
Grid1=GridSearchCV(RR,parameters1,cv=4)

# Fit the object with the data
Grid1.fit(x_data[['horsepower','curb-weight','engine-size','highway-mpg']],y_data)

#find the best values for the free parameters using the attribute best_estimator_
Grid1.best_estimator_

# get information like mean score on validation data using the attribure cv_results_
scores = Grid1.cv_results_
scores['mean_test_score']

#output:
>>> array([0.6654,0.6655,0.666,0.669,0.673,0.658,0.658])
```
* RR has the option to normailize the data:

```py
parameters=[{'alpha':[1,10,100,1000], 'normalize':[True, False]}]
```
![](./3.png)  

![](./4.png)

```py
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

parameters2=[{'alpha':[0.001,0.1,1,10,100], 'normalize':[True, False]}]

RR=Ridge()

Grid1=GridSearchCV(RR,parameters2,cv=4)

Grid1.fit(x_data[['horsepower','curb-weight','engine-size','highway-mpg']],ydata)

Grid1.best_estimator_

scores=Grid1.cv_results
```
* print out the scores:

```py
for param,mean_val_mean_test in zip(scores['params'],scores['mean_test_score'],scores['mean_train_score']):
    print(param,"R^2 on test data: ", mean_val, " R^2 on train data: ",mean_test)
```