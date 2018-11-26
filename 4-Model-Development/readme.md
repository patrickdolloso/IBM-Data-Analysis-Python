# Week 4: Model Development
## Video 4.1 Model Development
### Learning Objectives
* Simple and Multiple Linear Regression
* Model Evaluation using Visualization
* Polynomial Regression and Pipelines
* R-squared and MSE for In-Sample Evaluation
* Prediction and Decision Making
* Question: how can you determine a fair value for a used car?

### Model Development
* A model can be thought of as a mathematical equation used to predict a value given one or more other values
* Relating one or more independent variables to dependent variables
* Usually the more relevant data you have, the more accurate your model is
* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression

## Video 4.2: Linear Regression and Multiple Linear Regression
### Introduction
* Linear regression will refer to one independent variable to make a prediction
* Multiple linear regression will refer to multiple independent variables to make a prediction
### Simple linear regression
1. The predictor (independent) variable - x
2. The target (dependent) variable - y
$$y = b_0 + b_1 x$$
$$b_0: intercept$$
$$b_1: slope$$

### Fitting a Simple Linear Model Estimator
* X: Predictor variable
* Y: Target variable

1. Import linear_model from scikit-learn

```py
from sklearn.linear_model import LinearRegression
```
2. Create a linear regression object using the constructor:

```py
lm=LinearRegression()
```
* We define the predictor variable and target variable
```py
X = df[['highway-mpg']]
Y = df['price']
```
* Then use lm.fit(X, Y) to fit the model, i.e. fine the parameters b0 and b1
```py
lm.fit(X, Y)
```
* We can obtain a prediciton
```py
Yhat=lm.preditc(X)
```
### SLR - Estimated Linear Model
```py
# We can view the intercept (b0):
lm.intercept_
>>> 38423.30585

# We can also view the slope
lm.coef_
>>> -821.73337832

# the relationship b/w price and highway mpg:
Price = 38423.31 - 821.73 * highway-mpg
```

### Multiple Linear Regression (MLR)
* This method is used to explain the relationship between:
* One continuous target (Y) variable
* 2+ predictor (X) variables

$$Y^*=b_0 + b_1 x_1 + b_2 x_2...$$
$$b_0:intercept$$
$$b_1: coeff x_1$$

### Fitting a Multiple Linear Model Estimator
1. We can extract 4 predictor variables and store them in the variable z
```py
z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
```
2. Then Train the model as before
```py
lm.fit(Z, df['price'])
```
3. We can also obtain a prediction
```py
Yhat=lm.predict(X)
```
### MLR - Estimated Linear Model
1. Find the intercept (b0)
```py
lm.intercept_
>>> -15678.74
```
2. Find the coefficients (b1,b2,b3,...)
```py
lm.coef_
>>> array([52.66,4.70,81.96,33.58])
```
3. The Estimated Linear Model
```py
Price = -15678.74 + 52.66*horsepower + 4.7*curb_weight + 81.96*engine_size + 33.58*highway-mpg
```
## Video 4.3: Model Evaluation using Visualization
### Regression plot
* Gives us a good estimate of:
    * The relationship between 2 variables
    * The strength of the correlation
    * The direction of the relationship (positive or negative)
* Regression Plot shows us a combination of:
    * The scatterplot: where ea. point represents a different y
    * The fitted linear regression line y_hat > represents the predicted value

![1](./1.png)
```py
# plot regression using seaborn
import seaborn as sns

sns.redplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
```
### Residual Plot
![2](./2.png)
* Variance = Predicted - Actual
* We expect no curvature, and evenly spread above and below the mean
* Look at the **spread of the residuals**
    * Randomly spread out around x-axis then a linear model is appropriate
* For example, in the below graph, the error values change with x

![3](./3.png)
* Not randomly spread out around x-axis
* Therefore, a non-linear model may be more appropriate
* In the below graph, the variance increases with x
* model is incorrect
* Not randomly spread out around the axis
* Variance appears to change with x-axis

![](./4.png)

```py
# use seaborn to produce residual plot
import seaborn as sns

# 1st parameter is dependent variable/target
# 2nd is target
sns.residplot(df['highway-mpg'], df['price'])
```

### Distribution Plots
* Counts the predicted values vs the actual values
* Useful for visualizing models with more than one independent variable or feature

![](./5.png)
* Compare the distribution plots:
    * the fitted values that result from the model (blue)
    * The actual values (red)   

![](./6.png)

MLR - Distribution Plots
```py
import seaborn as sns

axl = sns.distplot(df['price'], hist=False. color="r", label="Actual Value")

sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=axl)
```
## Video 4.4: Polynomial Regression and Pipelines
### Polynomial Regression and Pipelines
* When linear models are not the best fit for the data
* Pipelines simplify code
### Polynomial Regressions
* A special case of the general linear regression model
* Useful for describing curvilinear relationships

![](./7.png)
### Curvilinear relationships:
* By Squaring or setting higher-order terms of the predictor variables

![](./8.png)
* Quadratic - 2nd order
$$Y^*=b_0 + b_1 x_2 + b_2 x_1^2$$
* Cubic - 3rd order

$$Y^*=b_0 + b_1 x_2 + b_2 x_1^2+b_3 x_1^3$$
* Higher Order
$$Y^*=b_0 + b_1 x_2 + b_2 x_1^2+b_3 x_1^3...$$

![](./9.png)

```py
# 1. calculate polynomial of 3rd order
import numpy as np
f=np.polyfit(x,y,3)

# 2. We can print out the model
print(p)
```
### Polynomial Regression with more than 1 dimension
* Sometimes, we can have multi-dimensional polynomial linear regression unable to be processed by numpy's polyfit function

![](./10.png)

* The "preprocessing" library in scikit-learn

```py
from sklearn.preprocessing import PolynomialFeatures

# takes the degree of polynomial as the parameter (this case -create a 2nd-order polynomial transform object)
pr=PolynomialFeatures(degree=2)

x_polly=pr.fit_transform(x[['horsepower','curb-weight']], include_bias=False)
```
![](./11.png)

### Pre-processing
* As the dimension of the data gets larger, we may want to normalize multiple features in scikit-learn. Instead we can use the preprocessing module to simplify many tasks.
* Ex. Normalize ea. feature simultaneously
```py
from sklearn.preprocessing import StandardScaler

SCALE=StandardScaler()
SCALE.fit(x_data[['horsepower','highway-mpg']])

x_scale=SCALE.transform(x_data[['horsepower','highway-mpg']])
```

### Pipelines
* There are many steps to getting a prediction
* Normalization > Polynomial Transform > Linear Regression

```py
#  1. First we import all the modules we need:
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 2. Then import library pipeline
from sklearn.pipeline import Pipeline

# 3. Create a list of tuples (1st element in touple contains the name of the estimator model, 2nd is model constructor):
Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('mode',LinearRegression())]

# 4. input the list in the pipeline constructor
pipe=Pipeline(Input)

# 5. We can train the pipeline object
Pipe.train(X['horsepower','curb-weight','engine-size','highway-mpg'], y)

# 6. Produce a prediction
yhat=Pipe.predict(X[['horsepower','curb-weight','engine-size','highway-mpg']])
```

![](./12.png)

## Video 4.5: Measures for In-Sample Evaluation
* A way to numerically determine how good the model fits on dataset
* 2 important measures to determine the fit of a model:
    * Mean Squared Errpr (MSE)
    * R-squared (R^2)
### MSE Mean Squared Error
```py
# in python library scikit-learn, we can measure the MSE as follows:
from sklearn.metrixs import mean_squared_error

mean_squared_error(df['price'],Y_predict_simple_fit)

>>> 3163502.944639888
```

### R-squared (Coef of Determination)
* The coefficient of Determination or R-squared(R^2)
* Is a measure to determine how close the data is to the fitted regression line
* R^2: the percentage of variation of the target variable (Y) that is explained by the linear model
* Think about as comparing a regression model to a simple model i.e. the mean of the data points.

![](./13.png)

### R^2
* The blue line represents the regression line
* The blue squares represents the MSE of the regression line
* The red line represents the average value of the data points
* The red squares represent the MSE of the red line
* We see the area of the blue squares is much smaller than the area of the red squares 
* This case, the ratio of the areas of the MSE is close to zero (small)

![](./15.png)

![](./16.png)
* Generally the values of the MSE are between 0 and 1
* We can calculate the R^2 as follows:
```py
X = df[['highway-mpg']]
Y=df['price']
lm.fit(X, Y)

lm.score(X,Y)
>>> 0.496591188
```
## Video 4.6: Prediction and Decision Making
### Decision Making: Determining a Good Model Fit
* To determine final best fit, we look at a combination of:
    * Do the predicted values make sense
    * Visualization
    * Numerical measures for evaluation
    * Comparing Models
### Do the predicted values make sense?
```py
# 1. Train the model
lm.fit(df['highway-mpg'],df['prices'])

# 2. predict the price of a car with 30 highway-mpg
lm.predict(30)

# 3. Result
>>> $13771.30

# 4. look at coefficients
lm.coef_
>>> -821.73337832
```
* Generate a sequence of predictions:

```py
import numpy as np

# use the arrange numpy function to generate a sequence from 1 to 100
new_input=np.arange(1,101,1).reshape(-1,1)

# output predicted new values
yhat=lm.predict(new_input)
```

### Visualization
* Simply visualizing your data with a regression is the first method you should try

### Comparing MLR and SLR
1. Is a lower MSE always implying a better fit? -not necessarily
2. MSE for an MLR model will be smaller than the MSE for an SLR model, since the errors of the data will decrease when more variables are included in the model
3. Polynomial regression will also have a smaller MSE then regular regression
4. A similar inverse relationship holds for R^2