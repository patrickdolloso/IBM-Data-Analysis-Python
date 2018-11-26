# Week 3: Exploratory Data Analysis
## Video 3.1: Exploratory Data Analysis
### EDA
* Preliminary Step in Data Analysis to:
    * Supparize main characteristics of data
    * Gain better understanding of the data set
    * Uncover relationships between variables
    * Extract important variables
* Question ex: what are the most important characteristics of car price?
### Learning Objectives
* Descriptive Statistics
* GroupBy
* ANOVA
* Correlation
* Correlation - Statistics

## Video 3.2: Descriptive Statistics
### Descriptive Statistics
* Describe basic features of data
* Giving short summaries about the sample and measures of the data
```py
import pandas as pd

# Summarize statistics using pandas .describe() method
df.describe()

# summarize categorical data using the value_counts method
drive_wheels_counts = df["drive-wheels"].values_counts()

drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'}, inplace=True)
drive_wheelscounts.index.name='drive-wheels'
```
### Box plots
![1](./1.png)

```py
# box plot example
sns.boxplot(x="drive-wheels",y="prices",data=df)
```
![2](./2.png)
### Scatter Plot
* Ea. observation represented as a point
* Scatter plots show the relationship between 2 variables:
    1. Predictor/independent variables on x-axis
    2. Targer/dependent variables on y-axis
```py
x=df["price"]
y=df["engine-size"]
plt.scatter(x,y)

plt.title("Scatterplot of Engine Size vs Price")
plt.xlable("Engine Size")
plt.xlable("Price")
```
![3](./3.png)
<!-- Next video -->
## Video 3.3: GroupBy in Python
### Groupung Data
* Use Panda dataframe.Groupby()
    * Can be applied on categorical variables
    * Group data into categories
    * Single or multiple variables
```py
df_test = df['drive-wheels', 'body-style', 'price']

df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()

df_grp
```
![4](./4.png)

### Pandas method - Pivot()
* One variable displayed along the columns and the other variable displayed along the rows
```py
df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
```
![5](./5.png)
### Heatmap
* Plot target variable over multiple variables
```py
plt.pcolor(df_pivot, cmap='RdBBu')
plt.colorbar()
plt.show()
```
![6](./6.png)
## Video 3.4: Analysis of Variance (ANOVA)
### ANOVA
* Statistical comparison of groups
![7](./7.png)
* Why do we perform ANOVA?
    * Finding correlation between defferent groups of a categorical variable
* What we obtain from ANOVA?
    * F-test score: variation between sample group means divided by variation within sample group
    * p-value: confidence degree
### F-test
* Small F - imply poor correlation between variable categories and target variable  
![](./8.png)
* Large F - imply strong correlation between variable categories and target variable
![](./9.png)
* Anova between "Honda" and "Subary"

```py
df_anova=df[["make","price"]]
grouped_anova=df_anova.groupb(["make"])

anova_results_1=stats.f_oneway(grouped_anova.get_group("honda")["price"],grouped_anova.get_group("subaru")["price"])
>>> ANOVA results:
F=0.19744031275, p=F_onewayResult(statistic=0.1974430127), pvalue=0.660947824
```
## Video 3.4: Correlation
### What is Correlation?
* Measures to what extent different variables are interdependent
* For example:
    * Lung cancer > Smoking
    * Rain > umbrella
* Does't imply causation
### Positive Linear Relationship
* Between 2 features (engine-size and price)
```py
sns.regplot(x="engin-size", y="prices", data=df)
Plt.ylim(0,)
```
### Pearson Correlation
* Measure the strength of the correlation between 2 features
    * Correlation coefficient
    * P-value
* Pearson Correlation coefficient
    * Close to +1: Large positive relationship
    * Close to -1: Large Negative relationship
    * Close to 0: No relationship
* P-value
    * P < 0.001 Strong certainty in result
    * P < 0.05 Moderate certainty
    * P < 0.01 Weak certainty
    * P > 0.1 No certainty
### Example
```py
Pearson_coef,p_value=stats.personr[['horsepower'],df['price']]

>>> Pearson corr: 0.81, P-val: 9.35e-48
```