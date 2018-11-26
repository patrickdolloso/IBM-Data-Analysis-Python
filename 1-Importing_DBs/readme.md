# Week 1: Importing Datasets
## Learning Objectives
* Understanding the Data
* Importing and Exporting Data in Python
* Detting Started Analyzing Data in Python
* Python Packages for Data Science
## Video 1.1: The Problem
### Why Data Analysis?
* Data is everywhere
* Data is not information
* Data analysis/data science helps us answer questions from data
* Plays an important role in:
    * Discoverint useful information
    * answering questions
    * Predicting future or the unknown
### Estimating used car prices
* How can we help tom determine the best price for his car?
* Is there data on the prices of other cars and their characteristics?
* What features of cars affect their prices?
    * Color? Brand? Horsepower? Something Else?
* Asking the right questions in terms of data.
## Video 1.2: Understanding the Data
### Each of the attributes in the dataset
* Cars are initially assigned a risk factor symbol associated with their price
* the name of the attribute what we want to predict is the target
### Python packages for Data Science
* Scientific computing libraries
    *  Pandas: Data structures and tools, dataframe, easy
    *  NumPy: Arrays & matrices
    *  SciPy: Integrals, solving differential equations, optimization
* Visualization Libraries:
    * Matploylib: plots, graphs - most popular
    * Seaborn: heat maps, time series, violin plots
* Algorithmic Libraries in Python
    * Scikit-learn: Machine Learning, regression, classification
    * Statsmodels: Explore data, estimate statistical models, and perform statistical tests
* A Python library is a collection of functions and methods that allow you to perform lots of actions without writing your code
## Video 1.3: Importing and Exporting Data in Python
### Importing Data
* Process of loading and reading data into Python from various resources
* Two format properties:
    * Format: .csv, .json, .xlsx, .hdf
    * File path:
        * computer: ./myfolder/myfile.csv
        * Internet: https://archive.ics.edu/imports/data.csv
* Importing a csv into Python
```py
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-database/autos/imports-85.data"

df = pd.read_csv(url)
```
or
```py
import pandas as pd
# read the online file by the URL provides above, and assign it to variable "df"
path="https://archive.ics.uci.edu/ml/machine-learning-database/autos/imports-85.data"

df = read_csv(path,header=None)
```
### Printing the dataframe in Python
* df prints the entire dataframe (not recommended for large databases)
* df.head(n) to show the first n rows of dataframe
* df.tail(n) shows the bottom n rows of dataframe
### Adding headers
* replace default header by df.columns = headers
```py
headers = ["Header 1","Header 2","Header 3"]

df.columns = headers
```
### Exporting a Pandas dataframe to CSV
```py
# preserve progress anytime by saving modified dataset using

path="c:windows\..\file.csv"

df.to_csv(path)
```
### Exporting to different formats in Python
```py
# to read
pd.read_csv()

# to save
df.to_csv()

# many file types, replace "csv" to:

json, excel, sqk

```
## Video 1.4: Getting Started Analyzing Data in Python
### Basic insights from the data
* Understand your data before you begin any analysis
* Should check:
    * Data types
    * Data distribution
* locate potential issues with the data
### Basic Insights of Dataset - Data types
![1](./1.png)
* Why check data types?
* pandas automatically assigns data
* therefore we need to check for potential info and type mismatch

### Basic insighs of Dataset - Datatypes
* in pandas, we use .dtypes to check for datatypes
```py
df.dtypes
```
### Check statistical summary
```py
# Returns a statistical summary
df.describe()

# provides full summary statistics of all columns, plus "unique", "top", "freq"
df.describe(include="all")
```
* unique is the number of unique entries in the column
* top is the most frequent entry in the column
* freq is how many times the most frequent entry appears
* "NaN" is not a number
### Basic Insights of Dataset - Info
* provides a concise summary of your Dataframe
```py
df.info
```