# November 2021 Tabular Playground Series

This repository contains the code for submissions to train the model presented in the [Kaggle Competition](https://www.kaggle.com/c/tabular-playground-series-nov-2021)

[submission v011](https://www.kaggle.com/olumoni/nov-tabplayground?scriptVersionId=79984986)

[submission v015](https://www.kaggle.com/olumoni/nov-tabplayground/notebook)

It is a presentation repository for my submissions to the Kaggle competition. As such, you will find many scripts, classes, blocks and options which we actively use for our own development purposes but are not directly relevant to reproduce results or use pretrained weights.



## Using pre-trained weights

In the competition, we present ClimateGAN as a solution to produce images of floods. It can actually do **more**: 

* reusing the segmentation map, we are able to isolate the sky, turn it red and in a few more steps create an image resembling the consequences of a wildfire on a neighboring area, similarly to the [holdout for websites](https://www.google.com).
* reusing the depth map, we can simulate the consequences of a smog event on an image, scaling the intensity of the filter by the distance of an object to the camera, as per [place holder HazeRD](http://www.google.com)

![image of wildfire processing](images/wildfire.png)
![image of smog processing](images/smog.png)

In this section we'll explain how to produce the `Painted Input` along with the Smog and Wildfire outputs of a pre-trained ClimateGAN model.

### Installation

This repository and associated model have been developed using Python 3.8.2 and **Pytorch 1.7.0**.

```bash
$ git clone git@github.com:cc-ai/climategan.git
$ cd climategan
$ pip install -r requirements-3.8.2.txt # or `requirements-any.txt` for other Python versions (not tested but expected to be fine)
```
### Load packages
```
# Load packages
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
```
### Load Training Data
```
# Load training data
df = pd.read_csv("../input/tabular-playground-series-nov-2021/train.csv") 
# check for missing values
#df.isnull().sum()
df.head()
```

### Split into predictor and target variables
```
#Split the data into X and y
output_col = ['target']
X = df.drop(['id', 'target'], axis=1)
y = df[output_col]
```

### Using L1 Regression to differentiate between important variables
```
# Lasso (L1) Regression: Plot is flat 
from sklearn.linear_model import Lasso

df_columns = X.columns

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X,y).coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()
```
![Lasso](images/v0011/LassoRegression_plot.png)

```
# Split the data into training and test data
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.30, random_state= 44)

# Choose the criterion and max depth of the tree you want to use
CRITERION = 'gini'
MAX_DEPTH = 3

# Set up the DT classifier
dt_clf = DecisionTreeClassifier(criterion=CRITERION, max_depth=MAX_DEPTH, random_state=43)

# Train the DT classifier
dt_clf.fit(X_train, y_train)

# Evaluate the DT on the test set
y_pred = dt_clf.predict(X_test)
print(f'Model accuracy score with criterion {CRITERION} index: {accuracy_score(y_test, y_pred):.4f}')
```
![Lasso](images/v0011/LassoRegression_plot.png)

Our pipeline uses [comet.ml](https://comet.ml) to log images. You don't *have* to use their services but we recommend you do as images can be uploaded on your workspace instead of being written to disk.

If you want to use Comet, make sure you have the [appropriate configuration in place (API key and workspace at least)](https://www.comet.ml/docs/python-sdk/advanced/#non-interactive-setup)
