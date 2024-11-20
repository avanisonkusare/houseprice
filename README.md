# Housing Price Prediction using Linear Regression

## Overview

This project aims to predict the price of houses in the USA based on various features such as average area income, number of rooms, house age, and population. A linear regression model is built to predict house prices based on the provided features in the `USA_Housing.csv` dataset.

## Dataset

The dataset used for this project is `USA_Housing.csv` and contains the following columns:

| Column                          | Description                                                   |
|----------------------------------|---------------------------------------------------------------|
| **Avg. Area Income**             | The average income in the area (in dollars).                  |
| **Avg. Area House Age**          | The average age of houses in the area.                        |
| **Avg. Area Number of Rooms**    | The average number of rooms in houses in the area.            |
| **Avg. Area Number of Bedrooms** | The average number of bedrooms in houses in the area.         |
| **Area Population**              | The population of the area.                                   |
| **Price**                        | The price of the house (dependent variable).                  |
| **Address**                      | The address of the house (not used for prediction).           |

### Example Data:

| Avg. Area Income | Avg. Area House Age | Avg. Area Number of Rooms | Avg. Area Number of Bedrooms | Area Population | Price        |
|------------------|---------------------|---------------------------|------------------------------|-----------------|--------------|
| 79545.46         | 5.68                | 7.01                      | 4.09                         | 23086.80        | 1,059,034.00 |
| 79248.64         | 6.00                | 6.73                      | 3.09                         | 40173.07        | 1,505,891.00 |
| 61287.07         | 5.87                | 8.51                      | 5.13                         | 36882.16        | 1,058,988.00 |
| 63345.24         | 7.19                | 5.59                      | 3.26                         | 34310.24        | 1,260,617.00 |
| 59982.20         | 5.04                | 7.84                      | 4.23                         | 26354.11        | 630,943.50   |

## Steps to Run the Code

### 1. Install Dependencies

To run the code, install the necessary Python libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## 2. Load and Explore the Data
The data is loaded into a pandas DataFrame and explored with the following code:

```python
Copy code
import pandas as pd

dataset = pd.read_csv("USA_Housing.csv")
print(dataset.head())
print(dataset.info())
print(dataset.describe())
```
## 3. Data Visualization
We use Seaborn's pairplot to visualize relationships between the features:

```python
Copy code
import seaborn as sns

sns.pairplot(dataset)
This gives a visual representation of how features relate to each other, helping to detect any patterns in the dataset.
```
## 4. Train-Test Split
We split the dataset into training and testing data. The model will be trained on 60% of the data, and evaluated on the remaining 40%.

```python
Copy code
from sklearn.model_selection import train_test_split

X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
             'Avg. Area Number of Bedrooms', 'Area Population']]
y = dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```
## 5. Train a Linear Regression Model
We use the LinearRegression model from Scikit-learn to fit the model on the training data:

```python
Copy code
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)
```
## 6. Make Predictions
Once the model is trained, we use it to make predictions on the test data:

```python
Copy code
predictions = lm.predict(X_test)
```
## 7. Evaluate the Model
We evaluate the model's performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE):

```python
Copy code
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
The output will provide these error metrics to understand how well the model is performing.
```
## 8. Example Predictions
You can also make predictions for specific data points. For example, if you want to predict the price of a house with the following features:

Avg. Area Income = 60000
Avg. Area House Age = 6
Avg. Area Number of Rooms = 6
Avg. Area Number of Bedrooms = 3
Area Population = 30000
The prediction can be made as follows:

```python
Copy code
example_data = np.array([[60000, 6, 6, 3, 30000]])
example_predictions = lm.predict(example_data)

print("Predictions for example data:", example_predictions)
This will output the predicted house price.
```

## Results
For the test data, the following error metrics were computed:
makefile
Copy code
MAE: 82288.22
MSE: 10460958907.21
RMSE: 102278.83
These errors show the average difference between predicted and actual house prices.

# Conclusion
This project demonstrates how to build a linear regression model to predict house prices based on various features of the housing market. The model achieves reasonable accuracy and can be further improved by adding more features, using different models, or tuning hyperparameters.


