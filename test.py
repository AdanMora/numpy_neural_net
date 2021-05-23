# from data.config import *
# from data.dataset import *
# from report.dumps import *
# from nn.model import model
# from nn.funcs import *
# import numpy as np

# from sklearn.metrics import mean_squared_error


# y_true = np.random.rand(4)
# y_pred = np.random.randn(4)

# print(y_true)
# print(y_true.shape)


# print(MSE(y_true, y_pred))
# print(mean_squared_error(y_true, y_pred))

# x = np.random.randn(2)
# print(tanh(x))
# print(np.tanh(x))

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = MLPRegressor(16, activation='tanh', learning_rate_init=0.0085, batch_size=8, max_iter=1000, alpha=0)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
# print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()