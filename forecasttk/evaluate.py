__author__ = "Christoph Schauer"
__date__ = "2020-07-01"


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def eval_model(y_true, y_pred):
    """
    Prints out several metrics for evaluating the performance of a regression model:
    - Mean of true values
    - Mean of predicted values
    - Mean absolute error
    - Mean relative error (mean absolute error divided by the mean of true values)
    - Root mean squared error (RMSE)
    - Coefficient of determination (R²)
    """

    print("Mean of true values:          {:.4f}".format(np.mean(y_true)))
    print("Mean of predicted values:     {:.4f}".format(np.mean(y_pred)))
    print("Mean absolute error:          {:.4f}".format(mean_absolute_error(y_true, y_pred)))
    print("Mean relative error:          {:.4f}".format(mean_absolute_error(y_true, y_pred)/np.mean(y_true)))
    print("Root mean squared error:      {:.4f}".format(mean_squared_error(y_true, y_pred)**0.5))
    print("Coefficient of determination: {:.4f}".format(r2_score(y_true, y_pred)))
