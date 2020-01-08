# Forecasting Toolkit

Author: Christoph Schauer <br>
Uploaded: 2019/11/16 <br>
Last update: 2020/08/01  <br>


## Introduction

This repository is a collection of notebooks plus a package for helping with handling time series and forecasting in Python. At present it contains notebooks on handling time series data, exploratory analysis, and forecasting with a number of classic statistical and machine learning models, including gradient boosting regression, ARMA and VAR models, and Fourier Transforms.


## Table of Contents

### Notebooks

* `01-data-prepatation.ipynb`: A collection of common data preparation and transformation operations for time series analyis.
* `02-exploratory-analysis.ipynb`: A collection of commonly used exploratory analysis methods and visualizations for time series analysis.
* `03-model-evaluation.ipynb`: A collection of commonly used metrics and visualizations for evaluating the performance of time series forecasting models.
* `11-linear-polynomial-trends.ipynb`: Showcases the custom <i>LinearTrend</i> class for modeling and forecasting time series with linear and polynomial regression models.
* `13-arma-models.ipynb`: Showcases models of the ARMA family (ARIMA, SARIMA, and SARIMAX) using <i>statsmodels.tsa.statespace.sarimax.SARIMAX</i>.
* `14-var-models.ipynb`: Showcases models of the VAR family (VAR, VARMA, and VARMAX) using <i>statsmodels.tsa.statespace.varmax.VARMAX</i>.
* `15-gradient-boosting-models.ipynb`: Showcases the custom <i>TimeSeriesGBR</i> class for modelling and forecasting time series with gradient boosting regression models.
* `16-fourier-models.ipynb`: Showcases the custom <i>FourierWave</i> class for modeling and forecasting time series with Fourier Transforms.
* `31-example-trend-fourier-sarima.ipynb`: Showcases several exploratory techniques and how to aggregate (sort of; add really) three different univariate models (linear trend, Fourier Transform, SARIMA) to capture three different types of patterns in a time series and combine their predictions to one forecast.


## Package forecasttk

The forecast toolkit package currently includes the following modules:
* `visualize.py`: Contains functions for visualizing time series, seasonal decomposition, autocorrelation functions, model forecasts, and residuals. All functions for plotting include an argument for saving the plots as jpeg file.
* `evaluate.py`: Contains functions for quickly printing out performance metrics.
* `lineartrend.py`: Contains the custom class <i>LinearTrend</i>, a child class of <i>sklearn.linear_model.LinearRegression</i>, inheriting everything from this class. It extends this class with several attributes and methods for easy-to-use modeling and forecasting of time series with linear and polynomial regression models.
* `tsgbr.py`: Contains the custom class <i>SeasonalGBRX</i>, a child class of <i>sklearn.ensemble.GradientBoostingRegressor</i>, inheriting everything from this class. It extends this class with several attributes and methods for easy-to-use modeling and forecasting of time series with gradient boosting regression models.
* `fourierwave.py`: Contains the custom class <i>FourierWave</i> which encapsulates several attributes and methods for applying a Fourier Transform to a time series, visualizing its main frequencies, fitting a combination of cosine waves for selected frequencies, and generating a forecast with it.


## References / a few good books, guides, and tutorials for further studies

* [Working with time series in Python](https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html) by Jake VanderPlas
* [Forecasting: Principles and Practice](https://otexts.com/fpp2/)
https://otexts.com/fpp2/ by Rob Hyndman and George Athanasopoulos
* [Forecasting with long seasonal periods](https://robjhyndman.com/hyndsight/longseasonality/) by Rob Hyndman
* [A Gentle Introduction to Autocorrelation and Partial Autocorrelation](https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/) by Jason Brownlee
* [11 Classical Time Series Forecasting Methods in Python (Cheat Sheet)](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/) by Jason Brownlee
* [How to Create an ARIMA Model for Time Series Forecasting in Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/) by Jason Brownlee
* [A Gentle Introduction to SARIMA for Time Series Forecasting in Python](https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/) by Jason Brownlee
* [Analyzing the frequency components of a signal with a Fast Fourier Transform](https://ipython-books.github.io/101-analyzing-the-frequency-components-of-a-signal-with-a-fast-fourier-transform/) by Cyrille Rossant


## Next Steps

#### Additions
* Module for a "custom" SARIMA algorithm based on linear regression plus lagged variables to circumvent the periodicity/number of lags limits of the statsmodels module
* Exponential smoothing notebook
* Time series cross validation / forecast robustness notebook
* Model blending notebook
* Notebook for forecasting many targets in parallel
* Facebook Prophet notebook

#### Updates
* Data preparation: Expand datetime parts
* Data preparation: Add section for doing maths with dates
* Exploratory analysis: Add more methods
* Model evaluation: Add more methods
* Add more explanations for everything
* Add more links to useful tutorials / guides from others
* Update gradient boosting regression class to accept exogenous variables
* Update Fourier Transform class to accept weekly data
* Update all classes to accept data with datetime indices

