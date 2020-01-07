"""
This script contains helper functions for visualizing:
- Time series
- Seasonal decomposition
- Autocorrelation & partial autocorrelation functions
- Model forecasts
- Residuals
All functions for plotting include an argument for saving the plots as jpeg file.
"""


__author__ = "Christoph Schauer"
__date__ = "2019-11-29"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots


def plot_series(*y_args, xlabel=None, ylabel=None, title=None, figsize=(15,5), saveas=None):
    """
    Visualizes one or more time series in one plot.
    """

    plt.figure(figsize=figsize)
    for y in y_args:
        y.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if len(y_args) > 1:
        plt.legend(loc="upper left", frameon=True)
    if saveas is not None:
        plt.savefig(saveas, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()


def plot_seasonal_decompose(y, freq, figsize=(12,8), saveas=None):
    """
    Plots the seasonal decomposition of a time series.
    """

    # Seasonal decompose
    sc = seasonal_decompose(y, freq=freq)

    # Set up plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=figsize, constrained_layout=True)

    # 1 time series
    y.plot(ax=ax1)
    ax1.set_xlabel(None)
    ax1.set_title("Time series")

    # 2 trend component
    sc.trend.plot(ax=ax2, c="tab:orange")
    ax2.set_xlabel(None)
    ax2.set_title("Trend component")

    # 3 seasonal component
    sc.seasonal.plot(ax=ax3, c="tab:orange")
    ax3.set_xlabel(None)
    ax3.set_title("Seasonal component")

    # 4 residual component
    sc.resid.plot(ax=ax4, c="tab:orange")
    ax4.set_xlabel("Date")
    ax4.set_title("Residual component")

    if saveas is not None:
        plt.savefig(saveas,  dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.show()


def plot_acf_pacf(y, lags, alpha=0.05, figsize=(12,8), saveas=None):
    """
    Plots the autocorrelation and partical autocorrelation functions.
    """

    fig, (ax1, ax2) = plt.subplots(2, figsize=figsize, constrained_layout=True)

    # ACF
    tsaplots.plot_acf(y, lags=lags, alpha=alpha, ax=ax1)
    ax1.set_xlabel(lags)
    ax1.set_xlim(0, lags + 1)

    # PACF
    tsaplots.plot_pacf(y, lags=lags, alpha=alpha, ax=ax2)
    ax2.set_xlabel(lags)
    ax2.set_xlim(0, lags + 1)

    if saveas is not None:
        plt.savefig(saveas,  dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.show()


def plot_model(
    y, y_pred, y_fcst, xlabel=None, ylabel=None, title=None, figsize=(15,5), saveas=None):
    """
    Plots three time series: True values, predicted values, and forecasted values.
    """

    plt.figure(figsize=figsize)
    y.plot(c="tab:blue", label="True values")
    y_pred.plot(c="tab:orange", lw=2, label="Model fit")
    y_fcst.plot(c="tab:orange", lw=2, ls="--", label="Model forecast")
    plt.axvline(y_fcst.index[0], c="tab:gray", ls="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper left", frameon=True)
    if saveas is not None:
        plt.savefig(saveas,  dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.show()


def plot_residuals(residuals, figsize=(15,5), saveas=None):
    """
    Plots residuals over time and the kernel density estimate of a time series of residuals.
    """

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[5, 2])

    # Residuals over time plot
    ax1 = fig.add_subplot(gs[0])
    residuals.plot(ax=ax1)
    plt.axhline(0, c="tab:gray", ls="--")
    ax1.set_xlabel(None)
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals over time")

    # KDE plot
    ax2 = fig.add_subplot(gs[1])
    residuals.plot.kde(ax=ax2)
    plt.axvline(0, c="tab:gray", ls="--")
    ax2.set_title("Kernel density estimate of residuals")

    if saveas is not None:
        plt.savefig(saveas,  dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.show()