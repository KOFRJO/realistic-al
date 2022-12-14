from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def plot_standard_dev(
    ax,
    data,
    x: str = "num_samples",
    y: str = "test_acc",
    hue: str = "Sampling",
    style=None,
    units=None,
    ci="sd",
    legend="auto",
    palette=None,
    markers=True,
    dashes=True,
    linestyle=None,
    err_kws=None,
) -> plt.Axes:
    """Creates a lineplot from dataframe with sns.lineplot.
    For information see:
    https://seaborn.pydata.org/generated/seaborn.lineplot.html

    Returns:
        plt.Axes: _description_
    """
    sns.lineplot(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        ci=ci,
        marker="o",
        units=units,
        style=style,
        legend=legend,
        palette=palette,
        markers=markers,
        dashes=dashes,
        linestyle=linestyle,
        err_kws=err_kws,
    )
    return ax
