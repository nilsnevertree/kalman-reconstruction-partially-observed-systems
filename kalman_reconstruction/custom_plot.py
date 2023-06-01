from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def set_custom_rcParams():
    """
    Set the default configuration parameters for matplotlib. The colorblind-
    save colors were choosen with the help of
    https://davidmathlogic.com/colorblind.

    Returns:
    --------
    None

    Note:
    -----
    This function modifies the global matplotlib configuration.

    Examples:
    ---------
    >>> default_rcParams()
    """

    # Set default figure size
    plt.rcParams["figure.figsize"] = (10, 8)

    # Set font sizes
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 15
    HUGHER_SIZE = 18
    plt.rc("font", size=MEDIUM_SIZE)  # Default text sizes
    plt.rc("figure", titlesize=HUGHER_SIZE)  # Axes title size
    plt.rc("figure", labelsize=BIGGER_SIZE)  # X and Y labels size
    plt.rc("axes", titlesize=BIGGER_SIZE)  # Axes title size
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # X and Y labels size
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # X tick labels size
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # Y tick labels size
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # Legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # Figure title size

    # Set axis spines visibility
    plt.rc(
        "axes.spines",
        **{
            "left": True,
            "right": False,
            "bottom": True,
            "top": False,
        },
    )

    # Set legend location
    plt.rc("legend", loc="upper right")

    # Use colorblind-safe colors
    colors = [
        "#CC6677",
        "#6E9CB3",
        "#CA8727",
        "#44AA99",
        "#AA4499",
        "#D6BE49",
        "#A494F5",
    ]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def plot_colors(colors):
    """
    Plot a scatter plot of colors.

    Parameters:
    -----------
    colors : list
        List of color values to plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axs : matplotlib.axes.Axes
        The matplotlib Axes object.

    Examples:
    ---------
    >>> colors = ['#FF0000', '#00FF00', '#0000FF']
    >>> fig, axs = plot_colors(colors)
    """
    fig, axs = plt.subplots(figsize=(5, 1))
    for idx, color in enumerate(colors):
        axs.scatter(idx, 1, c=color, s=300)

    axs.set_yticks([])
    return fig, axs


def plot_state_with_probability(
    ax,
    x_value,
    state,
    prob,
    stds=1.96,
    line_kwargs={},
    fill_kwargs=dict(alpha=0.3, label=None),
    output: bool = False,
) -> Union[None, Tuple]:
    """
    Plot the state variable with its corresponding probability distribution.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object to plot on.
    x_value : np.ndarray
        The x-axis values.
    state : np.ndarray
        The state variable values.
    prob : np.ndarray
        The probability distribution of the state variable.
    stds : float, optional
        The number of standard deviations to use for the probability interval (default: 1.96).
    line_kwargs : dict, optional
        Additional keyword arguments for the line plot (default: {}).
    fill_kwargs : dict, optional
        Additional keyword arguments for the fill_between plot (default: {'alpha': 0.3, 'label': None}).

    Returns:
    --------
    None

    Examples:
    ---------
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 10, 100)
    >>> state = np.sin(x)
    >>> prob = np.abs(np.cos(x))
    >>> fig, ax = plt.subplots()
    >>> plot_state_with_probability(ax, x, state, prob)
    """
    p = ax.plot(x_value, state, **line_kwargs)
    f = ax.fill_between(
        x_value,
        state - stds * np.sqrt(prob),
        state + stds * np.sqrt(prob),
        color=p[0].get_color(),
        **fill_kwargs,
    )
    if output:
        return p, f
