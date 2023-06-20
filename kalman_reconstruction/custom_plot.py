from colorsys import hls_to_rgb, rgb_to_hls
from typing import Tuple, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import cnames, to_hex, to_rgb
from matplotlib.legend_handler import (
    HandlerLine2D,
    HandlerPathCollection,
    HandlerPolyCollection,
)
from matplotlib.patches import Polygon


def set_custom_rcParams():
    """
    Set the default configuration parameters for matplotlib. The colorblind-
    save colors were chosen with the help of
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
    plt.rcParams["figure.figsize"] = (16 / 1.5, 9 / 1.5)

    # Set font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 15
    HUGHER_SIZE = 18
    plt.rc("font", size=MEDIUM_SIZE)  # Default text sizes
    plt.rc("figure", titlesize=HUGHER_SIZE)  # Axes title size
    plt.rc("figure", labelsize=BIGGER_SIZE)  # X and Y labels size
    plt.rc("axes", titlesize=MEDIUM_SIZE)  # Axes title size
    plt.rc("axes", labelsize=SMALL_SIZE)  # X and Y labels size
    plt.rc("xtick", labelsize=SMALL_SIZE)  # X tick labels size
    plt.rc("ytick", labelsize=SMALL_SIZE)  # Y tick labels size
    plt.rc("legend", fontsize=SMALL_SIZE)  # Legend fontsize
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


def adjust_lightness(color: str, amount: float = 0.75) -> str:
    """
    Adjusts the lightness of a color by the specified amount.

    This function takes a color name or a hexadecimal color code as input and adjusts its lightness
    by the specified amount. The color name can be one of the predefined color names from the Matplotlib
    `cnames` dictionary or a custom color name. If the input color is a hexadecimal color code, it will
    be converted to the corresponding RGB values before adjusting the lightness.

    The lightness adjustment is performed by converting the color to the HLS (Hue, Lightness, Saturation)
    color space, modifying the lightness component by the specified amount, and then converting it back
    to the RGB color space.

    Parameters:
        color (str): The color name or hexadecimal color code to adjust the lightness of.
        amount (float, optional): The amount by which to adjust the lightness.
            Positive values increase the lightness, while negative values decrease it.
            Default is 0.75.

    Returns:
        str: The adjusted color as a hexadecimal color code.

    Example usage:
    >>> color = "red"
    >>> adjusted_color = adjust_lightness(color, 0.5)
    >>> print(f"Adjusted color: {adjusted_color}")

    References:
        - Function created by Ian Hincks, available at:
          https://stackoverflow.com/a/49601444/16372843
    """
    try:
        try:
            c = cnames[color]
        except:
            c = color
        c = rgb_to_hls(*to_rgb(c))
        c = hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
        return to_hex(c)
    except ValueError:
        return color  # Return the original color if conversion fails


def __set_handler_alpha_to_1__(handle, orig):
    """
    Set the alpha (transparency) of the handler to 1.

    This internal function is used to set the alpha value of a legend handler to 1.
    It is used as an update function in `handler_map_alpha` to modify the legend handler's alpha value.

    Parameters:
        handle: The legend handler object to update.
        orig: The original legend handler object.

    Returns:
        None

    Reference:
        https://stackoverflow.com/a/59629242/16372843
    """
    handle.update_from(orig)
    handle.set_alpha(1)


def handler_map_alpha():
    """
    Create a mapping of legend handler types to update functions.

    This function returns a dictionary that maps specific legend handler types to their corresponding
    update functions. The update functions are used to modify the legend handler's properties,
    such as the alpha (transparency) value.

    Returns:
        dict: A dictionary mapping legend handler types to their update functions.

    Example usage:
    >>> ax.legend(handhandler_map_alpha()
    >>> print(handler_map)
    """
    return {
        PathCollection: HandlerPathCollection(update_func=__set_handler_alpha_to_1__),
        plt.Line2D: HandlerLine2D(update_func=__set_handler_alpha_to_1__),
        Polygon: HandlerPolyCollection(update_func=__set_handler_alpha_to_1__),
    }


def ncols_nrows_from_N(N):
    """
    Calculate the number of columns and rows for a grid based on the total
    number of elements.

    Given the total number of elements `N`, this function calculates the optimal number of
    columns and rows for a grid layout that can accommodate all the elements.

    Parameters:
        N (int): The total number of elements.

    Returns:
        dict: A dictionary containing the number of columns (`ncols`) and rows (`nrows`) for the grid.

    Examples:
        >>> ncols_nrows_from_N(12)
        {'ncols': 4, 'nrows': 3}

        >>> ncols_nrows_from_N(8)
        {'ncols': 3, 'nrows': 3}

        >>> ncols_nrows_from_N(1)
        {'ncols': 1, 'nrows': 1}
    """
    if not isinstance(N, (int, float)):
        try:
            N = int(N)
            warn(
                f"N should be and type int but is {type(N)}\nConverted to int. N : {N}"
            )
        except Exception:
            raise ValueError(
                f"N should be and type int but is {type(N)}\nConvertion to int not possible"
            )
    if N <= 0:
        raise ValueError(f"N need to be greater than 1 but is {N}")

    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    return dict(ncols=cols, nrows=rows)


def symmetrize_axis(axes: Axes, axis: Union[int, str]) -> None:
    """
    Symmetrize the given axis of the matplotlib Axes object.

    This function adjusts the limits of the specified axis of the matplotlib Axes object
    to make it symmetrical by setting the minimum and maximum values to their absolute maximum.

    Parameters:
        axes (Axes): The matplotlib Axes object.
        axis (Union[int, str]): The axis to symmetrize. It can be specified either by index (0 for x-axis, 1 for y-axis)
            or by string identifier ("x" for x-axis, "y" for y-axis).

    Returns:
        None

    Examples:
        # Example 1: Symmetrize the x-axis
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> symmetrize_axis(ax, axis=0)
        >>> plt.show()

        # Example 2: Symmetrize the y-axis
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> symmetrize_axis(ax, axis="y")
        >>> plt.show()
    """
    if axis in [0, "x"]:
        maxi = np.abs(axes.get_xlim()).max()
        axes.set_xlim(xmin=-maxi, xmax=maxi)
    elif axis in [1, "y"]:
        maxi = np.abs(axes.get_ylim()).max()
        axes.set_ylim(ymin=-maxi, ymax=maxi)
