import matplotlib.pyplot as plt
import numpy as np


def plot_state_with_probability(
    ax,
    x_value,
    state,
    prob,
    stds=1.96,
    line_kwargs={},
    fill_kwargs=dict(alpha=0.3, label=None),
):
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
    ax.fill_between(
        x_value,
        state - stds * np.sqrt(prob),
        state + stds * np.sqrt(prob),
        color=p[0].get_color(),
        **fill_kwargs,
    )
