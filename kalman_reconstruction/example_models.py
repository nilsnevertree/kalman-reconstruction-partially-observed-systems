"""
This module contains functions to simulate the Lorenz-63 system [1]_ . They are
mainly used to test the Kalman reconstruction by the example of the Lorenz-63
system.

Functions:
-----------------
- _Lorenz_63_ODE: Calculate the derivative of the Lorenz-63 system at a given time.
- Lorenz_63: Simulate the Lorenz-63 system and returns the results as scipy.integrate.solve_ivp [2]_ .
- Lorenz_63_xarray: Simulate the Lorenz-63 system and returns the results as xarray.Dataset.

References:
-----------------
.. [1] Lorenz, Edward N., 1963: Deterministic Nonperiodic Flow. *Journal of the Atmospheric Sciences*, **20**, 130-141,
    `doi:0.1175/1520-0469(1963)020\<0130:DNF\>2.0.CO;2
    <https://doi.org/10.1175/1520-0469(1963)020\<0130:DNF\>2.0.CO;2>`__
"""

import numpy as np
import xarray as xr

from scipy.integrate import solve_ivp


def _Lorenz_63_ODE(t, x, sigma, rho, beta):
    """
    Calculate the derivative of the Lorenz-63 system at a given time.

    The ODEs were first investigated by Lorenz (1963) [1]_ .

    (1) :math:`\\frac{dx}{dt} = \sigma(y-x)`

    (2) :math:`\\frac{dy}{dt} = x(r-z)-y`

    (3) :math:`\\frac{dz}{dt} = xy-bz`

    Parameters:
        t (float): The current time.
        x (ndarray): The state variables [x, y, z].
        sigma (float): The sigma parameter of the Lorenz-63 system.
        rho (float): The rho parameter of the Lorenz-63 system.
        beta (float): The beta parameter of the Lorenz-63 system.

    Returns:
        ndarray: The derivative of the state variables [dx/dt, dy/dt, dz/dt].
    """
    dx = np.empty(3)
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0] * (rho - x[2]) - x[1]
    dx[2] = x[0] * x[1] - beta * x[2]
    return dx


def Lorenz_63(
    time_length=10,
    time_steps=None,
    dt=0.01,
    initial_condition=None,
    sigma=10,
    rho=28,
    beta=8 / 3,
    method="RK45",
):
    """
    Simulate the Lorenz-63 system and returns the results as
    scipy.integrate.solve_ivp [2]_ .

    The ODEs were first investigated by Lorenz (1963) [1]_ .

    (1) :math:`\\frac{dx}{dt} = \sigma(y-x)`

    (2) :math:`\\frac{dy}{dt} = x(r-z)-y`

    (3) :math:`\\frac{dz}{dt} = xy-bz`

    Parameters:
        time_length (float): The total duration of the simulation.
        time_steps (int): The number of time steps to use for the simulation.
        dt (float): The time step size.
        initial_condition (ndarray): The initial condition of the system [x0, y0, z0]. Default is [8, 0, 30].
        sigma (float): The sigma parameter of the Lorenz-63 system. Default is 10.
        rho (float): The rho parameter of the Lorenz-63 system. Default is 28.
        beta (float): The beta parameter of the Lorenz-63 system. Default is 8/3.
        method (str): The numerical integration method to use. Default is "RK45".

    Returns:
        scipy.integrate.OdeResult: The solution of the Lorenz-63 system.
    Raises:
        KeyError: If both `time_length` and `time_steps` are None.

    Example:
        >>> result = Lorenz_63(time_length=10, dt=0.01, initial_condition=[8, 0, 30], sigma=10, rho=28, beta=8/3)
        >>> values = result.y # Extract state variables [x, y, z] at each time step.
        >>> print(values) # Print the state variables [x, y, z] at each time step.

    References:

    .. [1] Lorenz, Edward N., 1963: Deterministic Nonperiodic Flow. *Journal of the
        Atmospheric Sciences*, **20**, 130-141,
        `doi:0.1175/1520-0469(1963)020\<0130:DNF\>2.0.CO;2
        <https://doi.org/10.1175/1520-0469(1963)020\<0130:DNF\>2.0.CO;2>`__
    .. [2] `scipy.integrate.solve_ivp
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`__
    """

    if initial_condition is None:
        initial_condition = np.array([8, 0, 30])
    else:
        initial_condition = np.array(initial_condition).flatten()

    assert (
        np.size(initial_condition) == 3
    ), f"Initial condition needs to have size == 3, but has size {np.size(initial_condition)}"
    assert (
        np.ndim(initial_condition) == 1
    ), f"Initial condition needs to have ndim == 1, but has ndim {np.ndim(initial_condition)}"

    args = (sigma, rho, beta)

    if time_length is not None:
        time = np.arange(0, time_length, dt)
    elif time_steps is not None:
        time = np.arange(0, time_steps) * dt
    else:
        raise KeyError(
            "You need to provide either time_length or time_steps. Both were None."
        )

    t_span = [np.min(time), np.max(time)]
    result = solve_ivp(
        fun=_Lorenz_63_ODE,
        t_span=t_span,
        t_eval=time,
        y0=initial_condition,
        method=method,
        args=args,
    )
    return result


def Lorenz_63_xarray(
    time_length=10,
    time_steps=None,
    dt=0.01,
    initial_condition=None,
    sigma=10,
    rho=28,
    beta=8 / 3,
    method="RK45",
):
    """
    Simulate the Lorenz-63 system and returns the results as xarray Dataset.
    This function is a wrapper around the ``Lorenz_63`` function, which uses the scipy.integrate.solve_ivp [b]_ .

    The ODEs were first investigated by Lorenz (1963) [a]_ .

    (1) :math:`\\frac{dx}{dt} = \sigma(y-x)`

    (2) :math:`\\frac{dy}{dt} = x(r-z)-y`

    (3) :math:`\\frac{dz}{dt} = xy-bz`

    Parameters:
        time_length (float): The total duration of the simulation.
        time_steps (int): The number of time steps to use for the simulation.
        dt (float): The time step size.
        initial_condition (ndarray): The initial condition of the system [x0, y0, z0]. Default is [8, 0, 30].
        sigma (float): The sigma parameter of the Lorenz-63 system. Default is 10.
        rho (float): The rho parameter of the Lorenz-63 system. Default is 28.
        beta (float): The beta parameter of the Lorenz-63 system. Default is 8/3.
        method (str): The numerical integration method to use. Default is "RK45".

    Returns:
        scipy.integrate.OdeResult: The solution of the Lorenz-63 system.

    Raises:
        KeyError: If both `time_length` and `time_steps` are None.

    Example:
        >>> results = Lorenz_63_xarray(
        ...     time_length=10,
        ...     dt=0.01,
        ...     initial_condition=[8, 0, 30],
        ...     sigma=10,
        ...     rho=28,
        ...     beta=8/3
        ... )
        >>> print(results)
        <xarray.Dataset>
        Dimensions:  (time: 1000)
        Coordinates:
        * time     (time) float64 0.0 0.01 0.02 0.03 0.04 ... 9.95 9.96 9.97 9.98 9.99
        Data variables:
            x1       (time) float64 8.0 7.232 6.53 5.892 ... -3.115 -2.785 -2.493 -2.236
            x2       (time) float64 0.0 -0.1217 -0.177 -0.1789 ... 0.3176 0.2502 0.1741
            x3       (time) float64 30.0 29.21 28.43 27.67 ... 26.6 25.89 25.2 24.53
        Attributes:
            description:  Lorenz Model.

    References:

    .. [a] Lorenz, Edward N., 1963: Deterministic Nonperiodic Flow. *Journal of the
        Atmospheric Sciences*, **20**, 130-141,
        `doi:0.1175/1520-0469(1963)020\<0130:DNF\>2.0.CO;2
        <https://doi.org/10.1175/1520-0469(1963)020\<0130:DNF\>2.0.CO;2>`__
    .. [b] `scipy.integrate.solve_ivp
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`__


    """
    result = Lorenz_63(
        time_length=time_length,
        time_steps=time_steps,
        dt=dt,
        initial_condition=initial_condition,
        sigma=sigma,
        rho=rho,
        beta=beta,
        method=method,
    )

    return xr.Dataset(
        data_vars=dict(
            x1=(["time"], result.y[0]),
            x2=(["time"], result.y[1]),
            x3=(["time"], result.y[2]),
        ),
        coords=dict(
            time=result.t,
        ),
        attrs=dict(description="Lorenz Model."),
    )
