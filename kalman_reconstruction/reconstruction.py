import numpy as np


def calculate_x1_from_z1(x2, x3, z1, z2, a, b, sigma, beta, rho, **kwargs):
    """Calculate the missing component x1 based on the equation connecting x1 to z1:
    z1 = a2* d/dt(x2) + a3* d/dt(x3)
    For a and b refer to eq.6(a+b) in the paper Tandeo et. al 2023.

    Note :
    - The parameters were derived using SymPy.
    - The following true PDEs from the Lorenz Model were used to derive the coefficients:
        - d/dt(x2) = ...
        - d/dt(x3) = ...
    - The solution is of the form coef_1 * x1 + coef_0 = 0.
        where the coefficient can depend on all input parameters.

    Parameters
    ----------
    x2: np.ndarray
        x2 of the Lorenz model.
        Needs to be dimension 1 with length N.
    x3: np.ndarray
        x3 of the Lorenz model.
        Needs to be dimension 1 with length N.
    z1: np.ndarray
        1st additional component used for the Kalman_SEM().
        Needs to be dimension 1 with length N.
    z2: np.ndarray
        2nd additional component used for the Kalman_SEM().
        Needs to be dimension 1 with length N.
    a: np.ndarray
        Array containing the coefficients [a2,a3] for the equation:
        z1 = a2* d/dt(x2) + a3* d/dt(x3)
    b: np.ndarray
        Array containing the coefficients [b1, b2, b3] for the equation:
        z2 = b1* d/dt(z1) + b2* d2/dt2(x3) + b3* d2/dt2(x3)

    Returns
    -------
    np.ndarray
        Estimate of x1 based the coefficients and variabels provided.
    """
    a2 = a[0]
    a3 = a[1]

    x1 = (a2 * x2 + a3 * beta * x3 + z1) / (a2 * rho - a2 * x3 + a3 * x2)
    return x1


def calculate_x1_from_z2(x2, x3, z1, z2, a, b, sigma, beta, rho, **kwargs):
    """Calculate the missing component x1 based on the equation connecting x1 to z2:
    z2 = b1* d/dt(z1) + b2* d2/dt2(x3) + b3* d2/dt2(x3)
    For a and b refer to eq.6(a+b) in the paper Tandeo et. al 2023.

    Note :
    - The parameters were derived using SymPy.
    - The following true PDEs from the Lorenz Model were used to derive the coefficients:
        - d/dt(x1) = ...
        - d/dt(x2) = ...
        - d/dt(x3) = ...
    - The solution is of the form coef_2 * x1**2 + coef_1 * x1 + coef_0 = 0.
        where the coefficient can depend on all input parameters and the non linear combinations of them up to O(2).
    - Therefore the two solutions are returned
    Parameters
    ----------
    x2: np.ndarray
        x2 of the Lorenz model.
        Needs to be dimension 1 with length N.
    x3: np.ndarray
        x3 of the Lorenz model.
        Needs to be dimension 1 with length N.
    z1: np.ndarray
        1st additional component used for the Kalman_SEM().
        Needs to be dimension 1 with length N.
    z2: np.ndarray
        2nd additional component used for the Kalman_SEM().
        Needs to be dimension 1 with length N.
    a: np.ndarray
        Array containing the coefficients [a2,a3] for the equation:
        z1 = a2* d/dt(x2) + a3* d/dt(x3)
    b: np.ndarray
        Array containing the coefficients [b1, b2, b3] for the equation:
        z2 = b1* d/dt(z1) + b2* d2/dt2(x3) + b3* d2/dt2(x3)

    Returns
    -------
    np.ndarray
        Estimate of x1 based the coefficients and variabels provided.
        Corresponds to the 1st solution of the quadratic equation.
    np.ndarray
        Estimate of x1 based the coefficients and variabels provided.
        Corresponds to the 2nd solution of the quadratic equation.
    """
    a2 = a[0]
    a3 = a[1]
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]
    x1_res_1 = (
        a2 * b1 * beta * x3
        - a2 * b1 * rho * sigma
        - a2 * b1 * rho
        + a2 * b1 * sigma * x3
        + a2 * b1 * x3
        - a3 * b1 * beta * x2
        - a3 * b1 * sigma * x2
        - a3 * b1 * x2
        + b2 * rho
        - b2 * x3
        + b3 * x2
        - np.sqrt(
            a2**2 * b1**2 * beta**2 * x3**2
            - 2 * a2**2 * b1**2 * beta * rho * sigma * x3
            - 2 * a2**2 * b1**2 * beta * rho * x3
            + 2 * a2**2 * b1**2 * beta * sigma * x3**2
            + 2 * a2**2 * b1**2 * beta * x3**2
            + a2**2 * b1**2 * rho**2 * sigma**2
            + 2 * a2**2 * b1**2 * rho**2 * sigma
            + a2**2 * b1**2 * rho**2
            - 2 * a2**2 * b1**2 * rho * sigma**2 * x3
            + 4 * a2**2 * b1**2 * rho * sigma * x2**2
            - 4 * a2**2 * b1**2 * rho * sigma * x3
            - 2 * a2**2 * b1**2 * rho * x3
            + a2**2 * b1**2 * sigma**2 * x3**2
            - 4 * a2**2 * b1**2 * sigma * x2**2 * x3
            + 2 * a2**2 * b1**2 * sigma * x3**2
            + 4 * a2**2 * b1**2 * x2**2
            + a2**2 * b1**2 * x3**2
            + 2 * a2 * a3 * b1**2 * beta**2 * x2 * x3
            + 2 * a2 * a3 * b1**2 * beta * rho * sigma * x2
            + 2 * a2 * a3 * b1**2 * beta * rho * x2
            - 4 * a2 * a3 * b1**2 * beta * sigma * x2 * x3
            - 4 * a2 * a3 * b1**2 * beta * x2 * x3
            - 4 * a2 * a3 * b1**2 * rho**2 * sigma * x2
            + 2 * a2 * a3 * b1**2 * rho * sigma**2 * x2
            + 8 * a2 * a3 * b1**2 * rho * sigma * x2 * x3
            + 4 * a2 * a3 * b1**2 * rho * sigma * x2
            - 2 * a2 * a3 * b1**2 * rho * x2
            - 2 * a2 * a3 * b1**2 * sigma**2 * x2 * x3
            + 4 * a2 * a3 * b1**2 * sigma * x2**3
            - 4 * a2 * a3 * b1**2 * sigma * x2 * x3**2
            - 4 * a2 * a3 * b1**2 * sigma * x2 * x3
            + 2 * a2 * a3 * b1**2 * x2 * x3
            + 2 * a2 * b1 * b2 * beta * rho * x3
            - 2 * a2 * b1 * b2 * beta * x3**2
            - 2 * a2 * b1 * b2 * rho**2 * sigma
            - 2 * a2 * b1 * b2 * rho**2
            + 4 * a2 * b1 * b2 * rho * sigma * x3
            + 4 * a2 * b1 * b2 * rho * x3
            - 2 * a2 * b1 * b2 * sigma * x3**2
            - 4 * a2 * b1 * b2 * x2**2
            - 2 * a2 * b1 * b2 * x3**2
            - 2 * a2 * b1 * b3 * beta * x2 * x3
            - 2 * a2 * b1 * b3 * rho * sigma * x2
            - 2 * a2 * b1 * b3 * rho * x2
            + 2 * a2 * b1 * b3 * sigma * x2 * x3
            + 2 * a2 * b1 * b3 * x2 * x3
            - 4 * a2 * b1 * x2 * z2
            - 4 * a3**2 * b1**2 * beta**2 * rho * x3
            + a3**2 * b1**2 * beta**2 * x2**2
            + 4 * a3**2 * b1**2 * beta**2 * x3**2
            + 2 * a3**2 * b1**2 * beta * sigma * x2**2
            + 2 * a3**2 * b1**2 * beta * x2**2
            - 4 * a3**2 * b1**2 * rho * sigma * x2**2
            + a3**2 * b1**2 * sigma**2 * x2**2
            + 4 * a3**2 * b1**2 * sigma * x2**2 * x3
            + 2 * a3**2 * b1**2 * sigma * x2**2
            + a3**2 * b1**2 * x2**2
            - 2 * a3 * b1 * b2 * beta * rho * x2
            + 2 * a3 * b1 * b2 * beta * x2 * x3
            - 2 * a3 * b1 * b2 * rho * sigma * x2
            + 2 * a3 * b1 * b2 * rho * x2
            + 2 * a3 * b1 * b2 * sigma * x2 * x3
            - 2 * a3 * b1 * b2 * x2 * x3
            + 4 * a3 * b1 * b3 * beta * rho * x3
            - 2 * a3 * b1 * b3 * beta * x2**2
            - 4 * a3 * b1 * b3 * beta * x3**2
            - 2 * a3 * b1 * b3 * sigma * x2**2
            - 2 * a3 * b1 * b3 * x2**2
            + 4 * a3 * b1 * rho * z2
            - 4 * a3 * b1 * x3 * z2
            + b2**2 * rho**2
            - 2 * b2**2 * rho * x3
            + b2**2 * x3**2
            + 2 * b2 * b3 * rho * x2
            - 2 * b2 * b3 * x2 * x3
            + b3**2 * x2**2
        )
    ) / (2 * b1 * (a2 * x2 - a3 * rho + a3 * x3))
    x1_res_2 = (
        a2 * b1 * beta * x3
        - a2 * b1 * rho * sigma
        - a2 * b1 * rho
        + a2 * b1 * sigma * x3
        + a2 * b1 * x3
        - a3 * b1 * beta * x2
        - a3 * b1 * sigma * x2
        - a3 * b1 * x2
        + b2 * rho
        - b2 * x3
        + b3 * x2
        + np.sqrt(
            a2**2 * b1**2 * beta**2 * x3**2
            - 2 * a2**2 * b1**2 * beta * rho * sigma * x3
            - 2 * a2**2 * b1**2 * beta * rho * x3
            + 2 * a2**2 * b1**2 * beta * sigma * x3**2
            + 2 * a2**2 * b1**2 * beta * x3**2
            + a2**2 * b1**2 * rho**2 * sigma**2
            + 2 * a2**2 * b1**2 * rho**2 * sigma
            + a2**2 * b1**2 * rho**2
            - 2 * a2**2 * b1**2 * rho * sigma**2 * x3
            + 4 * a2**2 * b1**2 * rho * sigma * x2**2
            - 4 * a2**2 * b1**2 * rho * sigma * x3
            - 2 * a2**2 * b1**2 * rho * x3
            + a2**2 * b1**2 * sigma**2 * x3**2
            - 4 * a2**2 * b1**2 * sigma * x2**2 * x3
            + 2 * a2**2 * b1**2 * sigma * x3**2
            + 4 * a2**2 * b1**2 * x2**2
            + a2**2 * b1**2 * x3**2
            + 2 * a2 * a3 * b1**2 * beta**2 * x2 * x3
            + 2 * a2 * a3 * b1**2 * beta * rho * sigma * x2
            + 2 * a2 * a3 * b1**2 * beta * rho * x2
            - 4 * a2 * a3 * b1**2 * beta * sigma * x2 * x3
            - 4 * a2 * a3 * b1**2 * beta * x2 * x3
            - 4 * a2 * a3 * b1**2 * rho**2 * sigma * x2
            + 2 * a2 * a3 * b1**2 * rho * sigma**2 * x2
            + 8 * a2 * a3 * b1**2 * rho * sigma * x2 * x3
            + 4 * a2 * a3 * b1**2 * rho * sigma * x2
            - 2 * a2 * a3 * b1**2 * rho * x2
            - 2 * a2 * a3 * b1**2 * sigma**2 * x2 * x3
            + 4 * a2 * a3 * b1**2 * sigma * x2**3
            - 4 * a2 * a3 * b1**2 * sigma * x2 * x3**2
            - 4 * a2 * a3 * b1**2 * sigma * x2 * x3
            + 2 * a2 * a3 * b1**2 * x2 * x3
            + 2 * a2 * b1 * b2 * beta * rho * x3
            - 2 * a2 * b1 * b2 * beta * x3**2
            - 2 * a2 * b1 * b2 * rho**2 * sigma
            - 2 * a2 * b1 * b2 * rho**2
            + 4 * a2 * b1 * b2 * rho * sigma * x3
            + 4 * a2 * b1 * b2 * rho * x3
            - 2 * a2 * b1 * b2 * sigma * x3**2
            - 4 * a2 * b1 * b2 * x2**2
            - 2 * a2 * b1 * b2 * x3**2
            - 2 * a2 * b1 * b3 * beta * x2 * x3
            - 2 * a2 * b1 * b3 * rho * sigma * x2
            - 2 * a2 * b1 * b3 * rho * x2
            + 2 * a2 * b1 * b3 * sigma * x2 * x3
            + 2 * a2 * b1 * b3 * x2 * x3
            - 4 * a2 * b1 * x2 * z2
            - 4 * a3**2 * b1**2 * beta**2 * rho * x3
            + a3**2 * b1**2 * beta**2 * x2**2
            + 4 * a3**2 * b1**2 * beta**2 * x3**2
            + 2 * a3**2 * b1**2 * beta * sigma * x2**2
            + 2 * a3**2 * b1**2 * beta * x2**2
            - 4 * a3**2 * b1**2 * rho * sigma * x2**2
            + a3**2 * b1**2 * sigma**2 * x2**2
            + 4 * a3**2 * b1**2 * sigma * x2**2 * x3
            + 2 * a3**2 * b1**2 * sigma * x2**2
            + a3**2 * b1**2 * x2**2
            - 2 * a3 * b1 * b2 * beta * rho * x2
            + 2 * a3 * b1 * b2 * beta * x2 * x3
            - 2 * a3 * b1 * b2 * rho * sigma * x2
            + 2 * a3 * b1 * b2 * rho * x2
            + 2 * a3 * b1 * b2 * sigma * x2 * x3
            - 2 * a3 * b1 * b2 * x2 * x3
            + 4 * a3 * b1 * b3 * beta * rho * x3
            - 2 * a3 * b1 * b3 * beta * x2**2
            - 4 * a3 * b1 * b3 * beta * x3**2
            - 2 * a3 * b1 * b3 * sigma * x2**2
            - 2 * a3 * b1 * b3 * x2**2
            + 4 * a3 * b1 * rho * z2
            - 4 * a3 * b1 * x3 * z2
            + b2**2 * rho**2
            - 2 * b2**2 * rho * x3
            + b2**2 * x3**2
            + 2 * b2 * b3 * rho * x2
            - 2 * b2 * b3 * x2 * x3
            + b3**2 * x2**2
        )
    ) / (2 * b1 * (a2 * x2 - a3 * rho + a3 * x3))
    return x1_res_1, x1_res_2
