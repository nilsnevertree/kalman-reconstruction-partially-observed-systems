kalman (time dependent)
=======================================================
Time dependent implementation of the Kalman Algorithms (filter, smoother, SEM, etc.)
The main difference to the time independent implementation is that the time dependent implementation uses time dependent model matrix :math:`M(t)` and model uncertainty matrix :math:`Q(t)`.
Thus both have an additional "time" dimension. For each time step the model matrix and model uncertainty matrix are used to calculate the Kalman gain and the Kalman gain is used to calculate the state estimate and state uncertainty.

Functions
---------
.. automodapi:: kalman_reconstruction.kalman_time_dependent
   :members:
   :no-heading:
   :inheritance-diagram:
   :no-imported-members:
