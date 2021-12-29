from time import time
from typing import Tuple, Optional
import numpy as np


class PID(object):
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0, output_limits: Tuple[float, float] = (-1, 1)) -> None:
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.output_limits = output_limits
        self._last_input = 0
        self._integral = 0

    def __call__(self, input_: float, setpoint: float, output_limits: Optional[Tuple[float, float]] = None) -> float:
        try:
            self._last_time
        except AttributeError:
            self._last_time = time()

        if output_limits is not None:
            self.output_limits = output_limits

        now = time()
        dt = now - self._last_time if (now - self._last_time) else 1e-16

        # Compute error terms
        error = setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)

        self._proportional = -self.Kp * error

        # Compute integral and derivative terms
        self._integral += self.Ki * error * dt
        self._integral = np.clip(self._integral, *self.output_limits)

        self._derivative = self.Kd * d_input / dt

        # Compute final output
        output = self._proportional + self._integral + self._derivative
        output = np.clip(output, *self.output_limits)

        # Keep track of state
        self._last_output = output
        self._last_input = input_
        self._last_time = now

        return output
