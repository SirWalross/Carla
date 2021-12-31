from typing import Tuple, Optional
import numpy as np


class PID():
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.0, output_limits: Tuple[float, float] = (-1, 1), dt: float = 0.05) -> None:
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.output_limits = output_limits
        self._last_input = 0
        self._integral = 0
        self.dt = dt

    def __call__(self, input_: float, setpoint: float, output_limits: Optional[Tuple[float, float]] = None, verbose: bool = False) -> float:

        if output_limits is not None:
            self.output_limits = output_limits

        # Compute error terms
        error = setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)

        self._proportional = -self.Kp * error

        # Compute integral and derivative terms
        self._integral -= self.Ki * error * self.dt
        self._integral = np.clip(self._integral, *self.output_limits)

        self._derivative = self.Kd * d_input / self.dt

        # Compute output
        output = self._proportional + self._integral + self._derivative
        output = np.clip(output, *self.output_limits)

        # Save last state
        self._last_output = output
        self._last_input = input_

        if verbose:
            print(
                f"prop: {self._proportional:.3f}, int: {self._integral:.3f}, der: {self._derivative:.3f}, err: {error:.3f}, out: {output:.3f}",
                end="\033[0K\r",
            )

        return output
