from typing import Tuple, Optional
import numpy as np


class PID:
    """Simple pid controller."""

    def __init__(
        self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, output_limits: Tuple[float, float] = (-1, 1), dt: float = 0.05
    ) -> None:
        """[summary]

        Args:
            kp (float, optional): The proportional gain. Defaults to 1.0.
            ki (float, optional): The integral gain. Defaults to 0.0.
            kd (float, optional): The differential gain. Defaults to 0.0.
            output_limits (Tuple[float, float], optional): Output limits of the controller. Defaults to (-1, 1).
            dt (float, optional): The time difference between each output of the controller. Defaults to 0.05.
        """
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limits = output_limits
        self._last_input = 0
        self._integral = 0
        self.dt = dt

    def __call__(self, input_: float, setpoint: float, output_limits: Optional[Tuple[float, float]] = None, verbose: bool = False) -> float:
        """Generate output of the controller.

        Args:
            input_ (float): The input value.
            setpoint (float): The setpoint.
            output_limits (Optional[Tuple[float, float]], optional): Optional new output limits of the controller. Defaults to None.
            verbose (bool, optional): Wether to verbosely print the values of each termn. Defaults to False.

        Returns:
            float: The generated output of the controller.
        """

        if output_limits is not None:
            self.output_limits = output_limits

        # Compute error terms
        error = setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)

        self._proportional = -self.kp * error

        # Compute integral and derivative terms
        self._integral -= self.ki * error * self.dt
        self._integral = np.clip(self._integral, *self.output_limits)

        self._derivative = self.kd * d_input / self.dt

        # Compute output
        output = self._proportional + self._integral + self._derivative
        output = np.clip(output, *self.output_limits)

        # Save last state
        self._last_output = output
        self._last_input = input_

        if verbose:
            print(
                f"prop: {self._proportional:.3f}, int: {self._integral:.3f}, der: {self._derivative:.3f}, err: {error:.3f}, out:"
                f" {output:.3f}",
                end="\033[0K\r",
            )

        return output
