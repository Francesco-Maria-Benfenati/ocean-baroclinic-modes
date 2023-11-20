import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


class Interpolation:
    """
    This class is for interpolating vertical profiles.
    """

    def __init__(self, depth: NDArray, *fields: tuple[NDArray]) -> None:
        """
        Interpolation object constructor.

        Args:
            depth (NDArray): depth array
            *fields (tuple[NDARRAY]): vertical profiles to be interpolated
        """

        self.depth = depth
        self.fields = fields

    def apply_interpolation(
        self,
        start: float,
        stop: float,
        step: float,
    ) -> tuple[NDArray]:
        """
        Apply interpolaiton to fields.

        Args:
            start (float): upper depth
            stop (float): bottom depth
            step (float): grid step
        """

        interp_fields = ()
        for field in self.fields:
            (interp_field, interp_depth) = self.vert_interp(field, start, stop, step)
            interp_fields += ((interp_field, interp_depth))
        return interp_fields

    def vert_interp(
        self,
        field: NDArray,
        start: float,
        stop: float,
        step: float,
    ) -> NDArray:
        """
        Interpolate on a new equally spaced depth grid.

        Args:
            start (float): upper depth
            stop (float): bottom depth
            step (float): grid step
        """

        # Create new equally spaced depth array (interpolation depth levels).
        interp_depth_levels = np.arange(start, stop, step)
        # Remove NaN values
        depth = np.delete(self.depth, np.where(np.isnan(field)))
        field = np.delete(field, np.where(np.isnan(field)), axis=-1)

        # Interpolate (distinguish between 3d and 1d fields)
        if field.ndim == 3:
            # Remove NaN values
            x = np.arange(field.shape[0])
            y = np.arange(field.shape[1])
            interp = RegularGridInterpolator(
                (x, y, depth), field, bounds_error=False, fill_value=None
            )
            X, Y, Z = np.meshgrid(x, y, interp_depth_levels, indexing="ij")
            interp_profile = interp((X, Y, Z))

        else:
            interp_func = sp.interpolate.interp1d(
                depth,
                field,
                fill_value="extrapolate",
                kind="linear",
            )
            interp_profile = interp_func(interp_depth_levels)
        # Return interpolated profile and interpolation depth levels.
        return interp_profile, interp_depth_levels


if __name__ == "__main__":
    n_steps = 50
    H = 1000
    step = 1.0
    z = np.linspace(1, H, n_steps)

    # Test interpolation gives correct output when all elements except two are NaN,
    # using linear interpolation.
    field = np.full(n_steps, np.nan)
    a = 3.5e-05
    c = 2.3e-05
    new_z = np.arange(0, H, step)
    expected_field = a * new_z / H + c  # linear BV freq. sq.
    field[0] = expected_field[1]
    field[-1] = expected_field[-1]
    interpolation = Interpolation(z, field)
    (interpolated_field, interp_depth) = interpolation.apply_interpolation(0, H, step)
    assert np.allclose(interpolated_field, expected_field, atol=1e-07)
    print("OK: Nan values are treated well.")

    # Test interpolation gives the same results for two different grid steps.
    (interp_field_2, interp_depth_2) = interpolation.apply_interpolation(0, H, step / 2)
    assert np.allclose(interp_field_2[::2], interpolated_field, atol=1e-08)
    print("OK: same results with different grid steps.")

    # Test if interpolation gives the same output length.
    H_3 = 5000.01
    H_4 = 5000.99
    z_3 = np.linspace(0, H_3, n_steps)
    z_4 = np.linspace(0, H_4, n_steps)
    interp_3 = Interpolation(z_3, field)
    interp_4 = Interpolation(z_4, field)
    (interp_field_3, interp_depth_3) = interp_3.apply_interpolation(0, H_3, step)
    (interp_field_4, interp_depth_4) = interp_4.apply_interpolation(0, H_4, step)
    assert interp_field_3.shape == interp_field_4.shape
    print(
        f"OK: if mean depth diff {H_4-H_3} is less than step {step}, output arrays have same lengths."
    )
    # Test 3D array
    arr3d = np.ones((12, 13, n_steps))
    interp3d = Interpolation(z, arr3d)
    (interp_result, interp_depth_result) = interp3d.apply_interpolation(0, 100, 1)
    assert interp_result.shape == (12, 13, 100)
    print("OK: vertical interpolation works for 3D array.")

    # Test NaN interpolation
    test_arr = np.array([0, 1, 2, np.nan, 4, np.nan, 6])
    interp = Interpolation(np.arange(7), test_arr)
    (interp_arr, interp_depth_arr) = interp.apply_interpolation(0, 7, 1)
    assert np.allclose(interp_arr, np.arange(7))
