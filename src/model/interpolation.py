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
            interp_field = self.vert_interp(field, start, stop, step)
            interp_fields += (interp_field,)
        return interp_fields

    def vert_interp(
        self,
        field: NDArray,
        start: float,
        stop: float,
        step: float,
    ) -> NDArray:
        """
        Interpolate B-V freq. squared on a new equally spaced depth grid.

        Args:
            start (float): upper depth
            stop (float): bottom depth
            step (float): grid step
        """

        # Delete NaN elements from field (and corresponding dept values).
        where_nan_field = np.where(np.isnan(field))
        field_nan_excl = np.delete(field, where_nan_field, None)
        depth_nan_excl = np.delete(self.depth, where_nan_field, None)

        # Create new equally spaced depth array.
        z = np.arange(start=start, stop=stop, step=step)

        # Interpolate (distinguish between 3d and 1d fields)
        if field.ndim == 3:
            x = np.arange(field.shape[0])
            y = np.arange(field.shape[1])
            interp = RegularGridInterpolator(
                (x, y, self.depth), field, bounds_error=False, fill_value=None
            )
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            interp_profile = interp((X, Y, Z))
        else:
            f = sp.interpolate.interp1d(
                depth_nan_excl,
                field_nan_excl,
                fill_value="extrapolate",
                kind="linear",
            )
            interp_profile = f(z)
        # Return interpolated profile
        return interp_profile

    @staticmethod
    def interpolate_at_interface(depth: NDArray, field: NDArray) -> NDArray:
        """
        Interpolate vertically at interfaces +/- 1/2.

        :returns : interface depths and interpolated field
        """

        depth_levels = np.arange(len(depth))
        interface_levels = np.arange(len(depth) + 1)
        f1 = sp.interpolate.interp1d(
            depth_levels, depth, fill_value="extrapolate", kind="linear"
        )
        interface_depth = f1(interface_levels)
        if field.ndim == 3:
            x = np.arange(field.shape[0])
            y = np.arange(field.shape[1])
            f2 = RegularGridInterpolator(
                (x, y, depth_levels), field, bounds_error=False, fill_value=None
            )
            X, Y, Z = np.meshgrid(x, y, interface_levels, indexing="ij")
            interface_field = f2((X, Y, Z))
        else:
            f2 = sp.interpolate.interp1d(
                depth_levels, field, fill_value="extrapolate", kind="linear"
            )
            interface_field = f2(interface_levels)
        return interface_depth, interface_field


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
    interpolated_field = interpolation.apply_interpolation(0, H, step)[0]
    assert np.allclose(interpolated_field, expected_field, atol=1e-07)
    print("OK: Nan values are treated well.")

    # Test interpolation gives the same results for two different grid steps.
    interp_field_2 = interpolation.apply_interpolation(0, H, step / 2)[0]
    assert np.allclose(interp_field_2[::2], interpolated_field, atol=1e-08)
    print("OK: same results with different grid steps.")

    # Test if interpolation gives the same output length.
    H_3 = 5000.01
    H_4 = 5000.99
    z_3 = np.linspace(0, H_3, n_steps)
    z_4 = np.linspace(0, H_4, n_steps)
    interp_3 = Interpolation(z_3, field)
    interp_4 = Interpolation(z_4, field)
    interp_field_3 = interp_3.apply_interpolation(0, H_3, step)[0]
    interp_field_4 = interp_4.apply_interpolation(0, H_4, step)[0]
    assert interp_field_3.shape == interp_field_4.shape
    print(
        f"OK: if mean depth diff {H_4-H_3} is less than step {step}, output arrays have same lengths."
    )
    # Test 3D array
    arr3d = np.ones((12, 13, n_steps))
    interp3d = Interpolation(z, arr3d)
    interp_result = interp3d.apply_interpolation(0, 100, 1)[0]
    assert interp_result.shape == (12, 13, 100)
    print("OK: vertical interpolation works for 3D array.")
    interface_depth, interface_field = Interpolation.interpolate_at_interface(z, arr3d)
    assert interface_field.shape == (12, 13, n_steps + 1)
    print("OK: vertical interpolation *at interfaces* works for 3D array.")
