import numpy as np
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
        return_depth=True,
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
            if return_depth:
                interp_fields += (interp_field, interp_depth)
            else:
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
        Interpolate on a new equally spaced depth grid.

        Args:
            start (float): upper depth
            stop (float): bottom depth
            step (float): grid step
        """

        # Create new equally spaced depth array (interpolation depth levels).
        interp_depth_levels = np.arange(start, stop + step, step)

        # Interpolate (distinguish between 3d and 1d fields)
        # 3D array
        if field.ndim == 3:
            depth = self.depth
            x = np.arange(field.shape[0])
            y = np.arange(field.shape[1])
            interp = RegularGridInterpolator(
                (x, y, depth),
                field,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            X, Y, Z = np.meshgrid(x, y, interp_depth_levels, indexing="ij")
            interp_profile = interp((X, Y, Z))
        # 1D array
        else:
            # Remove NaN values
            depth = np.delete(self.depth, np.where(np.isnan(field)))
            field = np.delete(field, np.where(np.isnan(field)), axis=-1)
            # Interpolate
            interp = RegularGridInterpolator(
                (depth,),
                field,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            interp_profile = interp(interp_depth_levels)
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
    new_z = np.arange(0, H + step, step)
    expected_field = a * new_z / H + c  # linear BV freq. sq.
    field[0] = expected_field[1]
    field[-1] = expected_field[-1]
    interpolation = Interpolation(z, field)
    (interpolated_field, interp_depth) = interpolation.apply_interpolation(0, H, step)
    print(interpolated_field[:10], expected_field[:10])
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
    arr3d = np.random.rand(12, 13, n_steps)
    interp3d = Interpolation(z, arr3d)
    (interp_result, interp_depth_result) = interp3d.apply_interpolation(0, 100, 1)
    assert interp_result.shape == (12, 13, 101)
    print("OK: vertical interpolation works for 3D array.")

    # Test NaN interpolation
    test_arr = np.array([0, 1, 2, np.nan, 4, np.nan, 6, np.nan, np.nan, np.nan])
    depth = np.arange(10)
    correct_interpolation = np.arange(10)
    interp = Interpolation(depth, test_arr)
    (interp_arr, interp_depth_arr) = interp.apply_interpolation(0, depth[-1], 1)
    print(interp_arr)
    assert np.allclose(interp_arr, correct_interpolation)

    # Test with 2D bathymetry
    bathy_2D = np.ones((12, 13)) * 1000
    try:
        (interp_with_2d_bathy,) = interp3d.apply_interpolation(0, bathy_2D, 1)
    except ValueError:
        print("not working with 2d bathymetry")

    # Test for 3D array with NaNs
    arr3d[1, 1, 40:] = np.nan
    arr3d[1, 1, -3:] = 0.0
    arr3d[3, 4, 30:] = np.nan
    arr3d[5, 8, 20:] = np.nan
    arr3d[7, 12, :] = np.nan
    print(arr3d[1, 1], arr3d[3, 4], arr3d[5, 8], arr3d[7, 12])
    interp3d_with_nans = Interpolation(z, arr3d)
    (result_with_nans,) = interp3d_with_nans.apply_interpolation(
        0, H, 20, return_depth=False
    )
    print(result_with_nans[1, 1], arr3d[1, 1])
    assert np.isnan(result_with_nans[3, 4, 30:]).all()
    assert np.isnan(result_with_nans[5, 8, 20:]).all()
    assert np.isnan(result_with_nans[7, 12]).all()

    # Experiment with all Nan
    nan_array = np.ones(100) * np.nan
    interp_nan = Interpolation(np.arange(100), nan_array)
    try:
        (interp_nan_result,) = interp_nan.apply_interpolation(
            0, 100, 1, return_depth=False
        )
    except IndexError:
        pass

    # Test with different bottom depths
    field = np.ones_like(z)
    h1 = 1000
    h2 = 1000.1
    h3 = 1000.9
    bulk, bottom1 = interpolation.vert_interp(field, 0, h1, step=1)
    bulk, bottom2 = interpolation.vert_interp(field, 0, h2, step=1)
    bulk, bottom3 = interpolation.vert_interp(field, 0, h3, step=1)
    assert bottom1[-1] == 1000.0
    assert bottom2[-1] == bottom3[-1] == 1001.0
