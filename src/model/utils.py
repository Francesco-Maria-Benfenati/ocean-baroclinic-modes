from xarray import Dataset


class Utils:
    """
    This class contains different functions for utility purposes.
    """

    @staticmethod
    def drop_dims_from_dataset(dataset: Dataset, drop_dims: list[str]) -> Dataset:
        """
        Drop dimensions from datasets which would give troubles due to incompatible sizes.
        Preprocessing for Climatologies with different number of observations, from WOD-18.
        """

        return dataset.drop_dims(drop_dims=drop_dims)

    @staticmethod
    def andor(a: bool, b: bool) -> bool:
        """
        And/Or logical statement.
        """

        return a and b | a or b

if __name__ == "__main__":
    a = True
    b = False
    assert Utils.andor(a,a) and Utils.andor(a,b) and Utils.andor(b,a)
    assert not Utils.andor(b,b)
