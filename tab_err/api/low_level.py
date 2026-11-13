from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from tab_err._utils import check_data_emptiness, check_error_rate, set_column

if TYPE_CHECKING:
    import pandas as pd

    from tab_err import ErrorMechanism, ErrorType


def create_errors(
    data: pd.DataFrame, column: str | int, error_rate: float, error_mechanism: ErrorMechanism, error_type: ErrorType | None, error_types: list[ErrorType] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given column of a pandas DataFrame.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        column (str | int): The column to create errors in.
        error_rate (float): The rate at which errors will be created.
        error_mechanism (ErrorMechanism): The mechanism, controls the error distribution.
        error_type (ErrorType | None): The type of the error that will be distributed.
        error_types (list[ErrorType] | None): The types of errors that will be equally distributed. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
    """
    check_error_rate(error_rate)
    check_data_emptiness(data)
    data_copy: pd.DataFrame = data.copy()

    if (error_type is None and error_types is None) or (error_types is not None and error_types is not None) :
        raise ValueError("Exactly one of `error_type` or `error_types` must be not None.")

    error_mask: pd.DataFrame = error_mechanism.sample(data_copy, column, error_rate, error_mask=None)

    if (error_type is not None) or (error_type is None and error_types is not None and len(error_types) == 1):
        if error_types is not None:
            error_type = error_types[0]
        series = error_type.apply(data_copy, error_mask, column)
        set_column(data_copy, column, series)

    else:
        assert error_type is None
        assert error_types is not None
        assert len(error_types) > 1

        def split_1d(series, n):
            ones = np.where(series.to_numpy() == 1)[0]
            chunks = np.array_split(ones, n)
            res = np.zeros((n, len(series)), dtype=series.dtype)
            for i, chunk in enumerate(chunks):
                res[i, chunk] = 1
            return res

        split_arrays = np.stack(
            error_mask.apply(split_1d, n=len(error_types), axis=0).to_numpy(), axis=2
        )

        sub_masks = [
            pd.DataFrame(arr, index=error_mask.index, columns=error_mask.columns)
            for arr in split_arrays
        ]

        for i in range(len(error_types)):
            series = error_types[i].apply(data_copy, sub_masks[i], column)
            set_column(data_copy, column, series)

    return data_copy, error_mask
