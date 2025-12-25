# aegis/drift/psi.py

import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    """
    expected: baseline feature values (training data)
    actual: new feature values (test/live data)
    """

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    # Constant feature → no drift
    if expected.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, quantiles)

    # Remove duplicate bin edges
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = pd.cut(
        expected,
        breakpoints,
        duplicates="drop"
    ).value_counts(normalize=True)

    actual_counts = pd.cut(
        actual,
        breakpoints,
        duplicates="drop"
    ).value_counts(normalize=True)

    psi = np.sum(
        (actual_counts - expected_counts)
        * np.log((actual_counts + 1e-8) / (expected_counts + 1e-8))
    )

    return float(psi)
