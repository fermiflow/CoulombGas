import numpy as np
import pandas as pd

def running_average(data, alpha):
    """ Exponentially-weighted moving average of data.

        Adapted from the ferminet code repo:
    https://github.com/deepmind/ferminet/blob/main/ferminet/utils/tests/statistics_test.py
    https://github.com/deepmind/ferminet/blob/main/ferminet/utils/statistics.py

        See the notes refered therein https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
    as well as the `test` function below for interpretation of the results.
    """
    ewm = pd.Series(data).ewm(adjust=False, alpha=alpha)
    expected_mean = ewm.mean(bias=True)
    expected_variance = ewm.var(bias=True)
    return np.array(expected_mean), np.array(expected_variance)

def test():
    n = 10
    alpha = 0.1
    weight = np.concatenate([ [(1-alpha)**n], alpha * (1-alpha)**np.arange(n)[::-1] ])
    assert np.allclose(weight.sum(), 1.0)

    data = np.random.randn(n+1)
    ewm_mean, ewm_variance = running_average(data, alpha)
    #print("ewm_mean:", ewm_mean, "\newm_variance:", ewm_variance)

    mean = (weight * data).sum() 
    variance = (weight * (data - mean)**2).sum()
    #print("mean:", mean, "\nvariance:", variance)

    assert np.allclose(mean, ewm_mean[-1])
    assert np.allclose(variance, ewm_variance[-1])

if __name__ == "__main__":
    test()