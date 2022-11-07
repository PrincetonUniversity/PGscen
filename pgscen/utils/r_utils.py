"""Utility functions for working with distributions and models created in R."""

import warnings
from typing import Union, Tuple
from rpy2.robjects.methods import RS4 as GPD

import numpy as np
import pandas as pd

from scipy.stats import norm
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# import R packages
base = importr('base')
Rsafd = importr('Rsafd')
glasso = importr('glasso')
stats = importr('stats')


class PGscenECDF:
    """
    class object to store emperical CDF (ECDF)
    """

    def __init__(self, data: np.array, n: int = 1000) -> None:
        self.rclass = ['ecdf']
        self.data = data
        self.ecdf = stats.ecdf(robjects.FloatVector(data))

        quants = robjects.FloatVector(np.linspace(0, 1, n + 1))
        self.approxfun = stats.approxfun(quants,
                                         stats.quantile(self.ecdf, quants))

    def quantfun(self, data: np.array) -> np.array:
        return self.approxfun(data)

def qgpd(dist: GPD, x: np.array) -> np.array:
    """Wrapper for Rsafd qgpd function; gets quantiles at all values in x."""

    try:
        return np.array(Rsafd.qgpd(dist, robjects.FloatVector(x)))

    except:
        # compute quantiles using PDF
        ll = min(dist.slots['data'])
        rr = max(dist.slots['data'])
        xx = np.linspace(ll, rr, 1001)

        # rule=2 for extrapolation of values outside min and max
        ff = stats.approxfun(Rsafd.pgpd(dist, xx), xx, rule=2)

        return ff(x)


def fit_gpd(data: np.array) -> Union[GPD, PGscenECDF]:
    """Fit a GPD if possible (fitting converge), otherwise use emperical distribution function."""

    try:
        ## try fit two tails
        dist = Rsafd.fit_gpd(robjects.FloatVector(data),
                                    tail='two', plot=False)

        upper = dist.slots['upper.converged'][0]
        lower = dist.slots['lower.converged'][0]

        if upper and lower:
            return dist
        elif upper:
            return Rsafd.fit_gpd(robjects.FloatVector(data),
                                    tail='left', plot=False)
        elif lower:
            return Rsafd.fit_gpd(robjects.FloatVector(data),
                                    tail='right', plot=False)
        else:
            return PGscenECDF(data)
        
    except:
        warnings.warn(f'tail has been detected, but unable to fit '
                          f'GPD, using ECDF instead', RuntimeWarning)
        return PGscenECDF(data)

def qdist(dist: Union[GPD, PGscenECDF], x: np.array, gpd_max_extension: float=0.15) -> np.array:
    """Compute the quantiles of the distribution.
    If input distribution is GPD, output will be clipped. 
    """

    if tuple(dist.rclass)[0][0:3] == 'gpd':
        # GPD
        data_min, data_max = np.min(dist.slots['data']), np.max(dist.slots['data'])
        clip_min = data_min - gpd_max_extension * (data_max - data_min)
        clip_max = data_max + gpd_max_extension * (data_max - data_min)

        return np.clip(qgpd(dist, x), clip_min, clip_max)

    elif tuple(dist.rclass)[0] == 'ecdf':
        # Empirical CDF
        return stats.quantile(dist.ecdf, robjects.FloatVector(x))

    else:
        raise RuntimeError(
            "Unrecognized distribution class {}".format(tuple(dist.rclass)))


def standardize(table: pd.DataFrame,
                ignore_pointmass: bool = True) -> Tuple[pd.Series, pd.Series,
                                                        pd.DataFrame]:
    avg, std = table.mean(), table.std()

    if ignore_pointmass:
        std[std < 1e-2] = 1.

    else:
        if (std < 1e-2).any():
            raise RuntimeError(f'encountered point masses in '
                               f'columns {std[std < 1e-2].index.tolist()}')

    return avg, std, (table - avg) / std


def gaussianize(df: pd.DataFrame, gpd: bool = False) -> Tuple[dict, pd.DataFrame]:
    """Transform the data to fit a Gaussian distribution."""

    unif_df = pd.DataFrame(columns=df.columns, index=df.index)
    dist_dict = dict()

    for col in df.columns:

        data = np.ascontiguousarray(df[col].values)

        if gpd:
            dist_dict[col] = fit_gpd(data)
            unif_df[col] = np.array(Rsafd.pgpd(
                dist_dict[col], robjects.FloatVector(data)))
        else:
            dist_dict[col] = PGscenECDF(data)
            unif_df[col] = np.array(
                dist_dict[col].ecdf(robjects.FloatVector(data)))

    unif_df.clip(lower=1e-5, upper=0.99999, inplace=True)
    gauss_df = unif_df.apply(norm.ppf)

    return dist_dict, gauss_df

def graphical_lasso(df: pd.DataFrame, m: int, rho: float):
    """
    Wrapper for the glasso model.

    Arguments
    ---------
        df
            The input dataset.
        m
            Number of input dimensions.
        rho
            LASSO regularization penalty.

    """
    assert df.shape[1] == m, (
        "Expected a DataFrame with {} columns, got {}".format(m, df.shape[1]))

    f = glasso.glasso
    cov = df.cov().values
    rcov = robjects.r.matrix(cov, nrow=m, ncol=m)
    res = f(rcov, rho=rho, penalize_diagonal=False)

    return dict(zip(res.names, list(res)))['wi']


def gemini(df: pd.DataFrame,
           m: int, f: int, pA: float, pB: float) -> Tuple[np.array, np.array]:
    """
    A wrapper for the GEMINI model.

    Arguments
    ---------
        df
            The input dataset.
        m, f
            The number of spatial and temporal dimensions respectively.
        pA, pB
            The spatial and temporal regularization penalties.

    Returns
    -------
        A, B
            The spatial and temporal precision matrices.

    """
    assert df.shape[1] == m * f, (
        "Expected a DataFrame with {} columns, found {} "
        "columns instead!".format(f * m, df.shape[1])
        )

    n = len(df)
    XTX = np.zeros((m, m))
    XXT = np.zeros((f, f))

    for _, row in df.iterrows():
        X = np.reshape(row.values, (f, m), order='F')
        XTX += X.T @ X
        XXT += X @ X.T

    WA = np.diag(XTX)
    WB = np.diag(XXT)
    GA = XTX / np.sqrt(np.outer(WA, WA))
    GB = XXT / np.sqrt(np.outer(WB, WB))

    GAr = robjects.r.matrix(GA, nrow=m, ncol=m)
    GBr = robjects.r.matrix(GB, nrow=f, ncol=f)
    rA = glasso.glasso(GAr, rho=pA, penalize_diagonal=False)
    rB = glasso.glasso(GBr, rho=pB, penalize_diagonal=False)

    rA_dict = dict(zip(rA.names, list(rA)))
    Arho = rA_dict['wi']
    rB_dict = dict(zip(rB.names, list(rB)))
    Brho = rB_dict['wi']
    fact = np.sum(np.multiply(df.values, df.values)) / n

    WA = np.diag(np.sqrt(n / WA))
    WB = np.diag(np.sqrt(n / WB))
    A = np.sqrt(fact) * WA @ Arho @ WA
    B = np.sqrt(fact) * WB @ Brho @ WB

    return A, B
