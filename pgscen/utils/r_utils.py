"""Utility functions for working with distributions and models created in R."""

import warnings
from typing import Union, Tuple
from rpy2.robjects.functions import SignatureTranslatedFunction as ECDF
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
timeDate = importr('timeDate')
Rsafd = importr('Rsafd')
glasso = importr('glasso')
qgraph = importr('qgraph')
splines = importr('splines')
stats = importr('stats')


def point_mass(data: np.array, masspt: float, threshold: float = 0.05) -> bool:
    """
    Check if the input data has a point mass at a location

    Arguments
    ---------
        data
            A 1D array of numeric values.
        masspt
            Location of potential point mass.
        threshold
            Minimum percentage to be considered a point mass.

    """
    return (data == masspt).sum() / len(data) >= threshold


def gpd_tail(data: np.array, lower: float = 0.15, upper: float = 0.85,
             bins: int = 3, bin_threshold: int = 3,
             range_threshold: float = 0.05) -> str:
    """
    Check if the data's tails conform to the Gaussian distribution.

    Arguments
    ---------
        data
            A 1D array of numeric values.
        lower, upper
            Percentile cutoffs to use for defining the tails of the data.
        bins
            How many bins to discretize the tails into when
            checking their shape.
        bin_threshold
            The maximum number of values by which a bin closer to the tail can
            exceed its neighbor before the tail is considered non-conforming.
        range_threshold
            If the range of the lower or upper tails is less than this
            proportion of the range of the whole data, they conform.

    Returns
    -------
        tail_bool
            One of 'none', 'lower', 'upper', or 'two', specifying which of the
            data's tails are non-conforming.

    """

    # do not try to fit GPD if sample size < 100
    if len(data) < 100:
        return 'none'

    # get "lower" and "upper" parts of input data
    sdata = sorted(data)
    drange = sdata[-1] - sdata[0]
    n = len(data)
    nlow = int(lower * n)
    nup = int(upper * n)

    # find range of lower and upper tails
    lower_range = sdata[nlow - 1] - sdata[0]
    upper_range = sdata[-1] - sdata[nup]

    # if the range of the lower tail is small, it is conforming...
    if lower_range < range_threshold * drange:
        ll = False

    # ...otherwise, we discretize the values in the lower tail into bins
    else:
        lower_df = pd.DataFrame({'lower': sdata[:nlow]})
        lower_counts = lower_df.groupby(pd.cut(
            lower_df['lower'], bins)).count().sort_index().values.ravel()

        # if a bin has much more values than the neighbour to its right in the
        # distribution, the lower tail is non-conforming
        ll = True
        for i in range(len(lower_counts) - 1):
            if lower_counts[i] > lower_counts[i + 1] + bin_threshold:
                ll = False
                break

    # likewise, if the range of the upper tail is small it is conforming
    if upper_range < range_threshold * drange:
        uu = False

    # otherwise, check that bins are not much bigger than their left neighbors
    else:
        upper_df = pd.DataFrame({'upper': sdata[nup:]})
        upper_counts = upper_df.groupby(pd.cut(
            upper_df['upper'], bins)).count().sort_index().values.ravel()
        
        uu = True
        for i in range(len(upper_counts) - 1):
            if upper_counts[i + 1] > upper_counts[i] + bin_threshold:
                uu = False
                break

    if ll and uu:
        return 'two'
    elif ll:
        return 'lower'
    elif uu:
        return 'upper'
    else:
        return 'none'


def pgpd(dist: GPD, x: np.array) -> np.array:
    """Wrapper for pgpd function in Rsafd; computes CDF at all values in x."""

    f = Rsafd.pgpd
    return np.array(f(dist,robjects.FloatVector(x)))


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
    

def fit_dist(data: np.array) -> Union[GPD, ECDF]:
    """Fit a distribution (GPD or the emperical distribution) function."""

    # perturb the data if it has a point mass at zero
    tiny = 1e-3
    if point_mass(data, 0., threshold=0.05):
        data += tiny * (2 * np.random.rand(len(data)) - 1)

    # determine tails
    tail = gpd_tail(data)

    if tail != 'none':
        try:
            return Rsafd.fit_gpd(robjects.FloatVector(data),
                                 tail=tail, plot=False)

        except:
            warnings.warn(f'{tail} tail has been detected, but unable to fit '
                          f'GPD, using ECDF instead', RuntimeWarning)

            return stats.ecdf(robjects.FloatVector(data))

    else:
        return stats.ecdf(robjects.FloatVector(data))


def pdist(dist: Union[GPD, ECDF], x: np.array) -> np.array:
    """Evaluate a CDF function."""

    if tuple(dist.rclass)[0][0:3] == 'gpd':
        return np.array(Rsafd.pgpd(dist,robjects.FloatVector(x)))

    elif tuple(dist.rclass)[0] == 'ecdf':
        return np.array(dist(robjects.FloatVector(x)))

    else:
        raise(RuntimeError('Unrecognized distribution class {}'.format(
            tuple(dist.rclass))))
    

def qdist(dist: Union[GPD, ECDF], x: np.array) -> np.array:
    """Compute the quantiles of the distribution."""

    if tuple(dist.rclass)[0][0:3]=='gpd':
        # GPD
        return qgpd(dist,x)

    elif tuple(dist.rclass)[0] == 'ecdf':
        # Empirical CDF
        return stats.quantile(dist,robjects.FloatVector(x))

    else:
        raise(RuntimeError('Unrecognized distribution class {}'.format(
            tuple(dist.rclass))))
    

# def gaussianize(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
#     """Transform the data to fit a Gaussian distribution."""

#     unif_df = pd.DataFrame(columns=df.columns,index=df.index)
#     dist_dict = dict()

#     for col in df.columns:
#         data = np.ascontiguousarray(df[col].values)

#         dist_dict[col] = stats.ecdf(data)
#         unif_df[col] = np.array(dist_dict[col](robjects.FloatVector(data)))

#     unif_df.clip(lower=1e-5, upper=0.99999, inplace=True)
#     gauss_df = unif_df.apply(norm.ppf)

#     return dist_dict, gauss_df

def gaussianize(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """Transform the data to fit a Gaussian distribution."""

    unif_df = pd.DataFrame(columns=df.columns,index=df.index)
    dist_dict = dict()

    for col in df.columns:
        data = np.ascontiguousarray(df[col].values)

        dist_dict[col] = fit_dist(data)
        unif_df[col] = np.array(pdist(dist_dict[col],data))

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
    COV = df.cov().values
    RCOV = robjects.r.matrix(COV, nrow=m, ncol=m)

    RES = f(RCOV, rho=rho, penalize_diagonal=False)
    RES_DICT = dict(zip(RES.names, list(RES)))
    INVCOV = RES_DICT['wi']

    return INVCOV


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
        "Expected a DataFrame with {} columns, got {}".format(f * m,
                                                              df.shape[1])
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


def get_ecdf_data(cdf: ECDF) -> np.array:
    return stats.knots(cdf)

