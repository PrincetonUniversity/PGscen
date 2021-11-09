"""Utility functions for working with distributions and models created in R."""

import warnings
import pandas as pd
import numpy as np
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


def point_mass(data, masspt, threshold=0.05):
    """
    Check if the input data has a point mass at a location

    :param data: input data
    :type data: 1d numpy array
    :param masspt: location of potential point mass
    :type masspt: float
    :param threshold: minimum percentage to be considered a point mass, default to 0.05 
    :type threshold: float

    :return: true of false
    """
    return (data == masspt).sum() / len(data) >= threshold


def gpd_tail(data,lower=0.15,upper=0.85,bins=3,bin_threshold=3,range_threshold=0.05):

    # Do not try to fit GPD if sample size < 100
    if len(data) < 100:
        return 'none'

    # Get ``lower`` and ``upper`` parts of input data
    sdata = sorted(data)
    drange = sdata[-1]-sdata[0]

    n = len(data)
    nlow = int(lower*n)
    nup = int(upper*n)
    
    # Find range of lower and upper tails
    lower_range = np.ptp(sdata[:nlow])
    upper_range = np.ptp(sdata[nup:])
    
    
    if lower_range < range_threshold*drange:
        ll = False
    else:
        lower_df = pd.DataFrame({'lower':sdata[:nlow]})
        lower_counts = lower_df.groupby(pd.cut(lower_df['lower'],bins)).count().sort_index().values.ravel()
#         print(lower_counts)
        
        ll = True
        for i in range(len(lower_counts)-1):
            if lower_counts[i]>lower_counts[i+1]+bin_threshold:
                ll = False
                break
        
    if upper_range < range_threshold*drange:
        uu = False
    else:
        upper_df = pd.DataFrame({'upper':sdata[nup:]})
        upper_counts = upper_df.groupby(pd.cut(upper_df['upper'],bins)).count().sort_index().values.ravel()
        
#         print(upper_counts)
        uu = True
        for i in range(len(upper_counts)-1):
            if upper_counts[i+1]>upper_counts[i]+bin_threshold:
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


def pgpd(dist,x):
    """
    Wrapper for pgpd function in Rsafd. Compute CDF at all values in x

    :param dist: distribtution fitted by fit_gpd
    :type dist: R object sotres GPD 
    :param x: A numeric vector of values at which the cdf is computed
    :type x: 1d numpy array

    :return: 1d numpy array, values of the CDF
    """

    f = Rsafd.pgpd
    return np.array(f(dist,robjects.FloatVector(x)))


def qgpd(dist,x):
    """
    Wrapper for qgpd function in Rsafd. Compute quantiles at all valuess in x

    :param dist: distribtution fitted by fit_gpd
    :type dist: R object sotres GPD 
    :param x: A numeric vector of values at which quantiles are computed
    :type x: 1d numpy array

    :return: 1d numpy array, values of the quantiles
    """
    
    try:
        # Compute quantiles using Rsafd function qgpd
        return np.array(f(dist,robjects.FloatVector(x)))

    except:
        # Compute quantiles using PDF
        ll = min(dist.slots['data'])
        rr = max(dist.slots['data'])
        xx = np.linspace(ll,rr,1001)

        # rule=2 for extrapolation of values outside min and max
        ff = stats.approxfun(Rsafd.pgpd(dist,xx),xx,rule=2)

        return ff(x)
    

def fit_dist(data):
    """
    Fit a distribution (GPD or the emperical distribution) function to the give data set

    :param data: input data
    :type data: 1d numpy array
    :param gpd: wheter to fit GPD
    :type gpd: boolean

    :return: GPD object or ECDF object in R
    """

    # Has point mass at zero?
    tiny = 1e-3
    if point_mass(data,0.,threshold=0.05):
        data += tiny*(2*np.random.rand(len(data))-1)

    # Determine tails
    tail = gpd_tail(data)

    if tail != 'none':
        try:
            return Rsafd.fit_gpd(robjects.FloatVector(data),tail=tail,plot=False)
        except:
            warnings.warn(f'{tail} tail has been detected, but unable to fit GPD, using ECDF instead',RuntimeWarning)
            return stats.ecdf(robjects.FloatVector(data))
    else:
        return stats.ecdf(robjects.FloatVector(data))


def pdist(dist,x):
    """
    Evaluate CDF function

    :param dist: a distribution function
    :type dist: R object stores GPD or ECDF
    :param x: A numeric vector of values at which CDF are computed
    :type x: 1d numpy array

    :return: 1d numpy array, values of the CDF
    """
    if tuple(dist.rclass)[0][0:3] == 'gpd':
        return np.array(Rsafd.pgpd(dist,robjects.FloatVector(x)))
    elif tuple(dist.rclass)[0] == 'ecdf':
        return np.array(dist(robjects.FloatVector(x)))
    else:
        raise(RuntimeError('Unrecognized distribution class {}'.format(tuple(dist.rclass))))
    

def qdist(dist,x):
    """
    Compute the quantiles of the distribution

    :param dist: distribtution fitted by fit_gpd
    :type dist: R object sotres GPD or ECDF
    :param x: A numeric vector of values at which quantiles are computed
    :type x: 1d numpy array

    :return: 1d numpy array, values of the quantiles
    """
    if tuple(dist.rclass)[0][0:3]=='gpd':
        # GPD
        return qgpd(dist,x)

    elif tuple(dist.rclass)[0] == 'ecdf':
        # Emperical CDF
        return stats.quantile(dist,robjects.FloatVector(x))
    else:
        raise(RuntimeError('Unrecognized distribution class {}'.format(tuple(dist.rclass))))
    

def gaussianize(df):
    """
    Make data to be Gaussian.

    :param df: data to be Gaussianized
    :type df: pandas DataFrame

    :return: a dictionary of distributions and 
            a pandas DataFrame sotres ``Gaussian`` data
            both keyed by column name in df
    """
    unif_df = pd.DataFrame(columns=df.columns,index=df.index)
    dist_dict = dict()
    for col in df.columns:
        data = np.ascontiguousarray(df[col].values)

        dist_dict[col] = stats.ecdf(data)
        unif_df[col] = np.array(dist_dict[col](robjects.FloatVector(data)))

    unif_df.clip(lower=1e-5,upper=0.99999,inplace=True)
    gauss_df = unif_df.apply(norm.ppf)
    return dist_dict,gauss_df


def graphical_lasso(df,m,rho):
    """
    Wrapper for glasso
    """
    assert df.shape[1]==m, 'Expected a DataFrame with {} columns, got {}'.format(m)

    f = glasso.glasso
    COV = df.cov().values
    RCOV = robjects.r.matrix(COV,nrow=m,ncol=m)
    RES = f(RCOV,rho=rho,penalize_diagonal=False)
    RES_DICT = dict(zip(RES.names,list(RES)))
    INVCOV = RES_DICT['wi']

    return INVCOV

"""
GEMINI

:param df: input data
:type df: pandas DataFrame
:m: size of the spatial dimension
:type m: int
:param f: size of the temporal dimension
:type f: int
:param pA: spatial penalty 
:type pA: float
:param pB: temporal penalty
:type pB: float

:return: precision matrices A (spatial) and B (temporal)

"""
def gemini(df,m,f,pA,pB):
    assert df.shape[1]==m*f, 'Expected a DataFrame with {} columns, got {}'.format(f*m,df.shape[1])
        
    n = len(df)
    XTX = np.zeros((m,m))
    XXT = np.zeros((f,f))
    for _,row in df.iterrows():
        X = np.reshape(row.values,(f,m),order='F')
        XTX += X.T@X
        XXT += X@X.T

    WA = np.diag(XTX)
    WB = np.diag(XXT)

    GA = XTX/np.sqrt(np.outer(WA,WA))
    GB = XXT/np.sqrt(np.outer(WB,WB))
    
    GAr = robjects.r.matrix(GA,nrow=m,ncol=m)
    GBr = robjects.r.matrix(GB,nrow=f,ncol=f)

    f = glasso.glasso
    rA = f(GAr,rho=pA,penalize_diagonal=False)
    rB = f(GBr,rho=pB,penalize_diagonal=False)

    rA_dict = dict(zip(rA.names,list(rA)))
    Arho = rA_dict['wi']
    rB_dict = dict(zip(rB.names,list(rB)))
    Brho = rB_dict['wi']

    fact = np.sum(np.multiply(df.values,df.values))/n

    WA = np.diag(np.sqrt(n/WA))
    WB = np.diag(np.sqrt(n/WB))

    A = np.sqrt(fact)*WA@Arho@WA
    B = np.sqrt(fact)*WB@Brho@WB

    return A,B

"""
wi2net wrapper
"""
def wi2net(prec):
    return np.array(base.as_matrix(qgraph.wi2net(robjects.r.matrix(robjects.FloatVector(prec.ravel()),nrow=prec.shape[0]))))

"""
"""
def save_corgraph(pcor,filename,labels,minimum=0,title=None,font_size=2.0):
    assert pcor.shape[0]==pcor.shape[1],\
        'Expected a square matrix, got one with size {}'.format(pcor.shape)

    if not title:
        title = ''

    r_pcor = robjects.r.matrix(robjects.FloatVector(pcor.ravel()),nrow=pcor.shape[0])
    qgraph.qgraph(r_pcor,minimum=minimum,layout='spring',labels=labels,label_cex=font_size,
                  filetype='pdf',filename=filename,title=title)


def get_ecdf_data(cdf):
    return stats.knots(cdf)

