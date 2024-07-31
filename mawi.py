import numpy as np
from scipy.special import chndtrix
import math

#import rpy2.robjects as ro
#from rpy2.robjects import pandas2ri

# Activate the automatic conversion of pandas objects to R objects
#pandas2ri.activate()
# Define the R function as a string
#r_code = 
"""mawi <- function(alpha,m,n,eps1_,eps2_,x,y) {
    eqctr <- 0.5 + (eps2_-eps1_)/2 
    eqleng <- eps1_ + eps2_

    wxy <- 0
    pihxxy <- 0
    pihxyy <- 0

    for (i in 1:m)
        for (j in 1:n)
            wxy <- wxy + trunc(0.5*(sign(x[i] - y[j]) + 1))

    for (i in 1:m)
        for (j1 in 1:(n-1))
            for (j2 in (j1+1):n)
                pihxyy <- pihxyy + trunc(0.5*(sign(x[i] - max(y[j1],y[j2])) + 1))

    for (i1 in 1:(m-1))
        for (i2 in (i1+1):m)
            for (j in 1:n)
                pihxxy <- pihxxy + trunc(0.5*(sign(min(x[i1],x[i2]) - y[j]) + 1))

    wxy <- wxy / (m*n)
    pihxxy <- pihxxy*2 / (m*(m-1)*n)
    pihxyy <- pihxyy*2 / (n*(n-1)*m)
    sigmah <- sqrt((wxy-(m+n-1)*wxy**2+(m-1)*pihxxy+(n-1)*pihxyy)/(m*n))

    crit <- sqrt(qchisq(alpha,1,(eqleng/2/sigmah)**2))

    if (abs((wxy-eqctr)/sigmah) >= crit) rej <- 0
    if (abs((wxy-eqctr)/sigmah) < crit)  rej <- 1

    if (is.na(sigmah) || is.na(crit)) rej <- 0
    return(c(rej, crit, sigmah, pihxxy, pihxyy))
}
"""

#def sign(x):
#    if x>0:
#        return 1
#    elif x==0:
#        return 0
#    else:
#        return -1

def mawi(x, y, alpha=.05, eps1_=0.25, eps2_=0.25):
    """
    Implements the Mann-Whitney-Wilcoxon non parametric equivalence test.

    Args:
        alpha: Significance level for the test.
        eps1_: Lower bound for the equivalence zone.
        eps2_: Upper bound for the equivalence zone.
        x: Array of m observations from the first sample.
        y: Array of n observations from the second sample.

    Returns:
        rej: 0 if null hypothesis is rejected, 1 if accepted (due to equivalence).
        (optional) wxy, sigma_h, crit: Calculated test statistics (for potential debugging).

    Raises:
        ValueError: If alpha is not between 0 and 1 or if m or n is non-positive.
    """
    #TODO: assert two samples are single vectors
    #TODO: assert alpha

    m = x.shape[0]
    n = y.shape[0]

    eq_ctr = 0.5 + (eps2_ - eps1_) / 2
    eq_leng = eps1_ + eps2_

    wxy = 0
    pihxx = 0  # Renamed for clarity and consistency
    pihyy = 0  # Renamed for clarity and consistency

    # Vectorize loops for efficiency
    xy_signs = np.sign(x[:, np.newaxis] - y)  # Avoid nested loops
    wxy = np.mean(xy_signs==1)

    #np.array([np.maximum(y[i], y[i:]) for i in range(n)]).flatten()

    x2 = x[:, np.newaxis]
    mask = np.nonzero(np.tril(np.ones((n-1,n-1)))==1)
    y_max = np.maximum(y[:-1],y[1:, np.newaxis])[mask[0], mask[1]].reshape(-1)
    pihyy = 2*np.sum(np.sign(x2 - y_max) == 1)/(n*(n-1)*m)

    mask = np.nonzero(np.tril(np.ones((m-1,m-1)))==1)
    x_min = np.minimum(x[:-1],x[1:, np.newaxis])[mask[0], mask[1]].reshape(-1)
    pihxx = 2*np.sum(np.sign(x_min[:, np.newaxis] - y) == 1)/(m*(m-1)*n)

    temp = (wxy - (m + n - 1) * wxy**2 + (m - 1) * pihxx + (n - 1) * pihyy) / (m * n)

    sigma_h = np.sqrt(temp)

    if not np.isfinite(sigma_h):  # Handle potential NaN/Inf results
        crit = np.nan
        rej = 0
    else:
        crit = np.sqrt(chndtrix(alpha, 1, (eq_leng/2/sigma_h)**2))

        rej = 1 if np.abs((wxy - eq_ctr) / sigma_h) < crit else 0

    # Print results (optional)
    #print(" alpha =", alpha, "  m =", m, "  n =", n, "  eps1_ =", eps1_, "  eps2_ =", eps2_)
    #print(" W+ =", wxy, "  SIGMAH =", sigma_h, "  CRIT =", crit, "  REJ =", rej)
    return rej

"""
def mawi_slow(x, y, alpha=.05, eps1_=0.25, eps2_=0.25):
    
    #Implements the Mann-Whitney-Wilcoxon non parametric equivalence test.

    #Args:
    #    alpha: Significance level for the test.
    #    eps1_: Lower bound for the equivalence zone.
    #    eps2_: Upper bound for the equivalence zone.
    #    x: Array of m observations from the first sample.
    #    y: Array of n observations from the second sample.

    #Returns:
    #    rej: 0 if null hypothesis is rejected, 1 if accepted (due to equivalence).
    #    (optional) wxy, sigma_h, crit: Calculated test statistics (for potential debugging).

    #Raises:
    #    ValueError: If alpha is not between 0 and 1 or if m or n is non-positive.
    #
    #TODO: assert two samples are single vectors
    #TODO: assert alpha

    m = x.shape[0]
    n = y.shape[0]

    eq_ctr = 0.5 + (eps2_ - eps1_) / 2
    eq_leng = eps1_ + eps2_

    wxy = 0
    phixyy = 0  # Renamed for clarity and consistency
    phixxy = 0  # Renamed for clarity and consistency

    # Vectorize loops for efficiency
    for i in range(m):
        for j in range(n):
            wxy += math.trunc(0.5*(sign(x[i] - y[j])+1)) 
    
    for i in range(m):
        for j1 in range(n-1):
            for j2 in range(j1+1, n):
                phixyy += math.trunc(0.5*(sign(x[i]-max(y[j1], y[j2]))+1))

    for i1 in range(m-1):
        for i2 in range(i1+1,m):
            for j in range(n):
                phixxy += math.trunc(0.5*(sign(min(x[i1], x[i2])-y[j])+1))
    
    wxy = wxy/(m*n)
    phixyy = phixyy*2/(n*(n-1)*m)
    phixxy = phixxy*2 / (m*(m-1)*n)

    temp = (wxy - (m + n - 1) * wxy**2 + (m - 1) * phixxy + (n - 1) * phixyy) / (m * n)

    sigma_h = np.sqrt(temp)

    if not np.isfinite(sigma_h):  # Handle potential NaN/Inf results
        crit = np.nan
        rej = 0
    else:
        crit = np.sqrt(chndtrix(alpha, 1, (eq_leng/2/sigma_h)**2))

        rej = 1 if np.abs((wxy - eq_ctr) / sigma_h) < crit else 0

    # Print results (optional)
    print(" alpha =", alpha, "  m =", m, "  n =", n, "  eps1_ =", eps1_, "  eps2_ =", eps2_)
    print(" W+ =", wxy, "  SIGMAH =", sigma_h, "  CRIT =", crit, "  REJ =", rej)
    return rej
"""
# Load the R function into the R global environment
#ro.r(r_code)

# Function to call the R mawi function from Python
#def call_mawi_from_r(x, y, alpha=0.05, eps1_=0.25, eps2_=0.25):
#    m = len(x)
#    n = len(y)
#    result = ro.r['mawi'](alpha, m, n, eps1_, eps2_, ro.FloatVector(x), ro.FloatVector(y))
#    return list(result)


if __name__ == "__main__":
        # Example usage
        x = np.random.randn(100)
        y = np.random.randn(100)
        #print(call_mawi_from_r(x,y))
        print(mawi(x,y))
        #print(mawi_slow(x,y))
