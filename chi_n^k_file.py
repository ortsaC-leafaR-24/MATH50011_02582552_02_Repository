import numpy as np
from numpy.random import uniform, normal
import scipy.fft as fft
import matplotlib.pyplot as plt
from statsmodels.api import qqplot_2samples
from scipy.stats import ks_2samp

def chi_1_k_gen(k):
    """
    returns pdf of chi_1^k distribution. 
    
    k is assumed to be a positive integer
    """
    if k % 2 == 1:
        def _out(x):
            """pdf returned, input x is assumed to be real"""
            yscale = k*(2*np.pi)**0.5
            #the chi-1 k distribution is not defined for 0 for k > 1
            if x ==0 and k > 1:
                return np.nan
            elif x < 0:
                return _out(-x)
            return x**(1/k-1)*np.exp(-0.5*(x**(2/k)))/yscale
    else:
        def _out(x):
            """pdf returned, input x is assumed to be real"""
            yscale = k*(2*np.pi)**0.5/2
            if x <= 0:
                return 0
            else:
                return x**(1/k-1)*np.exp(-0.5*x**(2/k))/yscale
    return _out


def chi_n_k_gen(k, n, x_step = 0.00025, x_ext=None):
    """
    returns value of fft on chi_1_k_gen(k) evaluated at points evenly spaced 
    at x_step from each other within x_ext of the origin
    """
    #if x_ext is unassigned we use an upper bound assuming the underlying
    #distribution generated points 4 standard deviations away from the mean
    #every time
    if x_ext == None:
        x_ext = n*(4-(n-1)/n)**k
    n_x = int(x_ext/x_step)
    b = np.linspace(-x_ext, 0, n_x, endpoint=False)
    x = np.array(-b[::-1])
    _alpha = chi_1_k_gen(k)
    f_1 = np.array([_alpha(j) for j in x])
    if n == 1:
        f_n = f_1
    #use convolution theorem to find FFT for chi_n_k, the inverse FFT
    #to find values of chi_n_k at each x. rescale this such the points draw an 
    #area of 1 to find discrete approximation for PDF of chi_n^k
    else:
        if k % 2 == 1:
            fourier_f1 = fft.dct(f_1)
            f_n = fft.idct(fourier_f1**n)
        else:
            fourier_f1 = fft.rfft(f_1)
            f_n = fft.irfft(fourier_f1**n)
    if k % 2 == 1:
        x, f_n = np.array([*-x[::-1], *x]), np.array([*f_n[::-1], *f_n])
    return x, f_n/(np.sum(f_n)*x_step)


def fft_chi_n_k_cdf(k, n, c_args = None):
    """"
    models arrays representing CDF of chi_n_k distribution at discrete values
    discrete values
    """
    if c_args ==None:
        x_, f_ = chi_n_k_gen(k, n)
    else:
        x_, f_ = chi_n_k_gen(k, n, *c_args)
    f_cdf = np.cumsum(f_)*(x_[1]-x_[0])
    return x_, f_cdf

def fft_chi_n_k_genvals(k, n, _ret=3000, c_args = None):
    """
    approximate underlying random behaviour of chi_n_k distribution
    using value from FFT. Only x values used to calculate that will be
    returned.
    """
    #first, generate the underlying distribution modifying arguments
    #if thse are set by the user
    pvals = uniform(0.0, 1.0,_ret)
    x, f_cdf = fft_chi_n_k_cdf(k, n, c_args)
    return np.array([x[f_cdf < p_][-1] for p_ in pvals])

def true_chi_n_k(k, n, vals=3000):
    beta_list = np.array([normal(size=n) for p in range(vals)])**k
    return np.array([np.sum(beta_list[j]) for j in range(vals)])


def r_tail(xder, cnk_arr, lim_ = 1e-6):
    """
    removes a region of points extremely close to zero from a returned
    chi_n_k_gen function for plotting purposes.
    """
    c_max = lim_*max(cnk_arr) 
    return xder[cnk_arr >= c_max], cnk_arr[cnk_arr >= c_max]


def r_head(xder, cnk_arr, lim_ = 0.25):
    """
    removes a region of points with extreme distribution values from 
    a returned chi_n_k_gen function for plotting puposes
    """
    return xder[cnk_arr <= lim_], cnk_arr[cnk_arr <= lim_]


def plot_and_test(k, n, lim_ = 0.25):
    vals, chi_n_k = chi_n_k_gen(k, n)
    datapoints_tru = true_chi_n_k(k, n)
    datapoints_mdl = fft_chi_n_k_genvals(k, n)
    
    plt.scatter(*r_tail(vals, chi_n_k), s=3, linewidth=0, color="b")
    plt.show()
    plt.scatter(*r_head(*r_tail(vals, chi_n_k), lim_=lim_), s=3, linewidth=0, color="b")
    plt.hist(datapoints_tru, bins=100, density=True, alpha=0.25, color="g")
    plt.hist(datapoints_mdl, bins=100, density=True, alpha=0.25, color="r")
    plt.show()
    plt.hist(datapoints_tru, bins=100, density=True, alpha=0.25, color="g")
    plt.hist(datapoints_mdl, bins=100, density=True, alpha=0.25, color="r")
    plt.show()
    qqplot_2samples(datapoints_tru, datapoints_mdl, line='45')
    plt.show()
    
    return ks_2samp(datapoints_tru, datapoints_mdl)

print(plot_and_test(1, 4, lim_ = 1))
print(plot_and_test(2, 6, lim_ = 2))
print(plot_and_test(3, 6))
print(plot_and_test(4, 5, lim_ = 0.6))
print(plot_and_test(5, 2))
