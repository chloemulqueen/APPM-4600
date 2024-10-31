import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad #type:ignore


def driver():

#  function you want to approximate
    f = lambda x: math.exp(x)
    f2 = lambda x: 1/ (1+x**2)

# Interval of interest    
    a = -1
    b = 1
# weight function    
    w = lambda x: 1.

# order of approximation
    n = 2

#  Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
      pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err_l = abs(pval-fex)
    plt.semilogy(xeval,err_l,'ro--',label='error')
    plt.legend()
    plt.show()
    
      
    

def eval_legendre_expansion(f,a,b,w,n,x): 
#   This subroutine evaluates the Legendre expansion

#  Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab 
  p = eval_legendre(n,x)
  # initialize the sum to 0 
  pval = 0.0    
  for j in range(0,n+1):
      # make a function handle for evaluating phi_j(x)
      phi_j = lambda x: eval_legendre(n,x)[j]
      # make a function handle for evaluating phi_j^2(x)*w(x)
      phi_j_sq = lambda x: (phi_j(x))**2 * w(x)
      # use the quad function from scipy to evaluate normalizations
      norm_fac,err = quad(phi_j_sq, a, b)
      # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
      func_j = lambda x: (phi_j(x) * f(x) * w(x))/ norm_fac
      # use the quad function from scipy to evaluate coeffs
      aj,err = quad(func_j, a, b)
      # accumulate into pval
      pval = pval+aj*p[j] 
       
  return pval
    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()               


def eval_legendre(n, x):
    
    p = np.zeros(n + 1)
    
    p[0] = 1  # sigma0(x) = 1
    if n >= 1:
        p[1] = x  #sigma1(x) = x
    
    # Recursion
    for k in range(1, n):
        p[k+1] = (1/(k+1)) * ((2*k+1) * x * p[k] - k*p[k-1])
    
    return p



