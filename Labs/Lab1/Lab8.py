import numpy as np
import numpy.linalg as la
import math
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt


def driver():
    
    f = lambda x: 1/ (1 + (10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
#    for j in range(Neval):
#        fex(j) = f(xeval(j)) 
    plt.figure()
    plt.plot(xeval, fex, 'ro-')
    plt.plot(xeval, yeval, 'bs-')
    plt.legend()
    plt.show()   
     
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval, err, 'ro-')
    plt.show()            

def line_eval(a1, fa1, b1, fb1, xk):
    f0 = lambda x: (x - b1) / (a1 - b1)
    f1 = lambda x: (x - a1) / (b1 - a1)
    return (fa1 * f0(xk) + (fb1 * f1(xk)))    
    
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint[jint]
        b1 = xint[jint+1]
        
        ind= np.where((xeval >= a1) & (xeval <= b1))
        xloc = xeval[ind]
        n = len(xloc)

        fa = f(a1)
        fb = f(b1)
        yloc = np.zeros(len(xloc))
        
        for kk in range(n):
           '''use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           yloc[kk] = line_eval(a1, fa, b1, fb, xeval[kk])
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
        yeval[ind] = yloc
          
        return yeval  
    
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               



