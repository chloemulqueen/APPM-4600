# import libraries
import numpy as np
    


# computing order of convergence fo the algorithm that created the approximation
def compute_order(x, xstar):
    diff1 = np.abs(x[1::]+ xstar)
    
    diff2 = np.abs(x[0:-1] - xstar)
    
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
    print('the order equation is')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha * log(|p_n-p|) where')
    print('lambda = ', str(np.exp(fit[1])))
    print('alpha = ', str(fit[0]))
    
    return [fit, diff1, diff2]


# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    
    x = np.zeros((Nmax,1))
    #x[0] = x0
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       
       x[count-1] = x1
       
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return xstar,ier, x[:count], count
       x0 = x1

    xstar = x1
    ier = 1
    
    return xstar, ier, x[:count], count
    
def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**(1/2)
# fixed point is alpha1 = 1.4987....

     f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     xstar,ier, x, iter = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
     print('number of iterations: ', iter)
     print('compute order:', compute_order(x, xstar))
     
     
    
#test f2 '''
#     x0 = 0.0
 #    [xstar,ier, x, count] = fixedpt(f2,x0,tol,Nmax)
 #    print('the approximate fixed point is:',xstar)
 #    print('f2(xstar):',f2(xstar))
 #    print('Error message reads:',ier)

driver()




