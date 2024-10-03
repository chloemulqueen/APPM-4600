import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 


## Pre-Lab
f = lambda x: np.cos(x)

h = 0.01 * 2. ** (-np.arange(0,10))
s = np.pi/2

f_approx = []
c_approx = []

true_deriv = -np.sin(s)

for h in h:
    forward_approx = (f(s+h) - f(s))/h
    f_approx.append(forward_approx)
    
    centered_approx = (f(s+h) - f(s-h)) / (2*h)
    c_approx.append(centered_approx)
 
#Calculate orders of approximation   
forward_errors = np.abs(f_approx - true_deriv)
centered_errors = np.abs(c_approx - true_deriv)


print("Forward Difference Orders:",forward_errors )
print("Centered Difference Orders:", centered_errors)

## 3.2
def driver():
    x0 = np.array([1,0])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  SlackerNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)
    
    
def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 4*x[0]**2 + x[1]**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0]- x[1])

    return F

def evalJ(x): 

    
    J = np.array([[8*x[0], 2*x[1]], 
        [1 - np.cos(x[0]-x[1]), 1+np.cos(x[0]-x[1])]])
    return J

def SlackerNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    xst = x0
    
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
       
       if(norm(x1 - xst) > 0.1):
           J = evalJ(x1)
           Jinv = inv(J)
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   

driver()