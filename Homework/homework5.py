import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# Problem 1


def iteration():
    # Initial guesses
    x_n = 1.0
    y_n = 1.0

    # Matrix - (where inverse jacobian goes)
    A = np.array([[1/6, 1/18],
                [0, 1/6]])

    # Iterate
    for n in range(20): 
        f = 3*x_n**2 - y_n**2
        g = 3*x_n*y_n**2 - x_n**3 - 1
        F = np.array([[f],
                    [g]])
        
        # Update the values
        P_k = np.matmul(A, F)
        x_n = x_n - P_k[0,0]
        y_n = y_n - P_k[1,0]

        print("Iteration", n + 1,": x = ",x_n,"y = ", y_n)

    print("Final approx: x = ", x_n, "y = ", y_n)
    
    
def driver():

    x0 = np.array([1.0, 1.0])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Newton: number of iterations is:',its)
     
def evalF(x): 
    F = np.zeros(2)
    
    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 - 1
    
    return F


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    J = np.array([[1/6, 1/18],
                [0, 1/6]])
    
    for its in range(Nmax):
       
       F = evalF(x0)
 
       x1 = x0 - np.matmul(J,F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

driver()