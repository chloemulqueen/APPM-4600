import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv 
from numpy.linalg import norm 

# Question 1 Part b 

def driver_prob1(): 
    
    f = lambda x: 1/ (1+ (10*x)**2)
    
    N = 23
    a = -1
    b = 1
    
    ''' Create interpolation nodes'''
    #h = 2 / (N - 1)
    #xint = np.array([-1 + (i - 1) * h for i in range(1, N + 1)])
    xint = np.linspace(a,b,N+1)
#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
#    print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
#    print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    
#    print('coef = ', coef)

# No validate the code
    Neval = 100    
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)
    plt.figure()
    plt.title("N = 23")    
    plt.plot(xeval,yeval,'o-', label = 'Approximation')
    

# exact function
    yex = f(xeval)
    
    err =  norm(yex-yeval) 
    print('err = ', err)
    plt.plot(xeval,yex,'r-',label = 'actual f(x)')
    plt.legend()
    plt.show()
    
    return

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

   
def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V     

driver_prob1()


# Question 2 

def f(x):
    return 1/ (1+ (10*x)**2)

def barycentric_interpolation(x, x_nodes, f_nodes):
    n = len(x_nodes)
    w = np.ones(n)
     
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (x_nodes[j] - x_nodes[i])
    
    numerator = np.zeros_like(x)
    denominator = np.zeros_like(x)
    
    for j in range(n):
        diff = x - x_nodes[j]
        numerator += w[j] * f_nodes[j] / diff
        denominator += w[j] / diff
    
    return numerator / denominator

n = 100

x_nodes = np.linspace(-1, 1, n+1)
f_nodes = f(x_nodes)

x_vals = np.linspace(-1, 1, 400)
f_vals = f(x_vals)
p_vals = barycentric_interpolation(x_vals, x_nodes, f_nodes)

# Plot the approximation and the actual f(x)
plt.figure()
plt.plot(x_vals, f_vals, label='f(x)')
plt.plot(x_vals, p_vals, label='Barycentric Interpolation')
plt.title('n = 100')
plt.legend()
plt.show()


# Question 3:

# Generate Chebyshev nodes
x_nodes = np.cos((2 * np.arange(n + 1) - 1) * np.pi / (2 * (n + 1)))
f_nodes = f(x_nodes)

x_vals = np.linspace(-1, 1, 400)
f_vals = f(x_vals)
p_vals = barycentric_interpolation(x_vals, x_nodes, f_nodes)

# Plot the approximation and the actual f(x)
plt.figure()
plt.plot(x_vals, f_vals, label='f(x)')
plt.plot(x_vals, p_vals, label='Barycentric Interpolation')
plt.title('n = 100')
plt.legend()
plt.show()

