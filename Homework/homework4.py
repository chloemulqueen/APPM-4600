
import numpy as np
import matplotlib.pyplot as plt



#Problem 1
t = 5184000
f = lambda s: np.exp(-s**2)
#f_x = lambda x: (2/np.sqrt(np.pi)) * np.exp(x/1.430784) - 3/7
x = np.linspace(0,10)
f_x = [(2/np.sqrt(np.pi)) * np.exp(x/1.430784) - 3/7 for x in x ]

plt.plot(x, f_x)
plt.show()


#1b 


def driver_bisection():

# use routines    
    f = lambda x: (2/np.sqrt(np.pi)) * np.exp(x/1.430784) - 3/7
    a = 0
    b = 2

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-13

    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))




# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]
      
#driver_bisection()               

def driver_newt():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2
    f = lambda x: (2/np.sqrt(np.pi)) * np.exp(x/1.430784) - 3/7
    fp = lambda x: (2/np.sqrt(np.pi)) * np.exp(-x/1.430784) * 1/1.430784
    p0 = 2
    Nmax = 100
    tol = 1e-13
    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

def newton(f,fp,p0,tol,Nmax):
#Newton iteration.
#Inputs:
#f,fp - function and derivative
#p0 - initial guess for root
#tol - iteration stops when p_n,p_{n+1} are within tol
#Nmax - max number of iterations
#Returns:
#p - an array of the iterates
#pstar - the last iterate
#info - success message
#- 0 if we met tol
#- 1 if we hit Nmax iterations (fail)

    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]
driver_newt()

# Problem 4
def problem4():

    f = lambda x: x**6 - x - 1
    fp = lambda x: 6 * x**5 - 1
    p0 = 1.2
    Nmax = 100
    tol = 1.e-14
    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

# Problem 5
def driver():

    f = lambda x: x**6 - x - 1
    fp = lambda x: 6 * x**5 - 1
    p0 = 1.2
    Nmax = 100
    tol = 1.e-14
    (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

def newton(f,fp,p0,tol,Nmax):
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

#driver()

import numpy as np
def driver_secant():
    f = lambda x: x**6 - x - 1
    p0 = 2
    p1 = 1
    Nmax = 100
    tol = 1.e-14
    (p,pstar,info,it) = secant(f,p0,p1,tol, Nmax)
    print('the approximate root is', '%16.16e' % pstar)
    print('the error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

def secant(f,p0,p1,tol,Nmax):
    if np.abs(f(p1)- f(p0)) == 0:
        ier = 1
        pstar = p1
        return 
    p = np.zeros(Nmax+1)
    p[0] = p0
    p[1] = p1
    for it in range(Nmax):
        p2 = p1 - (f(p1)* (p1 - p0)/(f(p1) - f(p0)))
        p[it+1] = p2
        if (abs(p2-p1) < tol):
            pstar = p2
            info = 0
            return [p,pstar,info,it]
        p0 = p1
        p1 = p2
    if abs(f(p1)- f(p0)) == 0:
        info = 1
        pstar = p2
        return
    pstar = p2
    info = 1
    return [p,pstar,info,it]
#driver_secant()


