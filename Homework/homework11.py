import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad #type:ignore

## Composite Trapezoidal Rule and Composite Simpsons
## Code from Canvas -- composite_int.py
def driver():
    func = lambda s : 1/(1+s**2)
    n1 = math.sqrt(3/5)
    n2 = 9 **(1/4)
    a = -5
    b = 5
    print("Trapezoidal solve: ", CompTrap(a,b,n1,func))
    print("Simpson solve: ", CompSimp(a,b,n2,func))

def driver_part2():
    func = lambda s : 1/(1+s**2)
    n = 5
    a = -5
    b = 5
    print("Trapezoidal solve, n set: ", CompTrap(a,b,n,func))
    print("Simpson solve, n set : ", CompSimp(a,b,n,func))
    
    print("Using Scipy:", scipy.integrate.quad(func, a, b))
    
    
def CompTrap(a,b,n,func):
    h = (b-a)/n

    xnode = a+np.arange(0,n+1)*h

    I_trap = h*func(xnode[0])*1/2

    for j in range(1,n):
        I_trap = I_trap+h*func(xnode[j])
        
    I_trap= I_trap + 1/2*h*func(xnode[n])
    return I_trap


def CompSimp(a,b,n,func):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = func(xnode[0])
    nhalf = n/2
    for j in range(1,int(nhalf)+1):
        # even part
        I_simp = I_simp+2*func(xnode[2*j])
        # odd part
        I_simp = I_simp +4*func(xnode[2*j-1])
    I_simp= I_simp + func(xnode[n])

    I_simp = h/3*I_simp
    return I_simp



#driver()
driver_part2()


