import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 100
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)
  
     x = scila.solve(A,b)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     print(r)

     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)
     
     A1 = LU_solve(N,M)
     print("A1:", A1)
     


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =  np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     
     return B 
 
def LU_solve(n,m):
    
    a = np.linspace(1,10,m)
    d = 10**(-a)
     
    D2 = np.zeros((n,m))
    for j in range(0,m):
        D2[j,j] = d[j]
        
    '''' create matrices needed to manufacture the low rank matrix'''
    A = np.random.rand(n,n)
    L1, U = scila.lu(A)
    test1 = scila.lu_solve(L1,U)
    
    A =  np.random.rand(m,m)
    L2,R = scila.lu(A)
    test2 = scila.lu_solve(L2,R)
    
    print("Original A:", A)
    B = np.matmul(L1,D2)
    B = np.matmul(B,L2)
     
    return B 
    
    
    
    
 
  
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
