import numpy as np
import matplotlib.pyplot as plt

# 3.1.1
x = [1, 2, 3]
print(x * 3)

y = np.array([1,2,3])
print(y * 3)
#3 * y 

#print('this is 3y', 3*y)

# 3.1.3

#X = np.linspace(0, 2*np.pi, 100)
#Ya = np.sin(X)
#Yb = np.cos(X)

#plt.plot(X,Ya)
#plt.plot(X, Yb)
#plt.show()

# X is size 

X = np.linspace(0, 2*np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)

#plt.plot(X,Ya)
#plt.plot(X, Yb)
#plt.show()
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#3.2 

x = np.linspace(1, 10, 10)
y = np.arange(1,21, 2)

print('x', x )
print('y', y)

print(x[0:3])
print('The first three entries of x are', x[0:3])

w =10 ** (-np.linspace(1,10,10))

print('w:', w)

x = np.arange(1, len(w), 1)
