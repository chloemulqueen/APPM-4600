# Chloe Mulqueen - Homework 2

import numpy as np
import math
import matplotlib.pyplot as plt
import random

# Problem 4: 
# Part a

t = np.arange(0,math.pi, math.pi/30)
y = [math.cos(curr) for curr in t]

sum = 0
curr = 0

for curr in range(len(t)):
    sum += t[curr] * y[curr]
    curr +=1 
    
#print("the sum is: ", sum)

# Part b
R = 1.2
dr = 0.1
f = 15
p = 0
theta = np.linspace(0, 2*math.pi)

x_theta = [R * (1 + dr * math.sin(f*t + p))* math.cos(t) for t in theta]
y_theta = [R * (1 + dr * math.sin(f*t + p))* math.sin(t) for t in theta]

plt.plot(theta, x_theta)
plt.plot(theta, y_theta)
plt.title("Plotting x(theta) and y(theta)")
plt.show()

for i in range(10):
    p = random.uniform(0,2)
    R = i
    dr = 0.05
    f = 2 + i
    
    x_theta2 = [R * (1 + dr * math.sin(f*t + p))* math.cos(t) for t in theta]
    y_theta2 = [R * (1 + dr * math.sin(f*t + p))* math.sin(t) for t in theta]
    plt.plot(theta, x_theta2)
    plt.plot(theta, y_theta2)    

plt.title("Second Figure with 10 curves")
plt.show()