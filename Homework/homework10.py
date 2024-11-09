import numpy as np
import matplotlib.pyplot as plt

sinx = lambda x: x - (x**3)/6 + (x**5)/120

pade_33 = lambda x: (x - (7/60)*x**3)/(1+ (1/20)*x**2)

pade_24 = lambda x: x/ (1 + (1/6)*x**2 + (7/360)*x**4)

pade_42 = lambda x: (x - (7/60)*x**3)/(1+ (1/20)*x**2)


# Generate x values
x = np.linspace(0, 5, 100)

# Calculate errors
error_33 = np.abs(sinx(x) - pade_33(x))
error_24 = np.abs(sinx(x) - pade_24(x))
error_42 = np.abs(sinx(x) - pade_42(x))

# Plotting the errors
plt.figure(figsize=(10, 6))
plt.plot(x, error_33, label='Cubic/Cubic', color='blue')
plt.plot(x, error_24, label='Quadratic/Fourth', color='red')
plt.plot(x, error_42, label='Fourth/Quadratic', color='green')
plt.title('Error Comparison')
plt.xlabel('x')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.show()