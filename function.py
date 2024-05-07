import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter


def f1(x):
    bias=0
    D = 2
    sum=0
    for i in range(D):
        sum += x[i]**2 - 10*np.cos(2*math.pi*x[i]) + 10
    print(sum.shape)
    return sum + bias#8

    D = 2
    result = np.sum([(10**6)**((i) / (D - 1)) * x[i]**2 for i in range(D)])+100
    return result#1


# Generate data
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = f1([X[i,j], Y[i,j]])
        

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f1(X, Y)')
ax.set_title('Surface plot of f1(X, Y)')
ax.zaxis.set_major_formatter(FormatStrFormatter('%.1e'))  # 设置刻度格式为科学计数法
plt.show()
plt.savefig('function_plot.png')

