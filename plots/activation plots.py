import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
    return [max(0,y) for y in x]

def leaky_relu(x):
    return [max(0.01*y,y) for y in x]

x = np.arange(-2.0,2.0,0.1)
y = sigmoid(x)
z = tanh(x)
a = relu(x)
b = leaky_relu(x)

plt.plot(x,b)
plt.grid(True)
plt.ylim(-0.2,1)
plt.xlabel("z")
plt.ylabel("Leaky ReLU(z)")
plt.show()