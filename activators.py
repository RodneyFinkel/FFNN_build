import numpy as np
import matplotlib.pyplot as plt 

# softplus function
def softplus(x):
    return np.log(1 + np.exp(x))

x = np.linspace(-10, 10, 100) # adjust range and density as needed
y = softplus(x) 
plt.plot(x, y, label='sofplus function')
plt.xlabel('x')
plt.ylabel('softplus(x)')
plt.title('Softplus Function')
plt.legend()
plt.grid(True)
plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100) # adjust range and density as needed
y = sigmoid(x) 
plt.plot(x, y, label='sigmoid function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.legend()
plt.grid(True)
plt.show()