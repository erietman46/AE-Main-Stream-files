import Neural_Networks_Built_in
import numpy as np
from matplotlib import pyplot as plt

a = np.arange(-20,20,0.01)

sigm=1 / (1+np.exp(a))

derivative = sigm * (1-sigm)

plt.figure()
plt.plot(a,sigm)
plt.plot(a,derivative)
plt.xlabel('a')
plt.ylabel('sigmoid')
plt.legend(['Sigmoid, $\sigma(a)$', 'Derivative'])
plt.show()
