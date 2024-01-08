import numpy as np
import matplotlib.pyplot as plt

def generate_data(n: int) -> np.ndarray:
    x = np.linspace(0, 1, n)
    x = x.reshape(len(x), 1)
    y = np.sin(2 * np.pi * x)
    return x, y

x, y = generate_data(100)
plt.plot(x, y)
plt.show()
