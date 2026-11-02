import numpy as np
import matplotlib.pyplot as plt

distance = np.array([0.64, 0.87, 1])

m = 42.4/1000
g = 9.808

def U(h):
    return m * g * h/100

def K(v):
    return 0.5 * m * (v/100)**2

init = U(116-92)
E1 = U(15) + K(24.52)
E2 = K(181.9)
E3 = U(9) + K(126.1)

E = np.array([E2, E3, E1 ])
E_loss = init - E

plt.plot(distance, E_loss, 'ro')
plt.xlabel('distance traveled ratio (no unit)')
plt.ylabel('energy loss (J)')
plt.show()