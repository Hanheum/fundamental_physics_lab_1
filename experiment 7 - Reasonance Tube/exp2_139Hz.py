import numpy as np
import matplotlib.pyplot as plt

tube_length = 121 #cm

maximum = np.array([60], dtype=np.float32)
minimum = np.array([0, 120], dtype=np.float32)

maximum = tube_length - maximum
minimum = tube_length - minimum

plt.plot(maximum, [0], 'ro')
plt.plot(minimum, [0, 0], 'bo')
plt.xlabel("length from speaker (cm)")
plt.show()

lam = 60