import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mass_lifter_heat_engine_raw_data.csv')

height, pressure, temperature, n = df.to_numpy().T
volume = height/10 * (3.25/2)**2 * np.pi + 190

plt.plot(volume, pressure, 'ro')
plt.plot(volume, pressure, 'r')
plt.xlabel('volume (cm^3)')
plt.ylabel('pressure (kPa)')
plt.show()

plt.plot(volume, temperature, 'ro')
plt.plot(volume, temperature, 'r')
plt.xlabel('volume (cm^3)')
plt.ylabel('temperature (K)')
plt.show()

works = []
for i in range(4):
    work = 0.5 * (volume[i+1] - volume[i]) * (pressure[i] + pressure[i+1])
    works.append(work)

print(works)
print(np.sum(works))

volume[-1] = volume[0]
pressure[-1] = pressure[0]

works2 = []
for i in range(4):
    work = 0.5 * (volume[i+1] - volume[i]) * (pressure[i] + pressure[i+1])
    works2.append(work)

print(works2)
print(np.sum(works2))
