import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('charles_law_raw_data.csv')

piston_height, temperature = df.to_numpy().T
volume = piston_height/10 * (3.25/2)**2 * np.pi + 190

pressure = 101.3

alpha=-1689871.6553291243
beta=-80958.77359583262
gamma=17051.195551298046

c = np.array([alpha, beta, gamma])

def relation(v):
    return (pressure + c[0]/v**2) * (v - c[1]) / c[2]

domain = np.linspace(np.min(volume), np.max(volume), 100)
image = relation(domain)

plt.plot(temperature, volume, 'ro')
plt.plot(image, domain, 'b')
plt.xlabel('temperature (K)')
plt.ylabel('volume (cm^3)')
plt.show()