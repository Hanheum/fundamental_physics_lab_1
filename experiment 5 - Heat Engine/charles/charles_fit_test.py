import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./charles_law_raw_data.csv')
height, temperature = df.to_numpy().T

piston_area = (3.25/2)**2 * 3.14 #in cm^2
volume = height/10 * piston_area + 190 #in cm^3

volume = volume.astype(np.float32)
temperature = temperature.astype(np.float32)

c = np.array([0.70, 0], dtype=np.float32)

def relation(t):
    return c[0] * t + c[1]

def MSE(t, v):
    N = len(v)
    v_pred = relation(t)
    return (1/N)*np.sum((v_pred - v)**2)

loss = MSE(temperature, volume)
print(loss**0.5)

domain = np.linspace(np.min(temperature), np.max(temperature), 100)
image = relation(domain)

plt.plot(temperature, volume, 'ro')
plt.plot(domain, image, 'b')
plt.xlabel('temperature (K)')
plt.ylabel('volume (cm^3)')
plt.show()

