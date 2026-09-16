import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('boyles_law_raw_data.csv')

mass, piston_height, pressure, temperature = df.to_numpy().T
volume = piston_height/10 * (3.25/2)**2 * np.pi

n = pressure * volume / 8314 / temperature
print(n)

plt.plot(1/volume, pressure, 'ro')

a=2.1502060917555355e-05
b=0.0011634975671768188

c = np.array([a, b], dtype=np.float32)

def n(v):
    return c[0] * v + c[1]

R = 8314
T = 296.5

def relation(v):
    return (R*T)*n(v)/v

domain = np.linspace(np.min(1/volume), np.max(1/volume), 100)
image = relation(1/domain)

plt.plot(domain, image, 'g')
plt.xlabel('1/volume (cm^(-3))')
plt.ylabel('pressure (kPa)')

a=2868.138916015625
b=53.15916442871094
c = np.array([a, b], dtype=np.float32)

def relation(v):
    return c[0] / v + c[1]

domain = np.linspace(np.min(1/volume), np.max(1/volume), 100)
image = relation(1/domain)

plt.plot(domain, image, 'b')
plt.show()