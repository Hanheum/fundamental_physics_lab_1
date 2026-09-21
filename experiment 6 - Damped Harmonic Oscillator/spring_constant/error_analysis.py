import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_oscillator = pd.read_csv('./simple_harmonic_disk.csv')

time, angle = df_oscillator.to_numpy().T

w0_natural = 4.0745
w0_exp = 4.0711783847815886

beta = (w0_natural**2 - w0_exp**2)**0.5

A = 1
t1 = 2.44

t0 = (1/w0_exp)*(np.pi - np.atan(beta/w0_exp))
phi = t1 - t0

def relation(t):
    return A * np.exp(-beta * (t-phi)) * np.cos(w0_exp * (t-phi))

step = round(t1 / 0.05)
A = angle[step]/relation(time[step])

step2 = round(len(time)/6)
result = relation(time[step2:])
plt.plot(time[step2:], result, 'b')
plt.plot(time, angle, 'r')
plt.show()