import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

starting_freq = 1
ending_freq = 0.7

duration = 146 #s

def freq(t):
    return (ending_freq - starting_freq)/duration * t + starting_freq

t = np.linspace(0, 146, round(146/0.05))
angle = 2.8*np.sin(2 * np.pi * freq(t) * t)

df = np.array([t, angle]).T
df = np.round(df, 3)
df = pd.DataFrame(df, columns=['time(s)', 'angle(rad)'])

df.to_csv('./driving_force.csv')

plt.plot(t, angle, 'r')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()