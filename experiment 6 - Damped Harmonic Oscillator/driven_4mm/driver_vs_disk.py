import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./damped_angle_data.csv')
time, angle = df.to_numpy().T

df = pd.read_csv('./driving_force.csv', usecols=['time(s)', 'angle(rad)'])
time2, angle2 = df.to_numpy().T

plt.plot(time, angle, 'b')
plt.plot(time2, angle2, 'r')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()