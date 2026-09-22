import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./damped_angle_data.csv')
time, angle = df.to_numpy().T

plt.plot(time, angle, 'r')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()
