import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./exp2_50g_4/angle.csv')
time, angle = df.to_numpy().T

plt.plot(time, angle, 'ro')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')

plt.show()