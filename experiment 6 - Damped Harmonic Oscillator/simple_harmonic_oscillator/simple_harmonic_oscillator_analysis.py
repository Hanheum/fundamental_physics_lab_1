import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

disk_radius = 48/1000 #m
disk_mass = 122.54/1000 #kg

I = 0.5 * disk_mass * disk_radius **2

df = pd.read_csv('./simple_harmonic_disk.csv')
time, angle = df.to_numpy().T

first_peak = 2.44
first_peak_value = -1.514
second_peak = 7.07
second_peak_value = 1.425

plt.plot(time, angle, 'r')
plt.plot([first_peak, second_peak], [first_peak_value, second_peak_value], 'bo')
plt.plot([first_peak, first_peak], [first_peak_value, second_peak_value], 'b')
plt.plot([second_peak, second_peak], [first_peak_value, second_peak_value], 'b')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()

T = (second_peak - first_peak)/3 #s
w0 = 2*np.pi / T #Hz
print(f"period = {T}")
print(f"angular_frequency = {w0}")
