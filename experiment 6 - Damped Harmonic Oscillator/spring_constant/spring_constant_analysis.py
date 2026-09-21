import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./spring_constant.csv')
time, angle = df.to_numpy().T

first_angle = 0
second_angle = 1.97
angular_displacement = second_angle - first_angle #rad

radius = 23.5/1000 #m
g = 9.81 #ms^(-2)

hooked_mass = 20.02/1000 #kg

torque = radius * hooked_mass * g

disk_mass = 122.54/1000 #kg
disk_radius = 48/1000 #m

I = 0.5 * disk_mass * disk_radius**2

torsional_spring_constant = torque / angular_displacement
w0 = (torsional_spring_constant/I) ** 0.5

print(f"angular_displacement={angular_displacement}\ntorque={torque}\nkappa={torsional_spring_constant}\nw0={w0}")

plt.plot(time, angle, 'r')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
#plt.show()