import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./exp2_50g_4/angle.csv')
time, angle = df.to_numpy()[150:225].T
time -= time[0]
angle -= angle[0]

c = np.array([-0.05, 4, 0, 0.06*3.14])

def theta(t):
    return c[0] * np.cos(c[1]*t+c[2]) + c[3]*t

def MSE(t, angle):
    angle_pred = theta(t)
    return np.mean((angle_pred - angle)**2)

def get_gradient(t, angle):
    angle_pred = theta(t)
    dLdA = np.mean((angle_pred - angle)*np.cos(c[1]*t+c[2]))
    dLdw = np.mean((angle_pred - angle)*(-c[0]*np.sin(c[1]*t+c[2])*t))
    dLdphi = np.mean((angle_pred - angle)*(-c[0]*np.sin(c[1]*t+c[2])))
    dLdB = np.mean((angle_pred - angle)*t)
    return np.array([dLdA, dLdw, dLdphi, dLdB/100])

i = 0
previous_loss = -1
learning_rate = 1e-6
equal_stack = 0

while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(time, angle)
    if previous_loss == loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss
    #print(get_gradient(time, angle))
    c -= learning_rate * get_gradient(time, angle)

    if equal_stack >= 5:
        break

    print(f"epoch:{i}, loss:{loss}")

print(f"A={c[0]}\nw={c[1]}\nphi={c[2]}\nB={c[3]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(angle**2)-np.mean(angle)**2)}")


plt.plot(time, angle, 'ro')
domain = np.linspace(0, time[-1], 100)
plt.plot(domain, theta(domain), 'b')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')

plt.show()