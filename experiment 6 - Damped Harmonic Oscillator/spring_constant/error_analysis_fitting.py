import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_oscillator = pd.read_csv('./simple_harmonic_disk.csv')

time, angle = df_oscillator.to_numpy().T

w = 4.0745
w0_exp = 4.0711783847815886

A = 1
t1 = 2.44

c = np.array([w, A], dtype=np.float32)

beta = 0
phi = 0

def relation(t):
    global beta, phi
    beta = (c[0]**2 - w0_exp**2)**0.5
    t0 = (1/w0_exp)*(np.pi - np.atan(beta/w0_exp))
    phi = t1 - t0
    return c[1] * np.exp(-beta * (t-phi)) * np.cos(w0_exp * (t-phi))

step = round(t1 / 0.05)
c[1] = angle[step]/relation(time[step])

def MSE(time, angle):
    angle_pred = relation(time)
    return np.mean((angle_pred - angle)**2)

def get_gradient(time, angle):
    angle_pred = relation(time)
    dLdA = np.mean((angle_pred - angle) * angle_pred / c[1])
    dLdw = np.mean((angle_pred - angle) * angle_pred * (1/c[0]-c[0]/beta*(time - phi)+np.tan(w0_exp*(time-phi))*w0_exp/(beta * c[0])))
    return np.array([dLdw/10, dLdA], dtype=np.float32)

step2 = round(len(time)/6)
time = time[step2:round(step2*4.5)]
angle = angle[step2:round(step2*4.5)]

i = 0
equal_stack = 0
learning_rate = 1e-6
previous_loss = -1

while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(time, angle)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss

    c -= learning_rate * get_gradient(time, angle)

    if equal_stack >= 5:
        break

    print(f"epoch:{i}, loss:{loss}")

print(f"w={c[0]}\nA={c[1]}\nRMSE={loss**2}\nR^2={1-loss/(np.mean(angle**2) - np.mean(angle)**2)}")

plt.plot(time, relation(time), 'b')
plt.plot(time, angle, 'r')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.show()