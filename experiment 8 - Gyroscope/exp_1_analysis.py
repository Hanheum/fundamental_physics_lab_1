import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

y = [0.27, 0.42, 0.55, 0.72, 0.95]
dt = [5.342, 6.412, 7.212, 8.062, 9.082]

y = np.array(y, dtype=np.float32)
dt = np.array(dt, dtype=np.float32)
y -= 0.02
dt -= 1.533

c = np.random.randn(2)

def distance(t):
    return c[0] * t**2 + c[1]

def MSE(t, y):
    y_pred = distance(t)
    return np.mean((y_pred - y)**2)

def get_gradient(t, y):
    y_pred = distance(t)
    dLdA = np.mean((y_pred - y)*t**2)
    dLdB = np.mean((y_pred - y))

    return np.array([dLdA/10, dLdB])

i = 0
previous_loss = -1
learning_rate = 1e-6
equal_stack = 0

while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(dt, y)
    if previous_loss == loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss

    c -= learning_rate * get_gradient(dt, y)

    if equal_stack >= 5:
        break

    print(f"epoch:{i}, loss:{loss}")

print(f"A={c[0]}\nB={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(y**2)-np.mean(y)**2)}")

plt.plot(dt, y, 'ro')
domain = np.linspace(0, np.max(dt)+1, 100)
plt.plot(domain, distance(domain), 'b')
plt.xlabel('delta t (s)')
plt.ylabel('y (m)')
plt.show()