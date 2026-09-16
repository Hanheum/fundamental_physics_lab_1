import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('boyles_law_raw_data.csv')

mass, piston_height, pressure, temperature = df.to_numpy().T
volume = piston_height/10 * (3.25/2)**2 * np.pi

n = pressure * volume / 8314 / temperature
print(n)

plt.plot(volume, n, 'ro')

mean_gradient = np.mean(pressure/8314/temperature)

a=2.1502060917555355e-05
b=0.0011634975671768188
c = np.array([a, b], dtype=np.float32)

def relation(v):
    return c[0] * v + c[1]

def MSE(v, n):
    n_pred = relation(v)
    return np.mean((n_pred - n)**2)

def get_gradient(v, n):
    n_pred = relation(v)
    dLda = np.mean((n_pred - n) * v) / 100
    dLdb = np.mean((n_pred - n))
    return np.array([dLda, dLdb])

i = 0
equal_stack = 0
previous_loss = -1
learning_rate = 1e-6
while True:
    new_lr = open('./learning_rate.txt', 'r').read()
    if len(new_lr) != 0:
        learning_rate = float(new_lr)

    i += 1

    loss = MSE(volume, n)
    if previous_loss == loss:
        equal_stack += 1
    else:
        equal_stack = 0

    c -= learning_rate * get_gradient(volume, n)

    if equal_stack >= 5:
        break

    previous_loss = loss
    print(f'epoch:{i}, loss:{loss}')

print(f"a={c[0]}\nb={c[1]}\nRMSE={loss**0.5}")

mean = np.mean(n)
mean_squared = np.mean(n**2)
var = mean_squared - mean**2
R2 = 1-(loss/var)
print(R2)

domain = np.linspace(np.min(volume), np.max(volume), 100)
image = relation(domain)

plt.plot(domain, image, 'b')
plt.xlabel('volume (cm^3)')
plt.ylabel('n (mol)')
plt.show()