import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

absolute_zero = -273.15

df = pd.read_csv('./charles_law_raw_data.csv')
height, temperature = df.to_numpy().T

temperature -= absolute_zero #convert to Kelvin

piston_area = (3.25/2)**2 * 3.14 #in cm^2
volume = height/10 * piston_area + 190 #in cm^3

volume = volume.astype(np.float32)
temperature = temperature.astype(np.float32)

c = np.array([0.75, 0], dtype=np.float32)

def relation(t):
    return c[0] * t + c[1]

def MSE(t, v):
    N = len(v)
    v_pred = relation(t)
    return (1/N)*np.sum((v_pred - v)**2)

def get_gradient(t, v):
    N = len(v)
    v_pred = relation(t)
    dLda = (2/N)*np.sum((v_pred - v)*t)/1000
    dLdb = (2/N)*np.sum(v_pred - v)
    return np.array([dLda, dLdb])

plt.plot(temperature, volume, 'ro')
plt.xlabel('temperature (K)')
plt.ylabel('volume (cm^3)')
plt.show()

learning_rate = 1e-6
i = 0
previous_loss = -1
equal_stack = 0
while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(temperature, volume)
    gradient = get_gradient(temperature, volume)
    c -= gradient * learning_rate
    print(f'epoch:{i}, loss:{loss}')

    if previous_loss == loss:
        equal_stack += 1
    else:
        equal_stack = 0
    if loss <= 0.1 or (previous_loss == loss and equal_stack >= 5):
        break
    previous_loss = loss

domain = np.linspace(np.min(temperature), np.max(temperature), 100)
range = relation(domain)

print(f"a={c[0]}\nb={c[1]}\nRMSE={loss**0.5}")

plt.plot(domain, range, 'b')
plt.plot(temperature, volume, 'ro')
plt.xlabel('temperature (K)')
plt.ylabel('volume (cm^3)')
plt.show()