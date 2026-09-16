import pandas as pd
import numpy as np

df = pd.read_csv('charles_law_raw_data.csv')

piston_height, temperature = df.to_numpy().T
volume = piston_height/10 * (3.25/2)**2 * np.pi + 190

pressure = 101.3

c = np.array([0, 0, 0.75])

def relation(v):
    return (pressure + c[0]/v**2) * (v - c[1]) / c[2]

def MSE(v, t):
    t_pred = relation(v)
    return np.mean((t_pred - t)**2)

def get_gradient(v, t):
    t_pred = relation(v)
    dLdalpha = 2 * np.mean((t_pred - t)*(v**(-2)*(v-c[1])/c[2]))
    dLdbeta = 2 * np.mean((t_pred - t)*(pressure + c[0]/v**2)*(-1)/c[2])/1000
    dLdgamma = 2 * np.mean((t_pred - t) * t_pred * (-1) / c[2]**2)
    return np.array([dLdalpha, dLdbeta, dLdgamma])

learning_rate = 1e-6
i = 0
equal_stack = 0
previous_loss = -1
while True:
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    i += 1
    loss = MSE(volume, temperature)
    c -= learning_rate * get_gradient(volume, temperature)
    if loss == previous_loss:
        equal_stack += 1

    if equal_stack >= 5:
        break

    previous_loss = loss
    print(f"epoch:{i}, loss:{loss}")

print(f"alpha={c[0]}\nbeta={c[1]}\ngamma={c[2]}\nRMSE={loss**0.5}")