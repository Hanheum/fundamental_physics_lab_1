import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./boyles_law_raw_data.csv')

mass, piston_height, pressure, temperature = df.to_numpy().T

volume = (3.25/2)**2*np.pi * piston_height/10 #in cm^3

plt.plot(1/volume, pressure, 'ro')
plt.xlabel('V^(-1) (cm^(-3))')
plt.ylabel('P (kPa)')
plt.show()

plt.plot(volume, pressure, 'ro')
plt.xlabel('V (cm)')
plt.ylabel('P (kPa)')
plt.show()

plt.plot(1/volume, pressure, 'ro')
plt.xlabel('V^(-1) (cm^(-3))')
plt.ylabel('P (kPa)')

c = np.array([volume[0]*pressure[0], 0], dtype=np.float32)

v_inverse = 1/volume

def relation(v_inverse):
    return c[0] * v_inverse + c[1]

def MSE(v_inverse, p):
    N = len(p)
    p_pred = relation(v_inverse)
    return (1/N)*np.sum((p_pred - p)**2)

def get_gradient(v_inverse, p):
    N = len(p)
    p_pred = relation(v_inverse)
    dLda = (2/N)*np.sum((p_pred - p)*v_inverse)
    dLdb = (2/N)*np.sum(p_pred - p)
    return np.array([dLda, dLdb])

i = 0
equal_stack = 0
learning_rate = 1e-6
previous_loss = -1
while True:
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)
    loss = MSE(v_inverse, pressure)
    c -= get_gradient(v_inverse, pressure) * learning_rate

    print(f"epoch:{i}, loss:{loss}")
    i += 1
    if previous_loss == loss:
        equal_stack += 1

    previous_loss = loss
    if loss <= 0.001 or equal_stack >= 5:
        break

print(f"a={c[0]}\nb={c[1]}\nRMSE={loss**0.5}")

domain = np.linspace(np.min(v_inverse), np.max(v_inverse), 100)
p_preds = relation(domain)
plt.plot(domain, p_preds, 'b')
plt.show()