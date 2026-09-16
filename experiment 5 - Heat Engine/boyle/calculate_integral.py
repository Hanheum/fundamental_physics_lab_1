import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./boyles_law_raw_data.csv')
mass, piston_heights, pressure, temperature = df.to_numpy().T

area = (3.25/2)**2*np.pi
volume = piston_heights/10 * area

R = 8314
T = 296.5
a=2.1502060917555355e-05
b=0.0011634975671768188
c = np.array([R*T*b, R*T*a], dtype=np.float32)

def relation(v):  #p = a/v + b
    return c[0]/v + c[1]

def MSE(v, p):
    p_pred = relation(v)
    return np.mean((p_pred - p)**2)

def get_gradient(v, p):
    p_pred = relation(v)
    dLda = 2*np.mean((p_pred - p)*(1/v))
    dLdb = 2*np.mean((p_pred - p))
    return np.array([dLda, dLdb/100])

i = 0
equal_stack = 0
learning_rate = 1e-6
previous_loss = 100000
while True:
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)
    loss = MSE(volume, pressure)
    c -= learning_rate * get_gradient(volume, pressure)

    if previous_loss == loss:
        equal_stack += 1
    else:
        equal_stack = 0

    if equal_stack >= 5 or loss >= previous_loss:
        break

    i += 1
    previous_loss = loss

    print(f"epoch:{i}, loss:{loss}")

print(f"a={c[0]}\nb={c[1]}\nRMSE={loss**0.5}")

plt.plot(volume, pressure, 'ro')
domain = np.linspace(np.min(volume), np.max(volume), 1000)
plt.plot(domain, relation(domain), 'b')
plt.xlabel('V (cm^3)')
plt.ylabel('P (kPa)')
plt.show()

domain = np.linspace(np.min(volume), np.max(volume), 1000000)
dv = domain[1] - domain[0]
numerical_integral = np.sum(dv * relation(domain))
analytic_integral = c[0] * np.log(np.max(volume)/np.min(volume)) + c[1]*(np.max(volume) - np.min(volume))

print('numerical integral:', numerical_integral)
print('analytic integral:', analytic_integral)