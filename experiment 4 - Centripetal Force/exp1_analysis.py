import numpy as np
import matplotlib.pyplot as plt

r = 7.2 #cm
m = 50 #g

T = np.array([0.78, 0.61, 0.51, 0.43, 0.37, 0.33, 0.3, 0.27, 0.25, 0.23])
v = 2*np.pi/T * (r/100)

F = np.array([0.14, 0.29, 0.44, 0.71, 1.0, 1.19, 1.59, 2.05, 2.43, 2.84])

c = np.random.randn(3)

def function(v):
    return c[0] * v**2 + c[1] * v + c[0]

def MSE(v, F):
    return np.mean((function(v)-F)**2)

def get_gradient(v, F):
    F_pred = function(v)
    dLdA = np.mean((F_pred - F)*v**2)
    dLdB = np.mean((F_pred - F)*v)
    dLdC = np.mean((F_pred - F))
    return np.array([dLdA, dLdB, dLdC])

i = 0
previous_loss = -1
learning_rate = 1e-6
equal_stack = 0
while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(v, F)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    if equal_stack >= 5:
        break

    previous_loss = loss

    c -= learning_rate * get_gradient(v, F)

    print(f"epoch:{i}, loss:{loss}")

print(f"A={c[0]}\nB={c[1]}\nC={c[2]}\nRMSE={loss**0.5}\nR^2={1-loss/np.var(F)}")
