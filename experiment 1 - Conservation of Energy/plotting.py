import matplotlib.pyplot as plt
import numpy as np

m = 42.4/1000
g = 9.808

def U(h):
    return m * g * h/100

def K(v):
    return 0.5 * m * (v/100)**2

highest = U(51) - K(294.1)
middle = U(40.5) - K(259.1)
lowest = U(24) - K(196)

h = np.array([24, 40.5, 51])
E = np.array([lowest, middle, highest])

c = np.random.randn(2)

def relation(h):
    return c[0] * h + c[1]

def MSE(h, E):
    return np.mean((relation(h)-E)**2)

def get_gradient(h, E):
    E_pred = relation(h)
    dLdc0 = np.mean((E_pred - E)*h)
    dLdc1 = np.mean((E_pred - E))
    return np.array([dLdc0/10, dLdc1])

i = 0
equal_stack = 0
previous_loss = -1
learning_rate = 1e-6

while True:
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(h, E)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    if equal_stack >= 5:
        break

    i += 1
    c -= learning_rate * get_gradient(h, E)

    previous_loss = loss

    print(f"epoch:{i}, loss:{loss}")

print(f"c0={c[0]}\nc1={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(E**2)-np.mean(E)**2)}")

plt.plot(h, E, 'ro')
domain = np.linspace(h[0], h[-1], 100)
plt.plot(domain, relation(domain), 'b')
plt.xlabel('relative height (cm)')
plt.ylabel('lost energy (J)')
plt.show()