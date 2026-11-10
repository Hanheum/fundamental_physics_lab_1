import numpy as np
import matplotlib.pyplot as plt

n = np.array([1, 3, 5, 7, 9], dtype=np.float32)
f = np.array([7.2, 49.6, 143.4, 276.8, 449.4], dtype=np.float32)

plt.plot(n, f, 'ro')
plt.xlabel('n')
plt.ylabel('f (Hz)')

c = np.random.randn(3)

def relation(n):
    return c[0] * n**2 + c[1] * n + c[2]

def MSE(n, f):
    return np.mean((relation(n) - f)**2)

def get_gradient(n, f):
    f_pred = relation(n)
    dLda = np.mean((f_pred - f)*n**2)
    dLdb = np.mean((f_pred - f)*n)
    dLdc = np.mean((f_pred - f))
    return np.array([dLda/100, dLdb/10, dLdc])

i = 0
equal_stack = 0
previous_loss = -1
learning_rate = 1e-6
while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(n, f)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    if equal_stack >= 5:
        break

    previous_loss = loss

    c -= learning_rate * get_gradient(n, f)

    print(f"epoch:{i}, loss:{loss}")

print(f"a={c[0]}\nb={c[1]}\nc={c[2]}\nRMSE={loss**0.5}\nR^2={1-loss/np.var(f)}")
domain = np.linspace(n[0], n[-1], 100)
plt.plot(domain, relation(domain), 'b')
plt.show()