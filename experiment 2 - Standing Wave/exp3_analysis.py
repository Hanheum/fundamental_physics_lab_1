import numpy as np
import matplotlib.pyplot as plt

L = np.array([7.5, 8.5, 9.5, 11.5, 12.5, 13.5], dtype=np.float32)
fn = np.array([62.3, 46.7, 42.1, 26.6, 23.9, 20.4], dtype=np.float32)

L = 1/L
L = L**2

plt.plot(L, fn, 'ro')
plt.xlabel('L^(-2) (cm^(-2))')
plt.ylabel('f_n (Hz)')

c = np.random.randn(2)

def f(n):
    return c[0] * n + c[1]

def MSE(n, fn):
    return np.mean((f(n) - fn)**2)

def get_gradient(n, fn):
    fn_pred = f(n)
    dLda = np.mean((fn_pred - fn)*n)
    dLdb = np.mean((fn_pred - fn))
    return np.array([dLda, dLdb/100])

i = 0
previous_loss = -1
equal_stack = 0
learning_rate = 1e-6
while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(L, fn)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    if equal_stack >= 5:
        break

    previous_loss = loss

    c -= learning_rate * get_gradient(L, fn)

    print(f"epoch:{i}, loss:{loss}")

print(f"c={c[0]}\nd={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.var(fn))}")

domain = np.linspace(L[0], L[-1], 100)
plt.plot(domain, f(domain), 'b')
plt.show()