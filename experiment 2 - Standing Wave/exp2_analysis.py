import numpy as np
import matplotlib.pyplot as plt

n = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
fn = np.array([16.2, 23.6, 33.2, 42.3, 47.7, 55.9], dtype=np.float32)

plt.plot(n, fn, 'ro')
plt.xlabel('n')
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
    return np.array([dLda, dLdb])

i = 0
previous_loss = -1
equal_stack = 0
learning_rate = 1e-6
while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(n, fn)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    if equal_stack >= 5:
        break

    previous_loss = loss

    c -= learning_rate * get_gradient(n, fn)

    print(f"epoch:{i}, loss:{loss}")

print(f"a={c[0]}\nb={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.var(fn))}")

domain = np.linspace(1, 6, 100)
plt.plot(domain, f(domain), 'b')
plt.show()