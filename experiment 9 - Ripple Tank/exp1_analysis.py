import numpy as np
import matplotlib.pyplot as plt

freq = np.array([13, 14, 15], dtype=np.float32)
lamda = np.array([1.243, 1.098, 1.032], dtype=np.float32)
l_inverse = 1/lamda

c = np.random.randn(2)

def f(l_1):
    return c[0] * l_1 + c[1]

def MSE(l_1, freq):
    return np.mean((f(l_1) - freq)**2)

def get_gradient(l_1, freq):
    f_pred = f(l_1)
    dLdv = np.mean((f_pred - freq)*l_1)
    dLdb = np.mean((f_pred - freq))
    return np.array([dLdv, dLdb])

i = 0
equal_stack = 0
learning_rate = 1e-6
previous_loss = -1

while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(l_inverse, freq)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss

    if equal_stack >= 5:
        break

    c -= learning_rate * get_gradient(l_inverse, freq)

    print(f"epoch:{i}, loss:{loss}")

print(f"v={c[0]}\nb={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(freq**2)-np.mean(freq)**2)}")

domain = np.linspace(np.min(l_inverse), np.max(l_inverse), 100)
plt.plot(l_inverse, freq, 'ro')
plt.plot(domain, f(domain), 'b')
plt.xlabel('lambda^(-1) (1/m)')
plt.ylabel('frequency (Hz)')
plt.show()