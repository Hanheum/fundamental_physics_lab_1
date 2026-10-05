import numpy as np
import matplotlib.pyplot as plt

frequencies = [139, 279, 419, 559, 700, 839]
frequencies = np.array(frequencies, dtype=np.float32)
n = [1, 2, 3, 4, 5, 6]
n = np.array(n, np.float32)

f_lowest = frequencies[0]

f_over_f_lowest = frequencies/f_lowest
print(np.round(f_over_f_lowest, 2))

c = np.array([f_lowest, 0], dtype=np.float32)

def frequency(n):
    return c[0] * n + c[1]

def MSE(n, f):
    f_pred = frequency(n)
    return np.mean((f_pred - f)**2)

def get_gradient(n, f):
    f_pred = frequency(n)
    dLdc0 = np.mean((f_pred - f)*n)
    dLdc1 = np.mean((f_pred - f))
    return np.array([dLdc0, dLdc1])

previous_loss = -1
equal_stack = 0
learning_rate = 1e-6
i = 0

plt.plot(n, frequencies, 'ro')
plt.xlabel('n')
plt.ylabel('frequency (Hz)')
plt.show()

while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(n, frequencies)
    if previous_loss == loss:
        equal_stack += 1
    else:
        equal_stack = 0

    if equal_stack >= 5:
        break

    previous_loss = loss

    c -= learning_rate * get_gradient(n, frequencies)

    print(f"epoch: {i}, loss: {loss}")

print(f"c0={c[0]}\nc1={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(frequencies**2)-np.mean(frequencies)**2)}")

plt.plot(n, frequencies, 'ro')
domain = np.linspace(1, 6, 100)
plt.plot(domain, frequency(domain), 'b')
plt.xlabel('n')
plt.ylabel('frequency (Hz)')
plt.show()