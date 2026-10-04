import numpy as np
import matplotlib.pyplot as plt

tube_length = 121 #cm

maximum = np.array([30, 89], dtype=np.float32)
minimum = np.array([0, 59, 120], dtype=np.float32)

maximum = tube_length - maximum
minimum = tube_length - minimum

plt.plot(maximum, [0]*2, 'ro')
plt.plot(minimum, [0]*3, 'bo')
plt.xlabel("length from speaker (cm)")
plt.show()

combined = np.concat([maximum, minimum])
combined = np.sort(combined)
bigger = combined[1:]
smaller = combined[0:len(combined)-1]

diffs = np.abs(bigger - smaller)

print(diffs)
input('hit enter to continue')

fit_value = 25

def MSE(diffs):
    return np.mean((diffs - fit_value)**2)

def get_gradient(diffs):
    return np.mean(fit_value - diffs)

equal_stack = 0
previous_loss = -1
i = 0
learning_rate = 1e-6
while True:
    new_learning_rate = open('learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(diffs)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss
    i += 1

    fit_value -= learning_rate * get_gradient(diffs)

    print(f"epoch:{i+1}, loss:{loss}")

    if equal_stack >= 5:
        break

print(f"fitted_value={fit_value}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(diffs**2) - np.mean(diffs)**2)}")
