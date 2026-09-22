import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./freq_vs_amp.csv')
freq_original, amp_original = df.to_numpy().T

freq = []
for i in range(len(freq_original)):
    if i%2 == 1:
        freq.append(freq_original[i])

freq = np.array(freq)[1:35]

amp = []
for i in range(len(amp_original)):
    if i%2 == 0:
        amp.append(amp_original[i])

amp = np.array(amp)[1:35]
amp_2 = amp**(-2)

w = freq * 2 * np.pi
w2 = w**2

w0_true=4.073808670043945
w0_true = w0_true**2
alpha = 0.003
beta = alpha * (-2) * w0_true
gamma = alpha * w0_true**2 + 0.01

#alpha=0.007597998142443122
#beta=-0.16446267311568977
#gamma=1.04304891916691
c = np.array([alpha, beta, gamma])

def relation(w2):
    return c[0]*w2**2 + c[1] * w2 + c[2]

def MSE(w2, amp_2):
    amp_pred = relation(w2)
    return np.mean((amp_pred - amp_2)**2)

def get_gradient(w2, amp_2):
    amp_pred = relation(w2)
    dLdalpha = np.mean((amp_pred - amp_2)*w2**2)
    dLdbeta = np.mean((amp_pred - amp_2)*w2)
    dLdgamma = np.mean((amp_pred - amp_2))
    return np.array([dLdalpha, dLdbeta/100, dLdgamma])

i = 0
equal_stack = 0
previous_loss = -1
learning_rate = 1e-5
while True:
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)
    i += 1
    loss = MSE(w2, amp_2)
    if previous_loss == loss:
        equal_stack += 1
    else:
        equal_stack = 0
    previous_loss = loss

    c -= learning_rate * get_gradient(w2, amp_2)

    if equal_stack >= 5:
        break

    print(f"epoch:{i}, loss:{loss}")

print(f"alpha={c[0]}\nbeta={c[1]}\ngamma={c[2]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(amp_2**2) - np.mean(amp_2)**2)}")

domain = np.linspace(0, 13, 100)
plt.plot(domain, relation(domain), 'b')

plt.plot(w2, amp**(-2), 'ro')
plt.xlabel('w^2 (Hz^2)')
plt.ylabel('1/amp^2 (rad^(-2))')
plt.show()