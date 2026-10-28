import numpy as np
import matplotlib.pyplot as plt

frequency = 19 #Hz
d = 6.5 #cm
lamda = 1.101 #cm

m = np.array([-1.5, -1, 0, 1, 1.5], dtype=np.float32)
theta = np.array([-15.03, -9.79, 0, 10.23, 14.91], dtype=np.float32)
theta *= np.pi/180

mlamda = m * lamda

c = np.array([mlamda[0]/np.sin(theta[0]), 0])

def relation(theta):
    return c[0] * np.sin(theta) + c[1]

def MSE(theta, mlamda):
    return np.mean((relation(theta)-mlamda)**2)

def get_gradient(theta, mlamda):
    mlamda_pred = relation(theta)
    dLdd = np.mean((mlamda_pred-mlamda)*np.sin(theta))
    dLde = np.mean((mlamda_pred-mlamda))
    return np.array([dLdd, dLde])

i = 0
equal_stack = 0
learning_rate = 1e-6
previous_loss = -1

while True:
    i += 1
    new_learning_rate = open('./learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(theta, mlamda)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss

    if equal_stack >= 5:
        break

    c -= learning_rate * get_gradient(theta, mlamda)

    print(f"epoch:{i}, loss:{loss}")

print(f"d={c[0]}\ne={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(mlamda**2)-np.mean(mlamda)**2)}")
plt.plot(theta, mlamda, 'ro')
domain = np.linspace(theta[0], theta[-1], 100)
plt.plot(domain, relation(domain), 'b')
plt.xlabel('theta (radians)')
plt.ylabel('m * lambda (cm)')
plt.show()