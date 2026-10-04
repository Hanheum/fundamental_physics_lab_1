import numpy as np
import matplotlib.pyplot as plt

tube_length = 121 #cm

loud_points = np.array([3, 25, 46.5, 67.5, 89, 111.5], dtype=np.float32)
loud_points = tube_length - loud_points
loud_points = np.sort(loud_points)

alpha = loud_points[0]
beta = 0
c = np.array([alpha, beta]) #alpha, beta

domain = list(range(len(loud_points)))
domain = np.array(domain)

def L(i):
    return c[0] * (2*(c[1] + i)-1)

def MSE(loud_points):    
    L_pred = L(domain)
    return np.mean((L_pred - loud_points)**2)

def get_gradient(loud_points):
    L_pred = L(domain)
    dLdalpha = np.mean((L_pred - loud_points)*L_pred/c[0])
    dLdbeta = np.mean((L_pred - loud_points)*2*c[0])
    return np.array([dLdalpha, dLdbeta], dtype=np.float32)

equal_stack = 0
previous_loss = -1
i = 0
learning_rate = 1e-6
while True:
    new_learning_rate = open('learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(loud_points)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss
    i += 1

    c -= learning_rate * get_gradient(loud_points)

    print(f"epoch:{i+1}, loss:{loss}")

    if equal_stack >= 5:
        break

print(f"alpha={c[0]}\nbeta={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(loud_points**2)-np.mean(loud_points)**2)}")

plt.plot(domain, loud_points, 'ro')
lin_domain = np.linspace(domain[0], domain[-1], 100)
plt.plot(lin_domain, L(lin_domain), 'b')
plt.xlabel('i')
plt.ylabel('L (cm)')
plt.show()
