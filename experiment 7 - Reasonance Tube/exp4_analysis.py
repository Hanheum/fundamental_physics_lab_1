import numpy as np
import matplotlib.pyplot as plt

distance = np.array([67, 81, 94, 112], dtype=np.float32)
delta_t = np.array([0.00368, 0.00462, 0.00532, 0.00634], dtype=np.float32)

distance /= 100 #in meter

v = distance[0] * 2 / delta_t[0]

def function(dt):
    return 0.5 * v * dt

def MSE(distance, delta_t):
    distance_pred = function(delta_t)
    return np.mean((distance_pred - distance)**2)

def get_gradient(distance, delta_t):
    distance_pred = function(delta_t)
    dLdv = np.mean((distance_pred - distance)*delta_t)
    return dLdv

equal_stack = 0
previous_loss = -1
i = 0
learning_rate = 1e-6
while True:
    new_learning_rate = open('learning_rate.txt', 'r').read()
    if len(new_learning_rate) != 0:
        learning_rate = float(new_learning_rate)

    loss = MSE(distance, delta_t)
    if loss == previous_loss:
        equal_stack += 1
    else:
        equal_stack = 0

    previous_loss = loss
    i += 1

    v -= learning_rate * get_gradient(distance, delta_t)

    print(f"epoch:{i+1}, loss:{loss}")

    if equal_stack >= 5:
        break

print(f"v={v}\nRMSE={loss**0.5}\nR^2={1-loss/(np.mean(distance**2)-np.mean(distance)**2)}")

plt.plot(delta_t, distance, 'ro')
domain = np.linspace(delta_t[0], delta_t[-1], 100)
plt.plot(domain, function(domain), 'b')
plt.xlabel('delta t (s)')
plt.ylabel('distance (m)')
plt.show()