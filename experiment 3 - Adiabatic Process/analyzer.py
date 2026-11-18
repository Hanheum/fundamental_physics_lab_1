import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = './co2/'
pressure_dir = dir + 'pressure.csv'
temperature_dir = dir + 'temperature.csv'
volume_dir = dir + 'volume.csv'

time, pressure = pd.read_csv(pressure_dir).to_numpy().T
_, temperature = pd.read_csv(temperature_dir).to_numpy().T
_, volume = pd.read_csv(volume_dir).to_numpy().T

starts, ends = [], []

df = []
T1, T2 = [], []
P1, P2 = [], []
V1, V2 = [], []
gamma = []
W = []

rising = False
for i, p in enumerate(pressure):
    if i == 0 or i == len(pressure)-1:
        continue

    dif1 = pressure[i+1] - p
    dif2 = abs(p - pressure[i-1])

    if dif1 >= 5 and rising == False:
        rising = True
        starts.append(i)

    if rising == True and dif1 <= 0:
        rising = False
        ends.append(i)

print(starts, ends)

repeat = True
while repeat:
    plt.plot(time, pressure, 'r')
    plt.xlabel('time (s)')
    for start in starts:
        plt.plot([time[start]], [pressure[start]], 'bo')
    for end in ends:
        plt.plot([time[end]], [pressure[end]], 'go')

    plt.show()

    command = input('merge or delete:')
    if command == 'merge':
        merging_idx = input("merging idx:")
        merging_idx1, merging_idx2 = map(int, merging_idx.split(', '))
        starts.pop(merging_idx2)
        ends.pop(merging_idx1)
    elif command == 'stop':
        break
    elif command == 'delete':
        deleting_idx = int(input('deleting_idx:'))
        starts.pop(deleting_idx)
        ends.pop(deleting_idx)

plt.plot(time, pressure, 'r')
plt.xlabel('time (s)')
plt.ylabel('pressure (kPa)')
for start in starts:
    plt.plot([time[start]], [pressure[start]], 'bo')
    P1.append(pressure[start])
for end in ends:
    plt.plot([time[end]], [pressure[end]], 'go')
    P2.append(pressure[end])

plt.show()


plt.plot(time, temperature, 'r')
plt.xlabel('time (s)')
plt.ylabel('temperature (K)')
for start in starts:
    plt.plot([time[start]], [temperature[start]], 'bo')
    T1.append(temperature[start])
for end in ends:
    plt.plot([time[end]], [temperature[end]], 'go')
    T2.append(temperature[end])

plt.show()


plt.plot(time, volume, 'r')
plt.xlabel('time (s)')
plt.ylabel('volume (m^3)')
for start in starts:
    plt.plot([time[start]], [volume[start]], 'bo')
    V1.append(volume[start])
for end in ends:
    plt.plot([time[end]], [volume[end]], 'go')
    V2.append(volume[end])

plt.show()

learning_rate = 1e-5
start_point_v = []
start_point_p = []
end_point_v = []
end_point_p = []

avg_c = np.array([0, 0], dtype=float)
for i in range(len(starts)):
    start = starts[i]
    end = ends[i]
    pressure_partition = pressure[start:end]
    volume_partition = volume[start:end]
    lnp = np.log(pressure_partition)
    lnv = np.log(volume_partition)

    start_point_v.append(volume_partition[0])
    start_point_p.append(pressure_partition[0])
    end_point_v.append(volume_partition[-1])
    end_point_p.append(pressure_partition[-1])
    c = np.array([(lnp[0] - lnp[-1])/(lnv[0] - lnv[-1]), (lnp[0] - lnp[-1])/(lnv[0] - lnv[-1])*(-lnv[0])+lnp[0]])

    def function(lnv):
        return c[0] * lnv + c[1]

    def MSE(lnv, lnp):
        return np.mean((function(lnv)-lnp)**2)

    def get_gradient(lnv, lnp):
        lnp_pred = function(lnv)
        dLda = np.mean((lnp_pred - lnp)*lnv)
        dLdb = np.mean((lnp_pred - lnp))
        return np.array([dLda, dLdb])

    j = 0
    equal_stack = 0
    previous_loss = -1
    while True:
        new_learning_rate = open('learning_rate.txt', 'r').read()
        if len(new_learning_rate) != 0:
            learning_rate = float(new_learning_rate)

        loss = MSE(lnv, lnp)
        if loss == previous_loss:
            equal_stack += 1
        else:
            equal_stack = 0

        if equal_stack >= 5 or (1-loss/np.var(lnp))>0.99:
            break

        j += 1

        c -= learning_rate * get_gradient(lnv, lnp)
        print(f"j={j}, loss={loss}")

    print(f"i={i}\ngamma={-c[0]}\nb={c[1]}\nRMSE={loss**0.5}\nR^2={1-loss/np.var(lnp)}\n")
    gamma.append(-c[0])
    avg_c += c

    def pv(v):
        return np.exp(c[1]) * v**(c[0])

    partition = np.linspace(volume[start], volume[end], 10000)
    w = 0
    for i in range(len(partition)-1):
        w += (partition[i+1]-partition[i]) * pv(partition[i])

    W.append(w)

avg_c /= 10
def relation(lnv):
    return avg_c[0] * lnv + avg_c[1]

start_point_p = np.array(start_point_p)
start_point_v = np.array(start_point_v)
end_point_p = np.array(end_point_p)
end_point_v = np.array(end_point_v)

start_point_lnv = np.log(start_point_v)
start_point_lnp = np.log(start_point_p)
end_point_lnv = np.log(end_point_v)
end_point_lnp = np.log(end_point_p)

mean_start_p = np.mean(start_point_p)
mean_start_v = np.mean(start_point_v)
mean_end_p = np.mean(end_point_p)
mean_end_v = np.mean(end_point_v)

plt.plot([np.mean(start_point_lnv), np.mean(end_point_lnv)], [np.mean(start_point_lnp), np.mean(end_point_lnp)], 'ro')
plt.errorbar([np.mean(start_point_lnv), np.mean(end_point_lnv)], [np.mean(start_point_lnp), np.mean(end_point_lnp)], 
             xerr=[np.var(start_point_v)**0.5/mean_start_v, np.var(end_point_v)**0.5/mean_end_v], yerr=[np.var(start_point_p)**0.5/mean_start_p, np.var(end_point_p)**0.5/mean_end_p],
             fmt="none", ecolor="gray", elinewidth=1.5, capsize=4, capthick=1.5,)

plt.xlabel('lnv')
plt.ylabel('lnp')
domain = np.linspace(np.mean(start_point_lnv), np.mean(end_point_lnv), 100)
plt.plot(domain, relation(domain), 'b')

plt.show()

def pv(v):
    return np.exp(avg_c[1]) * v**(avg_c[0])

plt.plot([mean_start_v, mean_end_v], [mean_start_p, mean_end_p], 'ro')
domain = np.linspace(mean_start_v, mean_end_v, 100)
plt.plot(domain, pv(domain), 'b')
plt.xlabel('v (m^3)')
plt.ylabel('p (kPa)')
plt.errorbar([mean_start_v, mean_end_v], [mean_start_p, mean_end_p], 
             xerr=[np.var(start_point_v)**0.5, np.var(end_point_v)**0.5], yerr=[np.var(start_point_p)**0.5, np.var(end_point_p)**0.5],
             fmt="none", ecolor="gray", elinewidth=1.5, capsize=4, capthick=1.5)
plt.show()

partition = np.linspace(mean_start_v, mean_end_v, 10000)
w = 0
for i in range(len(partition)-1):
    w += (partition[i+1]-partition[i]) * pv(partition[i])

print(f"W={w}")

T1 = np.array(T1)
T2 = np.array(T2)
P1 = np.array(P1)
P2 = np.array(P2)
V1 = np.array(V1)
V2 = np.array(V2)
gamma = np.array(gamma)
W = np.array(W)
n = 1000*P1*V1/T1/(1.380649e-23)
W2 = []
for i in range(10):
    W2.append(P1[i]*V1[i]**gamma[i]/(1-gamma[i])*(V2[i]**(1-gamma[i])-V1[i]**(1-gamma[i])))
W2 = np.array(W2)
df = np.array([T1, P1, V1, n, T2, P2, V2, gamma, W, W2])
avg = np.mean(df, axis=1)
error = np.var(df, axis=1)**0.5
df = np.concat([df.T, np.array([avg]), np.array([error])]).T
df = pd.DataFrame(df, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'mean', 'standard error'], index=['T1', 'P1', 'V1', 'n', 'T2', 'P2', 'V2', 'gamma', 'W', 'W2'])
df.to_csv('./co2/table.csv')