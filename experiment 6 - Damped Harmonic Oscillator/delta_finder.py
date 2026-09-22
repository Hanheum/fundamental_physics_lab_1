import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./driven_5mm/full_5mm.csv', usecols=['freq', 'amp'])

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

plt.plot(amp, freq, 'ro')
plt.xlabel('amp (rad)')
plt.ylabel('freq (Hz)')
plt.show()

wr = 0.619 * 2 * np.pi
w0 = 4.0718

w = float(input('w:')) * 2 * np.pi

beta = np.sqrt((w0**2-wr**2)/2)

def delta(w):
    return np.atan(2*w*beta/(w**2-w0**2))

print(delta(w) * 180 / np.pi)

