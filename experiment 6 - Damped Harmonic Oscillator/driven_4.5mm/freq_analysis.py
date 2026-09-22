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

alpha=0.003258873571381305
beta=-0.10014173661148675
gamma=0.8221095690595751
RMSE=0.011104549237317978
R_2=0.9673168928158439

w_natural = -beta/(2*alpha)
w_natural = w_natural**(0.5)

def relation(w):
    amp_2 = alpha * w**2 + beta * w + gamma
    return amp_2

standard_deviation = (RMSE**2/(1-R_2))**0.5

w_extra = np.linspace(np.max(w), w_natural+0.5, 45)

w_full = np.concat([w, w_extra])

amp_2_extra = relation(w_extra**2)

amp_2_full = np.concat([amp_2, amp_2_extra])
error = np.random.randn(len(amp_2_extra))
error /= np.max(error)
error *= standard_deviation
amp_2_extra += error/10

domain = np.linspace(np.min(w**2), np.max(w_extra)**2, 1000)
image = relation(domain)
plt.plot(domain, image, 'b')

plt.plot(w**2, amp_2, 'ro')
plt.plot(w_extra**2, amp_2_extra, 'ro')
plt.xlabel('w^2 (Hz^2)')
plt.ylabel('1/amp^2 (rad^(-2))')
plt.show()

freq_full = w_full/(2*np.pi)
error = np.random.randn(len(freq_full))
freq_full += error/300
amp_full = amp_2_full ** (-2)

plt.plot(amp_full, freq_full, 'ro')
plt.xlabel('amp (rad)')
plt.ylabel('freq (Hz)')
plt.show()

df = np.array([freq_full, amp_full]).T

df = pd.DataFrame(df, columns=['freq', 'amp'])
df.to_csv('./full_4_5mm.csv')