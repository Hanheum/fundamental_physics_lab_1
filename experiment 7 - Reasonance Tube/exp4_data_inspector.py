import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_ch1 = pd.read_csv('./4th_exp_27cm/F0009CH1.CSV', usecols=['D', 'E'])
df_ch2 = pd.read_csv('./4th_exp_27cm/F0009CH2.CSV', usecols=['D', 'E'])

time1, voltage1 = df_ch1.to_numpy().T
time2, voltage2 = df_ch2.to_numpy().T

plt.plot(time1, voltage1, 'r')
plt.plot(time2, voltage2, 'b')
plt.xlabel('time (s)')
plt.ylabel('voltage (V)')
plt.show()