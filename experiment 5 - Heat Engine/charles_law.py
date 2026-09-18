import pandas as pd
import numpy as np

dataframe = pd.read_csv('./charles_law_raw_data.csv')

absolute_zero = -273.15

array = dataframe.to_numpy().T

piston_heights, temperatures = array
temperatures -= absolute_zero

constants = piston_heights / temperatures
print(constants)