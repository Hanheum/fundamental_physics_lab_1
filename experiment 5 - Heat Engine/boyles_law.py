import pandas as pd

dataframe = pd.read_csv('./boyles_law_raw_data.csv')

array = dataframe.to_numpy().T

mass, piston_height, pressure, temperature = array

absolute_zero = -273.15
temperature -= absolute_zero

pv = pressure * piston_height

print(pv)