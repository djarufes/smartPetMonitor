''' 
System Controller - Data visualization
'''

import pandas as pd
import matplotlib.pyplot as plt

audio_data = pd.read_csv('audio_data.csv')
audio_data = audio_data.dropna(how='all')
audio_data = audio_data.dropna(axis='columns',how='all')
# print(audio_data)

time = audio_data['timestamp'].to_list()
eating = audio_data['eat'].to_list()
drinking = audio_data['drink'].to_list()
# print(time,eating,drinking)

# Eating Behavior Graph
# Time vs Time - On Off graph
plt.figure("Eating", figsize=(12,3))
plt.plot(time,eating, drinking)
plt.ylabel("Behavior ('1' if eating/drinking)")
plt.legend(["Eating","Drinking"])
plt.title("Record of behavior over time")
plt.xticks(ticks=time)
plt.locator_params(axis='x', nbins = 10)
plt.show()

# Drinking Behavior Graph
# Amount vs time
# Group 5 data, add total drinking seconds.
# Rate => 0.4mL water consumed per second

data_x = []
data_y = []


