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

# # Eating Behavior Graph
# # Time vs Time - On Off graph
# plt.figure("Eating", figsize=(12,3))
# plt.plot(time,eating, drinking)
# plt.ylabel("Behavior ('1' if eating/drinking)")
# plt.legend(["Eating","Drinking"])
# plt.title("Record of behavior over time")
# plt.xticks(ticks=time)
# plt.locator_params(axis='x', nbins = 10)
# plt.show()

# Drinking Behavior Graph
# Amount vs time
# Group 5 data, add total drinking seconds.
# Rate => 0.4mL water consumed per second

data_x = []
data_y = []
data_xx = []
data_yy = []
daily = []
daily_food
drink_sum = 0 # For the 4 second chunk
daily_sum = 0 # For the entire time period given

for index, value in enumerate(time):
    drink_sum = drink_sum + drinking[index] * 0.4
    daily_sum = daily_sum + drinking[index] * 0.4
#     print(index, drink_sum)
    if index % 8 == 0:
        data_x.append(value)
        data_y.append(drink_sum)
        daily.append(daily_sum)
        drink_sum = 0

print(data_x, "\n", data_y, daily)

#
# fig = plt.figure("Water Consumption Tracking", figsize=(12,3))
# ax1 = fig.add_subplot()
#
# ax1.bar(data_x, data_y)

plt.figure("Water Consumption Tracking", figsize=(12,3))
plt.plot(data_x,daily, color='orange')
plt.bar(data_x, data_y, color='blue')
plt.ylabel("Amount of water consumed, mL")
plt.title("Amount of water consumed over time")
plt.legend(['Total amount consumed', 'Instant consumption amount'])
plt.xticks(ticks=data_x)
plt.locator_params(axis='x', nbins = 18)
plt.show()
