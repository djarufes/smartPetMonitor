''' 
System Controller - Data visualization
'''

import pandas as pd
import matplotlib.pyplot as plt

audio_data = pd.read_csv('metadata2.csv')
audio_data = audio_data.dropna(how='all')
audio_data = audio_data.dropna(axis='columns',how='all')
# print(audio_data)
# timestamp	seconds	eat	drink	activity	x100	index

time = audio_data['timestamp'].to_list()
eating = audio_data['eat'].to_list()
drinking = audio_data['drink'].to_list()
activity = audio_data['activity'].to_list()


# #######################################
# Eating Behavior Graph
# Time vs Time - On Off graph
plt.figure("Eating", figsize=(12,3))
plt.plot(time,eating, drinking)
plt.ylabel("Behavior")
plt.legend(["Eating","Drinking"])
plt.title("Record of behavior over time")
plt.xticks(ticks=time)
plt.yticks(ticks=[0,1], labels=['None', 'Drinking/Eating'])
plt.locator_params(axis='x', nbins = 10)
plt.show()

#######################################
# Drinking Behavior Graph
# Amount vs time

data_x = []
data_y = []
data_y_food = []
daily = []
daily_food = []
drink_sum = 0 # For the 4 second chunk
daily_sum = 0 # For the entire time period given
eat_sum = 0
eat_daily = 0

# Group data (according to bin_size), add total drinking seconds.
# Rate => 0.4mL water consumed per second
bin_size = 8

for index, value in enumerate(time):
    drink_sum = drink_sum + drinking[index] * 0.2
    daily_sum = daily_sum + drinking[index] * 0.2
    eat_sum = eat_sum + eating[index] * 0.2
    eat_daily = eat_daily + eating[index] * 0.2
#     print(index, drink_sum)
    if index % bin_size == 0:
        data_x.append(value)
        data_y.append(drink_sum)
        data_y_food.append(eat_sum)
        daily.append(daily_sum)
        daily_food.append(eat_daily)
        drink_sum = 0
        eat_sum = 0

print(data_x, "\n", data_y, daily)

# ##########################################
# # Water consumption volume graph

figure_title = 'Water Consumption Tracking (Each bin = ' + str(bin_size) + ' seconds)'
fig = plt.figure("Water Consumption Tracking", figsize=(12,3))
ax1 = fig.add_subplot()
ax1.bar(data_x, data_y)
ax1.set_ylabel('Instant consumption, mL')
ax2 = ax1.twinx()
ax2.plot(data_x, daily, color='red', linestyle='--')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylabel('Total consumption, mL',color='red')
plt.xticks(ticks=data_x)
plt.locator_params(axis='x', nbins = 10)
plt.title(figure_title)
plt.legend(['Total amount consumed', 'Instant consumption amount'])
plt.show()

# plt.figure("Water Consumption Tracking", figsize=(12,3))
# plt.plot(data_x,daily, color='orange')
# plt.bar(data_x, data_y, color='blue')
# plt.ylabel("Amount of water consumed, mL")
# plt.title("Amount of water consumed over time")
# plt.legend(['Total amount consumed', 'Instant consumption amount'])
# plt.xticks(ticks=data_x)
# plt.locator_params(axis='x', nbins = 10)
# plt.show()

figure_title = 'Food Consumption Tracking (Each bin = ' + str(bin_size) + ' seconds)'
fig = plt.figure("Food Consumption Tracking", figsize=(12,3))
ax1 = fig.add_subplot()
ax1.bar(data_x, data_y_food)
ax1.set_ylabel('Instant time spent, sec')
ax2 = ax1.twinx()
ax2.plot(data_x, daily_food, color='red', linestyle='--')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylabel('Total Time spent eating, sec',color='red')
plt.xticks(ticks=data_x)
plt.locator_params(axis='x', nbins = 10)
plt.title(figure_title)
plt.legend(['Total amount consumed', 'Instant consumption amount'])
plt.show()

##########################################
# Activity Graph


activity_index_list = []
activity_instant = 0

for index, value in enumerate(time):
    if activity[index] == 1:
        if activity_instant < 100:
            activity_instant = activity_instant + 10
    else:
        if activity_instant > 0:
            activity_instant = activity_instant - 1
    # print(activity_instant)
    activity_index_list.append(activity_instant)

figure_title = 'Activity Monitoring'
fig = plt.figure("Activity Index over time", figsize=(16,4))
ax1 = fig.add_subplot()
ax1.bar(time, activity) #
ax1.set_xlabel("Timestamps (hh:mm:ss)")
ax1.set_ylabel('Instant Activity')
ax2 = ax1.twinx()
ax2.plot(time, activity_index_list, color='red', linestyle='--')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylabel('Activity Index',color='red')
plt.xlabel("Timestamp")
plt.xticks(ticks=data_x)
plt.locator_params(axis='x', nbins = 10)
plt.title(figure_title)
plt.yticks(ticks=[0,50,85], labels=['Low','Medium', 'High'])
plt.legend(['Activity Index'])
plt.show()