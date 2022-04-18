# Graphs

- [x] Behavior event graphs for eating / drinking 

- [x] Quantity of water consumed graph 

- [x] Time spent eating graph 

- [ ] Long term significant divergence graph 

- [ ] Statistics overlay: 
  
  - [ ] Total vol. water consumed 
  
  - [ ] Total time spent eating 
  
  - [ ] Current behavior: /drinking/eating/-- 

Video 1: Drinking and eating 

    Not the optimum camera placement, but still operates well. Relatively more challenging to recognize cat so misses a couple frames. But there are no issue in data collection because we have data correction feature (and microphone holds data for +- 5 seconds anyways)

Video 2: Eating and activity 

Video 3: Long term activity 

# Video Data Parameters

- Timestamp: 
  
  - Current system time of record. 
  
  - Interval of 0.5 seconds. 

- Eatordrink: 
  
  - Whether the video model recognizes cat as either drinking or eating. 
  
  - Save `1` if eating or drinking, `0` if neither. 
  
  - Doesn't need to distinguish eating with drinking, as Audio Model will do that. 

- Activity: 
  
  - Activity index recognized in the timeframe. 
  
  - 1 - Low, 50 - Medium, 100 - High

# Audio Data Parameters

- Timestamp: 
  
  - Time of audio input from Video data
    
    - Will be taken from video's csv data 

- Durations: 
  
  - File : Duration of the input audio file 
  
  - Eat : Duration of cat's time spent eating  
  
  - Drink : Duration of cat's time spent drinking 
