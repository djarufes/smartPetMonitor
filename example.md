# Software code segments used 
Please refer to the github repository for complete codes. 

## User Food and water bowl Selector
```python

# Code inserted here:
    # user selection for food and water bowl:
    boxes = []
    # Read image
    img = cv2.VideoCapture(source)
    ret, frame = img.read()
    # frame = cv2.resize(frame, None, fx = 0.3045,fy = 0.5778)

    while(len(boxes) < 2):
        frame_cpy = frame.copy()
        if(len(boxes) == 0): 
            im0 = cv2.rectangle(frame_cpy, (0,0), (600, 80) , (0,0,0), -1)
            im0 = cv2.putText(frame_cpy, 'Select Water Bowl', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            print('Water Bowl Selection')
        elif(len(boxes) == 1): 
            im0 = cv2.rectangle(frame_cpy, (0,0), (570, 80) , (0,0,0), -1)
            im0 = cv2.putText(frame_cpy, 'Select Food Bowl', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            print('Food Bowl Selection')
        # Select ROI
        cv2.namedWindow("User Selection", cv2.WINDOW_NORMAL)
        r = cv2.selectROI("User Selection", frame_cpy, fromCenter=False, showCrosshair=False) # returns (x, y, w, h)
        # Bounded image
        boxes.append(((int(r[0]),int(r[1])), (int(r[0]+r[2]),int(r[1]+r[3]))))
        print(boxes[-1])
        cv2.waitKey(1)
    cv2. destroyAllWindows()
```

## Activity Index Calculations
```python
# Activity Index
                frames = 15
                if unNorm_xyxy != [[0,0,0,0]]:
                    color = (255, 255, 0)
                    perc = 0.10
                    boundingBox_center_x = (int(unNorm_xyxy[0][0]) + int(unNorm_xyxy[0][2])) / 2
                    boundingBox_center_y = (int(unNorm_xyxy[0][1]) + int(unNorm_xyxy[0][3])) / 2
                    boundingBox_width = int(unNorm_xyxy[0][2]) - int(unNorm_xyxy[0][0])
                    boundingBox_height = int(unNorm_xyxy[0][3]) - int(unNorm_xyxy[0][1])
                    activityBox_x1 = int(boundingBox_center_x - (boundingBox_width * perc))
                    activityBox_y1 = int(boundingBox_center_y - (boundingBox_height * perc))
                    activityBox_x2 = int(boundingBox_center_x + (boundingBox_width * perc))
                    activityBox_y2 = int(boundingBox_center_y + (boundingBox_height * perc))
                    start_point = (activityBox_x1, activityBox_y1)
                    end_point = (activityBox_x2, activityBox_y2)

                    # Lagging activityBox
                    if seen - prev_seen > frames:
                        if prev_start_point != prev_end_point:
                            #########
                            # IOU between lagging & real-time activityBox
                            # http://ronny.rest/tutorials/module/localization_001/iou/
                            #########
                            x1 = max(start_point[0], prev_start_point[0])
                            y1 = max(start_point[1], prev_start_point[1])
                            x2 = min(end_point[0], prev_end_point[0])
                            y2 = min(end_point[1], prev_end_point[1])

                            # AREA OF OVERLAP - Area where the boxes intersect
                            width = (x2 - x1)
                            height = (y2 - y1)
                            # handle case where there is NO overlap
                            area_overlap = width * height

                            # COMBINED AREA
                            area_a = (end_point[0] - start_point[0]) * (end_point[1] - start_point[1])
                            area_b = (prev_end_point[0] - prev_start_point[0]) * (prev_end_point[1] - prev_start_point[1])
                            area_combined = area_a + area_b - area_overlap

                            # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
                            iou = 0
                            if area_combined > 0:
                                iou = area_overlap / area_combined
                                #print('----------', area_a, area_b, area_combined, area_overlap, iou)
                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            org = (950, 220)
                            fontScale = 2
                            if iou > 0: activity = 0    # low
                            else: activity = 1          # high

                        # Update lagging box
                        prev_start_point = start_point
                        prev_end_point = end_point
                        prev_seen = seen

                    elif prev_start_point != prev_end_point:
                        color = (255, 0, 255)
                        im0 = cv2.rectangle(im0, prev_start_point, prev_end_point, color, thickness)
                    # Real-time activityBox
                    color = (255, 255, 0)
                    im0 = cv2.rectangle(im0, start_point, end_point, color, thickness)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (10, 110)
                    fontScale = 1
                    if activity == 0: im0 = cv2.putText(im0, 'Activity: Low', org, font, fontScale, (0, 255, 255), 2, cv2.LINE_AA)
                    elif activity == 1: im0 = cv2.putText(im0, 'Activity: High', org, font, fontScale, (0, 255, 255), 2, cv2.LINE_AA)
                
                if unNorm_xyxy == [[0,0,0,0]] or activity == None:
                    im0 = cv2.putText(im0, 'Activity: --', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    activity = None
                
                if seen - prev_write > frames:
                    ### CSV File Export
                    file_exists = os.path.exists('metaData.csv')
                    header = ['system time', 'timestamp', 'eat_drink', 'activity']
                    print(iou_food+iou_water)
                    data = [datetime.now(),timestamp, float(iou_food+iou_water), activity]
                    timestamp += 0.5
                    with open('metaData.csv', 'a', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        # If file does not exists, write the header
                        if file_exists == False: writer.writerow(header)
                        writer.writerow(data)
                    prev_write = seen

```

## Water and Food Bowl IOU calculations with respect to the Cat Bounding Box
```python
 # iou calculations:
                        # bounding box info for the cat:
                        x1 = unNorm_xyxy[0][0]
                        x2 = unNorm_xyxy[0][2]
                        y1 = unNorm_xyxy[0][1]
                        y2 = unNorm_xyxy[0][3]
                        #print(x1,y1)
                        #print(x2,y2)
                        area_cat_box = (x2-x1)*(y2-y1)
                        width_waterBowl = boxes[0][1][0]-boxes[0][0][0]
                        height_waterBowl = boxes[0][1][1]-boxes[0][0][1]
                        area_water_box = width_waterBowl * height_waterBowl
                        xleft_water = max(x1,boxes[0][0][0])
                        xright_water = min(x2, boxes[0][1][0])
                        ytop_water = max(y1, boxes[0][0][1])
                        ybottom_water = min(y2, boxes[0][1][1])
                        #print(xleft, xright, ytop, ybottom)
                        if (xright_water<xleft_water) or (ybottom_water<ytop_water):
                          area_inter_water = 0
                        else:
                          area_inter_water = (xright_water-xleft_water)*(ybottom_water-ytop_water)
                        iou_water = area_inter_water/(area_cat_box+area_water_box-area_inter_water)


                        width_foodBowl = boxes[1][1][0]-boxes[1][0][0]
                        height_foodBowl = boxes[1][1][1]-boxes[1][0][1]
                        area_food_box = width_foodBowl * height_foodBowl
                        xleft_food = max(x1,boxes[1][0][0])
                        xright_food = min(x2, boxes[1][1][0])
                        ytop_food = max(y1, boxes[1][0][1])
                        ybottom_food = min(y2, boxes[1][1][1])
                        #print(xleft, xright, ytop, ybottom)
                        if (xright_food<xleft_food) or (ybottom_food<ytop_food):
                          area_inter_food = 0
                        else:
                          area_inter_food = (xright_food-xleft_food)*(ybottom_food-ytop_food)
                        iou_food = area_inter_food/(area_cat_box+area_food_box-area_inter_food)

```

## Long-Term Divergence Checker
```python
import csv

def avgWaterConsumed (WaterData):
  total = 0
  count = 0
  min = 100
  max = 0
  for index in WaterData:
    #print(int(index))
    total += int(index)
    count += 1
    if int(index) < min:
      min = int(index)
    if int(index) > max:
      max = int(index)
  range = max - min
  #print(range)
  #print(max, min)
  avgWaterConsumed = total/count
  return avgWaterConsumed, range

def avgEatingTime (EatingData):
  total = 0
  count = 0
  min = 100
  max = 0
  for index in EatingData:
    total += int(index)
    count += 1
    if int(index) < min:
      min = int(index)
    if int(index) > max:
      max = int(index)
  range = max - min
  avgEatingTime = total/count
  return avgEatingTime, range

def avgActivity (ActivityData):
  total = 0
  count = 0
  min = 100
  max = 0
  for index in ActivityData:
    total += int(index)
    count += 1
    if int(index) < min:
      min = int(index)
    if int(index) > max:
      max = int(index)
  range = max - min
  avgActivity = total/count
  return avgActivity, range

def CurrentData():
  Filename = "/content/drive/MyDrive/CurrentCorrectAvg.csv"
  with open(Filename) as csv_file:
    count = 0
    WaterData = []
    EatingData = []
    ActivityData = []
    for line in csv.reader(csv_file, delimiter = ","):
      if count == 0:
        headers = line
        #print(line)
        count += 1
      else: 
        WaterData.append(line[2])
        EatingData.append(line[3])
        ActivityData.append(line[4])
    
    Water, WaterRange = avgWaterConsumed(WaterData)
    Eating, EatingRange = avgEatingTime(EatingData)
    Activity, ActivityRange = avgActivity(ActivityData)


  return Water , Eating , Activity




LongTermFile = "/content/drive/MyDrive/longTermData.csv"

with open(LongTermFile) as csv_file:
  count = 0
  WaterData = []
  EatingData = []
  ActivityData = []
  for line in csv.reader(csv_file, delimiter = ","):
    if count == 0:
      headers = line
      #print(line)
      count += 1
    else: 
      WaterData.append(line[2])
      EatingData.append(line[3])
      ActivityData.append(line[4])
  
  Water, WaterRange = avgWaterConsumed(WaterData)
  Eating, EatingRange = avgEatingTime(EatingData)
  Activity, ActivityRange = avgActivity(ActivityData)

  CurrentWater , CurrentEating, CurrentActivity = CurrentData()
  print(f"Current amount of Water Consumed = {CurrentWater} mL")
  print(f"Average amount of water Consumed in the past week = {round(Water,2)} mL")
  print(f"Current time spent eating = {CurrentEating} minutes")
  print(f"Average time spent eating in the past week = {round(Eating,2)} minutes")
  print(f"Current measured Activity = {CurrentActivity}")
  print(f"Average measured activity in the past week = {round(Activity,2)}")
  #print((Water - (0.5*WaterRange)))
  #print(WaterRange)

  message1, message2, message3 = "", "", ""

  if (CurrentWater < (Water + (0.5*WaterRange))) & (CurrentWater > (Water - (0.5*WaterRange))):
    message1 = "No Divergence for water consumption."
  else:
    message1 = "Divergence for water consumption detected."

  if (CurrentEating < (Eating + (0.5*EatingRange))) & (CurrentEating > (Eating - (0.5*EatingRange))):
    message2 = "No Divergence for eating time."
  else:
    message2 = "Divergence for eting time detected."

  if (CurrentActivity < (Activity + (0.5*ActivityRange))) & (CurrentActivity > (Activity - (0.5*ActivityRange))):
    message3 = "No Divergence for Activity."
  else:
    message3 = "Divergence for activty detected."

  print('\n')
  print(message1)
  print(message2)
  print(message3)

```
