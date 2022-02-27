import cv2
from time import time
'''
# Attempt 1, not working as well, crashes suddently
boxes = []
def on_mouse(event, x, y, flags, params):
    # global img
    t = time()
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Start Mouse Position: " + str(x) + ", " + str(y))
        sbox = [x, y]
        boxes.append(sbox)

    elif event == cv2.EVENT_MOUSEMOVE and boxes != []:
        color = (0,0,0)
        thickness = 5
        boundedImg = img.copy()
        cv2.rectangle(boundedImg, (boxes[-1][0], boxes[-1][1]), (x,y), color, thickness)
        cv2.imshow('test', boundedImg) 
        k = cv2.waitKey(0) 

    elif event == cv2.EVENT_LBUTTONUP:
        print("End Mouse Position: " + str(x)+ ", " + str(y))
        ebox = [x, y]
        boxes.append(ebox)
        print(boxes)
        boundedImg = img.copy()
        
        color = (0,0,0)
        thickness = 5
        cv2.rectangle(boundedImg, (boxes[-1][0], boxes[-1][1]), (boxes[-2][0],boxes[-2][1]), color, thickness)
        #cv2.rectangle(boundedImg, (100,500), (125,80), color, thickness)
        cv2.imshow('test', boundedImg)        

        while len(boxes) > 0: boxes.pop()

        k = cv2.waitKey(0)
        if ord('r') == k:
            cv2.imwrite('Crop'+str(t)+'.jpg',crop)
            print("Written to file")

count = 0
while(True):
   count += 1
   img = cv2.imread('drawBoxTest.PNG')
   img = cv2.resize(img, None, fx = 0.5,fy = 0.5)

   cv2.namedWindow('test')
   cv2.setMouseCallback('test', on_mouse, 0)
   cv2.imshow('test', img)
   if count < 50:
       if cv2.waitKey(33) == 27:
           cv2.destroyAllWindows()
           break
       elif count >= 50:
          if cv2.waitKey(0) == 27:
             cv2.destroyAllWindows()
             break
          count = 0
'''

if __name__ == '__main__' :
    # Attempt 2, simpler and effective
    boxes = []
    # Read image
    img = cv2.imread("drawBoxTest.PNG")
    img = cv2.resize(img, None, fx = 0.5,fy = 0.5)

    while(len(boxes) < 2):
        if(len(boxes) == 0): print('Water Bowl Selection')
        elif(len(boxes) == 1): print('Food Bowl Selection')
        # Select ROI
        r = cv2.selectROI(img, fromCenter=False, showCrosshair=False)
        # Bounded image
        boxes.append(((int(r[1]),int(r[1]+r[3])), (int(r[0]),int(r[0]+r[2]))))
        print((int(r[1]),int(r[1]+r[3])), (int(r[0]),int(r[0]+r[2])), '\n')
        # Display cropped image
        cv2.waitKey(1)
    print(boxes)