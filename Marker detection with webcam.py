from cv2 import cv2
import numpy as np


cap = cv2.VideoCapture(0)

kernel1 = np.ones((5,5),dtype = 'uint8')
kernel2 = np.ones((3,3),dtype = 'uint8')

l = []


paint = False

while True:
    key = cv2.waitKey(1)
    if key == ord('p'): 
        paint = True          

    ret, frame = cap.read()
    # print(ret)
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low = np.array([107,173,127],dtype = 'uint8')
    high = np.array([120,255,225],dtype = 'uint8')

    mask = cv2.inRange(hsv,low,high)

    erode = cv2.erode(mask, kernel1, iterations= 1)
    dilate = cv2.dilate(erode, kernel1, iterations= 1)
    dilate = cv2.dilate(dilate, kernel2, iterations= 3)
    
    # img2 = cv2.bitwise_and(frame, frame,mask = dilate)

    contours, hierarchy = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    if len(contours)>0:

        x,y,h,w = cv2.boundingRect(contours[0])
        # print(cv2.contourArea(contours[0]))
        x1= int(x+h/2+20)
        y1= int(y+(w/2))
        cv2.circle(frame, (x1, y1), 10, (0,0,255), -1)
        if paint:         
            l.append((x1,y1))
        
    else:
        pass


    if paint:         
        # print(l)
        for x2,y2 in l:
            cv2.circle(frame, (x2,y2), 5, (0,255,250), -1)
        
        if len(l)>1:
            for i in range(1, len(l)):
                cv2.line(frame, (l[i-1][0], l[i-1][1]),(l[i][0], l[i][1]), (0,127,23), 5)
        
    cv2.imshow('window',frame)

    # cv2.imshow('win',img2)
    # cv2.imshow('mask',dilate)
    
    if key == ord('q'):
        break

cap.release()
cv2.waitKey(3000)
cv2.destroyAllWindows()

