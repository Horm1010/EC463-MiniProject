import cv2
import numpy as np

capture = cv2.VideoCapture('cars.mp4')
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:

    ret, frame = capture.read()

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray_scale, 1.1, 3)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)      

    cv2.imshow('video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

capture.release()
