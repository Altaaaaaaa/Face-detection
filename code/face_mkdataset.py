#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ') # 숫자만 가능
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    faces = face_detector.detectMultiScale(gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20))
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        #User 이름 폴더 안에 개수만큼 저장
        #cv2.imwrite("dataset/" + str(face_id) + '/User_' + str(face_id) + str(count) + ".jpg", gray[y:y+h,x:x+w]) 
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()