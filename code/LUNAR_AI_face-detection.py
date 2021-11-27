#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from PIL import Image
import os

face_detector = cv2.CascadeClassifier("./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml") # 경로 주의
recognizer = cv2.face.LBPHFaceRecognizer_create()

def datalrn(image_path):

    def data_learning(image_path):
        face_dataset =  []
        face_id = int(input('\n enter user id end press <return> ==>  ')) # 숫자만 가능
        imagePaths = [f'{image_path}/{f}' for f in os.listdir(image_path)]

        for imagePath in imagePaths:
            img = Image.open(imagePath).convert('L')
            img_numpy = np.array(img,'uint8')
            faces = face_detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces: 
                # Save the captured image into the datasets folder
                #User 이름 폴더 안에 개수만큼 저장

                face_dataset.append(img_numpy[y:y+h,x:x+w])

        
        recognizer.train(face_dataset, np.array([face_id]*len(face_dataset)))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(face_id))))

    image_path1 = input('\n enter image path : ')
    data_learning(image_path1)


#datalrn("C:/Users/user/Desktop/JIONI/COSMIC/Face-detection/test/2113849/1234") #data 파일 있는 위치


def face_recog():
    recognizer.read('trainer/trainer.yml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    #iniciate id counter
    id = 0

    # names related to ids: example ==> None2: id=2,  etc
    # 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.
    names = ['None0', 'jw', 'jh', 'None3', 'None4']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img =cam.read()
        img = cv2.flip(img, 1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (100 - confidence) > 35 and (100 - confidence) <= 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

#image_path2 = input('\n enter image path : ')
face_recog()
