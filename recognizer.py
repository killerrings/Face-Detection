from cv2 import cv2
import numpy as np
import os 
import pickle

face_cascade=cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0+cv2.CAP_DSHOW)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]

        #Recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf>=50:          
            font = cv2.FONT_HERSHEY_SIMPLEX            
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                        
        img_item = "7.jpg"
        cv2.imwrite(img_item, roi_gray)
      
        color=(0,0,255)
        stroke=2
        start_cord = x + w
        end_cord = y + h
        cv2.rectangle(frame, (x,y), (start_cord,end_cord), color,stroke)
        subitems = smile_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in subitems:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #Display the resulting frame
    cv2.imshow('face recognition',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#When everything is finally done release the capture
cap.release()
cv2.destroyAllWindows()
