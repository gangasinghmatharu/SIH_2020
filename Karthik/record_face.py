import numpy as np
import cv2
import os
import time

cap = cv2.VideoCapture(0)
student = input("Enter Your Name")
os.mkdir('train_img/'+student)
i=0
while(True):
    i=i+1
    ret, frame = cap.read()
    # print('D:/PROJECTS/Facenet/train_img/Karthik/'+student+'_'+str(i)+'.jpg')
    # cv2.imshow('image',frame)
    # cv2.imwrite('D:/PROJECTS/Facenet/train_img/Karthik/'+student+'_'+str(i)+'.jpg', frame)   
    img_uint8 = frame.astype(np.uint8) 
    if not cv2.imwrite('train_img/'+student+'/'+ str(i)+'.jpg', img_uint8):
        print("ERROR")    
    time.sleep(0.2)
    cv2.imshow("Image",frame)
    if i==200: 
        break
print("ALL DONE")
cap.release()
cv2.destroyAllWindows()