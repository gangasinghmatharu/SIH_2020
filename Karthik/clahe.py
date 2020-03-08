import numpy as np
import cv2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imshow('Clahe',bgr)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

