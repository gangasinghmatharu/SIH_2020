from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import os
import time
from functions.preprocess import preprocesses
import sys
from functions.classifier import training
from numba import cuda

cuda.current_context().reset()
device = cuda.get_current_device()
device.reset()

# Constants
number_of_samples = 100
cap = cv2.VideoCapture(0)
student = input("Enter Your Name : ")
os.mkdir('./data/train_img/'+student)
i=0
input_datadir = './data/train_img'
output_datadir = './data/pre_img'
datadir = './data/pre_img'
modeldir = './facenet/model/20170511-185253.pb'
classifier_filename = './facenet/class/classifier.pkl'

# Record Images of New User And Store it In Entered Folder 
while(True):
    i=i+1
    ret, frame = cap.read()
    cv2.imshow("Image",frame)
    img_uint8 = frame.astype(np.uint8)
    if not cv2.imwrite('./data/train_img/'+student+'/'+ str(i)+'.jpg', img_uint8):
        print("ERROR")
    time.sleep(0.2)
    if i==number_of_samples: 
        break
print("Image Uploading Done !!!")
cap.release()
cv2.destroyAllWindows()

# Data Preprocessing
print("Data Prepocessing Started!!!")
obj=preprocesses(input_datadir,output_datadir)
nrof_images_total,nrof_successfully_aligned=obj.collect_data()
print("Data Preprocessing Done !!!")

# Image Training
print("Model Training Started !!!")
obj=training(datadir,modeldir,classifier_filename)
get_file=obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")