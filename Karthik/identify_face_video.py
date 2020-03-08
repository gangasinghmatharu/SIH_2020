from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
from skimage.transform import resize
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
# import dlib
import posenet
from numba import cuda
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
count = 0
ear=0.3
eye_thr=0
x=0
count1=0
count2=0
st=[]
temp1=[]
start1 = []
def eye_aspect_ratio(eye):
        A=dist.euclidean(eye[1], eye[5])
        B=dist.euclidean(eye[2], eye[4])
        C=dist.euclidean(eye[0], eye[3])
        ear = (A+B)/(2.0*C)
        return ear    

def clahe_converted(frame_img):
    lab = cv2.cvtColor(frame_img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3,allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        
        model_cfg, model_outputs = posenet.load_model(50, sess)
        output_stride = model_cfg['output_stride']
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  
        threshold = [0.6, 0.7, 0.7] 
        factor = 0.709
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        video_capture = cv2.VideoCapture(0)
        # video_capture.set(cv2.CV_CAP_PROP_FPS, 30)
        c = 0


        print('Start Recognition')
        prevTime = 0
        while True:
            black_image=np.zeros((720,1280,3),np.uint8)
            ret, frame = video_capture.read()
            frame = clahe_converted(frame)
            orginal = frame
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)  

            curTime = time.time()+1   
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                print(bounding_boxes)
                nrof_faces = bounding_boxes.shape[0]
                # print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(resize(cropped[i], (image_size, image_size)))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        
                        

                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        # print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # print(best_class_indices,' with accuracy ',best_class_probabilities)

                        if best_class_probabilities>0.53:
                            
                            # POSENET STARTING 
                            # frame = cv2.resize(cropped[i],(1280,720))
                            frame = cropped[i]
                            input_image, display_image, output_scale = posenet.single_frame(frame,scale_factor=0.7125, output_stride=output_stride)
                            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                                model_outputs,feed_dict={'image:0': input_image})
                            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                                heatmaps_result.squeeze(axis=0),
                                offsets_result.squeeze(axis=0),
                                displacement_fwd_result.squeeze(axis=0),
                                displacement_bwd_result.squeeze(axis=0),
                                output_stride=output_stride,
                                max_pose_detections=10,
                                min_pose_score=0.15)
                            keypoint_coords *= output_scale
                            overlay_image = posenet.draw_skel_and_kp(
                                frame, pose_scores, keypoint_scores, keypoint_coords,
                                min_pose_score=0.15, min_part_score=0.1)
                            cv2.imshow(HumanNames[best_class_indices[0]] ,overlay_image )
                            
            
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2) 
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            # print('Result Indices: ', best_class_indices[0])
                            # print(HumanNames)


                            # Each Person
                            
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Alignment Failure')
            
            
            

            cv2.imshow('Video', orginal)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        device = cuda.get_current_device()
        device.reset()
