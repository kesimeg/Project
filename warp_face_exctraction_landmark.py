from skimage import transform as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import dlib
import cv2
import os
from skimage.transform import rotate
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import hm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def bounding_box(images):
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    landmark_list = []
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)


        x = rects[0].left()
        y = rects[0].top()
        w = rects[0].right()
        h = rects[0].bottom()

        return x,y,w,h


def all_landmarks(images):
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    landmark_list = []
    fixed_points = []
    try:
        rect = rects[0]

        shape = predictor(gray, rect)

        for n in range(0,shape.num_parts):
            landmark_list.append([shape.part(n).x,shape.part(n).y])
            if n >=27 and n<=45:
                fixed_points.append([shape.part(n).x,shape.part(n).y])
            if n>=0 and n<=3:
                fixed_points.append([shape.part(n).x,shape.part(n).y])
            if n>=13 and n<=16:
                fixed_points.append([shape.part(n).x,shape.part(n).y])
    except:
        pass
    return np.array(fixed_points),np.array(landmark_list)

def sliding_window1(arr1):
    arr1 = np.array(arr1)

    arr1_copy = arr1.copy()
    for i in range(len(arr1_copy)):
        if i==0 or i==len(arr1)-1:
            continue
        if i==1:
            arr1[i] = (arr1_copy[i-1]+arr1_copy[i]+arr1_copy[i+1])/3
        elif i+1==len(arr1_copy)-1:
            arr1[i] = (arr1_copy[i-1]+arr1_copy[i]+arr1_copy[i+1])/3
        else:
            arr1[i] = (arr1_copy[i-2:i+3].sum(axis=0))/5
    return arr1

def sliding_window2(arr1,arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    arr1_copy = arr1.copy()
    arr2_copy = arr2.copy()
    for i in range(len(arr1_copy)):
        if i==0 or i==len(arr1)-1:
            continue
        if i==1:
            arr1[i] = (arr1_copy[i-1]+arr1_copy[i]+arr1_copy[i+1])/3
            arr2[i] = (arr2[i-1]+arr2[i]+arr2[i+1])/3
        elif i+1==len(arr1_copy)-1:
            arr1[i] = (arr1_copy[i-1]+arr1_copy[i]+arr1_copy[i+1])/3
            arr2[i] = (arr2[i-1]+arr2[i]+arr2[i+1])/3
        else:
            arr1[i] = (arr1_copy[i-2:i+3].sum(axis=0))/5
            arr2[i] = (arr2[i-2:i+3].sum(axis=0))/5

    return arr1,arr2

video_dir = "video_frames"
video_dir_folders = os.listdir(video_dir)
try:
    video_dir_folders.remove("1047_IEO_FEA_LO.flv")
    video_dir_folders.remove("1047_IEO_SAD_LO.flv")
    video_dir_folders.remove("1076_MTI_SAD_XX.flv")
except:
    pass
video_dir_folders.sort()


num_skipped = 0

for i in tqdm(range(3840,len(video_dir_folders))):
    folder = video_dir_folders[i]
    print("folder name:{} iter:{}".format(folder,i))
    folder_dir = os.path.join(video_dir,folder)

    folder_name = folder.split(".")[0]

    write_img_dir = os.path.join("cropped_face_frames_land/imgs",folder_name)
    write_landmark_dir = os.path.join("cropped_face_frames_land/landmarks",folder_name)

    img_num = len(os.listdir(folder_dir))

    if os.path.exists(write_img_dir)==False:
        os.mkdir(write_img_dir)
    if os.path.exists(write_landmark_dir)==False:
        os.mkdir(write_landmark_dir)

    landmark_vid = []
    eye_landmarks = []

    for j in range(1,img_num+1):

        images = cv2.imread(os.path.join(folder_dir,"image-{}.png".format(j)))
        if j ==1:
            x,y,w,h = bounding_box(images)
        eyes,point_list = all_landmarks(images)
        if  point_list.size == 0:
            eye_landmarks.append(eye_landmarks[-1])
            landmark_vid.append(landmark_vid[-1])
        else:
            eye_landmarks.append(eyes)
            landmark_vid.append(point_list)

    landmark_vid,eye_landmarks = sliding_window2(landmark_vid,eye_landmarks)

    for j in range(0,27):
        eye_landmarks[:,j,0] = gaussian_filter1d(eye_landmarks[:,j,0] , sigma=5)
        eye_landmarks[:,j,1] = gaussian_filter1d(eye_landmarks[:,j,1] , sigma=5)

    transformed_landmark_list =[]
    warped_list = []

    x = landmark_vid[0,0,0]
    w = landmark_vid[0,16,0]
    h =landmark_vid[0,8,1]
    y = landmark_vid[0,27,1]

    for j in range(1,img_num+1):

        images = cv2.imread(os.path.join(folder_dir,"image-{}.png".format(j)))

        tform = tf.estimate_transform('similarity', eye_landmarks[0],eye_landmarks[j-1])  # find the transformation matrix
        warped = tf.warp(images, tform, output_shape=images.shape)

        transformed_landmark = tform.inverse(landmark_vid[j-1])
        transformed_landmark = np.array(transformed_landmark) #.astype(np.int64)

        warped = np.array(warped)*255
        warped = warped.astype(np.uint8)
        transformed_landmark[:,0]-=(x-30) #change coordinates to local
        transformed_landmark[:,1]-=(y-40)
        warped = warped[y-40:h+10,x-30:w+30]
        h_2, w_2 = warped.shape[:2]

        transformed_landmark*= np.array([128 / w_2, 128 / h_2])
        transformed_landmark_list.append(transformed_landmark)
        warped_list.append(warped)
    transformed_landmark_list = sliding_window1(transformed_landmark_list)

    for j in range(len(transformed_landmark_list)):

        landmarks = transformed_landmark_list[j].astype(np.int64) #np.round()
        warped = cv2.resize(warped_list[j],(128,128))
        """
        for k in range(68):
            cv2.circle(warped, (landmarks[k][0],landmarks[k][1]), 1, (0, 0, 255), -1)
        """
        cv2.imwrite(os.path.join(write_img_dir,"image-{}.jpg".format(j)),warped)
        heat_map = hm.generate_gtmap(landmarks,1,warped.shape)
        heat_map = heat_map.sum(axis=2)
        heat_map*=255
        #heat_map = cv2.resize(heat_map,(128,128))
        cv2.imwrite(os.path.join(write_landmark_dir,"image-{}.jpg".format(j)),heat_map)
    #if i == 5:
    #    break
