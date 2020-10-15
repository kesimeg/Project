
from skimage import transform as tf
import numpy as np

import dlib
import cv2
import os
from skimage.transform import rotate

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def img_rotate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    landmark_list = []

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)

        x = rects[0].left()
        y = rects[0].top()
        w = rects[0].right()
        h = rects[0].bottom()

        left_eye = []
        right_eye = []

        for n in range(0,shape.num_parts):
            if n >=36 and n<=41:
                left_eye.append([shape.part(n).x,shape.part(n).y])
            elif n >=42 and n<=47:
                right_eye.append([shape.part(n).x,shape.part(n).y])

        left_eye = np.array(left_eye)
        right_eye = np.array(right_eye)


        left_eye_center = left_eye.mean(axis=0)
        right_eye_center = right_eye.mean(axis=0)

        lx,ly=left_eye_center[0],left_eye_center[1]
        rx,ry=right_eye_center[0],right_eye_center[1]

        degree = (ry-ly)/(rx-lx)

        angle = np.degrees(np.arctan2(ry-ly, rx-lx)) #- 180
        img = rotate(img,angle)

    return img

def landmark_extract(images):
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    landmark_list = []

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)

        left_eye = []
        right_eye = []
        for n in range(0,shape.num_parts):
            if n >=36 and n<=41:
                left_eye.append([shape.part(n).x,shape.part(n).y])
            elif n >=42 and n<=47:
                right_eye.append([shape.part(n).x,shape.part(n).y])
            if n==33:
                landmark_list.append([shape.part(n).x,shape.part(n).y])
        left_eye = np.array(left_eye)
        right_eye = np.array(right_eye)


        left_eye_center = left_eye.mean(axis=0)
        right_eye_center = right_eye.mean(axis=0)
        landmark_list.append(left_eye_center)
        landmark_list.append(right_eye_center)
        x = rects[0].left()
        y = rects[0].top()
        w = rects[0].right()
        h = rects[0].bottom()

        return np.array(landmark_list),x,y,w,h


video_dir = "../data/video_frames"
video_dir_folders = os.listdir(video_dir)
video_dir_folders.remove("1047_IEO_FEA_LO.flv")
video_dir_folders.remove("1047_IEO_SAD_LO.flv")
video_dir_folders.remove("1076_MTI_SAD_XX.flv")
video_dir_folders.sort()

num_skipped = 0

#1047_IEO_FEA_LO iter 3766 (siyah kareler var)
#1047_IEO_SAD_LO.flv 3773 (siyah kaereler var)
# 1076_MTI_SAD_XX.flv iter 6187 (yazı var yüz yok yanlış)
# error = [1047_IEO_FEA_LO.flv,1047_IEO_SAD_LO.flv,1076_MTI_SAD_XX.flv]

for i in range(0,len(video_dir_folders)):
    folder = video_dir_folders[i]
    print("folder name:{} iter:{}".format(folder,i))
    folder_dir = os.path.join(video_dir,folder)

    folder_name = folder.split(".")[0]

    #none yada yüz bulunamazsa devam et

    images1 = cv2.imread(os.path.join(folder_dir,"image-1.png"))
    print(os.path.join(folder_dir,"image-1.png"))
    images1 = img_rotate(images1)*255
    images1 = images1.astype(np.uint8)

    src,x,y,w,h = landmark_extract(images1)

    write_dir = os.path.join("../data/video/cropped_face_frames2",folder_name)

    img_num = len(os.listdir(folder_dir))
    if os.path.exists(write_dir)==False:
        os.mkdir(write_dir)

    cv2.imwrite(os.path.join(write_dir,"image-1.jpeg"),images1[y-20:h+20,x-20:w+20])


    for j in range(1,img_num+1):
        #print(i)

        #none ise yada yüz yoksa devam et
        images2 = cv2.imread(os.path.join(folder_dir,"image-{}.png".format(j)))
        if j==1:
            dst,_,_,_,_ = landmark_extract(images2)

        tform = tf.estimate_transform('similarity', src,dst)  # find the transformation matrix
        warped = tf.warp(images2, tform, output_shape=images2.shape)

        warped_numpy = np.array(warped*255).astype(np.uint8)

        cv2.imwrite(os.path.join(write_dir,"image-{}.jpeg".format(j)),warped_numpy[y-20:h+20,x-20:w+20]) #orijinal x,y,w,h'ı kullanmak doğru mu?????
